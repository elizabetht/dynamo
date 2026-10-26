# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import copy
import hashlib
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import aiohttp
from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
from transformers import AutoTokenizer

import dynamo._core
import dynamo.frontend.sglang_prepost as prepost
import dynamo.frontend.sglang_processor as processor
from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    ModelInput,
    ModelType,
    PythonAsyncEngine,
    WorkerType,
    make_engine,
    register_model,
    run_input,
)
from dynamo.runtime import DistributedRuntime


async def main(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    model = str(args.model_path)
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert tok.chat_template is None
    preprocessed = []
    dispatched = []
    rows = []
    factory_calls = []

    def observed(request, **kwargs):
        result = prepost.preprocess_chat_request(request, **kwargs)
        preprocessed.append(
            {"request": copy.deepcopy(request), "tokens": result.prompt_token_ids}
        )
        return result

    async def worker(request, context):
        dispatched.append(request)
        yield {
            "token_ids": tok.encode("ok", add_special_tokens=False),
            "finish_reason": "stop",
        }

    async def factory(instance, card, routed):
        factory_calls.append(True)
        obj = processor.SglangProcessor(
            tok,
            routed,
            "hermes",
            None,
            processor._model_eos_token_ids(tok, model),
            stream_interval=1,
        )
        return PythonAsyncEngine(obj.generator, asyncio.get_running_loop())

    with patch.object(
        processor, "preprocess_chat_request", observed
    ), tempfile.TemporaryDirectory(
        dir=root, prefix="continuation-"
    ) as discovery, patch.dict(
        os.environ,
        {
            "DYN_FILE_KV": discovery,
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "1",
            "DYN_TCP_RPC_HOST": "127.0.0.1",
        },
    ):
        runtime = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        tasks = []
        try:
            ns = "continuation-" + uuid.uuid4().hex
            endpoint = runtime.endpoint(ns + ".worker.generate")
            tasks.append(asyncio.ensure_future(endpoint.serve_endpoint(worker)))
            await register_model(
                ModelInput.Tokens,
                ModelType.Chat,
                endpoint,
                model,
                "deepseek-v4-probe",
                worker_type=WorkerType.Aggregated,
                ignore_weights=True,
                kv_cache_block_size=16,
            )
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            entrypoint_args = EntrypointArgs(
                EngineType.Dynamic,
                http_host="127.0.0.1",
                http_port=port,
                namespace=ns,
                migration_limit=0,
                chat_engine_factory=factory,
            )
            engine = await make_engine(runtime, entrypoint_args)
            tasks.append(asyncio.ensure_future(run_input(runtime, "http", engine)))
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                for _ in range(100):
                    try:
                        async with session.get(
                            f"http://127.0.0.1:{port}/v1/models"
                        ) as response:
                            data = await response.json()
                            if any(
                                x["id"] == "deepseek-v4-probe"
                                for x in data.get("data", [])
                            ):
                                break
                    except aiohttp.ClientConnectorError:
                        # The task-owned HTTP listener may not have bound yet.
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("discovery timeout")
                for name, flags in [
                    (
                        "continuation_chat",
                        {
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                        },
                    ),
                    (
                        "continuation_thinking",
                        {
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                            "chat_template_kwargs": {"thinking": True},
                        },
                    ),
                ]:
                    for stream in [False, True]:
                        request = {
                            "model": "deepseek-v4-probe",
                            "messages": [
                                {"role": "user", "content": "Return JSON"},
                                {"role": "assistant", "content": '{"city": "'},
                            ],
                            "response_format": {"type": "json_object"},
                            "max_tokens": 8,
                            "stream": stream,
                            **flags,
                        }
                        before = len(preprocessed)
                        before_dispatch = len(dispatched)
                        async with session.post(
                            f"http://127.0.0.1:{port}/v1/chat/completions", json=request
                        ) as response:
                            status = response.status
                            body = await response.text()
                        row = {
                            "case": name,
                            "stream": stream,
                            "status": status,
                            "body": body,
                            "dispatches": len(dispatched) - before_dispatch,
                        }
                        if len(preprocessed) > before:
                            item = preprocessed[-1]
                            expected_prompt = encode_messages(
                                request["messages"][:-1],
                                thinking_mode="thinking"
                                if flags.get("chat_template_kwargs", {}).get("thinking")
                                else "chat",
                            )
                            expected = tok.encode(expected_prompt)
                            suffix = tok.encode(request["messages"][-1]["content"])
                            if suffix and suffix[0] == tok.bos_token_id:
                                suffix = suffix[1:]
                            expected += suffix
                            row.update(
                                python_request=item["request"],
                                actual_prompt=tok.decode(item["tokens"]),
                                expected_prompt=tok.decode(expected),
                                prompt_matches=item["tokens"] == list(expected),
                                worker_tokens_match=dispatched[-1]["token_ids"]
                                == item["tokens"],
                            )
                        rows.append(row)
        finally:
            runtime.shutdown()
            for task in tasks:
                try:
                    await asyncio.wait_for(task, 10)
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            result = {
                "rows": rows,
                "factory_calls": len(factory_calls),
                "preprocess_calls": len(preprocessed),
                "imports": {
                    m.__name__: {
                        "path": m.__file__,
                        "sha256": hashlib.sha256(
                            Path(m.__file__).read_bytes()
                        ).hexdigest(),
                    }
                    for m in [dynamo._core, prepost, processor]
                },
            }
            (root / "http.json").write_text(json.dumps(result, indent=2))
    assert len(rows) == 4
    assert all(r["status"] == 200 for r in rows)
    assert all("data: [DONE]" in r["body"] for r in rows if r["stream"])
    assert all(r["dispatches"] == 1 for r in rows)
    assert all(r["worker_tokens_match"] for r in rows)
    assert all(r["prompt_matches"] != args.expect_baseline for r in rows)
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local HTTP prompt-control replay with synthetic worker tokens; no model inference."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-baseline", action="store_true")
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), 180))
