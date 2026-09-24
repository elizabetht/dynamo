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
                "probe-model",
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
                                x["id"] == "probe-model" for x in data.get("data", [])
                            ):
                                break
                    except aiohttp.ClientConnectorError:
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("discovery timeout")
                for name, flags in [
                    ("default", {}),
                    ("closed", {"add_generation_prompt": False}),
                    (
                        "continue",
                        {
                            "add_generation_prompt": False,
                            "continue_final_message": True,
                        },
                    ),
                ]:
                    for stream in [False, True]:
                        request = {
                            "model": "probe-model",
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
                            expected = tok.apply_chat_template(
                                request["messages"],
                                tokenize=True,
                                add_generation_prompt=flags.get(
                                    "add_generation_prompt", True
                                ),
                                continue_final_message=flags.get(
                                    "continue_final_message", False
                                ),
                            )
                            expected = getattr(expected, "input_ids", expected)
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
    assert len(rows) == 6 and len(preprocessed) == 6 and factory_calls
    assert all(r["status"] == 200 and r["dispatches"] == 1 for r in rows)
    for row in rows:
        assert row["worker_tokens_match"], row
        events = (
            [
                json.loads(line[6:])
                for line in row["body"].splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            if row["stream"]
            else [json.loads(row["body"])]
        )
        choices = [choice for event in events for choice in event.get("choices", [])]
        assert (
            "".join(
                choice.get("delta", choice.get("message", {})).get("content") or ""
                for choice in choices
            )
            == "ok"
        ), row
        assert [
            choice["finish_reason"] for choice in choices if choice.get("finish_reason")
        ] == ["stop"], row
        if row["stream"]:
            assert row["body"].count("data: [DONE]") == 1, row
        assert row["prompt_matches"] is (
            not args.expect_baseline or row["case"] == "default"
        ), row
    print(
        json.dumps(
            [
                {k: r[k] for k in ["case", "stream", "status", "prompt_matches"]}
                for r in rows
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local HTTP prompt-control replay with synthetic worker tokens; no model inference."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-baseline", action="store_true")
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), 180))
