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
            {
                "request": copy.deepcopy(request),
                "tokens": result.prompt_token_ids,
                "guidance": result.guided_decoding,
            }
        )
        return result

    cases = {
        "two_closed": ' \n[{"name": "weather", "parameters": {"city": "東京 😀"}}, {"name": "weather", "parameters": {"city": "Rome"}}',
        "partial_second": '[{"name": "weather", "parameters": {"city": "Paris"}}, {"name": "weather", "parameters": {"city": "Ro',
        "complete": '[{"name": "weather", "parameters": {"city": "Paris"}}]',
    }

    async def worker(request, context):
        dispatched.append(request)
        text = cases[case_id]
        ids = tok.encode(text, add_special_tokens=False)
        for index, token in enumerate(ids):
            yield {
                "token_ids": [token],
                "finish_reason": "length" if index == len(ids) - 1 else None,
            }

    async def factory(instance, card, routed):
        factory_calls.append(True)
        obj = processor.SglangProcessor(
            tok,
            routed,
            "hermes",
            None,
            processor._model_eos_token_ids(tok, model),
            stream_interval=args.interval,
        )
        return PythonAsyncEngine(obj.generator, asyncio.get_running_loop())

    with patch.object(
        processor, "preprocess_chat_request", observed
    ), tempfile.TemporaryDirectory(
        dir=root, prefix="parallel-"
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
            ns = "parallel-" + uuid.uuid4().hex
            endpoint = runtime.endpoint(ns + ".worker.generate")
            tasks.append(asyncio.ensure_future(endpoint.serve_endpoint(worker)))
            await register_model(
                ModelInput.Tokens,
                ModelType.Chat,
                endpoint,
                model,
                "parallel-probe",
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
                                x["id"] == "parallel-probe"
                                for x in data.get("data", [])
                            ):
                                break
                    except aiohttp.ClientConnectorError:
                        # The task-owned HTTP listener may not have bound yet.
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("discovery timeout")
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ]
                for case_id in cases:
                    choice = "required"
                    for flag in [True]:
                        for stream in [False, True]:
                            request = {
                                "model": "parallel-probe",
                                "messages": [
                                    {"role": "user", "content": "Get weather"}
                                ],
                                "tools": tools,
                                "tool_choice": choice,
                                "stream": stream,
                                "max_tokens": 64,
                            }
                            if flag != "omitted":
                                request["parallel_tool_calls"] = flag
                            before = len(preprocessed)
                            before_dispatch = len(dispatched)
                            async with session.post(
                                f"http://127.0.0.1:{port}/v1/chat/completions",
                                json=request,
                            ) as response:
                                status = response.status
                                body = await response.text()
                            row = {
                                "case": case_id,
                                "expected_text": cases[case_id],
                                "choice": choice,
                                "parallel": flag,
                                "stream": stream,
                                "status": status,
                                "body": body,
                                "dispatches": len(dispatched) - before_dispatch,
                            }
                            if len(preprocessed) > before:
                                row["python_request"] = preprocessed[-1]["request"]
                                row["guidance"] = preprocessed[-1]["guidance"]
                                row["worker_sampling"] = (
                                    dispatched[-1]["sampling_options"]
                                    if len(dispatched) > before_dispatch
                                    else None
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
    assert len(rows) == 6
    assert all(r["status"] == 200 and r["dispatches"] == 1 for r in rows)
    summary = []
    for row in rows:
        events = (
            [
                json.loads(line[6:])
                for line in row["body"].splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            if row["stream"]
            else [json.loads(row["body"])]
        )
        calls = []
        finishes = []
        for event in events:
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
                message = choice.get("delta", choice.get("message", {}))
                calls.extend(message.get("tool_calls") or [])
        assert finishes == ["length"], finishes
        content = "".join(
            choice.get("delta", choice.get("message", {})).get("content") or ""
            for event in events
            for choice in event.get("choices", [])
        )
        passed = (
            (
                len(calls) == 1
                and calls[0]["function"]["name"] == "weather"
                and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
            )
            if row["case"] == "complete"
            else (not calls and content == row["expected_text"])
        )
        summary.append(
            {
                "case": row["case"],
                "stream": row["stream"],
                "calls": calls,
                "content": content,
                "finishes": finishes,
                "passed": passed,
            }
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    assert all(row["passed"] for row in summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local HTTP truncated-array replay with synthetic worker tokens; no model inference."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, choices=[1, 20, 1024], required=True)
    asyncio.run(asyncio.wait_for(main(parser.parse_args()), 180))
