# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU HTTP stop-inclusion regression with scripted tokens; no model weights or GPU."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path

import aiohttp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import dynamo._core
import dynamo.frontend.sglang_prepost as prepost_module
import dynamo.frontend.sglang_processor as processor_module
from dynamo.frontend.sglang_processor import SglangProcessor
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


async def main(output, expect_baseline_failure):
    output.mkdir(parents=True, exist_ok=True)
    model = snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = [
        ("included", "Hi STOP tail", True, "Hi STOP", "stop"),
        ("excluded", "Hi STOP tail", False, "Hi ", "stop"),
        ("omitted", "Hi STOP tail", None, "Hi ", "stop"),
    ]
    cases = [
        (name + "-" + shape, text, include_stop, expected, finish, extra)
        for name, text, include_stop, expected, finish in cases
        for shape, extra in [
            ("plain", {}),
            ("guided", {"guided_regex": "Hi STOP tail"}),
            (
                "tools",
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "greet",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                },
            ),
        ]
    ]
    captured = asyncio.Future()
    active = {}
    calls = []
    finalized = []
    rows = []

    async def worker(request, context):
        calls.append(request)
        ids = tokenizer.encode(active["text"], add_special_tokens=False)
        batch = active["batch"]
        try:
            for start in range(0, len(ids), batch):
                if context.is_stopped() or context.is_killed():
                    return
                yield {
                    "token_ids": ids[start : start + batch],
                    "finish_reason": "length" if start + batch >= len(ids) else None,
                }
                await asyncio.sleep(0.01)
        finally:
            finalized.append(True)

    async def factory(instance, card, routed):
        processor = SglangProcessor(
            tokenizer,
            routed,
            "hermes",
            None,
            processor_module._model_eos_token_ids(tokenizer, model),
            stream_interval=1,
        )
        captured.set_result(processor)
        return PythonAsyncEngine(processor.generator, asyncio.get_running_loop())

    with tempfile.TemporaryDirectory(prefix="stop-inclusion-", dir=output) as discovery:
        os.environ["DYN_FILE_KV"] = discovery
        os.environ["DYN_ROUTER_MIN_INITIAL_WORKERS"] = "1"
        os.environ["DYN_TCP_RPC_HOST"] = "127.0.0.1"
        runtime = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        tasks = []
        try:
            namespace = "stop-inclusion-" + uuid.uuid4().hex
            endpoint = runtime.endpoint(namespace + ".worker.generate")
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
            args = EntrypointArgs(
                EngineType.Dynamic,
                http_host="127.0.0.1",
                http_port=port,
                namespace=namespace,
                chat_engine_factory=factory,
                migration_limit=0,
            )
            engine = await make_engine(runtime, args)
            tasks.append(asyncio.ensure_future(run_input(runtime, "http", engine)))
            processor = await asyncio.wait_for(captured, 30)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                for _ in range(100):
                    async with session.get(
                        f"http://127.0.0.1:{port}/v1/models"
                    ) as response:
                        payload = await response.json()
                        if any(
                            m["id"] == "probe-model" for m in payload.get("data", [])
                        ):
                            break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("Model registration timed out")
                for name, text, include_stop, expected, finish, extra in cases:
                    for batch in [1, 7]:
                        active.update(text=text, batch=batch)
                        for interval in [1, 20]:
                            processor.stream_interval = interval
                            for streaming in [True, False]:
                                request = {
                                    **extra,
                                    "model": "probe-model",
                                    "messages": [
                                        {"role": "user", "content": "Say hello."}
                                    ],
                                    "stop": ["STOP"],
                                    **(
                                        {"include_stop_str_in_output": include_stop}
                                        if include_stop is not None
                                        else {}
                                    ),
                                    "max_tokens": 64,
                                    "stream": streaming,
                                }
                                before = len(calls)
                                async with session.post(
                                    f"http://127.0.0.1:{port}/v1/chat/completions",
                                    json=request,
                                ) as response:
                                    body = await response.text()
                                    status = response.status
                                for _ in range(100):
                                    if len(finalized) == len(calls):
                                        break
                                    await asyncio.sleep(0.01)
                                events = (
                                    [
                                        json.loads(line[6:])
                                        for line in body.splitlines()
                                        if line.startswith("data: ")
                                        and line != "data: [DONE]"
                                    ]
                                    if streaming
                                    else [json.loads(body)]
                                )
                                choices = [
                                    c
                                    for event in events
                                    for c in event.get("choices", [])
                                ]
                                content = "".join(
                                    c.get("delta", c.get("message", {})).get("content")
                                    or ""
                                    for c in choices
                                )
                                finishes = [
                                    c["finish_reason"]
                                    for c in choices
                                    if c.get("finish_reason")
                                ]
                                row = {
                                    "case": name,
                                    "batch": batch,
                                    "interval": interval,
                                    "stream": streaming,
                                    "status": status,
                                    "body": body,
                                    "content": content,
                                    "finishes": finishes,
                                    "worker_request": calls[-1]
                                    if len(calls) > before
                                    else None,
                                    "passed": status == 200
                                    and content == expected
                                    and finishes == [finish],
                                }
                                assert len(calls) == before + 1 and len(
                                    finalized
                                ) == len(calls)
                                assert not streaming or "data: [DONE]" in body
                                rows.append(row)
        finally:
            runtime.shutdown()
            for task in tasks:
                try:
                    await asyncio.wait_for(task, 10)
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            (output / "results.json").write_text(
                json.dumps({"rows": rows, "model": model}, indent=2)
            )
            (output / "imports.json").write_text(
                json.dumps(
                    {
                        m.__name__: {
                            "path": m.__file__,
                            "sha256": hashlib.sha256(
                                Path(m.__file__).read_bytes()
                            ).hexdigest(),
                        }
                        for m in [dynamo._core, processor_module, prepost_module]
                    },
                    indent=2,
                )
            )
    assert len(rows) == 72
    failed = [row for row in rows if not row["passed"]]
    if expect_baseline_failure:
        assert len(failed) == 24 and all(
            row["case"].startswith("included-") for row in failed
        )
    else:
        assert not failed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-baseline-failure", action="store_true")
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(main(args.output, args.expect_baseline_failure), 150))
