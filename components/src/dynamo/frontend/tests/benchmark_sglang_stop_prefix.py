# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU HTTP stop-prefix benchmark with scripted tokens; no model weights or GPU."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

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


async def main(output, baseline_source):
    output.mkdir(parents=True, exist_ok=True)
    model = snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = [
        (f"prefix-{n}", ("a" * n + "x ") * 8, None, ("a" * n + "x ") * 8, "length")
        for n in [8, 64, 256, 1024]
    ]
    cases = [
        (name + "-" + shape, text, include_stop, expected, finish, extra)
        for name, text, include_stop, expected, finish in cases
        for shape, extra in [
            ("plain", {}),
            ("guided", {"guided_regex": text}),
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
    cases += [
        (
            "control-0-plain",
            "Hello world. " * 64,
            None,
            "Hello world. " * 64,
            "length",
            {},
        ),
        (
            "control-16-plain",
            "Hello world. " * 64,
            None,
            "Hello world. " * 64,
            "length",
            {},
        ),
    ]
    cases = [
        (name + f"-repeat-{repeat}-variant-{variant}", *rest)
        for repeat in range(-1, 3)
        for variant in (
            ["baseline", "candidate"] if repeat % 2 == 0 else ["candidate", "baseline"]
        )
        for name, *rest in cases
    ]
    spec = importlib.util.spec_from_file_location(
        "baseline_engine_response", baseline_source
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    variants = {
        "baseline": baseline.trailing_stop_prefix_len,
        "candidate": prepost_module.trailing_stop_prefix_len,
    }
    original = prepost_module.SglangStreamingPostProcessor.process_output
    call_times = []

    def profiled(self, response):
        before = time.perf_counter_ns()
        try:
            return original(self, response)
        finally:
            call_times.append(time.perf_counter_ns() - before)

    prepost_module.SglangStreamingPostProcessor.process_output = profiled
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
                    "finish_reason": None,
                }
                await asyncio.sleep(0)
            yield {"token_ids": [], "finish_reason": active["finish"]}
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

    with tempfile.TemporaryDirectory(
        prefix="stop-prefix-", dir=output
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
                    batch, interval, streaming = 1, 1, True
                    active.update(text=text, batch=batch, finish=finish)
                    processor.stream_interval = interval
                    request = {
                        **extra,
                        "model": "probe-model",
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "stop": (
                            []
                            if name.startswith("control-0")
                            else ["STOP", "</answer>"]
                            if name.startswith("control-16")
                            else [
                                "a" * int(name.split("-")[1]) + suffix
                                for suffix in ["b", "c", "d", "e"]
                            ]
                        ),
                        **(
                            {"include_stop_str_in_output": include_stop}
                            if include_stop is not None
                            else {}
                        ),
                        "max_tokens": 4096,
                        "stream": streaming,
                    }
                    prepost_module.trailing_stop_prefix_len = variants[
                        name.split("-variant-")[-1]
                    ]
                    if name.startswith("control-0"):
                        request.pop("stop")
                    call_times.clear()
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
                            if line.startswith("data: ") and line != "data: [DONE]"
                        ]
                        if streaming
                        else [json.loads(body)]
                    )
                    choices = [c for event in events for c in event.get("choices", [])]
                    content = "".join(
                        c.get("delta", c.get("message", {})).get("content") or ""
                        for c in choices
                    )
                    finishes = [
                        c["finish_reason"] for c in choices if c.get("finish_reason")
                    ]
                    row = {
                        "process_output_calls": len(call_times),
                        "process_output_ns": sum(call_times),
                        "case": name,
                        "batch": batch,
                        "interval": interval,
                        "stream": streaming,
                        "status": status,
                        "body": body,
                        "content": content,
                        "finishes": finishes,
                        "worker_request": calls[-1] if len(calls) > before else None,
                        "passed": status == 200
                        and content == expected
                        and finishes == [finish],
                    }
                    assert len(calls) == before + 1 and len(finalized) == len(calls), (
                        name,
                        status,
                        body,
                        len(calls),
                        before,
                        len(finalized),
                    )
                    assert not streaming or "data: [DONE]" in body
                    assert row["passed"], row
                    if "-repeat--1-" not in name:
                        rows.append(row)
        finally:
            runtime.shutdown()
            for task in tasks:
                try:
                    await asyncio.wait_for(task, 10)
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            prepost_module.SglangStreamingPostProcessor.process_output = original
            prepost_module.trailing_stop_prefix_len = variants["candidate"]
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
    assert len(rows) == 84
    failed = [row for row in rows if not row["passed"]]
    assert not failed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(main(args.output, args.baseline_source), 240))
