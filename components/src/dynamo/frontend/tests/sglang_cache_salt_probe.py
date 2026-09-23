# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU localhost HTTP cache-salt probe with a synthetic SGLang token worker.

Requires Dynamo bindings, SGLang frontend dependencies, and a cached Qwen3-0.6B
tokenizer. No weights are loaded. See the frontend README for commands.
"""

import argparse
import asyncio
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path

import aiohttp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

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
from dynamo.sglang.request_utils import request_cache_salt

CASES = [
    ("control", {}, {}, None),
    ("canonical", {"nvext": {"cache_salt": "tenant-b"}}, {}, "tenant-b"),
    (
        "precedence",
        {"cache_salt": "tenant-a", "nvext": {"cache_salt": "tenant-b"}},
        {},
        "tenant-b",
    ),
    (
        "header",
        {"nvext": {"cache_salt": "tenant-b"}},
        {"x-tenant-id": " tenant-c "},
        "tenant-c",
    ),
    (
        "guided",
        {
            "nvext": {"cache_salt": "tenant-a"},
            "response_format": {"type": "json_object"},
        },
        {},
        "tenant-a",
    ),
    (
        "tool",
        {
            "nvext": {"cache_salt": "tenant-tools"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        },
        {},
        "tenant-tools",
    ),
    ("empty", {"nvext": {"cache_salt": ""}}, {}, None),
]


async def main(run: Path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    model = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    captured = asyncio.Future()
    calls = []
    rows = []
    finalized = []

    async def worker(request, context):
        calls.append(request)
        try:
            ids = tokenizer.encode("OK", add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            for position, token_id in enumerate(ids):
                yield {
                    "token_ids": [token_id],
                    "finish_reason": "stop" if position == len(ids) - 1 else None,
                }
                await asyncio.sleep(0)
        finally:
            finalized.append(request)

    async def factory(instance, card, routed):
        processor = SglangProcessor(
            tokenizer=tokenizer,
            routed_engine=routed,
            tool_call_parser_name="hermes",
            reasoning_parser_name=None,
            eos_token_ids=processor_module._model_eos_token_ids(tokenizer, model),
            stream_interval=1,
        )
        if not captured.done():
            captured.set_result(processor)
        return PythonAsyncEngine(processor.generator, asyncio.get_running_loop())

    with tempfile.TemporaryDirectory(prefix="cache-salt-", dir=run) as discovery:
        os.environ["DYN_FILE_KV"] = discovery
        os.environ["DYN_ROUTER_MIN_INITIAL_WORKERS"] = "1"
        os.environ["DYN_TCP_RPC_HOST"] = "127.0.0.1"
        rt = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        tasks = []
        try:
            namespace = "cache-salt-" + uuid.uuid4().hex
            endpoint = rt.endpoint(namespace + ".worker.generate")
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
            engine = await make_engine(rt, args)
            tasks.append(asyncio.ensure_future(run_input(rt, "http", engine)))
            processor = await asyncio.wait_for(captured, 30)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                for attempt in range(100):
                    async with session.get(
                        f"http://127.0.0.1:{port}/v1/models"
                    ) as response:
                        payload = await response.json()
                        if any(
                            m.get("id") == "probe-model"
                            for m in payload.get("data", [])
                        ):
                            break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("Model registration timeout")
            for name, fields, headers, expected in CASES:
                for interval in [1, 20]:
                    processor.stream_interval = interval
                    for streaming in [True, False]:
                        request = {
                            "model": "probe-model",
                            "messages": [{"role": "user", "content": "Reply briefly."}],
                            "chat_template_kwargs": {"enable_thinking": False},
                            "max_completion_tokens": 24,
                            **fields,
                            "stream": streaming,
                        }
                        before = len(calls)
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=15)
                        ) as session:
                            async with session.post(
                                f"http://127.0.0.1:{port}/v1/chat/completions",
                                json=request,
                                headers=headers,
                            ) as response:
                                status = response.status
                                body = await response.text()
                        await asyncio.sleep(0.1)
                        assert status == 200 and len(calls) == before + 1, body
                        actual = request_cache_salt(calls[-1])
                        rows.append(
                            {
                                "case": name,
                                "interval": interval,
                                "stream": streaming,
                                "expected_salt": expected,
                                "actual_salt": actual,
                                "passed": actual == expected,
                            }
                        )
                        (run / "results.json").write_text(
                            json.dumps(rows, indent=2) + "\n"
                        )
                        assert actual == expected, rows[-1]
                        assert calls[-1]["routing"].get("cache_salt") == expected
                        assert (
                            calls[-1]
                            .get("extra_args", {})
                            .get("nvext", {})
                            .get("cache_salt")
                            == expected
                        )
                        assert len(finalized) == len(calls)
        finally:
            rt.shutdown()
            for task in tasks:
                try:
                    await asyncio.wait_for(task, 10)
                except (Exception, asyncio.CancelledError):
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    (run / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} native HTTP cache-salt checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    asyncio.run(asyncio.wait_for(main(run), 150))
