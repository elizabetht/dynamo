# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check literal U+FFFD delivery before worker finish over native localhost HTTP."""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import tempfile
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


async def main(output):
    output.mkdir(parents=True, exist_ok=True)
    model = snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    cases = [
        (name, text, ids, extra)
        for name, text, ids in [
            ("literal", "�", tokenizer.encode("�", add_special_tokens=False)),
            ("prefix-literal", "A�", tokenizer.encode("A�", add_special_tokens=False)),
            ("double-literal", "��", tokenizer.encode("��", add_special_tokens=False)),
            ("ordinary", "Hello", tokenizer.encode("Hello", add_special_tokens=False)),
            ("incomplete", "�", tokenizer.encode("있다", add_special_tokens=False)[:-1]),
        ]
        for extra in [{}, {"guided_regex": ".*"}]
    ]
    release = asyncio.Event()
    captured = asyncio.Future()
    active = {}
    calls = []
    rows = []

    async def worker(request, context):
        calls.append(request)
        yield {"token_ids": active["ids"], "finish_reason": None}
        try:
            await asyncio.wait_for(release.wait(), 0.5)
            active["released_by_client"] = True
        except asyncio.TimeoutError:
            active["released_by_client"] = False
        yield {"token_ids": [], "finish_reason": "length"}

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
        prefix="literal-streaming-", dir=output
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
            namespace = "literal-streaming-" + uuid.uuid4().hex
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
            await asyncio.wait_for(captured, 30)
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
                for name, text, ids, extra in cases:
                    release.clear()
                    active.clear()
                    active.update(ids=ids)
                    request = {
                        **extra,
                        "model": "probe-model",
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 128,
                        "stream": True,
                    }
                    events = []
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", json=request
                    ) as response:
                        status = response.status
                        async for line in response.content:
                            if (
                                not line.startswith(b"data: ")
                                or line.strip() == b"data: [DONE]"
                            ):
                                continue
                            event = json.loads(line[6:])
                            events.append(event)
                            if any(
                                c.get("delta", {}).get("content")
                                for c in event.get("choices", [])
                            ):
                                release.set()
                    choices = [c for e in events for c in e.get("choices", [])]
                    content = "".join(
                        c.get("delta", {}).get("content") or "" for c in choices
                    )
                    finishes = [
                        c["finish_reason"] for c in choices if c.get("finish_reason")
                    ]
                    row = dict(
                        case=name,
                        guided=bool(extra),
                        status=status,
                        content=content,
                        events=events,
                        worker_request=calls[-1],
                        released_by_client=active["released_by_client"],
                        passed=status == 200
                        and content == text
                        and finishes == ["length"],
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
    assert len(rows) == len(cases)
    failed = [row for row in rows if not row["passed"]]
    assert not failed, failed
    assert all(
        row["released_by_client"] == (row["case"] != "incomplete") for row in rows
    ), rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(main(args.output), 120))
