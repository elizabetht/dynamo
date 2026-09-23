# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU localhost HTTP logit-bias probe with a synthetic SGLang token worker.

Requires Dynamo bindings, SGLang frontend dependencies, and a cached Qwen3-0.6B
tokenizer. No weights are loaded. See the frontend README for commands.
"""

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from huggingface_hub import snapshot_download
from sglang.srt.sampling.sampling_params import SamplingParams
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
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

CASES = (
    ("control", {}, {}, None),
    ("positive", {"logit_bias": {"32": 100}}, {}, {"32": 100}),
    ("negative", {"logit_bias": {"32": -100}}, {}, {"32": -100}),
    (
        "guided",
        {"logit_bias": {"32": -100}, "response_format": {"type": "json_object"}},
        {},
        {"32": -100},
    ),
    ("empty", {"logit_bias": {}}, {}, {}),
    ("invalid_range", {"logit_bias": {"32": 101}}, {}, "rejected"),
)


async def main(run: Path, baseline: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", revision=MODEL_REVISION, local_files_only=True
    )
    model = snapshot_download(
        "Qwen/Qwen3-0.6B", revision=MODEL_REVISION, local_files_only=True
    )
    modules = [
        processor_module,
        sys.modules[DecodeWorkerHandler.__module__],
        sys.modules[SamplingParams.__module__],
        sys.modules["dynamo._core"],
    ]
    (run / "provenance.json").write_text(
        json.dumps(
            {
                module.__name__: {
                    "path": module.__file__,
                    "sha256": hashlib.sha256(
                        Path(module.__file__).read_bytes()
                    ).hexdigest(),
                }
                for module in modules
            },
            indent=2,
        )
        + "\n"
    )
    captured = asyncio.Future()
    calls = []
    engine_params = []
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = False
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(skip_tokenizer_init=False)
    )
    rows = []
    finalized = []

    async def worker(request, context):
        calls.append(request)
        params = handler._build_sampling_params(request)
        SamplingParams(**params).verify(tokenizer.vocab_size)
        engine_params.append(params)

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

    with tempfile.TemporaryDirectory(
        prefix="logit-bias-", dir=run
    ) as discovery, patch.dict(
        os.environ,
        {
            "DYN_FILE_KV": discovery,
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "1",
            "DYN_TCP_RPC_HOST": "127.0.0.1",
        },
    ):
        rt = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        tasks = []
        try:
            namespace = "logit-bias-" + uuid.uuid4().hex
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

                        actual = (
                            engine_params[-1].get("logit_bias")
                            if len(calls) > before
                            else "rejected"
                        )
                        rows.append(
                            {
                                "case": name,
                                "interval": interval,
                                "stream": streaming,
                                "expected_bias": expected,
                                "http_status": status,
                                "body": body,
                                "worker_called": len(calls) > before,
                                "worker_request": calls[-1]
                                if len(calls) > before
                                else None,
                                "actual_bias": actual,
                                "engine_params": engine_params[-1]
                                if len(calls) > before
                                else None,
                                "passed": actual
                                == (
                                    "rejected"
                                    if expected == "rejected"
                                    else (None if baseline else expected)
                                ),
                            }
                        )
                        (run / "results.json").write_text(
                            json.dumps(rows, indent=2) + "\n"
                        )
                        expected_bias = (
                            "rejected"
                            if expected == "rejected"
                            else (None if baseline else expected)
                        )
                        assert actual == expected_bias, rows[-1]
                        assert status == (400 if expected == "rejected" else 200), body
                        assert len(calls) == before + (expected != "rejected"), body
                        assert len(finalized) == len(calls)
        finally:
            rt.shutdown()
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), 10)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    (run / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} native HTTP logit-bias observations recorded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", action="store_true", help="Expect main to drop biases"
    )
    args = parser.parse_args()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    asyncio.run(asyncio.wait_for(main(run, args.baseline), 150))
