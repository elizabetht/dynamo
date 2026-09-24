# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce Unicode logprob fidelity over native localhost HTTP, without GPUs."""

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
from dynamo.common.backend.logprobs import (
    build_sglang_logprob_kwargs,
    extract_from_sglang_meta,
)
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


async def main(output, truncate=False, topk=False):
    output.mkdir(parents=True, exist_ok=True)
    model = snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def output_ids(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[:-1] if truncate else ids

    cases = [
        (
            shape + "-" + str(as_ids),
            text,
            None,
            text,
            None,
            {
                **extra,
                "logprobs": True,
                "top_logprobs": int(topk),
                "return_tokens_as_token_ids": as_ids,
            },
            batch,
            interval,
            streaming,
        )
        for text in (
            ["😊", "A😊", "있다", "ः�숗𐁩"]
            if truncate
            else [
                "Hello world",
                "中文😊",
                "\ufffd",
                "A\ufffdB",
                "\ufffd😊",
                "😊\ufffd",
                "있다",
                "A있다B",
            ]
        )
        for shape, extra in [("plain", {}), ("guided", {"guided_regex": ".*"})]
        for as_ids in [False]
        for batch in [1, 7]
        for interval in [1, 20]
        for streaming in [False, True]
    ]
    captured = asyncio.Future()
    active = {}
    calls = []
    finalized = []
    rows = []

    async def worker(request, context):
        calls.append(request)
        assert build_sglang_logprob_kwargs(
            request["output_options"], allow_top_logprobs=topk
        ) == {"return_logprob": True, "top_logprobs_num": int(topk)}
        ids = output_ids(active["text"])
        batch = active["batch"]
        try:
            for start in range(0, len(ids), batch):
                if context.is_stopped() or context.is_killed():
                    return
                lp, top = extract_from_sglang_meta(
                    {
                        "output_top_logprobs": [
                            [(-0.5, tid, tokenizer.decode([tid]))]
                            for tid in ids[start : start + batch]
                        ]
                        if topk
                        else None,
                        "output_token_logprobs": [
                            (-0.5, tid, tokenizer.decode([tid]))
                            for tid in ids[start : start + batch]
                        ],
                    },
                    return_tokens_as_token_ids=request["output_options"][
                        "return_tokens_as_token_ids"
                    ],
                )
                yield {
                    "token_ids": ids[start : start + batch],
                    "finish_reason": None,
                    "log_probs": lp,
                    "top_logprobs": top,
                }
                await asyncio.sleep(0)
            yield {
                "token_ids": [],
                "finish_reason": active["finish"],
                "stop_reason": active["reason"],
            }
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
        prefix="unicode-logprob-", dir=output
    ) as discovery, patch.dict(
        os.environ,
        {
            "DYN_FILE_KV": discovery,
            "DYN_SGL_ALLOW_TOP_LOGPROBS": "1" if topk else "0",
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "1",
            "DYN_TCP_RPC_HOST": "127.0.0.1",
        },
    ):
        runtime = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        tasks = []
        try:
            namespace = "unicode-logprob-" + uuid.uuid4().hex
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
                for (
                    name,
                    text,
                    stops,
                    expected,
                    reason,
                    extra,
                    batch,
                    interval,
                    streaming,
                ) in cases:
                    finish = "stop" if reason else "length"
                    active.update(text=text, batch=batch, finish=finish, reason=reason)
                    processor.stream_interval = interval
                    request = {
                        **extra,
                        "model": "probe-model",
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "stop": stops,
                        "max_tokens": 128,
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
                    entries = [
                        entry
                        for c in choices
                        for entry in (c.get("logprobs") or {}).get("content", [])
                    ]
                    expected_ids = output_ids(text)
                    id_tokens = ["token_id:" + str(tid) for tid in expected_ids]
                    expected = tokenizer.decode(expected_ids, skip_special_tokens=False)
                    row = {
                        "id_format_correct": [e["token"] for e in entries] == id_tokens
                        if extra["return_tokens_as_token_ids"]
                        else None,
                        "logprob_entries": entries,
                        "logprob_text": "".join(e["token"] for e in entries),
                        "source_text": text,
                        "expected_text": expected,
                        "text_fidelity": "".join(e["token"] for e in entries)
                        == expected,
                        "expected_id_tokens": id_tokens,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="End output one token before the complete text",
    )
    parser.add_argument(
        "--topk", action="store_true", help="Opt in to one top-logprob alternative"
    )
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(main(args.output, args.truncate, args.topk), 240))
