# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local HTTP shutdown reproduction with a scripted engine; no GPU or inference."""

import argparse
import asyncio
import gc
import json
import os
import socket
import tempfile
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from transformers import AutoTokenizer

import dynamo.sglang.request_handlers.cancellation as cancellation
import dynamo.sglang.request_handlers.llm.decode_handler as wh
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.input_params import InputParamManager
from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    ModelInput,
    ModelType,
    WorkerType,
    make_engine,
    register_model,
    run_input,
)
from dynamo.runtime import DistributedRuntime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-metadata", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--mode", choices=["normal", "eof", "error"], required=True)
parser.add_argument("--baseline-helper", type=Path)
args = parser.parse_args()
MODEL = args.model_metadata
if args.baseline_helper:
    baseline = types.ModuleType("baseline_cancellation")
    exec(args.baseline_helper.read_text(), baseline.__dict__)
    cancellation.CancellationMixin._stream_until_cancelled = (
        baseline.CancellationMixin._stream_until_cancelled
    )


async def main():
    contexts = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, ctx: contexts.append(
            {
                "message": ctx.get("message"),
                "exception": type(ctx.get("exception")).__name__,
            }
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = []
    tasks = []
    active = {}

    async def worker(request, context):
        state = active["state"]
        state["request"] = request
        state["context"] = context

        async def chunks():
            yield {
                "output_ids": tokenizer.encode("hello", add_special_tokens=False),
                "meta_info": {"id": "actual-rid", "finish_reason": None},
            }
            if args.mode != "normal":
                state["shutdown"].set()
                # Let the real monitor and next-item task finish in the same wait.
                for _ in range(4):
                    await asyncio.sleep(0)
                if args.mode == "error":
                    raise ValueError("scripted engine failure")
                return
            yield {
                "output_ids": [],
                "meta_info": {"id": "actual-rid", "finish_reason": {"type": "stop"}},
            }

        def abort(**kw):
            state["aborts"].append(kw)
            state["aborted"].set()

        async def generate(**kwargs):
            state["params"] = kwargs["sampling_params"]
            return chunks()

        try:
            handler = wh.DecodeWorkerHandler.__new__(wh.DecodeWorkerHandler)
            handler.serving_mode = DisaggregationMode.AGGREGATED
            handler.enable_trace = False
            handler.engine = SimpleNamespace(
                async_generate=generate,
                tokenizer_manager=SimpleNamespace(abort_request=abort),
            )
            handler.shutdown_event = state["shutdown"]
            handler._abort_tasks = set()
            handler._supports_ordered_cancellation = False
            handler.use_sglang_tokenizer = False
            handler._first_token_source = None
            handler.config = SimpleNamespace(
                server_args=SimpleNamespace(skip_tokenizer_init=True),
                dynamo_args=SimpleNamespace(enable_rl=False),
            )
            handler.input_param_manager = InputParamManager(None)
            handler.lora_id_for_name = {}
            handler._engine_supports_priority = False
            handler._enable_frontend_decoding = False
            handler._mm_hashes_supported = False
            handler._routed_experts_kwargs = {}
            async for data in handler.generate(request, context):
                state["raw"].append(dict(data))
                yield data
        finally:
            state["done"].set()

    with tempfile.TemporaryDirectory(prefix="dynamo-shutdown-") as temp, patch.dict(
        os.environ,
        {
            "DYN_FILE_KV": temp,
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "1",
            "DYN_TCP_RPC_HOST": "127.0.0.1",
        },
    ):
        runtime = DistributedRuntime(
            asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
        )
        try:
            namespace = "cancel-probe-" + uuid.uuid4().hex
            endpoint = runtime.endpoint(namespace + ".worker.generate")
            tasks.append(asyncio.ensure_future(endpoint.serve_endpoint(worker)))
            await register_model(
                ModelInput.Tokens,
                ModelType.Chat,
                endpoint,
                str(MODEL),
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
                namespace=namespace,
                migration_limit=0,
            )
            engine = await make_engine(runtime, entrypoint_args)
            tasks.append(asyncio.ensure_future(run_input(runtime, "http", engine)))
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                for _ in range(100):
                    try:
                        async with session.get(
                            f"http://127.0.0.1:{port}/v1/models"
                        ) as response:
                            if any(
                                x["id"] == "probe-model"
                                for x in (await response.json()).get("data", [])
                            ):
                                break
                    except aiohttp.ClientConnectorError:
                        pass
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("registration timeout")
                for mode in [args.mode]:
                    state = {
                        "started": asyncio.Event(),
                        "done": asyncio.Event(),
                        "release": asyncio.Event(),
                        "shutdown": asyncio.Event(),
                        "aborts": [],
                        "raw": [],
                        "aborted": asyncio.Event(),
                    }
                    active.update(state=state, control=False, mode=mode)
                    request = {
                        "model": "probe-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "guided_regex": "[a-z]+",
                        "stream": True,
                        "max_tokens": 8,
                    }
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", json=request
                    ) as response:
                        body = await response.text()
                        rows.append(
                            {
                                "mode": mode,
                                "status": response.status,
                                "body": body,
                                "handler_outputs": state["raw"],
                            }
                        )
                    await asyncio.wait_for(state["done"].wait(), 3)
                    gc.collect()
                    await asyncio.sleep(0)
                    rows[-1]["loop_errors"] = list(contexts)
                    rows[-1]["sampling_params"] = state.get("params")
                    rows[-1]["aborts"] = state["aborts"]
                    # Retain serialized exceptions through runtime finalization.
        finally:
            if active.get("state"):
                active["state"]["release"].set()
            runtime.shutdown()
            for task in tasks:
                try:
                    await asyncio.wait_for(task, 10)
                except (TimeoutError, asyncio.CancelledError):
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            gc.collect()
            for _ in range(20):
                await asyncio.sleep(0)
            gc.collect()
            if not rows:
                raise AssertionError("HTTP request did not complete")
            rows[-1]["final_loop_errors"] = list(contexts)
            row = rows[-1]
            assert row["status"] == 200
            assert row["sampling_params"]["regex"] == "[a-z]+"
            events = [
                json.loads(line[6:])
                for line in row["body"].splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            choices = [
                choice for event in events for choice in event.get("choices", [])
            ]
            assert (
                "".join(
                    choice.get("delta", {}).get("content", "") for choice in choices
                )
                == "hello"
            )
            assert [event["error"]["code"] for event in events if "error" in event] == (
                [] if args.mode == "normal" else [503]
            )
            assert [
                choice["finish_reason"]
                for choice in choices
                if choice.get("finish_reason")
            ] == (["stop"] if args.mode == "normal" else [])
            expected = []
            if args.baseline_helper and args.mode != "normal":
                expected = [
                    {
                        "message": "Task exception was never retrieved",
                        "exception": "StopAsyncIteration"
                        if args.mode == "eof"
                        else "ValueError",
                    }
                ]
            assert contexts == expected, contexts
            args.output.write_text(
                json.dumps(
                    {
                        "observations": rows,
                        "scope": "Native localhost HTTP; real admission, sampling and cancellation; scripted engine, no generation",
                        "cleanup": "Worker finalized; runtime shutdown requested; full interpreter shutdown not qualified",
                    },
                    indent=2,
                )
                + "\n"
            )
    print(
        json.dumps(
            [{k: v for k, v in x.items() if k not in ["worker_request"]} for x in rows],
            indent=2,
        )
    )


asyncio.run(asyncio.wait_for(main(), 90))
