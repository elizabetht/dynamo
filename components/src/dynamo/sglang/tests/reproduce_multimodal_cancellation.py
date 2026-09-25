# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import hashlib
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from transformers import AutoTokenizer

import dynamo._core
import dynamo.sglang.request_handlers.multimodal.worker_handler as wh
from dynamo.common.constants import DisaggregationMode
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
from dynamo.sglang.protocol import PreprocessedRequest, SglangMultimodalRequest

parser = argparse.ArgumentParser(
    description="CPU native HTTP multimodal cancellation replay with scripted generation"
)
parser.add_argument("--model-metadata", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expect-cancellation", action="store_true")
parser.add_argument("--disaggregated", action="store_true")
args = parser.parse_args()
MODEL = args.model_metadata


async def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows = []
    tasks = []
    active = {}

    async def worker(request, context):
        state = active["state"]
        state["request"] = request
        state["context"] = context

        async def chunks():
            state["started"].set()
            yield {
                "output_ids": tokenizer.encode("hello", add_special_tokens=False),
                "meta_info": {"id": "actual-rid", "finish_reason": None},
            }
            await state["release"].wait()
            yield {
                "output_ids": [],
                "meta_info": {"id": "actual-rid", "finish_reason": {"type": "stop"}},
            }

        async def generate(**kwargs):
            state["params"] = kwargs["sampling_params"]
            return chunks()

        try:
            if active["control"]:
                yield {"token_ids": tokenizer.encode("hello", add_special_tokens=False)}
                await context.async_killed_or_stopped()
                return
            handler = wh.MultimodalWorkerHandler.__new__(wh.MultimodalWorkerHandler)
            handler.serving_mode = (
                DisaggregationMode.DECODE
                if args.disaggregated
                else DisaggregationMode.AGGREGATED
            )

            async def bootstrap():
                yield {
                    "bootstrap_host": "localhost",
                    "bootstrap_port": 1234,
                    "bootstrap_room": 1,
                }

            async def prefill_generate(*args, **kwargs):
                return bootstrap()

            handler.prefill_client = SimpleNamespace(generate=prefill_generate)
            handler.enable_trace = False
            handler.engine = SimpleNamespace(
                async_generate=generate,
                tokenizer_manager=SimpleNamespace(
                    abort_request=lambda **kw: state["aborts"].append(kw)
                ),
            )
            handler.embeddings_processor = None
            handler.shutdown_event = None
            handler._abort_tasks = set()
            handler._supports_ordered_cancellation = False
            typed = SglangMultimodalRequest(
                request=PreprocessedRequest.model_validate(request)
            )
            async for item in handler.generate(typed.model_dump_json(), context):
                data = json.loads(item)
                data.pop("finished", None)
                if not data.get("text"):
                    data.pop("text", None)
                yield data
        finally:
            state["done"].set()

    with (
        tempfile.TemporaryDirectory(prefix="discovery-") as temp,
        patch.dict(
            os.environ,
            {
                "DYN_FILE_KV": temp,
                "DYN_ROUTER_MIN_INITIAL_WORKERS": "1",
                "DYN_TCP_RPC_HOST": "127.0.0.1",
            },
        ),
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
                for control in [True, False]:
                    state = {
                        "started": asyncio.Event(),
                        "done": asyncio.Event(),
                        "release": asyncio.Event(),
                        "aborts": [],
                    }
                    active.update(state=state, control=control)
                    request = {
                        "model": "probe-model",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe this image."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC"
                                        },
                                    },
                                ],
                            }
                        ],
                        "guided_regex": "[a-z]+",
                        "stream": True,
                        "max_tokens": 128,
                    }
                    response = await session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", json=request
                    )
                    assert response.status == 200
                    async for raw in response.content:
                        if raw.startswith(b"data: ") and raw.strip() != b"data: [DONE]":
                            event = json.loads(raw[6:])
                            if any(
                                c.get("delta", {}).get("content")
                                for c in event.get("choices", [])
                            ):
                                break
                    else:
                        raise AssertionError("no content")
                    response.close()
                    try:
                        for _ in range(200):
                            if (
                                state["context"].is_stopped()
                                or state["context"].is_killed()
                            ):
                                break
                            await asyncio.sleep(0.01)
                        stopped = (
                            state["context"].is_stopped()
                            or state["context"].is_killed()
                        )
                        assert stopped, "disconnect signal did not reach handler"
                        try:
                            await asyncio.wait_for(state["done"].wait(), 1.5)
                        except TimeoutError:
                            # A blocked baseline is an expected observation.
                            pass
                        rows.append(
                            {
                                "control": control,
                                "context_cancelled": stopped,
                                "handler_completed_before_manual_release": state[
                                    "done"
                                ].is_set(),
                                "aborts": list(state["aborts"]),
                                "worker_request": state["request"],
                                "sampling_params": state.get("params"),
                            }
                        )
                    finally:
                        state["release"].set()
                        await asyncio.wait_for(state["done"].wait(), 3)
                    assert rows[-1]["handler_completed_before_manual_release"] == (
                        control or args.expect_cancellation
                    )
                    assert rows[-1]["aborts"] == (
                        [{"rid": "actual-rid", "abort_all": False}]
                        if args.expect_cancellation and not control
                        else []
                    )
                    assert state["request"]["multi_modal_data"]["image_url"]
                    assert state["request"]["sampling_options"]["guided_decoding"] == {
                        "regex": "[a-z]+"
                    }
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
            args.output.write_text(
                json.dumps(
                    {
                        "observations": rows,
                        "imports": {
                            m.__name__: {
                                "module_file": Path(m.__file__).name,
                                "sha256": hashlib.sha256(
                                    Path(m.__file__).read_bytes()
                                ).hexdigest(),
                            }
                            for m in [wh, dynamo._core]
                        },
                        "scope": "Native localhost HTTP disconnect and real handler; encoder bypassed, engine and prefill scripted, no embeddings or inference",
                        "mode": "decode" if args.disaggregated else "aggregated",
                        "cleanup": "Every scripted stream released and worker finalized; runtime shutdown requested",
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
