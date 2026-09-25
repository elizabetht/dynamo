# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handler cancellation tests with a scripted engine, without vision inference."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import EngineShutdown
from dynamo.sglang.request_handlers.multimodal.worker_handler import (
    MultimodalWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


class RequestContext:
    trace_id = "submitted-id"

    def __init__(self, signal):
        self.signal = signal
        self.cancelled = asyncio.Event()

    def id(self):
        return self.trace_id

    def is_stopped(self):
        return self.signal == "stop" and self.cancelled.is_set()

    def is_killed(self):
        return self.signal == "kill" and self.cancelled.is_set()

    def async_killed_or_stopped(self):
        return asyncio.create_task(self.cancelled.wait())


@pytest.mark.asyncio
@pytest.mark.parametrize("signal", ["normal", "stop", "kill", "shutdown", "close"])
@pytest.mark.parametrize(
    "mode", [DisaggregationMode.AGGREGATED, DisaggregationMode.DECODE]
)
async def test_stream_aborts_only_active_engine_id(signal, mode):
    context = RequestContext(signal)
    aborted = asyncio.Event()
    aborts = []

    def abort_request(*, rid, abort_all):
        aborts.append((rid, abort_all))
        aborted.set()

    async def chunks():
        yield {"output_ids": [42], "meta_info": {"id": "actual-id"}}
        if signal != "normal":
            await aborted.wait()
        yield {
            "output_ids": [],
            "meta_info": {
                "id": "actual-id",
                "finish_reason": {"type": "stop" if signal == "normal" else "abort"},
            },
        }

    async def generate(**kwargs):
        assert kwargs["rid"] == "submitted-id"
        return chunks()

    handler = MultimodalWorkerHandler.__new__(MultimodalWorkerHandler)
    handler.serving_mode = mode

    async def bootstrap():
        yield {
            "bootstrap_host": "localhost",
            "bootstrap_port": 1234,
            "bootstrap_room": 1,
        }

    handler.prefill_client = SimpleNamespace(
        generate=AsyncMock(return_value=bootstrap())
    )
    handler.enable_trace = False
    handler.shutdown_event = asyncio.Event()
    handler._abort_tasks = set()
    handler._supports_ordered_cancellation = False
    handler.embeddings_processor = Mock()
    handler.engine = SimpleNamespace(
        async_generate=generate,
        tokenizer_manager=SimpleNamespace(abort_request=abort_request),
    )
    handler.config = SimpleNamespace(server_args=SimpleNamespace(pp_size=1))
    outputs = []

    async def consume():
        stream = handler.generate(
            {
                "request": {
                    "token_ids": [1, 2],
                    "sampling_options": {},
                    "stop_conditions": {"max_tokens": 8},
                },
                "multimodal_inputs": [],
            },
            context,
        )
        if signal == "close":
            outputs.append(json.loads(await anext(stream)))
            await stream.aclose()
            assert aborts == [("actual-id", False)]
            return
        async for output in stream:
            outputs.append(json.loads(output))
            if signal == "shutdown":
                handler.shutdown_event.set()
            elif signal != "normal":
                context.cancelled.set()

    await asyncio.wait_for(consume(), timeout=2)
    if mode == DisaggregationMode.DECODE:
        handler.prefill_client.generate.assert_awaited_once()
        assert handler.prefill_client.generate.call_args.kwargs["context"] is context
    assert outputs[0]["token_ids"] == [42]
    if signal == "normal":
        assert aborts == []
        assert outputs[-1]["finished"]
    else:
        assert aborts == [("actual-id", False)]
        if signal == "shutdown":
            assert len(outputs) == 2
            assert outputs[-1]["finish_reason"] == "error"
            assert "shut down" in outputs[-1]["error"]
        else:
            assert len(outputs) == 1
    assert not handler._abort_tasks
    handler.embeddings_processor.release_embeddings.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("choices,completed", [(1, 1), (2, 1), (2, 2)])
async def test_shutdown_after_terminal_preserves_completed_choices(choices, completed):
    context = RequestContext("normal")
    handler = MultimodalWorkerHandler.__new__(MultimodalWorkerHandler)
    handler.shutdown_event = asyncio.Event()
    handler._abort_tasks = set()
    handler._supports_ordered_cancellation = False
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(abort_request=Mock())
    )
    outputs = []

    async def chunks():
        for index in range(completed):
            yield {
                "output_ids": [],
                "index": index,
                "meta_info": {
                    "id": f"choice-{index}",
                    "finish_reason": {"type": "stop"},
                },
            }
        handler.shutdown_event.set()

    async def consume():
        async for chunk in handler._cancel_aware_stream(
            chunks(), context, {"n": choices}
        ):
            outputs.append(chunk)

    if completed < choices:
        with pytest.raises(EngineShutdown):
            await asyncio.wait_for(consume(), 2)
    else:
        await asyncio.wait_for(consume(), 2)
    assert [chunk["index"] for chunk in outputs] == list(range(completed))
    handler.engine.tokenizer_manager.abort_request.assert_not_called()
