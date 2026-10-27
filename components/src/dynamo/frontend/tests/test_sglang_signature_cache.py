# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from dynamo.frontend import sglang_prepost as prepost

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_tool_guidance_does_not_retain_request_parsers(monkeypatch):
    parsers = []
    original = FunctionCallParser.__init__

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        parsers.append(weakref.ref(self))

    monkeypatch.setattr(FunctionCallParser, "__init__", observed)
    request = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": "auto",
    }
    for _ in range(3):
        prepost.build_tool_call_guided_decoding(
            request,
            tool_call_parser_name="hermes",
            sglang_tools=prepost.convert_tools(request["tools"]),
        )
    gc.collect()
    assert len(parsers) == 3
    assert all(parser() is None for parser in parsers)


def test_bound_and_unbound_signatures_remain_distinct():
    class Helper:
        def explicit(self, value, *, parallel_tool_calls=None):
            return value, parallel_tool_calls

        def legacy(self, value):
            return value

        def unusual(parallel_tool_calls, value):
            return value

    helper = Helper()
    assert prepost._call_with_optional_parallel_tool_calls(
        helper.explicit, 1, parallel_tool_calls=False
    ) == (1, False)
    assert (
        prepost._call_with_optional_parallel_tool_calls(
            helper.legacy, 1, parallel_tool_calls=False
        )
        == 1
    )
    assert not prepost._callable_accepts_kwarg(helper.unusual, "parallel_tool_calls")
    assert prepost._callable_accepts_kwarg(Helper.unusual, "parallel_tool_calls")


def test_variadic_and_positional_only_methods():
    class Helper:
        def variadic(*args, **kwargs):
            return kwargs

        def positional(self, parallel_tool_calls, /):
            return parallel_tool_calls

        @classmethod
        def class_method(cls, *, parallel_tool_calls=True):
            return parallel_tool_calls

        @staticmethod
        def static_method(*, parallel_tool_calls=True):
            return parallel_tool_calls

    helper = Helper()
    assert prepost._call_with_optional_parallel_tool_calls(
        helper.variadic, parallel_tool_calls=False
    ) == {"parallel_tool_calls": False}
    assert not prepost._callable_accepts_kwarg(helper.positional, "parallel_tool_calls")
    assert (
        prepost._call_with_optional_parallel_tool_calls(
            helper.class_method, parallel_tool_calls=False
        )
        is False
    )
    assert (
        prepost._call_with_optional_parallel_tool_calls(
            helper.static_method, parallel_tool_calls=False
        )
        is False
    )
