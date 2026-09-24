# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from transformers import AutoTokenizer

from dynamo.frontend import sglang_prepost as prepost
from dynamo.frontend.utils import PreprocessError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.timeout(30),
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.profiled_vram_gib(0),
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture
def request_body():
    return {
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


def preprocess(request_body, tokenizer):
    return prepost.preprocess_chat_request(
        request_body,
        tokenizer=tokenizer,
        tool_call_parser_name="hermes",
        reasoning_parser_name=None,
    )


@pytest.mark.parametrize("guidance", ["response_format", "guided_json"])
def test_explicit_guidance_skips_unused_tool_grammar(
    tokenizer, request_body, monkeypatch, guidance
):
    schema = {"type": "object"}
    request_body[guidance] = (
        {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}
        if guidance == "response_format"
        else schema
    )
    before = copy.deepcopy(request_body)
    calls = []
    original = prepost.build_tool_call_guided_decoding

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(prepost, "build_tool_call_guided_decoding", observed)
    result = preprocess(request_body, tokenizer)
    assert not calls
    assert result.guided_decoding == {"json": schema}
    assert isinstance(result.tool_call_parser, FunctionCallParser)
    assert (
        result.tool_call_parser.parse_non_stream(
            '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'
        )[1][0].name
        == "weather"
    )
    assert request_body == before


@pytest.mark.parametrize("forced", [False, True])
def test_selected_tool_grammar_is_preserved(tokenizer, request_body, forced):
    if forced:
        request_body.update(
            tool_choice="required", response_format={"type": "json_object"}
        )
    result = preprocess(request_body, tokenizer)
    assert result.guided_decoding == prepost.build_tool_call_guided_decoding(
        request_body,
        tool_call_parser_name="hermes",
        sglang_tools=prepost.convert_tools(request_body["tools"]),
    )
    assert isinstance(
        result.tool_call_parser, JsonArrayParser if forced else FunctionCallParser
    )


def test_response_validation_still_runs(tokenizer, request_body):
    request_body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": []},
    }
    with pytest.raises(PreprocessError, match="schema must be a JSON object"):
        preprocess(request_body, tokenizer)
