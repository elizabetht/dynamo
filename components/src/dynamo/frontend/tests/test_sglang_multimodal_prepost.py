# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal content normalization on the sglang chat-processor path.

Regression coverage for the bug where ``apply_chat_template`` ran on raw
OpenAI messages: VLM templates branch on ``type == 'image'`` while OpenAI
sends ``image_url``, so placeholders never rendered.
"""

import pytest

from dynamo.frontend.sglang_prepost import (
    _normalize_messages_for_template,
    preprocess_chat_request,
)

# Needs sglang packages (gpu_1 container) but allocates no GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]

# A list-iterating template; this is what makes sglang's detector report the
# "openai" content format that drives normalization.
_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.content is iterable and message.content is not string %}"
    "{% for chunk in message.content %}"
    "{% if chunk.type == 'image' %}<IMG>"
    "{% elif chunk.type == 'video' %}<VID>"
    "{% elif chunk.type == 'audio' %}<AUD>"
    "{% elif chunk.type == 'text' %}{{ chunk.text }}"
    "{% endif %}{% endfor %}{% endif %}{% endfor %}"
)


def _url_chunk(kind: str) -> dict:
    return {"type": f"{kind}_url", f"{kind}_url": {"url": "x://m"}}


@pytest.fixture
def tokenizer():
    class Tokenizer:
        chat_template = _TEMPLATE

    return Tokenizer()


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_url_chunk_normalized_to_bare_type(tokenizer, kind):
    messages = [{"role": "user", "content": [_url_chunk(kind)]}]
    out = _normalize_messages_for_template(messages, tokenizer)
    assert [c["type"] for c in out[0]["content"]] == [kind]


def test_text_only_passes_through(tokenizer):
    messages = [{"role": "user", "content": "hello"}]
    assert _normalize_messages_for_template(messages, tokenizer) == messages


def test_preprocess_feeds_normalized_chunks_to_template():
    class Tokenizer:
        chat_template = _TEMPLATE

        def apply_chat_template(self, messages, **_):
            self.seen = messages[0]["content"]
            return [42 if c["type"] == "image" else 1 for c in self.seen]

        def encode(self, prompt):
            raise AssertionError("text path must not be taken")

    tok = Tokenizer()
    request = {
        "model": "fake-vlm",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    _url_chunk("image"),
                ],
            }
        ],
    }

    result = preprocess_chat_request(
        request,
        tokenizer=tok,
        tool_call_parser_name=None,
        reasoning_parser_name=None,
    )

    assert 42 in result.prompt_token_ids
    assert [c["type"] for c in tok.seen] == ["text", "image"]


def test_template_format_cache_tracks_template_text(monkeypatch):
    from types import SimpleNamespace

    import dynamo.frontend.sglang_prepost as prepost

    detector = prepost.detect_jinja_template_content_format
    detected = []

    def record(template):
        detected.append(template)
        return detector(template)

    monkeypatch.setattr(prepost, "detect_jinja_template_content_format", record)
    prepost._cached_template_content_format.cache_clear()
    string_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    list_template = (
        "{% for message in messages %}{% for item in message.content %}"
        "{{ item.text }}{% endfor %}{% endfor %}"
    )
    tokenizer = SimpleNamespace(chat_template=string_template)
    messages = [{"role": "user", "content": [{"type": "text", "text": "中文😊"}]}]
    try:
        for _ in range(2):
            assert prepost._normalize_messages_for_template(messages, tokenizer) == [
                {"role": "user", "content": "中文😊"}
            ]
        tokenizer.chat_template = list_template
        assert prepost._normalize_messages_for_template(messages, tokenizer) == messages
        tokenizer.chat_template = string_template
        assert (
            prepost._normalize_messages_for_template(messages, tokenizer)[0]["content"]
            == "中文😊"
        )
        assert detected == [string_template, list_template]
        assert messages[0]["content"] == [{"type": "text", "text": "中文😊"}]
    finally:
        prepost._cached_template_content_format.cache_clear()


def test_unhashable_template_preserves_detector_fallback(tokenizer):
    tokenizer.chat_template = {"default": "{{ messages }}"}
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert _normalize_messages_for_template(messages, tokenizer) == [
        {"role": "user", "content": "hello"}
    ]
