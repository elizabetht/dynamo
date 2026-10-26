# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.frontend import sglang_prepost as prepost

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.sglang,
    pytest.mark.core,
]

TEXT = "{% for message in messages %}{{ message.content }}{% endfor %}"
PARTS = "{% for message in messages %}{% for part in message.content %}{{ part.text }}{% endfor %}{% endfor %}"


@pytest.fixture(autouse=True)
def clear_format_cache():
    prepost._cached_template_content_format.cache_clear()
    yield
    prepost._cached_template_content_format.cache_clear()


def test_reuses_detection_without_caching_messages():
    tokenizer = SimpleNamespace(chat_template=TEXT + " cache-test")
    first = [{"role": "user", "content": [{"type": "text", "text": "first"}]}]
    second = [{"role": "user", "content": [{"type": "text", "text": "second"}]}]
    with patch.object(
        prepost,
        "detect_jinja_template_content_format",
        wraps=prepost.detect_jinja_template_content_format,
    ) as detect:
        assert (
            prepost._normalize_messages_for_template(first, tokenizer)[0]["content"]
            == "first"
        )
        assert (
            prepost._normalize_messages_for_template(second, tokenizer)[0]["content"]
            == "second"
        )
        assert detect.call_count == 1
    assert first[0]["content"] == [{"type": "text", "text": "first"}]


def test_template_changes_preserve_text_and_media_normalization():
    tokenizer = SimpleNamespace(chat_template=TEXT)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "héllo"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,eA=="},
                },
            ],
        }
    ]
    original = copy.deepcopy(messages)
    for template, expected in [
        (TEXT, "héllo"),
        (PARTS, [{"type": "text", "text": "héllo"}, {"type": "image"}]),
        ("{{ image }}", [{"type": "text", "text": "héllo"}, {"type": "image"}]),
        (TEXT, "héllo"),
    ]:
        tokenizer.chat_template = template
        assert (
            prepost._normalize_messages_for_template(messages, tokenizer)[0]["content"]
            == expected
        )
        assert messages == original


@pytest.mark.parametrize("template", [None, "{% broken", {"default": TEXT}])
def test_detector_fallback_is_preserved(template):
    tokenizer = SimpleNamespace(chat_template=template)
    messages = [{"role": "user", "content": [{"type": "text", "text": "ok"}]}]
    for _ in range(2):
        assert prepost._normalize_messages_for_template(messages, tokenizer) == [
            {"role": "user", "content": "ok"}
        ]
