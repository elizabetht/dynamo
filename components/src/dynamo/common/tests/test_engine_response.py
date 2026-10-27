# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import random

import pytest

from dynamo.common.utils.engine_response import trailing_stop_prefix_len

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "text,stops,expected",
    [
        ("", {"STOP"}, 0),
        ("text", set(), 0),
        ("text", {""}, 0),
        ("xabab", {"ababz", "abz"}, 4),
        ("xabba", {"ababz", "baz"}, 2),
        ("你好世界", {"世界末", "界外"}, 2),
        ("xSTOP", {"STOP", "STOPS"}, 4),
        ("a" * 1024 + "x", {"a" * 1024 + "b"}, 0),
    ],
)
def test_trailing_stop_prefix(text, stops, expected):
    assert trailing_stop_prefix_len(text, stops) == expected


def test_stop_prefix_matches_brute_force():
    strings = [
        "".join(chars)
        for length in range(5)
        for chars in itertools.product("ab界", repeat=length)
    ]
    rng = random.Random(41)
    for text in strings:
        for stop in strings:
            stops = {stop, rng.choice(strings), ""}
            expected = max(
                [0]
                + [
                    length
                    for length in range(1, len(text) + 1)
                    if any(item.startswith(text[-length:]) for item in stops)
                ]
            )
            assert trailing_stop_prefix_len(text, stops) == expected
