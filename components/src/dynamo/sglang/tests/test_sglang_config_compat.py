# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise configuration compatibility against the installed SGLang release."""

import json
from types import SimpleNamespace

import pytest

# The separately pinned XPU 0.5.11 predates declarative config resolution.
overrides = pytest.importorskip("sglang.srt.arg_groups.overrides")

from sglang.srt.server_args import ServerArgs  # noqa: E402

from dynamo.common.config_dump.config_dumper import canonical_json_encoder  # noqa: E402
from dynamo.sglang._compat import (  # noqa: E402
    override_server_args,
    resolved_server_args,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
]


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_memory_saver", True),
        ("tokenizer_worker_num", 2),
        ("forward_pass_metrics_worker_id", "123"),
    ],
)
def test_launcher_overrides_preserve_raw_input(monkeypatch, field, value):
    # Construction is model-free in both supported releases. Seal the record
    # as resolution would, without downloading a model or initializing CUDA.
    server_args = ServerArgs(model_path="test-model")
    server_args._resolution_finished = True
    original = getattr(server_args, field)
    monkeypatch.setattr(
        overrides, "get_context", lambda: SimpleNamespace(server_args=None)
    )

    override_server_args(server_args, "dynamo.test", **{field: value})

    assert getattr(server_args, field) == original
    assert getattr(resolved_server_args(server_args), field) == value


def test_launcher_override_rejects_published_config(monkeypatch):
    server_args = ServerArgs(model_path="test-model")
    monkeypatch.setattr(
        overrides, "get_context", lambda: SimpleNamespace(server_args=server_args)
    )

    with pytest.raises(ValueError, match="published config"):
        override_server_args(server_args, "dynamo.test", enable_memory_saver=True)


def test_server_args_config_dump_preserves_declared_fields():
    server_args = ServerArgs(model_path="test-model", tp_size=2)
    server_args._model_config = object()

    dumped = json.loads(canonical_json_encoder.encode(server_args))

    assert dumped["model_path"] == "test-model"
    assert dumped["tp_size"] == 2
    assert "_model_config" not in dumped


def test_gms_launcher_override_preserves_raw_input(monkeypatch):
    gms = pytest.importorskip("gpu_memory_service.integrations.sglang")
    server_args = ServerArgs(model_path="test-model")
    server_args._resolution_finished = True
    monkeypatch.setattr(
        overrides, "get_context", lambda: SimpleNamespace(server_args=None)
    )

    assert gms.declare_late_resolution is not None
    gms.declare_late_resolution(server_args, "dynamo.gms", enable_memory_saver=True)

    assert server_args.enable_memory_saver is False
    assert resolved_server_args(server_args).enable_memory_saver is True
