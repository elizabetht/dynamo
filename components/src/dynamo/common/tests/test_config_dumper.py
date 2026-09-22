# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import msgspec
import pytest

from dynamo.common.config_dump.config_dumper import canonical_json_encoder

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class ModelConfig(msgspec.Struct):
    model_path: str


class ServerConfig(msgspec.Struct, dict=True):
    model: ModelConfig
    tp_size: int = 1


@dataclass
class BackendConfig:
    server_args: ServerConfig


def test_config_dump_preserves_struct_fields_without_private_bookkeeping():
    server_args = ServerConfig(ModelConfig("test-model"), tp_size=2)
    server_args._input_frozen = True
    server_args._model_config = object()

    dumped = json.loads(canonical_json_encoder.encode(BackendConfig(server_args)))

    assert dumped == {
        "server_args": {"model": {"model_path": "test-model"}, "tp_size": 2}
    }
