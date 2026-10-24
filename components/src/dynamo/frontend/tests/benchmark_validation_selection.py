# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.renderers.hf import HfRenderer
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

from dynamo.frontend import prepost

parser = argparse.ArgumentParser(
    description="Compare CPU preprocessing with a saved baseline prepost.py"
)
parser.add_argument("--baseline-source", type=Path, required=True)
parser.add_argument(
    "--model", required=True, help="Local Qwen3-0.6B tokenizer/config snapshot"
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model_config = ModelConfig(
    model=args.model, tokenizer=args.model, dtype="bfloat16", max_model_len=32768
)
renderer = HfRenderer(
    VllmConfig(model_config=model_config, device_config=DeviceConfig(device="cpu")),
    tokenizer,
)

spec = importlib.util.spec_from_file_location(
    "dynamo.frontend._baseline_prepost", args.baseline_source
)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)


def observable(result):
    return {
        "request": result.request_for_sampling.model_dump(),
        "tokens": result.prompt_token_ids,
        "prompt": result.engine_prompt,
        "guidance": result.guided_decoding,
        "fallback": result.uses_dynamo_json_tool_call_fallback,
    }


async def main():
    rows = []
    for size in (1, 32, 256):
        schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": 'Unicode café and escaping \\".',
                }
                for i in range(size)
            },
            "additionalProperties": False,
        }
        for route in (
            "chat",
            "response_format",
            "structured_outputs",
            "legacy",
            "auto",
            "required",
            "named",
        ):
            request = {
                "request_id": "validation-benchmark",
                "model": "Qwen/Qwen3-0.6B",
                "messages": [{"role": "user", "content": "Return an answer."}],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            parser = None
            if route == "response_format":
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema},
                }
            if route == "structured_outputs":
                request["structured_outputs"] = {"json": schema}
            if route == "legacy":
                request["guided_json"] = schema
            if route in ("auto", "required", "named"):
                request["tools"] = [
                    {
                        "type": "function",
                        "function": {"name": "answer", "parameters": schema},
                    }
                ]
                request["tool_choice"] = (
                    route
                    if route != "named"
                    else {"type": "function", "function": {"name": "answer"}}
                )
                parser = Hermes2ProToolParser
            original = copy.deepcopy(request)

            async def invoke(module):
                return await module.preprocess_chat_request(
                    request,
                    tokenizer=tokenizer,
                    renderer=renderer,
                    tool_parser_class=parser,
                    model_config=model_config,
                )

            expected = observable(await invoke(baseline))
            actual = observable(await invoke(prepost))
            assert actual == expected
            for module in (baseline, prepost):
                for _ in range(5):
                    await invoke(module)
            measurements = []
            for repetition in range(4):
                pair = {}
                order = (
                    ("baseline", "candidate")
                    if repetition % 2 == 0
                    else ("candidate", "baseline")
                )
                for label in order:
                    module = baseline if label == "baseline" else prepost
                    samples = []
                    for _ in range(100):
                        start = time.perf_counter_ns()
                        result = await invoke(module)
                        samples.append((time.perf_counter_ns() - start) / 1000)
                        assert observable(result) == expected
                        assert request == original
                    pair[label] = {
                        "samples_us": samples,
                        "median_us": statistics.median(samples),
                    }
                measurements.append({"repetition": repetition, "order": order, **pair})
            b = statistics.median(x["baseline"]["median_us"] for x in measurements)
            c = statistics.median(x["candidate"]["median_us"] for x in measurements)
            row = {
                "route": route,
                "properties": size,
                "baseline_us": b,
                "candidate_us": c,
                "percent_change": (c / b - 1) * 100,
                "paired_repetitions": measurements,
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "paired_repetitions"}),
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            {
                "scope": "CPU preprocessing with real HfRenderer/tokenizer/Hermes; no HTTP or inference",
                "samples_per_variant_per_case": 400,
                "source_hashes": {
                    "baseline_prepost": hashlib.sha256(
                        args.baseline_source.read_bytes()
                    ).hexdigest(),
                    "candidate_prepost": hashlib.sha256(
                        Path(prepost.__file__).read_bytes()
                    ).hexdigest(),
                    "benchmark": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                },
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
