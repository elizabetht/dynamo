# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare real SGLang preprocessing CPU time; no model generation is performed."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import dynamo.frontend.sglang_prepost as candidate


def load_baseline(path):
    spec = importlib.util.spec_from_file_location(
        "dynamo.frontend._baseline_prepost", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def requests():
    base = {
        "model": "probe-model",
        "messages": [{"role": "user", "content": "Report the weather in 中文😊."}],
    }
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tool = {
        "type": "function",
        "function": {"name": "weather", "parameters": schema},
    }
    yield "plain", base
    yield "text_parts", {
        **base,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "中文😊"}]}],
    }
    yield "json_schema", {
        **base,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": schema},
        },
    }
    for choice in [
        "auto",
        "required",
        {"type": "function", "function": {"name": "weather"}},
    ]:
        yield str(choice) if isinstance(choice, str) else "named", {
            **base,
            "tools": [tool],
            "tool_choice": choice,
        }
    yield "tools32", {
        **base,
        "tools": [
            {
                "type": "function",
                "function": {"name": f"weather{i}", "parameters": schema},
            }
            for i in range(32)
        ],
        "tool_choice": "auto",
    }


def observable(result):
    return {
        "prompt_token_ids": result.prompt_token_ids,
        "guided_decoding": result.guided_decoding,
        "request": result.request,
        "force_reasoning": result.force_reasoning,
        "named_zero_arg_tool": result.named_zero_arg_tool,
        "tool_parser": type(result.tool_call_parser).__name__,
        "reasoning_parser": type(result.reasoning_parser).__name__,
    }


def main(args):
    baseline = load_baseline(args.baseline)
    model = snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = []
    for name, request in requests():

        def once(module):
            return module.preprocess_chat_request(
                request,
                tokenizer=tokenizer,
                tool_call_parser_name="hermes",
                reasoning_parser_name=None,
            )

        expected = observable(once(baseline))
        assert observable(once(candidate)) == expected, name
        cold_samples = {"baseline": [], "candidate": []}
        for repetition in range(args.pairs):
            order = [("baseline", baseline), ("candidate", candidate)]
            if repetition % 2:
                order.reverse()
            for label, module in order:
                if label == "candidate":
                    candidate._cached_template_content_format.cache_clear()
                start = time.process_time_ns()
                result = once(module)
                cold_samples[label].append((time.process_time_ns() - start) / 1000)
                assert observable(result) == expected, name
        samples = {"baseline": [], "candidate": []}
        for repetition in range(args.pairs):
            order = [("baseline", baseline), ("candidate", candidate)]
            if repetition % 2:
                order.reverse()
            for label, module in order:
                start = time.process_time_ns()
                results = [once(module) for _ in range(args.requests)]
                elapsed = (time.process_time_ns() - start) / 1000 / args.requests
                assert all(observable(result) == expected for result in results), name
                samples[label].append(elapsed)
        medians = {
            label: statistics.median(values) for label, values in samples.items()
        }
        rows.append(
            {
                "case": name,
                "detector_cold_samples_us": cold_samples,
                "samples_us_per_request": samples,
                "median_us_per_request": medians,
                "reduction_percent": 100
                * (1 - medians["candidate"] / medians["baseline"]),
                "observable_sha256": hashlib.sha256(
                    json.dumps(expected, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    result = {
        "scope": "Warm real Python SGLang preprocessing CPU benchmark, no HTTP or generation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "jinja2": importlib.metadata.version("jinja2"),
        "pairs": args.pairs,
        "requests_per_pair_arm": args.requests,
        "tokenizer": "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
        "source_sha256": {
            label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for label, module in [("baseline", baseline), ("candidate", candidate)]
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--requests", type=int, default=40)
    main(parser.parse_args())
