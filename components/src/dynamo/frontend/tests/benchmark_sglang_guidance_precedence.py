# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from sglang.srt.function_call import function_call_parser
from transformers import AutoTokenizer

import dynamo.frontend.sglang_prepost as p

parser = argparse.ArgumentParser(
    description="Paired CPU preprocessing benchmark; no model generation"
)
parser.add_argument("--baseline-source", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--strict", action="store_true")
args = parser.parse_args()
ROOT = args.output
ROOT.mkdir(parents=True, exist_ok=True)
MODEL = str(args.tokenizer)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
spec = importlib.util.spec_from_file_location(
    "dynamo.frontend._benchmark_baseline", args.baseline_source
)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
functions = {
    "baseline": baseline.preprocess_chat_request,
    "candidate": p.preprocess_chat_request,
}
logging.getLogger(baseline.__name__).setLevel(logging.ERROR)
logging.getLogger(p.__name__).setLevel(logging.ERROR)
base = {
    "messages": [{"role": "user", "content": "Return the weather as JSON."}],
    "max_tokens": 32,
}
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
cases = []
for count in [1, 16, 64]:
    for mode in ["auto", "response_json", "legacy_json", "required"]:
        request = copy.deepcopy(base)
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": f"weather_{i}",
                    "description": "Report weather for the selected city.",
                    "parameters": copy.deepcopy(schema),
                },
            }
            for i in range(count)
        ]
        if args.strict:
            for tool in request["tools"]:
                tool["function"]["strict"] = True
        request["tool_choice"] = "required" if mode == "required" else "auto"
        if mode == "response_json":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "weather", "schema": schema},
            }
        if mode == "legacy_json":
            request["guided_json"] = schema
        cases.append((f"{mode}_{count}", request))
cases.append(("ordinary", copy.deepcopy(base)))
kwargs = {
    "tokenizer": tokenizer,
    "tool_call_parser_name": "hermes",
    "reasoning_parser_name": None,
}
rows = []
parity = []
for name, request in cases:
    original = copy.deepcopy(request)
    results = {arm: fn(request, **kwargs) for arm, fn in functions.items()}
    a, b = results.values()
    assert (
        a.prompt_token_ids == b.prompt_token_ids
        and a.guided_decoding == b.guided_decoding
    )
    assert type(a.tool_call_parser) is type(b.tool_call_parser) and type(
        a.reasoning_parser
    ) is type(b.reasoning_parser)
    assert a.request == b.request == original == request
    parity.append(
        {
            "case": name,
            "token_count": len(a.prompt_token_ids),
            "guidance": a.guided_decoding,
            "parser": type(a.tool_call_parser).__name__,
        }
    )
    for fn in functions.values():
        for _ in range(5):
            fn(request, **kwargs)
    for pair in range(4):
        for arm in (
            ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        ):
            fn = functions[arm]
            for sample in range(100):
                start = time.perf_counter_ns()
                fn(request, **kwargs)
                elapsed = time.perf_counter_ns() - start
                rows.append(
                    {
                        "case": name,
                        "pair": pair,
                        "arm": arm,
                        "sample": sample,
                        "ns": elapsed,
                    }
                )
summary = []
for name, _ in cases:
    medians = {
        arm: statistics.median(
            r["ns"] for r in rows if r["case"] == name and r["arm"] == arm
        )
        / 1000
        for arm in functions
    }
    pairs = []
    for pair in range(4):
        pair_medians = {
            arm: statistics.median(
                r["ns"]
                for r in rows
                if r["case"] == name and r["arm"] == arm and r["pair"] == pair
            )
            / 1000
            for arm in functions
        }
        pairs.append(pair_medians)
    summary.append(
        {
            "case": name,
            "median_us": medians,
            "change_percent": 100 * (medians["candidate"] / medians["baseline"] - 1),
            "pairs": pairs,
        }
    )
imports = {
    m.__name__: {
        "path": m.__file__,
        "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
    }
    for m in [p, baseline, function_call_parser]
}
(ROOT / "profile.json").write_text(
    json.dumps(
        {
            "scope": "CPU full preprocessing incl real tokenization; warnings suppressed in both arms; source baseline loaded separately",
            "rows": rows,
            "parity": parity,
            "summary": summary,
            "imports": imports,
            "tokenizer": MODEL,
            "seed": None,
            "warmup": 5,
            "repetitions": 4,
            "samples_per_arm_case": 400,
        },
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
