# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU postprocessing cost with an earlier sglang_prepost.py revision."""

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

from transformers import AutoTokenizer

import dynamo.frontend.sglang_prepost as candidate

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CORPUS = {
    "ascii": "Explain how streaming text works. " * 32,
    "korean": "있다 없는 숗 한글 문장입니다. " * 32,
    "emoji": "😊🚀🤖👩🏽‍💻 " * 32,
    "literal": "A�B� C�D� " * 32,
    "json": json.dumps({"message": "한국어 😊�" * 32}, ensure_ascii=False),
}


def process(cls, tokenizer, ids, batch):
    post = cls(tokenizer=tokenizer, tool_call_parser=None, reasoning_parser=None)
    events = []
    for index in range(0, len(ids), batch):
        event = post.process_output({"token_ids": ids[index : index + batch]})
        if event is not None:
            events.append(event)
    events.append(post.process_output({"token_ids": [], "finish_reason": "length"}))
    return events


def main(args):
    spec = importlib.util.spec_from_file_location(
        "dynamo.frontend.literal_baseline", args.baseline_source
    )
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    classes = {
        "baseline": baseline.SglangStreamingPostProcessor,
        "candidate": candidate.SglangStreamingPostProcessor,
    }
    rows = []
    for name, text in CORPUS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        for batch in [1, 20]:
            expected = process(classes["baseline"], tokenizer, ids, batch)
            observed = process(classes["candidate"], tokenizer, ids, batch)
            assert observed == expected, (name, batch)
            assert "".join(e["delta"].get("content", "") for e in observed) == text
            for _ in range(args.warmup):
                for cls in classes.values():
                    process(cls, tokenizer, ids, batch)
            for pair in range(args.pairs):
                order = list(classes) if pair % 2 == 0 else list(reversed(classes))
                for arm in order:
                    start = time.perf_counter_ns()
                    for _ in range(args.requests):
                        process(classes[arm], tokenizer, ids, batch)
                    rows.append(
                        {
                            "corpus": name,
                            "tokens": len(ids),
                            "batch": batch,
                            "pair": pair,
                            "arm": arm,
                            "us_per_request": (time.perf_counter_ns() - start)
                            / (args.requests * 1000),
                        }
                    )
    summary = []
    for name in CORPUS:
        for batch in [1, 20]:
            samples = {
                arm: [
                    row["us_per_request"]
                    for row in rows
                    if row["corpus"] == name
                    and row["batch"] == batch
                    and row["arm"] == arm
                ]
                for arm in classes
            }
            medians = {arm: statistics.median(times) for arm, times in samples.items()}
            summary.append(
                {
                    "corpus": name,
                    "batch": batch,
                    "median_us_per_request": medians,
                    "range_us_per_request": {
                        arm: [min(times), max(times)] for arm, times in samples.items()
                    },
                    "reduction_pct": 100
                    * (1 - medians["candidate"] / medians["baseline"]),
                }
            )
    result = {
        "scope": "CPU postprocessor microbenchmark; no inference or HTTP timing",
        "model": MODEL,
        "tokenizer_revision": REVISION,
        "source_sha256": {
            "baseline": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
            "candidate": hashlib.sha256(
                Path(candidate.__file__).read_bytes()
            ).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["sglang", "transformers", "tokenizers"]
            },
        },
        "pairs": args.pairs,
        "requests_per_arm_per_pair": args.requests,
        "warmup_requests_per_arm": args.warmup,
        "corpus": CORPUS,
        "event_parity_cases": 10,
        "rows": rows,
        "summary": summary,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 3 or args.requests < 1 or args.warmup < 0:
        parser.error("require pairs >= 3, requests >= 1 and warmup >= 0")
    main(args)
