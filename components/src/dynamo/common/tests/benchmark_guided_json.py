# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired CPU cycle-check benchmark; no engine, tokenizer, or GPU involved."""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import statistics
import time
from pathlib import Path

import dynamo.common.utils.guided_json as candidate
from dynamo.llm import HttpError


def outcome(module, payload):
    try:
        module.reject_nonprogressing_guided_json_ref_cycles(payload)
    except HttpError as error:
        return error.code, str(error)
    return None


def make_corpus():
    corpus = []
    for size in (16, 1024):
        for kind, name, pointer in (
            ("plain", "Value", "Value"),
            ("escaped", "a/b", "a~1b"),
            ("no-ref", "Value", None),
        ):
            child = {"$ref": f"#/$defs/{pointer}"} if pointer else {"type": "string"}
            schema = {
                "type": "object",
                "$defs": {name: {"type": "string"}},
                "properties": {f"p{i}": dict(child) for i in range(size)},
            }
            corpus.append((f"{kind}-{size}", schema))
    return corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_guided_json", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    source_path = Path(candidate.__file__).resolve()
    expected_path = Path(__file__).resolve().parents[1] / "utils/guided_json.py"
    assert source_path == expected_path, source_path
    args.output.mkdir(parents=True, exist_ok=True)
    token_count = 0
    for length in range(6):
        for chars in itertools.product("a~01/é😀", repeat=length):
            token = "".join(chars)
            assert baseline._decode_pointer_token(
                token
            ) == candidate._decode_pointer_token(token)
            token_count += 1
    corpus = make_corpus()
    parity = []
    checks = list(corpus)
    for name, pointer in (
        ("plain", "plain"),
        ("a/b", "a~1b"),
        ("a~b", "a~0b"),
        ("a~1b", "a~01b"),
        ("é😀", "%C3%A9%F0%9F%98%80"),
        ("", ""),
        ("bad~2", "bad~2"),
        ("bad~", "bad~"),
    ):
        ref = f"#/$defs/{pointer}"
        checks.append(
            (f"cycle-{pointer}", {"$defs": {name: {"$ref": ref}}, "$ref": ref})
        )
    for case, schema in checks:
        for representation, payload in (
            ("dict", schema),
            ("string", json.dumps(schema)),
        ):
            before = json.dumps(payload, sort_keys=True)
            left, right = outcome(baseline, payload), outcome(candidate, payload)
            assert left == right, (case, left, right)
            assert before == json.dumps(payload, sort_keys=True)
            parity.append(
                {"case": case, "representation": representation, "outcome": left}
            )
    summaries = []
    with (args.output / "timings.csv").open("w", newline="") as raw:
        writer = csv.writer(raw, lineterminator="\n")
        writer.writerow(
            ("case", "representation", "pair", "variant", "sample", "elapsed_ns")
        )
        for case, schema in corpus:
            for representation, payload in (
                ("dict", schema),
                ("string", json.dumps(schema)),
            ):
                for module in (baseline, candidate):
                    for _ in range(20):
                        module.reject_nonprogressing_guided_json_ref_cycles(payload)
                pairs = []
                for pair in range(4):
                    variants = [("baseline", baseline), ("candidate", candidate)]
                    if pair % 2:
                        variants.reverse()
                    medians = {}
                    for variant, module in variants:
                        samples = []
                        for sample in range(100):
                            start = time.perf_counter_ns()
                            module.reject_nonprogressing_guided_json_ref_cycles(payload)
                            elapsed = time.perf_counter_ns() - start
                            samples.append(elapsed / 1000)
                            writer.writerow(
                                (case, representation, pair, variant, sample, elapsed)
                            )
                        medians[variant] = statistics.median(samples)
                    pairs.append(medians)
                before = statistics.median(p["baseline"] for p in pairs)
                after = statistics.median(p["candidate"] for p in pairs)
                summaries.append(
                    {
                        "case": case,
                        "representation": representation,
                        "unit": "us/call",
                        "baseline": before,
                        "candidate": after,
                        "change_percent": 100 * (after / before - 1),
                        "pair_medians": pairs,
                    }
                )
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    report = {
        "scope": "CPU guided JSON cycle check; not serving performance",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": cpu,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "baseline_source_sha256": hashlib.sha256(
            args.baseline.read_bytes()
        ).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, sort_keys=True).encode()
        ).hexdigest(),
        "seed": 0,
        "pairs": 4,
        "order": "AB BA AB BA",
        "warmups_per_variant_case": 20,
        "samples_per_variant_case": 400,
        "token_parity_cases": token_count,
        "schema_parity": parity,
        "results": summaries,
        "limitations": "Shared CPU; no affinity or frequency controls. Warm function timings, not schema compilation. No model, image, tokenizer, GPU, HTTP or engine execution.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
