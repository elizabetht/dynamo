<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang template-format preprocessing

SGLang message normalization previously inspected the configured Jinja template
on every request. Its compiled-template cache still reparses the AST to detect
whether message content should be strings or structured parts. Dynamo now caches
that format result for up to 32 template strings per process. Each request still
normalizes its own messages and tokenizes its own prompt. Replacing the template
selects a new cache key; non-string values retain the existing detector fallback.
The same function is used by direct and spawned-worker preprocessing.

A CPU full-preprocessing benchmark on an AMD Ryzen 7 6800H, Python 3.11.15,
SGLang 0.5.19, Transformers 5.12.1 and Jinja2 3.1.6 measured the following
warm-cache medians. It uses the Qwen/Qwen3-0.6B tokenizer at revision
`c1899de289a04d12100db370d81485cdf75e47ca`; no model weights are loaded.
Baseline is `a76e12e6584f6029aeaa551c7cc8967671f67345`.

| Input | Baseline µs/request | Candidate µs/request | Reduction |
| --- | ---: | ---: | ---: |
| Ordinary chat | 4381.38 | 102.56 | 97.66% |
| One automatic tool | 4831.96 | 366.39 | 92.42% |
| JSON response format, 16 tools | 6324.59 | 1768.52 | 72.04% |
| Required tool choice, 64 tools | 10915.39 | 6362.89 | 41.71% |
| Strict tools: JSON response format, 16 tools | 6493.06 | 1832.03 | 71.78% |

Each case has four alternating AB/BA pairs, 30 samples per arm/pair and five
warmups per arm: 120 samples per arm/case, 6,240 timings across 26 cases.
The corpus covers automatic/required tools, response-format/legacy JSON,
1/16/64 tool counts, strict/non-strict tools and ordinary chat. Every paired
median improved; per-pair values and variability are in the
[measurement summary](tests/sglang_template_format_evidence.json), with
[raw nanosecond timings](tests/sglang_template_format_timings.csv).
The host was not isolated; these numbers describe this CPU component and corpus.
**No end-to-end serving performance improvement has been measured. Cluster
validation is pending.** First use and eviction still require detection; media
keyword templates already have a fast path and may benefit less.

The small cache avoids adding template snapshots and invalidation plumbing to
both processor and worker initialization. It retains template strings, not
request bodies or tokenizer objects. Its bound is an entry count, not a byte
limit; concurrent cold misses may duplicate pure detection. Malformed string
templates retain their fallback result, with fewer repeated debug diagnostics.

To reproduce in a Dynamo development environment with these dependencies, set
`MODEL_PATH` to a local snapshot of the tokenizer revision above, then run from
the repository root:

```bash
scratch=$(mktemp -d)
git show a76e12e6584f6029aeaa551c7cc8967671f67345:components/src/dynamo/frontend/sglang_prepost.py > "$scratch/baseline.py"
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/frontend/tests/test_sglang_template_format.py \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'not byte_fallback'
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/benchmark_sglang_template_format.py \
  --baseline-source "$scratch/baseline.py" --tokenizer "$MODEL_PATH" --output "$scratch/normal"
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/benchmark_sglang_template_format.py \
  --baseline-source "$scratch/baseline.py" --tokenizer "$MODEL_PATH" --output "$scratch/strict" --strict
```

Five focused tests cover reuse, fresh messages, template replacement, text/media
normalization and missing/malformed/dictionary fallbacks. The available broader
CPU suite passed 291 tests; one existing byte-fallback test needs an uncached
alternate tokenizer and was deselected. Full preprocessing comparisons preserve
token IDs, guidance, parser classes and input requests. Supplemental local native
HTTP/TCP replay matched eight unary/streaming baseline/candidate cases, including
worker guidance and ordered response choices; a real spawned worker matched the
direct path for three requests. Those replays used synthetic responses and do not
qualify model generation. Both HTTP arms reuse native bindings built from
`70e094f650ab89eb81d359825adcb78fc9256aba`, rather than a fresh main build.
Native HTTP shutdown reported remaining runtime tasks
at interpreter exit. Review was two-pass in-context self-review, not independent
review; full CI and GPU qualification remain pending.
