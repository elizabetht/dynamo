<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Avoid constructing discarded SGLang tool guidance

For automatic tool choice, explicit `response_format` or legacy `guided_*`
guidance already takes precedence over tool grammar. Preprocessing previously
constructed and serialized the tool grammar before discarding it. It now skips
that unused construction. Forced tool choices still build their constraints;
response-format validation, prompt rendering and response parser creation remain.

The automatic override warning and engine diagnostics from constructing a
**discarded** grammar are no longer emitted. In the pinned engine, conflicting
`$defs` in that discarded grammar are logged and caught, not rejected. Local HTTP
replays verify those requests still reach the worker with explicit JSON guidance.
This change does not add support for conflicting definitions in a selected grammar.

### CPU evidence

Compared with `a76e12e6584f6029aeaa551c7cc8967671f67345`, full preprocessing
of strict automatic tools with explicit JSON-schema response format measured:

| Tool count | Baseline µs/request | Candidate µs/request | Median change |
| --- | ---: | ---: | ---: |
| 16 | 6399.52 | 6211.58 | -2.94% |
| 64 | 11214.28 | 10792.68 | -3.76% |

The 64-tool case improved in all four pairs (2.73–4.58%). Ordinary chat in that
run measured 4404.94 → 4404.87 µs/request. Some unchanged controls varied by
about 1%, and individual pairs varied more; no speedup is claimed for every
workload. The separate non-strict corpus and all controls are included in
[the evidence](tests/sglang_guidance_precedence_evidence.json) and
[raw timings](tests/sglang_guidance_precedence_timings.csv).

These are serial CPU component measurements on an AMD Ryzen 7 6800H,
Python 3.11.15, SGLang 0.5.19 (`303def2253ae36fad89eb34dc66a7245003743f8`),
Transformers 5.12.1 and Jinja2 3.1.6, using the Qwen3-0.6B tokenizer at
`c1899de289a04d12100db370d81485cdf75e47ca`. Each of 26 strict/non-strict
cases has four alternating AB/BA pairs, 100 requests per arm/pair and five
warmups: 20,800 timings total. CPU affinity was not isolated. Warnings were
suppressed equally in both timed arms. This candidate is independent of the
separate template-format-cache proposal.

No serving performance improvement has been measured. Cluster validation,
real constrained generation, TTFT, inter-token latency and GPU memory are
pending. Local HTTP uses native Dynamo admission/transport and the real
SGLang processor with synthetic worker tokens; it cannot validate model output.
The local native build reports outstanding runtime tasks at interpreter shutdown;
the bounded replay process exits. Full CI is not established by these local tests.

### Reproduce

Use a Dynamo development environment with compatible native bindings and the
versions above. Pre-cache the pinned tokenizer; no model weights are needed.
From the repository root:

```bash
BENCH_TMP=$(mktemp -d)
TOKENIZER_DIR=/path/to/pinned/Qwen3-0.6B/tokenizer
export PYTHONPATH=components/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
git show a76e12e6584f6029aeaa551c7cc8967671f67345:components/src/dynamo/frontend/sglang_prepost.py > "$BENCH_TMP/baseline.py"
python -m pytest -q components/src/dynamo/frontend/tests/test_sglang_guidance_precedence.py
python components/src/dynamo/frontend/tests/benchmark_sglang_guidance_precedence.py --baseline-source "$BENCH_TMP/baseline.py" --tokenizer "$TOKENIZER_DIR" --output "$BENCH_TMP/strict" --strict
python components/src/dynamo/frontend/tests/benchmark_sglang_guidance_precedence.py --baseline-source "$BENCH_TMP/baseline.py" --tokenizer "$TOKENIZER_DIR" --output "$BENCH_TMP/non-strict"
python components/src/dynamo/frontend/tests/replay_sglang_guidance_precedence.py --baseline-source "$BENCH_TMP/baseline.py" --model-path "$TOKENIZER_DIR" --output "$BENCH_TMP/http-baseline"
python components/src/dynamo/frontend/tests/replay_sglang_guidance_precedence.py --model-path "$TOKENIZER_DIR" --output "$BENCH_TMP/http-candidate"
```

Five focused tests pass (two fail on baseline's redundant work). Another 317
existing processor/tool-parser tests pass; one uncached byte-fallback tokenizer
test was excluded. Twelve paired HTTP cases preserve prompt IDs and worker
guidance. A diagnostic 144-case parser/schema comparison found matching results
and the expected log differences; it is not all-parser or engine qualification.

The selected approach adds no schema cache or cross-request state. Reusing the
live response parser still constructs an unused grammar and couples its state
to grammar generation; caching grammars adds schema retention and invalidation
complexity. In-context self-review traced both direct and process-worker callers,
forced conflicts, validation and diagnostics. It is not an independent review.
