<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Minimum tokens and SGLang stop strings

The Python SGLang chat frontend now applies local stop-string matching only after
`min_tokens` output tokens. Previously, a request with `min_tokens: 8` and
`stop: ["STOP"]` could end at an earlier `STOP`, even though the minimum reached
the worker correctly. For the scripted output
`Hi STOP one two three four five six seven eight STOP tail`, the response was
`Hi `; it now preserves the first marker and ends before the second one.

Both inline and pooled preprocessing pass the minimum to the postprocessor. It
counts original token IDs and decodes protected tokens individually so a partial
UTF-8 character cannot defer earlier text into stop matching. Ordinary requests
without stop strings or with a zero minimum retain the existing decode path.
A minimum is not a guarantee against an engine error or engine-initiated finish.

CPU validation against base `70e094f650ab89eb81d359825adcb78fc9256aba`:

- 300 frontend unit tests passed; the unavailable TinyLlama byte-fallback fixture
  was excluded. New coverage includes inline/pool paths, batch sizes 1/7/64,
  UTF-8 across the minimum, terminal output below the minimum, and logprob alignment.
- A localhost native HTTP probe covers Rust admission, Python processing, native
  routing/TCP, and a scripted token worker. Baseline passed 8/24 cases; candidate
  passed 24/24. All 16 positive-minimum cases are corrected; eight zero-minimum
  controls are unchanged. Worker iterators finish before runtime shutdown.
- The probe uses the Qwen3-0.6B tokenizer at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, without weights. CPU engine dependencies
  came from SGLang commit `3c82e48e825bae504527ed28e2442b1429d36650`.

With built Dynamo bindings, SGLang frontend dependencies, and the pinned tokenizer
metadata cached, run from the repository root:

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  --confcutdir=components/src/dynamo \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'not byte_fallback'
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/sglang_minimum_stop_probe.py \
  --output /tmp/sglang-minimum-stop-candidate
```

To reproduce the baseline, copy the probe to a temporary location, check out the
base in a separate worktree, point `PYTHONPATH` to its `components/src`, and run
that probe with `--expect-baseline-failure` and a different output directory.
The probe saves HTTP responses, forwarded requests, module paths and hashes.

The selected approach keeps enforcement in the existing stop matcher. Checking a
whole batch only at its end would mishandle batches crossing the minimum; removing
local matching would lose stop strings on tokenizer-free workers. The first
batch-split implementation failed a UTF-8 regression and was corrected before
publication. Review was an in-context self-review, not an independent review.

Cluster validation pending: no real SGLang generation, GPU or disaggregated test
was run for this change. Native runtime shutdown still reports live tasks at
interpreter exit; this does not establish runtime leak freedom. No performance
improvement has been measured. Per-token decoding adds work only while protecting
a positive minimum for a request with stop strings; its serving cost is unmeasured.
