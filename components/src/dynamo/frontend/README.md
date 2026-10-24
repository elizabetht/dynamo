<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang stop-string inclusion

The Python SGLang chat processor honors `include_stop_str_in_output: true`.
For generated text `Hi STOP tail` and `stop: ["STOP"]`, it returns `Hi STOP`
and finishes with `stop`. Previously it returned `Hi ` even when inclusion
was requested. False and omitted values keep the existing trimmed output.
This applies to streaming and non-streaming chat responses, including guided
requests and requests with automatic tools enabled.

Both preprocessing paths pass the option to the streaming postprocessor. Its
existing stop matcher selects the visible end after the matched string when
inclusion is enabled. It still cancels further generation, removes trailing
text, and keeps logprobs for the visible stop string. Token-ID stops and model
EOS handling are unchanged. This change does not depend on worker-side text
trimming: the token-based SGLang worker forwards raw output IDs and the Python
frontend owns stop-string matching.

CPU validation against base `70e094f650ab89eb81d359825adcb78fc9256aba`:

| Native localhost HTTP cases | Baseline | Candidate |
| --- | --- | --- |
| Inclusion enabled | 0/24 correct | 24/24 correct |
| Inclusion false or omitted | 48/48 correct | 48/48 correct |

The corpus covers plain, regex-guided and auto-tool requests; SSE and unary
responses; token batches of 1 and 7; frontend stream intervals of 1 and 20.
The worker emits scripted token IDs. Native HTTP admission/routing, the real
Python processor and cached Qwen3 tokenizer run locally without model weights.
The focused unit suite passes 289 tests (one unavailable tokenizer fixture
excluded), including inline/pool forwarding and split-marker logprobs.

Reproduce in a Dynamo development environment with the pinned SGLang 0.5.19
and vLLM 0.29.0 frontend dependencies and native bindings installed. Cache the
Qwen/Qwen3-0.6B tokenizer at revision
`c1899de289a04d12100db370d81485cdf75e47ca` first; the HTTP probe is offline:

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  --confcutdir=components/src/dynamo \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'not byte_fallback'
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/sglang_stop_inclusion_probe.py \
  --output /tmp/sglang-stop-inclusion
```

To reproduce the baseline, run the same probe against the base revision's
processor modules with `--expect-baseline-failure`. It asserts exactly the
24 inclusion failures and saves responses and imported module hashes.
Local validation used CPU dependency overlays, not a qualified serving image.
Pool tests use a thread executor to exercise the pool branch, not subprocess
startup. The native runtime reports nine tasks at interpreter exit; per-request
finalization passes, but whole-runtime leak freedom is not established.
Cluster generation and disaggregated serving validation remain pending.
No performance improvement has been measured.
