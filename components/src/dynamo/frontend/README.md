<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## vLLM frontend string stops

The vLLM Python frontend owns string-stop detection because the token-only worker
runs without detokenization. Once all choices for a request finish, the frontend
now releases its routed stream after emitting the final response and metrics.
Previously it ignored the native output processor's abort notification and kept
reading worker tokens until the worker finished, delaying HTTP completion and
leaving generation active after a local string stop. Other choices and concurrent
requests retain their own state.

Run the CPU regressions with vLLM installed and the Qwen3 tokenizer available:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -k local_stop_releases -q
```

The four cases use vLLM's native output processor and cover a single local stop,
multiple local stops, and a local stop followed by another choice's length limit.
They also check unrelated request state, local-stop usage fallback, and preservation
of worker-supplied usage. The baseline continues consuming the tail; the candidate
leaves it unread.

Local HTTP validation with vLLM 0.29.0 and the Qwen/Qwen3-0.6B tokenizer exercised
24 requests per revision: ordinary text, automatic tools, two staggered choices,
normal-completion controls, unary/SSE, and stream intervals 1 and 20. Text and tool
outputs matched. All 16 stop cases released the synthetic worker early, and all eight
normal-completion controls retained their output. These checks used real
Rust HTTP/router/TCP and native Python processing with synthetic worker tokens.
They do not establish real-engine abort or GPU release. Cluster validation is
pending. No performance improvement has been measured.

### Usage when generation stops locally

A string stop can finish the request before the token-only worker emits its final
usage. Previously those requests returned no usage even with
`stream_options.include_usage=true`. The final local-stop response now reports
prompt and completion counts from the original token IDs received by the frontend.
For multiple choices, completion usage includes all received token deltas,
including tokens from a stopped choice while another choice remains active.
Worker-supplied usage takes precedence. No text is retokenized, no cached-token
count is invented, and tokens generated remotely after cancellation are unknown.

With the same 24-case CPU HTTP corpus, 12 local-stop responses had no usage on
both the original processor and the initial stream-release candidate. All 12 now
include exact received-token counts; the other 12 retain their existing usage.
Text, parsed tool arguments, choice indices and finish reasons match in all 24
cases. Sixteen local-stop requests still release the synthetic worker early and
eight normal controls still finish naturally. The corpus uses unary and streamed
requests, native stream intervals 1 and 20, `stop=["STOP"]`, one or two choices,
and complete Hermes `get_weather` calls with `{"city":"Paris"}` arguments.

The regression command above reproduces missing usage for one and two locally
stopped choices on the initial candidate, and passes all four cases with this
correction. The affected processor/preprocessing suites pass 187 CPU tests with
vLLM 0.29.0. These are component correctness results, not actual engine generation
or serving performance measurements. Cluster validation remains pending.
