<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Worker-reported content filtering with the vLLM chat processor

A token worker can terminate a response with Dynamo's `content_filter` finish
reason. The Python vLLM chat processor previously translated that reason to
vLLM's `STOP` enum and returned `stop` to the client. After a parsed tool call,
it could instead return `tool_calls`, incorrectly reporting successful tool
completion. This affected streaming and unary chat responses.

vLLM 0.29.0 has no content-filter enum. The processor still uses `STOP` to
finalize vLLM detokenization and release request state, then restores the
worker's `content_filter` reason before tool postprocessing. Ordinary `stop`
and `length` behavior is unchanged. This preserves worker-reported filtering;
it does not add a content filter to vLLM or claim stock vLLM generates this reason.

### Reproduction and CPU evidence

With Dynamo Python bindings, vLLM 0.29.0 and the Qwen3-0.6B tokenizer available:

```bash
PYTHONPATH=components/src python -m pytest -q \
  --confcutdir=components/src/dynamo/frontend/tests \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py
```

The `test_worker_content_filter_finish` regression uses real vLLM
`EngineCoreRequest`, `OutputProcessor`, Hermes parser and Dynamo postprocessing,
with deterministic token-worker responses. On the unchanged source, four
filtered cases fail and four stop/length controls pass. The candidate passes
all eight cases and the complete 173-test processor suite.

A local native HTTP comparison also exercised admitted requests, token-worker
RPC, real vLLM processing and HTTP/SSE aggregation. It used synthetic worker
tokens, not model generation: 20 requests per revision, 40 observations total.
Each case was tested once streaming and once unary.

| Worker result | Baseline client finish | Candidate client finish |
| --- | --- | --- |
| Filtered plain text, terminal tokens or empty terminal chunk | `stop` | `content_filter` |
| Filtered complete tool call, terminal tokens or empty terminal chunk | `tool_calls` | `content_filter` |
| Filtered incomplete tool call | `tool_calls` streaming / `stop` unary | `content_filter` |
| Two choices: filtered first, length-limited second | `stop`, `length` | `content_filter`, `length` |
| Normal stop, plain text / tool call | `stop` / `tool_calls` | Unchanged |
| Length limit, plain text / tool call | `length` | Unchanged |

All observations released per-request output state and finalized their worker
generators. Partial tool arguments retain the existing parser representation;
the filter reason tells clients that the response did not complete normally.
This evidence establishes terminal-status propagation, not filtering policy,
whole-runtime leak freedom or real-model behavior. Cluster validation is pending.
No performance improvement has been measured.
