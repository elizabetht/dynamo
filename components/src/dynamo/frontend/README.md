<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang Hermes parallel-call finalization

With `--dyn-chat-processor sglang --tool-call-parser hermes`, the frontend
flushes the first generated token immediately, then batches tokens (20 by
default). SGLang 0.5.19 can finish parsing the first call while leaving a later
call buffered. Previously, complete first-call arguments suppressed Dynamo's
final reparse, so two complete calls could produce only one returned tool call.

Hermes responses now use the existing full-text final reparse even when the
first call is complete. Known-name filtering, sequential indexes, unique IDs,
and the existing `length` finish reason are retained. Tool calls are still
emitted only at finalization. Required/named JSON-array parsing and other parser
implementations retain their existing reparse conditions.

### CPU reproduction

Use the repository's SGLang test environment (SGLang 0.5.19 and transformers
5.12.1). Cache the Qwen/Qwen3-0.6B tokenizer at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, then run from the checkout:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py -q
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'not test_byte_fallback_sequence_longer_than_six_tokens' -q
```

The new tests call `SglangProcessor.generator` with real preprocessing,
tokenization, batching and SGLang parsing. Only the routed engine is a test
double supplying deterministic output tokens; no model is loaded.

| Check | Baseline | Candidate |
| --- | --- | --- |
| Tool-call suite including new generator regressions | 30 passed, 4 failed | 34 passed |
| Available combined processor/tool-call suite | Not rerun in full | 303 passed, 1 deselected |

The four baseline failures lose the second complete call at batch intervals 20
and 32, with either `stop` or `length`. Tokenwise, truncated-second-call and
ordinary-text controls pass. The deselected existing byte-fallback test needs
an uncached TinyLlama tokenizer; the initial full run had 303 passes and that
fixture error. With both tokenizers available, omit `-k` to run the full suite.

The fix reuses the existing finalization path instead of inspecting SGLang's
private buffer or forcing tokenwise processing. It adds one final parse for
Hermes tool output, with cost proportional to the response; that cost has not
been benchmarked. No performance improvement has been measured. HTTP transport,
real-engine output and GPU validation are pending; these CPU results are not
end-to-end serving qualification.
