<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## vLLM reasoning-to-tool streaming handoff

With the Python vLLM chat processor, a chunk containing both `</think>` and
`<tool_call>` previously switched to a separate buffered parser. When another
tool call followed, the streaming parser started again at index 0: clients could
receive conflicting call IDs, concatenated arguments and a literal closing tag.
The same response delivered in one chunk or without streaming could succeed.

The processor now gives the existing streaming tool parser the post-reasoning
text and original content token IDs. That parser retains call identity across
subsequent chunks. This also avoids withholding the first call until generation
finishes. Non-streaming parsing is unchanged; `length` remains `length`.

### Reproduction and CPU evidence

In the repository's vLLM test environment, with the Qwen3-0.6B tokenizer cached:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -q -k test_reasoning_tool_handoff_preserves_parallel_call_identity

PYTHONPATH=components/src python -m pytest \
  tests/frontend/test_prepost.py tests/frontend/test_prepost_mistral.py \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py -q
```

The regression uses real vLLM parsers and tokenizer, two `get_weather` calls,
Unicode/escaped arguments, and a reasoning-end/tool-start chunk followed by
separate closing-tag and second-call chunks. Baseline: two streaming failures,
two non-streaming controls passed. Candidate: all four passed. The broader
Hermes, Qwen and Mistral suites passed 187 tests.

A separate localhost HTTP replay used real Rust admission/routing, TCP worker
transport, Python `VllmProcessor`, vLLM `InputProcessor`/`OutputProcessor`, and
unary/SSE conversion. Its worker supplied deterministic token IDs, appending
EOS for stop termination. Four corpora (parallel calls, Unicode/escaping,
length termination, ordinary text) crossed unary/SSE and output intervals
1, 8, 20, 32 and 4096:

| CPU HTTP replay | Baseline | Candidate |
| --- | --- | --- |
| Correct responses | 36/40 | 40/40 |
| Parallel calls, interval 8, SSE | incorrect arguments | two intact calls |
| Unicode calls, intervals 8 and 20, SSE | incorrect arguments | two intact calls |
| Length finish, interval 8, SSE | incorrect arguments | two intact calls; `length` retained |

Checks cover call indexes, stable IDs, arguments, reasoning, visible content,
finish reason, worker dispatch and SSE `[DONE]`. The replay used vLLM 0.29.0
source (`98dff2a81d747d1dba01a47f939f48c3526d4206`) and Qwen3-0.6B tokenizer
revision `c1899de289a04d12100db370d81485cdf75e47ca`. It did not generate model
outputs. Cluster validation is pending. No performance improvement has been
measured.
