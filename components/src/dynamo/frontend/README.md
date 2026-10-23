<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.


## SGLang detailed finish diagnostics

Request `"nvext": {"extra_fields": ["detailed_finish_reason"]}` to receive the
normalized backend finish reason before OpenAI conversion. The SGLang Python
processor now retains this existing extension when rebuilding responses. A
locally matched stop string reports `stop`; intermediate chunks and requests
that do not opt in omit the field. Both inline and pool preprocessing use the
same response builder. Older workers need only their existing `finish_reason`.

Previously, this processor omitted the requested field. Local HTTP tests through
the real router, Python processor and unary/SSE handlers observed:

| Worker reason | Baseline extension | Candidate extension | HTTP modes |
| --- | --- | --- | --- |
| `cancelled` | absent | `cancelled` | unary and SSE |
| `stop` | absent | `stop` | unary and SSE |
| `length` | absent | `length` | unary and SSE |

All 12 baseline/candidate responses were HTTP 200 and retained the complete
`get_weather` call with `{"city":"Paris"}` arguments. This experiment used
synthetic backend tokens, a cached Qwen3-0.6B tokenizer, SGLang 0.5.19 native
parsers and installed Dynamo bindings. It did not run model generation. The
ordinary OpenAI finish reason is unchanged: cancelled tool output can still
report `tool_calls`. This diagnostic extension does not fix that separate
cancellation-finalization issue.

Reproduce the processor regression in the SGLang test environment from the
repository root (with Dynamo bindings installed):

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -q -k test_processor_detailed_finish_reason
```

The initial baseline had three failing requested-field cases and three passing
opt-out controls. The final eight cases also cover a local stop. The affected
processor, tool-call and metrics suites passed 322 tests; one existing
byte-fallback tokenizer case was deselected and was not requalified in this run.
The new test uses the actual processor and tokenizer with synthetic routed
responses; it does not establish engine or GPU qualification.

For a running SGLang frontend, this request exercises the same extension:

```bash
curl "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Count to ten"}],"max_tokens":1,"nvext":{"extra_fields":["detailed_finish_reason"]}}'
```

Replace `MODEL` with the served model. A length-limited response should include
`"nvext":{"detailed_finish_reason":"length"}`; add `"stream":true` to check SSE.
This real-model check is pending cluster validation. No performance improvement
has been measured.
