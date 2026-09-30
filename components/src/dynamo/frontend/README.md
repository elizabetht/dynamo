<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Kimi renderer request metadata

The vLLM chat processor passes the parser-adjusted `tool_choice` and
`response_format` to the renderer's dedicated `ChatParams` fields. Kimi K3 uses
these fields to add model-visible tool-choice and JSON-format instructions.
Previously Dynamo forwarded neither field, so an admitted `tool_choice=required`
request rendered the same prompt as `auto`. JSON-format instructions were also
missing, even when the backend still received a JSON constraint.

Requests without tools leave the renderer's tool choice unset, matching vLLM's
request conversion. Tool stripping and parser adjustments retain their existing
order. This uses fields present in the pinned vLLM 0.29.0 API; it does not change
the grammar or make unsupported named-tool requests valid.

### CPU validation

With Dynamo's Python bindings and vLLM 0.29.0 installed, run from the repository:

```bash
PYTHONPATH=components/src python -m pytest -q   --confcutdir=components/src/dynamo/frontend/tests   components/src/dynamo/frontend/tests/test_vllm_processor_unit.py
```

The four new metadata regressions fail on the unchanged production source and
pass with the fix. The complete processor unit suite passes: **169 tests**.

A local CPU HTTP comparison used native admission, Dynamo's real vLLM processor,
`KimiK3Renderer`, the official Kimi tokenizer at revision
`f831ab66814297da540d832a5235f8e904f29d06`, and synthetic worker output. The engine
source was `d5870cfdac4973e2cda089591995a24fa96abcca` (vLLM 0.29.0). Each row below
was exercised once streaming and once unary on each revision. Counts are prompt
tokens, not timing measurements. The expected token IDs came from the official
tokenizer with the request metadata supplied.

| Request | Baseline tokens | Candidate/expected tokens | Exact candidate token IDs |
| --- | ---: | ---: | --- |
| Auto tool choice | 109 | 109 | Match |
| Required tool choice | 109 | 145 | Match |
| No tool calls, tools supplied | 38 | 76 | Match |
| JSON object response, no tools | 38 | 85 | Match |

Baseline matched 2/8 admitted prompts; candidate matched 8/8. Automatic-tool
controls stayed identical. Two additional named-tool observations per revision
were rejected before worker entry: unary HTTP 400 and a streaming error carrying
code 400. These controls do not demonstrate named-tool support. All successful
requests released processor state and finalized their synthetic worker generator.

The HTTP observations establish prompt construction, not model compliance with
those instructions. No model weights or GPUs were used. Cluster validation is
pending, including real required/none tool output, JSON output, and auto controls
on a feasible Kimi deployment. No performance improvement has been measured.
