<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Forced tool choices with parser-provided JSON

With the vLLM chat processor, a parser such as Hermes can install the standard
JSON schema for a named or required tool choice. Previously, Dynamo forwarded
that schema to the worker but sent the resulting bare JSON to the parser's
native `<tool_call>` decoder. For example, a named `get_weather` request returning
`{"city":"Paris"}` became ordinary message content with `finish_reason="stop"`.
It now becomes a `get_weather` tool call with those arguments and
`finish_reason="tool_calls"`.

Preprocessing selects the existing JSON tool decoder when the parser-provided
schema equals vLLM's standard schema for the requested tools. This avoids treating
an arbitrary parser-specific JSON format as the standard tool format. Existing
structural-tag and grammar routes retain their native decoders. The JSON decoder
separates reasoning before decoding arguments, preserves reasoning suppression,
and retains `length` for truncated output. It buffers until the terminal chunk;
this change does not introduce incremental forced-tool argument streaming.

### Reproduce the CPU regression

Use the repository's vLLM development environment (vLLM 0.29.0) and cache the
`Qwen/Qwen3-0.6B` tokenizer. No GPU or model weights are required:

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -k 'parser_json_guidance_uses_json_tool_response or parser_native_json_keeps_native_decoder'

PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/frontend/test_prepost.py tests/frontend/test_prepost_mistral.py \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py
```

The regression uses the real Hermes and Qwen3 reasoning parsers. Only prompt
rendering is replaced in the unit test. Copying the new regression onto baseline
`70e094f650ab89eb81d359825adcb78fc9256aba` exposes the incorrect decoder selection.
The custom-JSON control verifies that a different wire schema keeps its parser.
The internal `include_reasoning=False` cases exercise processor behavior; this
parameter is currently rejected by the public HTTP endpoint.

### Supported-path observations

A separate CPU replay used actual localhost HTTP admission, Rust routing/TCP
transport, `VllmProcessor`, and native vLLM input/output processors. A synthetic
worker returned fixed token sequences, using vLLM revision
`98dff2a81d747d1dba01a47f939f48c3526d4206` and Qwen3 tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The native binding's Rust sources match
the baseline. Baseline and candidate received identical worker-bound requests;
complete JSON payloads passed independent validation against the captured schemas.

| Check | Baseline | Candidate |
| --- | --- | --- |
| Valid response observations matching expected calls, arguments, reasoning and finish | 32/72 | 72/72 |
| Unsupported HTTP `include_reasoning` rejected before worker dispatch | 16/16 | 16/16 |

The 72 response observations cover named and required choices, parallel calls,
Unicode and escaping, empty arguments, truncation, and auto/none native-markup
controls. Each runs unary and SSE with token batching intervals 1, 8, 20, and
4096. Checks include distinct call IDs, indexes, exact decoded arguments, a single
terminal finish and a single SSE `[DONE]`. Forty baseline observations returned
plain content where a tool call was expected; all forty are corrected.

No performance improvement has been measured. The schema comparison adds work
once per applicable forced-tool request; its cost has not been benchmarked.
Cluster validation and real model generation remain pending. These CPU replays
do not establish grammar enforcement, serving throughput, cancellation behavior,
or shutdown leak freedom.
