<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## vLLM terminal error diagnostics

The vLLM chat processor accepts Dynamo's canonical worker finish reason,
`{"error": "controlled worker failure"}`, as a terminal internal error.
Previously, string-only finish mapping raised `AttributeError: 'dict' object has
no attribute 'startswith'`, replacing the worker diagnostic in internal error
frames and logs. Rust also normalizes legacy `error: ...` finishes to this map.

The processor now sends the diagnostic through the existing error-envelope
utility, stops consuming output, and releases registered vLLM request state.
It does not pass the diagnostic through the exception-based client-error
translator. Empty messages use the existing request-specific fallback.
HTTP error bodies remain sanitized; normal stop handling is unchanged.

Reproduce the CPU regression with the project's vLLM 0.29.0 dependencies and
Qwen3 tokenizer available:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -q -k canonical_error_finish
```

The original two diagnostic regressions fail on main; all three candidate cases
pass, covering a backend diagnostic, an empty diagnostic, and text resembling a
client-error envelope. They also check terminal consumption and request cleanup.
The full processor suite passes 168 tests.

Additional local HTTP/TCP validation used real Dynamo routing, vLLM
InputProcessor/OutputProcessor and Hermes parsing with synthetic worker tokens.
Canonical and legacy errors both lost their diagnostic on the baseline (2/2).
The candidate preserves it before and after visible text (4/4); 18 HTTP response
checks cover these errors and successful stop controls across baseline/candidate.
These CPU checks do not qualify real model generation. The installed native
binding's source revision was not verified. Cluster validation and full CI are
pending. No performance improvement has been measured.
