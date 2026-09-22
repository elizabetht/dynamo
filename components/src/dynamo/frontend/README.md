<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang terminal backend errors

Routed worker failures use `finish_reason: {"error": "<diagnostic>"}`.
Previously the SGLang frontend looked up this dictionary as a string finish
reason and raised `TypeError: unhashable type: 'dict'`, masking the backend
failure in frontend diagnostics. The frontend now handles this terminal value
before detokenization and emits the existing annotated error frame. It does not
flush pending parser text, process tokens attached to the error, or emit a
successful finish. Worker messages remain internal: HTTP responses retain the
existing sanitized 500 error policy.

The Rust routed boundary also normalizes legacy `error: <diagnostic>` strings to
the canonical dictionary. This change uses that normalization and requires no
engine patch. It does not change validation-abort handling or other processors.

With the repository's pinned SGLang dependencies and Qwen/Qwen3-0.6B tokenizer
available, run the CPU regression:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -q -k canonical_error_finish
```

Both regression cases fail on base `70e094f650ab89eb81d359825adcb78fc9256aba`
and pass with this change. The affected processor, tool-parser and metrics
suites passed 316 tests; one test requiring an unavailable TinyLlama tokenizer
was deselected. Local HTTP/TCP routing with synthetic token output additionally
checked canonical and legacy errors before and after visible text: all 18
baseline/candidate response assertions passed, preserving HTTP sanitization and
successful ordinary-stop controls. Four candidate error runs retained the
original internal diagnostic; two baseline error runs raised `TypeError`.
These checks used the installed native binding, whose exact source revision was
not verified, and stock SGLang 0.5.19. They do not qualify real model serving.
Cluster validation is pending. No performance improvement has been measured.
