<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.


## Legacy guided-choice validation

The shared Python frontend helper `legacy_guided_decoding` validates that
`guided_choice` is a list of strings before converting it into backend guidance.
Previously, it checked the list container but accepted non-string elements such
as `{"guided_choice": ["yes", 123]}`. The updated helper raises `InvalidArgument`:

```text
guided_choice must be a list of strings; element 1 has type int
```

The check reports the offending element's index and type without including its
contents. It preserves valid strings, including empty strings, Unicode and
duplicates. `null` and empty lists remain inactive constraints, and an explicitly
empty whitespace modifier is preserved. This change adds element validation to
the existing shared helper; it does not change the grammar engine.

### Validation

With the Dynamo Python bindings and pytest installed, run from the repository root:

```bash
PYTHONPATH=components/src python -m pytest components/src/dynamo/frontend/tests/test_frontend_utils.py -q
```

The regression cases cover seven invalid element types in both the first and
second positions. Against the original implementation, these 14 cases failed
while 39 cases passed. With the fix, all 53 focused CPU tests passed. Formatting,
lint, syntax compilation and diff checks also passed.

This evidence covers the shared Python helper using the real Dynamo exception
type. Backend integration and HTTP behavior have not been validated by this
experiment; vLLM and SGLang were not installed in its CPU test environment.

### Performance

**No performance improvement has been measured.** This is a correctness change.
The added validation scans at most all choice elements, stops at the first
invalid element, and uses O(n) time and O(1) auxiliary space. Its runtime cost has
not been benchmarked. No GPU serving, throughput, time-to-first-token or
inter-token latency comparison was performed, and CPU test duration is not a
serving-performance measurement.
