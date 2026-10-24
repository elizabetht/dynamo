<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Worker error classification at HTTP boundaries

A canonical worker error's class and safe public message determine the client
response. Private diagnostics do not override them, even when a diagnostic is
JSON shaped like an older worker's HTTP status envelope.

Previously, an `Internal` worker error with diagnostic
`{"message":"private worker detail","code":400}` returned HTTP 400 from unary
Chat Completions. A validation error could likewise become HTTP 500 or lose its
safe public message. Streaming preflight and the Responses terminal error path
shared the same extraction code.

The extractor now limits diagnostic status parsing to legacy fallback identities
without public details. Existing capacity-rejection handling remains unchanged.
Known legacy Python worker envelopes and untyped error annotations remain
supported within the N-2 compatibility window. Canonical errors use the existing
semantic renderer, which keeps diagnostics private.

## Validation

Run these CPU tests from the repository root:

```bash
cargo test --locked --offline -p dynamo-llm --no-default-features --test frontend_protocol_validation_http
cargo test --locked --offline -p dynamo-llm --no-default-features --lib http::service::
```

The regression uses the actual local HTTP service with a scripted worker and
wire-roundtripped errors. On the baseline, the internal-error case fails with
HTTP 400 instead of 500. Candidate checks cover canonical internal, validation,
and unavailable errors across Chat Completions, Responses, and Anthropic unary
and streaming-preflight requests, including safe public messages and metrics.
Separate controls cover legacy 415/503 envelopes and a late Responses failure.

These tests use scripted output, not model inference. Cluster validation is
pending compatible frontend images. No performance improvement has been measured.
