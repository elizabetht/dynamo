<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# HTTP streaming error compatibility

During mixed-version operation, older Python workers can report an HTTP exception
as `Backend(Unknown)` or `Backend(InvalidArgument)` with a JSON status envelope.
With the default streaming configuration, a legacy worker rejection with code 400
previously became a chat SSE error with code 500. Unary error extraction already
recognized the envelope.

The stream boundary now normalizes these legacy envelopes into the existing
semantic error classes. The SSE error code and request failure metrics reflect
the rejection. Anthropic streaming renders the same normalized error captured by
the stream signal. Canonical errors retain their classification and explicit
public messages; private worker diagnostics remain private. This adapter is
limited to the supported N-2 worker compatibility window.

Run the CPU HTTP regressions from the repository root:

```sh
cargo test --locked -p dynamo-llm --no-default-features \
  --test frontend_protocol_validation_http
DYN_HTTP_OVERLOAD_STATUS_CODE=503 cargo test --locked -p dynamo-llm \
  --no-default-features --test frontend_protocol_validation_http \
  legacy_worker_http_errors_keep_status_in_chat_sse
```

The regression uses the real HTTP service with a scripted worker and deserializes
the legacy worker wire representation. It checks early and late SSE failures,
status codes, terminal framing, failure metrics, diagnostic sanitization,
canonical precedence, malformed envelopes and Anthropic errors. On the baseline,
the supported JSON-object request receives an SSE error with code 500 instead of
400. This is CPU protocol coverage, not model inference or a benchmark.

Validation on the candidate: 13 HTTP protocol tests and 363 HTTP service unit
tests passed. The legacy-status HTTP test also passed in a separate process with
`DYN_HTTP_OVERLOAD_STATUS_CODE=503`. The baseline at
`70e094f650ab89eb81d359825adcb78fc9256aba` fails the new HTTP assertion with SSE
code 500 where 400 is expected.

No performance improvement has been measured. Cluster validation is pending.
A matched immutable frontend build is required before testing the fix with a
real inference backend. Legacy diagnostic text is intentionally not exposed, so
this change does not promise that a backend's raw validation message reaches the
client.
