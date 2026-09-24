<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Literal replacement characters in SGLang streams

A complete literal U+FFFD (`�`) token used to remain buffered until a following
non-replacement character or the finish event. The streaming postprocessor now
recognizes a final token whose decoded text round-trips to that same token ID and
releases the complete text immediately. Incomplete UTF-8 byte tokens remain
buffered, and stop-string filtering still runs before delivery.

The [CPU HTTP reproduction](tests/reproduce_sglang_literal_streaming.py) uses real
native HTTP/RPC transport, the SGLang processor, and the Qwen3 tokenizer with a
scripted worker. The worker sends tokens, waits for the HTTP client to acknowledge
content, then sends finish (with a 0.5-second timeout to let baseline complete).
This checks event order, not a model-serving latency estimate.

[Recorded results](tests/sglang_literal_streaming_results.json) compare main
`70e094f650ab89eb81d359825adcb78fc9256aba` with this change:

| Cases | Baseline delivers before finish | Candidate delivers before finish |
| --- | ---: | ---: |
| Literal `�`, `A�`, `��`, plain and guided requests | 0/6 | 6/6 |
| Ordinary text controls | 2/2 | 2/2 |
| Incomplete UTF-8 controls (must wait) | 0/2 | 0/2 |

All ten cases preserve final content and finish reason. Four new regressions fail
on main; the affected CPU suites pass 318 tests, excluding one unavailable
TinyLlama tokenizer fixture.

With Dynamo native bindings built, SGLang 0.5.19 installed, and the tokenizer
revision from the results file cached, run from the repository root:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py \
  -q -k 'not test_byte_fallback_sequence_longer_than_six_tokens'
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/reproduce_sglang_literal_streaming.py \
  --output /tmp/sglang-literal-streaming
```

No performance improvement has been measured. Cluster validation is pending;
these checks do not validate constrained model generation. Tokenizers that encode
literal U+FFFD as several byte tokens conservatively retain the old buffering
behavior. Existing logprob-formatting issues are outside this change. Native
interpreter-exit task warnings leave shutdown behavior unqualified.
