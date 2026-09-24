<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Literal replacement characters in SGLang streams

A complete literal U+FFFD (`�`) token used to remain buffered until a following
non-replacement character or the finish event. The streaming postprocessor now
recognizes a final token whose decoded text round-trips to that same token ID and
releases the complete text immediately. Classification is cached per token ID
within each request, including incomplete-byte results, to avoid repeated encoding.
Incomplete UTF-8 byte tokens remain
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
on main; the affected CPU suites pass 320 tests, excluding one unavailable
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

No serving performance improvement has been measured. Cluster validation is pending;
these checks do not validate constrained model generation. Tokenizers that encode
literal U+FFFD as several byte tokens conservatively retain the old buffering
behavior. Existing logprob-formatting issues are outside this change. Native
interpreter-exit task warnings leave shutdown behavior unqualified.


### CPU cost of literal classification

The uncached early-delivery implementation at `5501c05b4bc0919865c62e7aa0d496648784864d`
re-encoded recurring Unicode fragments. A per-request dictionary avoids repeating
that work. A one-entry cache was also measured, but alternating emoji and mixed
JSON fragment IDs evicted each other, leaving their cost largely unchanged.

The [CPU benchmark](tests/benchmark_sglang_literal_streaming.py) compares that
uncached feature revision with this correction. These are **not speedups over
main**: main lacks the early literal-delivery feature. [Raw samples and ranges](tests/sglang_literal_cpu_results.json)
include five alternating pairs, 20 requests per arm per pair, three warmups per
arm, and exact event parity across five texts and chunk sizes 1 and 20. Input
text is synthetic and repetitive; each measured request creates a fresh processor.
No model generation or HTTP timing is included.

On an AMD Ryzen 7 6800H CPU (x86_64), Python 3.11.15, SGLang 0.5.19,
Transformers 5.12.1 and tokenizers 0.22.2, median microseconds per request at
one token per chunk were:

| Corpus | Uncached | Cached | Reduction |
| --- | ---: | ---: | ---: |
| ascii | 1230.4 | 1233.2 | -0.2% |
| korean | 3160.8 | 2230.3 | 29.4% |
| emoji | 3603.9 | 2003.9 | 44.4% |
| literal | 5164.1 | 1742.7 | 66.3% |
| json | 3235.3 | 1396.6 | 56.8% |

ASCII control differences are within observed variability. Benefit depends on
repeated ambiguous token IDs and chunking; the raw results include the larger
chunks. The cache retains one boolean per encountered ambiguous token ID until
request completion, rather than sharing tokenizer state between requests.

To reproduce with the environment above and the cached tokenizer revision:

```bash
git show 5501c05b4bc0919865c62e7aa0d496648784864d:components/src/dynamo/frontend/sglang_prepost.py > /tmp/sglang-literal-baseline.py
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/benchmark_sglang_literal_streaming.py \
  --baseline-source /tmp/sglang-literal-baseline.py \
  --output /tmp/sglang-literal-cpu.json
```
