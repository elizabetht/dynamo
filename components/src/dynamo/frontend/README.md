<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Unicode text in SGLang logprobs

A generated literal U+FFFD (`�`) represented by a complete token previously
remained in response content but
became an empty logprob token with null bytes. The processor mistook the complete
token for an unfinished UTF-8 sequence. It now recognizes a complete token when
encoding its decoded text returns the same token ID. Context reconstruction also
keeps these literal tokens when completing a subsequent split-byte character.
Tokens containing a leading byte fragment followed by complete text also enter
context reconstruction. For example, Qwen tokenizes `있다` as two IDs whose
isolated decodes are `�` and `�다`; previously the logprob text was `�다` even
though response content was correct. The check runs only for token strings
containing U+FFFD; ordinary tokens retain the existing path. This does not add a
generic raw-byte tokenizer API, and a noncanonical token that does not round-trip still uses the existing fallback.
Literal U+FFFD assembled from multiple byte tokens is outside this narrow fix.

CPU validation: 322 frontend/tool tests passed (one uncached TinyLlama fixture
excluded). Ten Unicode regressions fail on the baseline and pass with the fix.
Native localhost HTTP admission, Rust routing, real SGLang preprocessing and the
Python response processor were exercised with a scripted token worker and the
cached Qwen3-0.6B tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Across plain/guided requests,
streaming/unary responses, worker batches 1/7 and frontend intervals 1/20,
logprob text fidelity improves from 32/128 to 128/128. All 128 response-content and
finish checks pass in both arms. The 32 ordinary Unicode/ASCII controls retain
correct output. [Grouped HTTP evidence](tests/sglang_unicode_logprobs_http_results.json)
is committed without private infrastructure details.

Reproduce in a Dynamo development environment with the pinned SGLang dependency
and native bindings, after caching the tokenizer above:

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py \
  -q -k 'not test_byte_fallback_sequence_longer_than_six_tokens'
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/reproduce_sglang_unicode_logprobs.py \
  --output /tmp/sglang-unicode-results
```

Run the HTTP observer on the baseline and candidate separately and inspect
`text_fidelity` in `results.json`. Its worker is scripted; no weights are loaded.
Cluster validation and actual generated-token coverage remain pending. The
round-trip checks add tokenization work on the rare replacement-character path.
No performance improvement has been measured.
