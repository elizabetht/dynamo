<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang logprob token-ID formatting

The Python SGLang frontend honors `return_tokens_as_token_ids=true` on chat
requests with logprobs. For example, selected-token logprobs for `Hello world`
now contain `token_id:9707` and `token_id:1879` with the Qwen3 tokenizer, instead
of `Hello` and ` world`. The response content remains `Hello world`. Omitted or
false options retain decoded text. Logprob `bytes` represent the emitted token
string, matching Dynamo's native chat formatter.

Both direct and preprocessing-pool paths pass the option to the response
processor. Selected-token IDs remain private while stop-prefix buffering needs
decoded text; formatting happens only when entries are emitted. Top alternatives
use their backend token IDs when available. Existing SGLang top-logprob admission
rules remain in effect; the HTTP reproduction uses supported chosen-token-only
`logprobs=true, top_logprobs=0`.

### CPU reproduction

Use a Dynamo development environment with native bindings, SGLang v0.5.19 and
Transformers installed. Cache the Qwen/Qwen3-0.6B tokenizer at revision
`c1899de289a04d12100db370d81485cdf75e47ca`; no model weights are loaded.

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py \
  -k 'not test_byte_fallback_sequence_longer_than_six_tokens'

PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/check_sglang_logprob_token_ids.py \
  --output /tmp/sglang-logprob-evidence
```

The manual check starts an isolated localhost HTTP frontend and scripted token
worker, exercises the native router and Python SGLang processor, and records
responses and import hashes. It verifies actual SGLang worker option validation
and metadata extraction. To reproduce the old behavior, copy this check into a
clean checkout of baseline `70e094f650ab89eb81d359825adcb78fc9256aba`, use that
checkout's `components/src`, and add `--expect-baseline`.

[Recorded CPU results](tests/sglang_logprob_token_ids_cpu_results.json): 64 HTTP
observations per revision, covering plain and guided requests, ASCII and Unicode,
SSE and unary, worker batches of 1/7 tokens and frontend intervals of 1/20.
All 32 token-ID cases change to the requested representation; 32 text-mode
controls retain identical logprob entries. Content and finish checks pass in all
cases. The focused suite passes 315 tests; one existing TinyLlama tokenizer test
was excluded because that tokenizer was unavailable offline. Unicode and split
stop-prefix regressions run in both formatting modes.

In-context self-review compared leaving the option ignored, converting tokens
before buffering, and formatting at emission. Early conversion would break
text-based stop matching; emission-time conversion preserves it. Review covered
request serialization, both processor construction paths, native byte semantics,
stop suppression, and all changed tests. No independent review is claimed.

Cluster validation is pending. These CPU checks use scripted token/logprob data,
not model generation or grammar enforcement, and do not qualify runtime images,
GPU sampling, preprocessing-pool execution, or whole-runtime shutdown (the local
binding reported remaining tasks at interpreter exit). Top alternatives are
covered by unit tests, not the chosen-token-only HTTP corpus.
No performance improvement has been measured.
