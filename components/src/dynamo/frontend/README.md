<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.


## SGLang assistant-message continuation

The Hugging Face template path now forwards explicit `add_generation_prompt` and
`continue_final_message` request fields. Previously it always added a new assistant
header and closed the supplied assistant prefix. With `add_generation_prompt=false`
and `continue_final_message=true`, the prompt now ends at the supplied prefix.
`add_generation_prompt=false` alone closes the last message without starting another.
Explicit top-level controls take precedence over conflicting `chat_template_kwargs`;
valid requests using only nested template settings retain their previous behavior.

The fix uses the tokenizer's continuation support, shared by the inline and process-pool
SGLang preprocessors. It does not change the custom DeepSeek-V4 encoder or claim support
for continuation inside tool-call arguments. Existing HTTP admission rejects incompatible
values of the two top-level flags. Conflicts supplied through `chat_template_kwargs`
or its `chat_template_args` alias are checked after explicit top-level overrides.
They now produce HTTP 400, or an SSE error with code 400 after streaming headers,
instead of a generic server error. No worker request is dispatched.

CPU validation uses the real SGLang 0.5.19 preprocessing code and a Qwen3-0.6B tokenizer
at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Ten regression cases cover ordinary
and JSON-guided requests, default behavior, closed messages, continuation and nested
settings. On the baseline, six fail and four pass. The local HTTP reproducer executes
native admission, the Python processor, TCP worker dispatch and unary/SSE conversion;
it uses synthetic worker tokens, not model inference. Before the fix, four of six
prompt comparisons fail; the default pair passes. After the fix, all six comparisons
pass and the worker receives the exact expected tokens. All ten original unit cases pass.
Four additional unit cases cover nested conflicts and valid top-level overrides.
Before the conflict check, two fail and two pass; afterward all four pass. Four local
HTTP cases cover both aliases in unary and streaming mode. All four change from a
server error to a client error without dispatching to a worker. The broader processor
suite passes 305 cases with the uncached TinyLlama byte-fallback case deselected.

Run in a Dynamo development environment containing the pinned SGLang dependencies:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k "test_chat_generation_controls or test_nested_continuation_conflict" -q
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/reproduce_sglang_continuation.py \
  --model-path /path/to/pinned/tokenizer-and-config --output /tmp/continuation
```

The model path needs tokenizer and model configuration files, not weights. Run the same
reproducer on the baseline with `--expect-baseline` to assert the observed four mismatches.
Add `--invalid-controls` to exercise nested conflicts; use `--expect-baseline` to
assert the previous server errors when running against the baseline implementation.
It binds ephemeral localhost ports, uses isolated in-memory discovery and shuts down its
runtime. Native logs report five to nine tasks at interpreter exit; complete native shutdown
is not qualified. Cluster/model-generation validation is pending.
No performance improvement has been measured.
