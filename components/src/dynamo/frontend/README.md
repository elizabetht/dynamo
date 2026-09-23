<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## vLLM token constraints through the Python chat processor

With `--dyn-chat-processor vllm`, accepted `allowed_token_ids` and
`bad_words_token_ids` previously disappeared when the Rust HTTP request was
serialized for Python. The vLLM adapter also omitted them from its worker
request. A client could receive a successful response without either constraint
reaching the engine.

Chat serialization now preserves the existing admitted extension fields and
continues to drop unknown fields. The vLLM adapter forwards token constraints
through the existing `extra_args.sampling_options` worker contract. Token IDs
are preserved without decoding or retokenization. Native vLLM validation now
rejects empty or out-of-vocabulary allowlists before worker dispatch.

### CPU validation

Run with the repository's pinned vLLM dependencies and a cached Qwen3-0.6B tokenizer:

```bash
cargo test --locked -p dynamo-llm --no-default-features test_passthrough --lib
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py -q
```

The regression tests exercise admitted-field serialization and the Python
request-to-worker conversion. The Python test substitutes rendering and dispatch;
it is not a model-generation test. The focused Rust tests passed (5 tests), and
the processor suite passed (168 tests). The new Python regressions fail twice on
the baseline and preserve the unconstrained control.

A separate CPU replay used rebuilt baseline/candidate native bindings, real
localhost HTTP/router/TCP, native vLLM preprocessing and the worker's
`build_sampling_params`, with synthetic output tokens. Across 36 requests per
revision, unary/SSE and stream intervals 1/20:

| Observation | Baseline | Candidate |
| --- | ---: | ---: |
| Exact token-constraint forwarding, including empty bad-word list | 0/20 | 20/20 |
| Empty/out-of-vocabulary allowlists rejected before dispatch | 0/8 | 8/8 |
| Malformed allowlist shape rejected | 4/4 | 4/4 |
| Unconstrained chat controls preserved | 4/4 | 4/4 |

The forwarding corpus includes ordinary chat, JSON-object guidance and automatic
tool choice. For example, `allowed_token_ids: [42, 43]` and
`bad_words_token_ids: [[42, 43]]` now arrive unchanged in the worker's native
`SamplingParams`. A compatible worker must implement the existing token
constraint contract. Other backends and mixed-version serving remain unqualified.

Cluster validation pending. These checks establish transport and parameter
construction, not constrained model generation or GPU scheduling behavior.
No performance improvement has been measured.
