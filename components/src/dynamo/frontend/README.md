<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Integer stop arrays with the vLLM chat processor

Dynamo accepts token IDs in the chat request's `stop` array. With
`--dyn-chat-processor vllm`, `"stop": [42]` previously reached vLLM as a
string-stop list and failed before worker dispatch. The frontend now translates
nonempty integer arrays to vLLM's `stop_token_ids` before request validation.
Integer `stop` takes precedence over an explicit `stop_token_ids`, matching
Dynamo's Rust request provider. String stops and empty arrays are unchanged;
the input dictionary is not mutated.

Reproduce the CPU regression with the repository's pinned vLLM dependencies:

```bash
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -k 'integer_stop_normalization or string_stop_preserves'
```

Against main `70e094f650ab89eb81d359825adcb78fc9256aba`, the integer-stop
regression fails in both validation modes (2 failures, 3 passing controls).
The candidate passes all 5 cases and the full affected suite (170 tests).

A CPU localhost HTTP check used the real Rust frontend, router, TCP transport,
vLLM 0.29.0 input/output processors and Dynamo worker sampling-parameter builder,
with the Qwen3-0.6B tokenizer at
`c1899de289a04d12100db370d81485cdf75e47ca` and synthetic worker tokens:

| Request cases | Main | Candidate |
| --- | --- | --- |
| Integer stop; plain, JSON-guided and auto-tool requests; unary and SSE (6) | Rejected before worker dispatch | Accepted; requested token ID preserved in worker sampling parameters |
| String-stop controls (6) | Accepted | Unchanged |
| Root-level `stop_token_ids` controls (6) | Field lost before Python | Unchanged; separate serialization fix in draft PR #22 |

SSE rejection uses HTTP 200 with an error event carrying code 400; unary rejection
uses HTTP 400. These observations establish request compatibility and parameter
preservation, not constrained generation or token-stop enforcement by a model.
With a running vLLM chat frontend, the public request to exercise is:

```bash
curl "$DYNAMO_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"Reply briefly."}],"stop":[42],"max_completion_tokens":24}'
```

Use a token ID from the served tokenizer and the registered model name.
Cluster validation is pending. No performance improvement has been measured.
