<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Model default output length with the vLLM chat processor

When neither `max_completion_tokens` nor `max_tokens` is supplied, the vLLM chat
processor now uses the model's `generation_config.json` `max_new_tokens` default.
Previously it sent the entire remaining context window as the worker's output
limit, overriding that model default. An explicit request limit still takes
precedence. With no model default, the remaining context window remains the limit.
The default is clipped using the processed decoder-input length, so expanded media
placeholders and prompt embeddings count toward the context window.

The change runs before vLLM input validation and forwards the resulting limit in
`stop_conditions.max_tokens`. It does not add worker CLI configuration propagation
or impose a model default as a hard cap on explicit client limits.

Reproduce the CPU regression with the repository's pinned vLLM dependencies and
cached `Qwen/Qwen3-0.6B` tokenizer/config (model weights and GPUs are not used):

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -k test_model_default_output_limit -q
```

Local native HTTP/router/TCP checks used real vLLM input processing and Dynamo's
worker sampling-parameter builder, with synthetic output tokens. For plain chat,
JSON mode and automatic tool choice, both unary and streaming requests omitted
client limits: all six baseline cases forwarded remaining-context limits
(32,753 or 32,628 tokens) despite a model default of two; all six candidate cases
forwarded two. Twelve explicit-limit controls per revision remained unchanged
(`max_completion_tokens=4` or `max_tokens=3`). Synthetic worker output did not
simulate enforcement: these observations establish parameter forwarding, not
actual model stopping. The committed regression also covers context clipping and
absence of a model default.

Cluster validation pending. Actual-engine enforcement and multimodal serving have
not been qualified. No performance improvement has been measured.
