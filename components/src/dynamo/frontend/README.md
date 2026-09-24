<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang parallel tool-call defaults

SGLang chat preprocessing defaults `parallel_tool_calls` to `true` when the
HTTP field is omitted or null. Only explicit `false` constrains a forced tool-call
array to one element. Previously the adapter passed `None` to the SGLang grammar
helper, which interpreted it as false and added `maxItems: 1` for otherwise
parallel-capable requests. Normalize the value once before selecting the required,
named, or auto tool constraint. Explicit true/false behavior and older helper
signature compatibility are preserved.

CPU reproduction (with Dynamo bindings, SGLang 0.5.19 and a cached Qwen3 tokenizer):

```bash
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k parallel_tool_default_preprocessing
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/reproduce_sglang_parallel_default.py \
  --model-path "$TOKENIZER_SNAPSHOT" --output /tmp/parallel-tool-replay
```

Use Qwen/Qwen3-0.6B tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The replay starts only task-owned
localhost HTTP/TCP services and returns synthetic worker tokens; it loads no
model weights. To reproduce the baseline, retain the replay script and tests
while reverting only the two-line normalization in `sglang_prepost.py`, then
pass `--expect-baseline` to the replay.

On base `a76e12e6584f6029aeaa551c7cc8967671f67345`, four regression cases fail
and four explicit-boolean controls pass. The candidate passes all eight.
The native HTTP replay covers required/named choice, omitted/null/false/true,
and unary/streaming: the baseline sends the wrong single-call grammar to the
worker in eight of sixteen cases; the candidate fixes all eight and preserves
the explicit-boolean controls. It verifies worker-facing guidance and complete
tool-call responses. The CPU runtime uses native bindings from `70e094f` and
SGLang source `0bcd822377da7b5718e674eaf9c870d349424dd1`.

The change restores allowed grammar cardinality; it does not promise that a model
will choose multiple calls. Model generation and cluster validation remain pending.
Native shutdown still reports remaining tasks at interpreter exit; complete native
shutdown is unqualified. No performance improvement has been measured.
