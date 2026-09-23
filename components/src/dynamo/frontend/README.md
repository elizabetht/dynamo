<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang logit-bias forwarding

The Python SGLang chat processor preserves OpenAI `logit_bias` through the
existing `extra_args.sampling_options` transport. SGLang decode and prefill
workers extract the bias into engine sampling parameters; the engine retains
responsibility for token vocabulary validation and sampling. Previously, valid
bias maps were accepted over HTTP but silently omitted before generation.
Absent and null biases remain omitted, and an empty map remains a no-op.

CPU validation against main `70e094f650ab89eb81d359825adcb78fc9256aba`:

- The focused regressions changed from 7 failures / 3 passes to passing.
- 479 frontend and worker tests passed; one uncached TinyLlama tokenizer test was
  deselected. The prefill test observes dispatch parameters using a fake engine.
- 24 native localhost HTTP requests per revision exercise positive/negative biases,
  guided JSON, absent/empty controls and invalid ranges, with SSE/unary responses
  and stream intervals 1/20. All 12 nonempty biases were lost on main and preserved
  by the candidate through the real worker parameter builder and SGLang validation.
  Invalid ranges remain HTTP 400 before worker dispatch.

Reproduce in a Dynamo development environment with SGLang frontend/worker
requirements and the cached Qwen3-0.6B tokenizer:

```bash
PYTHONPATH=components/src python -m pytest -q --confcutdir=components/src/dynamo \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  components/src/dynamo/sglang/tests/test_sglang_decode_handler.py \
  -k 'not byte_fallback'
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/sglang_logit_bias_probe.py \
  --output /tmp/sglang-logit-bias
```

The standalone probe uses native HTTP/routing and synthetic tokens, loads no
weights and allocates no GPU. To reproduce main's loss, run the same probe against
main's three production modules with `--baseline`. Local checks used SGLang source
`3c82e48e825bae504527ed28e2442b1429d36650` and Qwen tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with isolated CPU dependency overlays.
The native runtime reported pending tasks at interpreter exit; this is not a
whole-runtime lifecycle qualification.

Deploy the frontend and worker changes together: an old worker does not consume
the new passthrough. This change covers Python SGLang chat preprocessing and the
LLM decode/prefill handlers, not every backend or the Rust-only chat processor.
Real model sampling and distributed prefill/decode validation remain pending.
No performance improvement has been measured.
