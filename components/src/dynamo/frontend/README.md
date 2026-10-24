<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## CPU guidance snapshot cost

The vLLM chat processor previously deep-copied the normalized client schema on
all requests. It now skips that snapshot when neither a tool parser nor a
reasoning parser is configured. Every configured-parser path retains the deep
copy, including nested schema mutation protection. Request validation, guidance
precedence, rendering and tokenization are unchanged.

This affects `--dyn-chat-processor vllm` requests through
`VllmProcessor._generator_inner` → `preprocess_chat_request`. It does not optimize
the Rust-only chat preprocessing path or configured-parser deployments.

### CPU component evidence

Baseline: `07caf939988cfd4021e78c03aa934b34053af687`. The
[benchmark](tests/benchmark_guidance_snapshot.py) invokes real preprocessing,
vLLM HfRenderer, Hermes tool parser (tool controls) and the Qwen3-0.6B tokenizer.
It compares complete sampling requests, rendered prompts, token IDs, guidance,
fallback selection and input immutability on every request. No model is loaded.

Environment: AMD Ryzen 7 6800H, Linux x86_64, Python 3.11.15, process affinity to
one CPU. vLLM source tag v0.29.0
(`98dff2a81d747d1dba01a47f939f48c3526d4206`), with generated version metadata;
no engine source patch is needed. Model/tokenizer/config revision:
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
Real local dependencies included OpenAI 2.26.0 and ijson 3.4.0.post0.

Four alternating baseline/candidate pairs, 40 requests per variant per pair,
five warmups, concurrency one, fixed corpus (no sampling RNG). Schemas have
1, 32 or 256 string properties with Unicode/escaping; tool cases use one tool.
The table shows the median of four pair medians for 256 properties, in
microseconds of preprocessing wall time. The chat control has no schema.

| Request | Baseline µs | Candidate µs | Change | Paired change range |
| --- | ---: | ---: | ---: | ---: |
| chat | 580.77 | 590.34 | +1.6% | -5.6% to +9.5% |
| response_format | 1069.24 | 659.25 | -38.3% | -42.9% to -37.3% |
| structured_outputs | 1103.45 | 633.93 | -42.5% | -44.9% to -40.0% |
| legacy | 582.53 | 591.49 | +1.5% | +0.6% to +2.6% |
| auto | 7932.70 | 7815.53 | -1.5% | -4.4% to +2.3% |
| required | 7862.94 | 7849.30 | -0.2% | -3.3% to +2.7% |
| named | 7812.17 | 7832.83 | +0.3% | -0.1% to +1.3% |

[Raw timings and source hashes](tests/guidance_snapshot_cpu_results.json)
include all 21 cases and 6,720 timed invocations. Small-schema and control
variation is not claimed as an improvement. Earlier unpinned runs had noisier
controls; large-schema savings persisted after CPU affinity was fixed.

These are CPU component measurements, not HTTP or model-serving measurements.
No serving performance improvement has been measured. Cluster validation is
pending; TTFT, inter-token latency, throughput and GPU memory were not measured.
All 167 vLLM processor CPU tests pass, including nested-mutation controls for
both parser kinds, parser rewrites,
reasoning and streaming behavior. These tests do not substitute for GPU tests.

### Reproduce

Use a vLLM 0.29-compatible Python environment and a local tokenizer/config
snapshot at the revision above. From the repository root:

```bash
git show 07caf939988cfd4021e78c03aa934b34053af687:components/src/dynamo/frontend/prepost.py > /tmp/baseline-prepost.py
PYTHONPATH=components/src HF_HUB_OFFLINE=1 taskset -c 0 python \
  components/src/dynamo/frontend/tests/benchmark_guidance_snapshot.py \
  --baseline-source /tmp/baseline-prepost.py \
  --model /path/to/pinned/tokenizer-config-snapshot \
  --output /tmp/guidance-snapshot-results.json
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest \
  --confcutdir=components/src/dynamo/frontend/tests \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py -q
```

Select an allowed CPU instead of CPU 0 where necessary. The pytest suite uses
the cached Qwen3-0.6B tokenizer. No cluster endpoint or serving process is used.

### Decision and review

Removing all snapshots or shallow-copying would risk parser mutation semantics.
Gating on actual parser activation would duplicate tool/thinking conditions.
Checking only configured class presence is the smaller change. In-context
self-review covered the production caller, both parser gates, schema aliasing,
all changed hunks and benchmark controls; no independent review is claimed.
