<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Avoid duplicate request construction

With `--dyn-chat-processor vllm`, supported raw tool and structured-output
requests reach `preprocess_chat_request` as dictionaries. Previously,
`_validate_chat_completion_request` called `model_construct`, inspected its nested
fields, discarded that object, and then called `model_validate`. The unchecked
construction resolves every request default even though full validation will
resolve those defaults again.

Select that existing validation path from the raw fields first. Requests with
raw `tools`, `response_format`, or `structured_outputs` still receive the same
full validation. Already-typed requests, named-tool normalization, and the
`DYN_VLLM_SKIP_REQUEST_VALIDATION=0` path retain their behavior. This does not
remove schema validation, cache request state, or change rendering/tokenization.

### CPU evidence

Compared with `07caf939988cfd4021e78c03aa934b34053af687`, using the real
vLLM v0.29.0 `HfRenderer`, Qwen3-0.6B tokenizer, and Hermes tool parser.
Each row uses four alternating baseline/candidate pairs, 100 calls per variant
per pair (400 per variant), after five warmups. Values are medians of the four
per-pair medians, in microseconds per complete preprocessing call.

| Request | Properties | Baseline µs | Candidate µs | Change | Per-pair change range |
| --- | ---: | ---: | ---: | ---: | ---: |
| `response_format` | 1 | 637.08 | 341.00 | -46.5% | -51.5% to -39.0% |
| `response_format` | 256 | 1070.52 | 791.01 | -26.1% | -27.3% to -23.7% |
| `structured_outputs` | 256 | 1044.57 | 775.53 | -25.8% | -28.0% to -25.0% |
| Auto tool choice | 32 | 1674.40 | 1365.50 | -18.4% | -21.4% to -17.4% |
| Required tool choice | 32 | 1670.10 | 1368.80 | -18.0% | -26.8% to -13.1% |
| Named tool choice | 32 | 1736.42 | 1361.46 | -21.6% | -26.0% to -16.8% |

The 21-case corpus includes ordinary chat and legacy-guidance controls; all
16,800 timed calls preserve the sampling request, prompt, tokens, effective
guidance, fallback selection, and input payload. Control case medians vary from
-4.3% to +4.7%; no improvement is claimed for those paths. A separate six-pair
validation-only control measures roughly 180–188 µs for both variants, versus
roughly 220–238 to 22–28 µs for nested raw requests. Large tool prompts remain
mostly tokenizer work, so benefits shrink with prompt size.

[Raw CPU samples and environment](tests/validation_selection_cpu_results.json)
include every case, pair, source hash, control, and dependency revision. This ran
on a shared x86_64 Ryzen 7 6800H host, Python 3.11.15, Pydantic 2.13.5,
Transformers 5.12.1 and Tokenizers 0.22.2. It used stock vLLM source at
`98dff2a81d747d1dba01a47f939f48c3526d4206`; the local source overlay's generated
version metadata differs from the tag and is recorded in the evidence. No model
weights were loaded. These measurements do not include the separate guidance
snapshot optimization.

### Reproduce and limits

In a CPU environment with this checkout's Python source and pinned real vLLM
and Transformers dependencies, provide a local tokenizer/config snapshot of
`Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`:

```bash
git show 07caf939988cfd4021e78c03aa934b34053af687:components/src/dynamo/frontend/prepost.py > /tmp/validation-baseline.py
PYTHONPATH=components/src python -m pytest components/src/dynamo/frontend/tests/test_vllm_processor_unit.py -q
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/benchmark_validation_selection.py \
  --baseline-source /tmp/validation-baseline.py \
  --model /path/to/pinned/tokenizer-config-snapshot \
  --output /tmp/validation-selection-results.json
```

174 CPU tests pass, including both validation settings, nested defaults,
invalid tool-choice errors, and typed request identity. Benchmark parity is
component evidence, not an HTTP admission or real-backend test. HTTP, streaming,
GPU correctness and serving-performance validation remain pending. No serving
performance improvement has been measured.

Alternatives considered: keeping the extra construction preserves unnecessary
work; always validating would change the intentional skip-validation contract.
Reading the same three unaliased fields before constructing is the narrow change.
The diff received an in-context self-review against the frontend and systems
review guidance; this is not an independent review.
