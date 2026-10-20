<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang template format detection

SGLang preprocessing normalizes message content before rendering the chat
prompt. Its format detector reparses the template's Jinja AST on every call,
even when the tokenizer's template has not changed. The frontend now caches
the detected format for up to 32 template strings per process. A changed string
gets a new lookup; non-string template values keep the existing uncached
fallback. Messages, tool schemas, tokenizer objects and ASTs are not cached.

The supported path is native HTTP admission → `SglangProcessor.generator` →
`preprocess_chat_request` → message normalization → SGLang format detection.
Preprocessing worker processes each warm their own cache. Concurrent first
misses can still compute the same format more than once.

### CPU evidence

Measured against base `70e094f650ab89eb81d359825adcb78fc9256aba`, using stock
SGLang v0.5.19 (`0bcd822377da7b5718e674eaf9c870d349424dd1`), Transformers
5.12.1, Jinja2 3.1.6, Python 3.11.15 and an AMD Ryzen 7 6800H CPU.
The tokenizer was Qwen/Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`. No weights or GPU were used.

These are medians of five alternating baseline/candidate pairs, 40 requests
per arm per pair. Every result was checked for identical prompt token IDs,
guidance, request options and parser selection. All seven workloads passed.
The clock measures process CPU time for complete Python preprocessing, including
rendering, tokenization and parser construction; correctness checks are outside
the timed section. This is a warm component benchmark, not HTTP throughput.

| Workload | Baseline µs/request | Candidate µs/request | CPU reduction |
| --- | ---: | ---: | ---: |
| Plain chat | 4517.3 | 110.5 | 97.6% |
| Structured text content | 4510.0 | 107.9 | 97.6% |
| JSON schema | 4474.4 | 111.2 | 97.5% |
| Auto tool | 5008.9 | 372.9 | 92.6% |
| Required tool | 4843.1 | 336.1 | 93.1% |
| Named tool | 5267.7 | 371.4 | 92.9% |
| 32 tools | 7848.6 | 3043.9 | 61.2% |

[Raw samples, cold-cache observations and source hashes](tests/sglang_template_format_cpu_results.json)
include all pair results and variability. Cold-cache observations clear only
Dynamo's format cache, not Hugging Face's compiled-template cache; they show
that initial AST work remains. No cold-start improvement is claimed.
Results depend on the template: media templates that use SGLang's keyword
shortcut do not have this AST parsing cost.

Local validation passed 319 focused CPU tests. One existing byte-fallback
tokenizer test was excluded because its model files were unavailable offline.
A separate native localhost HTTP replay passed all 16 requests on both baseline
and candidate, preserving prompt IDs, guidance, Unicode arguments and streamed
call identity. The replay used scripted worker tokens, not an inference engine.
Its runtime reported nine tasks at interpreter exit, so it does not establish
complete runtime shutdown behavior.

### Reproduce

Use an environment with the repository's SGLang dependencies and the pinned
tokenizer cached. From the repository root:

```bash
validation_dir=$(mktemp -d)
git show 70e094f650ab89eb81d359825adcb78fc9256aba:components/src/dynamo/frontend/sglang_prepost.py > "$validation_dir/baseline.py"
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/frontend/tests/test_sglang_multimodal_prepost.py \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  components/src/dynamo/frontend/tests/test_sglang_tool_calls.py \
  -k 'not test_byte_fallback_sequence_longer_than_six_tokens'
PYTHONPATH=components/src python \
  components/src/dynamo/frontend/tests/benchmark_sglang_template_format.py \
  --baseline "$validation_dir/baseline.py" --output "$validation_dir/results.json" \
  --pairs 5 --requests 40
```

Keeping the repeated detector call would preserve the measured cost. Computing
once in the processor constructor would require passing derived state through
both direct and multiprocess preprocessing, and would need invalidation when
templates change. Skipping detection for scalar-only messages helps a subset
of traffic but still reparses structured text/tool conversations. The bounded
text-keyed cache reuses the existing detector for all those paths without
changing SGLang or its format rules.

The implementation and final diff received two passes of in-context self-review,
including non-string fallback, template replacement, media normalization and
request-path checks. This was not independent review. Cluster validation is
pending; no serving performance improvement has been measured.
