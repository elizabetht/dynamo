<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang signature-cache parser lifetime

Auto-tool preprocessing constructs a temporary SGLang parser to select its
structure constraint. Previously the 64-entry signature cache keyed on its bound
method, retaining each request's parser and tools until eviction and inspecting
the same method again for the next request. Cache Python methods by their
underlying function and binding state instead. On a miss, inspect a disposable
bound receiver to preserve Python's bound-method signature semantics. This does
not cache or share response parsers, schemas, or request data.

The supported path is `SglangProcessor._generator_inner` (also
`_preprocess_worker`) → `preprocess_chat_request` →
`build_tool_call_guided_decoding` → compatibility signature inspection.
Native localhost HTTP replay exercised 16 requests per arm: flat/namespaced
Responses tool subsets, auto/required choices, unary/SSE, Unicode arguments, and
two token batching policies. All 16 prompt/guidance pairs matched. Of eight
temporary auto-tool parsers observed per arm, baseline retained eight after
request completion; candidate retained zero. Workers replayed tokens; they did
not run a model. This is bounded retention, not an unbounded leak.

[Raw CPU samples and HTTP evidence](tests/sglang_signature_cache_cpu_results.json)
compare main `70e094f650ab89eb81d359825adcb78fc9256aba` with this candidate.
Five alternating baseline/candidate pairs, 100 calls per arm, stock SGLang
v0.5.19 (`0bcd822377da7b5718e674eaf9c870d349424dd1`), Python 3.11.15,
Transformers 5.12.1, Jinja2 3.1.6, and an AMD Ryzen 7 6800H produced:

| Guidance-construction CPU stage | Baseline median | Candidate median |
| --- | ---: | ---: |
| One auto tool | 17.268 µs/request | 3.630 µs/request |
| 32 auto tools | 18.618 µs/request | 5.449 µs/request |

The seven full-preprocessing workloads varied from a 1.51% slowdown to a 1.88%
reduction; no reliable full-preprocessing improvement was established. Named and
required guidance controls changed by less than 0.4 µs/request at stage scope.
No serving performance improvement has been measured. Cluster validation is
pending; TTFT, throughput, GPU memory, and real constraint enforcement were not
measured. This candidate is independent of the separate template-format cache.

Reproduce in a CPU environment with the repository's SGLang dependencies and the
cached `Qwen/Qwen3-0.6B` tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`:

```bash
git show 70e094f650ab89eb81d359825adcb78fc9256aba:components/src/dynamo/frontend/sglang_prepost.py > /tmp/sglang-signature-baseline.py
PYTHONPATH=components/src python -m pytest -q components/src/dynamo/frontend/tests/test_sglang_signature_cache.py
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/benchmark_sglang_signature_cache.py --baseline /tmp/sglang-signature-baseline.py --output /tmp/sglang-signature-results.json --pairs 5 --requests 100
```

The broader available SGLang frontend suites passed 320 tests; one existing
uncached tokenizer fixture was excluded. Regression tests cover parser release,
legacy/current helpers, bound/unbound, variadic, positional-only, class, and
static methods. Removing caching would release parsers but keep repeated
inspection; inspecting unbound signatures directly would mishandle the receiver.
Two passes of in-context self-review covered those alternatives and the final
diff; this was not independent review. Native replay reported nine runtime tasks
at interpreter exit, so it provides no shutdown-lifecycle qualification.
