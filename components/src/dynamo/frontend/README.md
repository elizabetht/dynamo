<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang stop-prefix CPU cost

The Python SGLang response path buffers suffixes that might complete a requested
stop string. Previously it constructed every possible suffix and checked every
stop string in Python, including suffixes whose final character could not match.
The utility now searches each stop string for that final character in C, then
checks only possible prefix lengths longer than the best match already found.
The longest-prefix result, Unicode handling, output, and stop semantics are unchanged.
No cache, per-request precomputation, engine change, or parser dependency is added.

The native HTTP benchmark below exercises Rust admission/routing, Python SGLang
processing, the pinned tokenizer and real Hermes parser with scripted worker tokens.
It compares only the old/new stop-prefix utility in the same process and records
summed `process_output` time, excluding network waits and generation. This is a CPU
component benchmark, not a model-serving throughput, TTFT, or inter-token latency result.
**No serving performance improvement has been measured. Cluster validation is pending.**

On AMD Ryzen 7 6800H / Python 3.11.15, three alternating pairs passed all 84 HTTP
observations (14 workloads × 2 implementations × 3 repetitions), after 28 warmup
requests covering both implementations. Values below are
median milliseconds per request, with min–max across the three repetitions:

| Workload | Baseline ms | Candidate ms | Median change |
|---|---:|---:|---:|
| prefix-1024-plain | 34.640 (34.516–34.907) | 28.179 (27.659–31.473) | -18.7% |
| prefix-1024-guided | 33.425 (33.343–33.908) | 28.199 (27.545–28.858) | -15.6% |
| prefix-1024-tools | 35.695 (34.935–38.643) | 28.894 (27.892–29.447) | -19.1% |
| control-16-plain | 5.601 (5.294–6.155) | 4.850 (4.831–5.245) | -13.4% |
| control-0-plain | 4.799 (4.583–4.894) | 4.575 (4.389–4.839) | -4.7% |

The stress corpus emits eight runs of `a` repeated N times followed by `x `, with
four overlapping stops ending in b/c/d/e, for N=8/64/256/1024. Plain, guided-regex,
and auto-tool request shapes use SSE, one token per worker chunk, interval 1,
concurrency 1, and max_tokens=4096. The guided worker is scripted: the benchmark
does not establish engine regex enforcement. Controls emit ordinary text with
short delimiters or no stop field. No-stop differences are measurement noise;
that path has the same early return. Runs are not CPU-isolated, and these
adversarial long prefixes do not establish the frequency of this cost in production.

[Raw samples and revisions](tests/sglang_stop_prefix_cpu_results.json) include all
measurements, source/binding hashes and the tokenizer revision. No model weights
or GPU are used; no container image is involved. The first 12-case cProfile pass
identified stop-prefix scanning as a substantial long-prefix cost. A simpler direct
Python loop was also tested: it improved less than reverse search on the 1024-prefix
cases (about 11–20% versus 19–25% in that exploratory run). Those exploratory
numbers are not the final-candidate table above. A stateful prefix automaton would
add setup and state complexity without evidence that it is needed.

### Reproduce

Use a Dynamo build from the baseline below, install the repository's pinned SGLang
extra (0.5.19), and set `PYTHONPATH=components/src` so the changed Python source is
loaded. Cache the Qwen/Qwen3-0.6B tokenizer at revision
`c1899de289a04d12100db370d81485cdf75e47ca` first; the script requires local files.
From the repository root:

```bash
git show 70e094f650ab89eb81d359825adcb78fc9256aba:components/src/dynamo/common/utils/engine_response.py > /tmp/baseline_engine_response.py
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/benchmark_sglang_stop_prefix.py --baseline-source /tmp/baseline_engine_response.py --output /tmp/stop-prefix-results
PYTHONPATH=components/src python -m pytest -q components/src/dynamo/common/tests/test_engine_response.py components/src/dynamo/frontend/tests/test_sglang_processor_unit.py components/src/dynamo/frontend/tests/test_sglang_tool_calls.py -k 'not test_byte_fallback_sequence_longer_than_six_tokens'
```

CPU validation passed 321 tests; the excluded byte-fallback test requires an uncached
TinyLlama tokenizer and errored offline when attempted. The helper regression checks
14,641 Unicode/overlap combinations against a brute-force oracle in addition to edge
cases. The HTTP script asserts exact content, terminal reason, one worker dispatch,
worker finalization and SSE completion for every request. It is a standalone process
because it temporarily instruments the processor and changes discovery environment
variables. The native runtime still reports nine tasks at interpreter exit; this
benchmark does not qualify whole-runtime shutdown. Two-pass in-context self-review
covered longest-match semantics, empty inputs, Unicode, alternatives and evidence;
no independent review is claimed.
