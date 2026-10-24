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
No serving performance improvement has been measured. The CPU measurements above
do not measure TTFT, throughput, GPU memory, or real constraint enforcement.
This candidate is independent of the separate template-format cache.

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

### Real-model Chat correctness

[Generation evidence and reproduction files](tests/sglang_signature_cache_gpu_results.json)
record 28 successful streamed requests: seven cases, one warmup and one retained
observation per case in each arm. Plain chat, text parts, JSON schema,
auto/required/named tools, and 32 tools all passed output/schema checks. All seven
non-warmup output pairs and token counts matched. The first plain-chat warmup
output differed, and four warmup cases had different cache usage; all remained
valid. Prompt lengths ranged from 14 to 2,076 tokens; output was capped at 128 tokens.

This used one allocated GPU, the pinned Qwen3-0.6B model/tokenizer above,
bfloat16, temperature 0, seed 17, and the immutable SGLang 0.5.18 image recorded
in the artifact. Both arms used the same worker sequentially. The artifact pins
all source files loaded over that image: main common modules and processor,
plus baseline or candidate preprocessing. It is compatibility validation of this
source closure, not a current-main native build or SGLang 0.5.19 qualification.
The older native image fails a separate Responses allowed-tools corpus in both
arms; this Chat result does not clear that blocker. Auto-tool cases returned
valid text without calls, so they exercise preprocessing but do not qualify
auto-selected tool emission. Required/named choices emitted valid calls.

To reconstruct the launch/client files from a checkout containing both
recorded commits:

```bash
python - <<'PY'
import hashlib, json, pathlib, subprocess
artifact = json.loads(pathlib.Path("components/src/dynamo/frontend/tests/sglang_signature_cache_gpu_results.json").read_text())
out = pathlib.Path("/tmp/signature-cache-repro")
for name, text in artifact["reproduction_files"].items():
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
for name, source in artifact["sources"].items():
    data = subprocess.check_output(["git", "show", source["revision"] + ":" + source["path"]])
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
PY
```

Inside the recorded image with one allocated GPU and the pinned model cache
mounted read-only at `/model-cache`, run
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /tmp/signature-cache-repro/generation.py`.
Mount the reconstructed directory at that path. The script starts the real
worker/frontend, validates streamed responses with JSON Schema, emits per-request
evidence, and stops its own process groups. The measured Job had a
900-second server-side deadline and a 4-GiB shared-memory volume. Use an isolated
request plane and a bounded execution environment for reproduction.

No serving performance improvement has been measured. One fixed-order pair with
a shared worker/cache cannot establish a latency or throughput improvement.
GPU-memory use and grammar compilation costs were not instrumented. Shutdown
produced engine diagnostics after process-group termination, so this experiment
does not qualify graceful shutdown behavior.
