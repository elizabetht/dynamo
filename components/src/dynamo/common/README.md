<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common guided JSON validation

The local-reference cycle check rejects reachable JSON Schema reference cycles
that cannot consume JSON. Both `sglang/request_handlers/handler_base.py`
(`_get_guided_decoding_params`) and `vllm/handlers.py` (`build_sampling_params`)
call it before constructing the engine's structured-output parameters.

## Unescaped JSON pointer fast path

Previously, every pointer token was decoded with a Python character loop, a
list, and a join, even when it contained no `~`. Tokens without `~` now return
unchanged. Tokens with `~` still use the existing decoder, preserving `~0`,
`~1`, malformed-escape handling, and the order of percent decoding. This avoids
work for common references such as `#/$defs/Value`; it adds no cache or shared
state and changes neither accepted schemas nor error messages.

The supported request path admits `guided_json` as a JSON value in
`CommonExt`, maps it into `GuidedDecodingOptions.json`, checks mutual exclusion,
and passes it to the worker sampling options. The two backend callers above
then run the measured check. Their subsequent engine validation and grammar
compilation are outside this benchmark. This is a CPU optimization, not a new
claim about HTTP error handling or engine schema support.

## Reproduce the CPU evidence

Use a development Python environment with the real Dynamo bindings and pytest.
The recorded run used CPython 3.11.15 on an AMD Ryzen 7 6800H, Linux x86_64.
Run from the repository root, with this candidate checked out:

```bash
git show 07caf939988cfd4021e78c03aa934b34053af687:components/src/dynamo/common/utils/guided_json.py > /tmp/guided-json-baseline.py
PYTHONPATH="$PWD/components/src" python -m pytest components/src/dynamo/common/tests/test_guided_json.py -q
PYTHONPATH="$PWD/components/src" python components/src/dynamo/common/tests/benchmark_guided_json.py --baseline /tmp/guided-json-baseline.py --output /tmp/guided-json-benchmark
```

The script verifies that the imported candidate comes from this checkout.
It generates deterministic schemas (seed 0, no random sampling), compares
19,608 pointer tokens and 28 dict/string schema outcomes, checks input
immutability, and measures the entire cycle-check function. The six timing
schemas also passed Draft 2020-12 meta-validation with jsonschema 4.26.0.
The 19 focused tests pass on both baseline and candidate: this change preserves
behavior and reduces measured cost rather than fixing a failing request.

Four paired repetitions alternate baseline/candidate order (AB, BA, AB, BA),
with 20 warmups and 100 samples per variant per pair: 400 samples per variant
per case/representation, 9,600 timed calls total. Summary values are medians
of the four pair medians. Raw nanosecond samples and all pair medians are in
[timings](tests/guided_json_benchmark.csv) and
[summary](tests/guided_json_benchmark.json), including exact source hashes.

| Workload (dict input) | Baseline µs/call | Candidate µs/call | Change | Pair change range |
| --- | ---: | ---: | ---: | ---: |
| 16 shared unescaped references | 66.28 | 53.35 | -19.5% | -19.7% to -18.2% |
| 1,024 shared unescaped references | 4014.74 | 3186.98 | -20.6% | -20.9% to -20.6% |
| 1,024 shared escaped references | 3957.86 | 3595.44 | -9.2% | -10.8% to -8.7% |
| 1,024 properties without references | 1310.32 | 1300.36 | -0.8% | -1.2% to +1.4% |

Escaped references still benefit from the unescaped `$defs` token. Treat the
small no-reference difference as noise, not an improvement. String inputs,
which include JSON parsing, are reported separately in the summary.

## Alternatives, review, and limits

Leaving the decoder unchanged preserves behavior but retains measured work.
Chained `replace` calls would mishandle invalid escapes without an additional
validation pass. Caching resolved references would introduce state and resource
scope concerns. The two-line early return is the smallest viable change.

An in-context self-review applied the systems and frontend/runtime review
guidance, traced the backend callers and upstream types, checked every changed
hunk, and reran focused tests and lint. It was not an independent review.

No serving performance improvement has been measured. This shared-host CPU
microbenchmark used no affinity or frequency controls, model, tokenizer,
container image, GPU, HTTP server, or inference engine. Native binding build
provenance is unverified; the timed Python module is verified by path and hash.
TTFT, inter-token latency, throughput, cold/warm schema compilation, GPU memory,
and generated-output correctness remain unmeasured. The change does not modify
streaming, cancellation, tool parsing, or backend capabilities.
