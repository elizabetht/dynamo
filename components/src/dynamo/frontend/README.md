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
will choose multiple calls. The real-generation check below confirms valid output,
but does not yet demonstrate multiple generated calls.
Native shutdown still reports remaining tasks at interpreter exit; complete native
shutdown is unqualified. No performance improvement has been measured.

### Real-generation check

The same baseline and candidate `c5b826607ab9f1494ad0eaf3c2fc0edd0c19e282`
passed 64 real-generation requests on one NVIDIA GB10. Two distinct prompts
cover required/named tools, omitted/null/false/true flags and unary/streaming
responses: 16 cases per prompt per arm. Each case ran once, without warmup.
Schemas, forced tool names, stable unique IDs, contiguous streaming indexes,
SSE completion and nontruncated finish reasons pass. Explicit false always
returns one call.

**Multi-call generation remains inconclusive.** Every response contains one
call, including explicit-true controls and the follow-up prompt containing
an explicit two-call example. These results establish no observable
parallel-generation benefit; the supported CPU HTTP grammar evidence above
remains the demonstrated reason for this fix. No performance improvement
has been measured.

The [original corpus](tests/sglang_parallel_generation_corpus.json),
[two-call prompt corpus](tests/sglang_parallel_two_city_corpus.json),
[saved outputs and pinned environment](tests/sglang_parallel_generation_evidence.json)
and [client](tests/reproduce_sglang_parallel_generation.py) are included.
Use Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`,
bfloat16, SGLang 0.5.19, `qwen25` tools (normalized to Hermes), `qwen3`
reasoning, thinking disabled, seed 17, temperature 0, maximum 256 output
tokens and concurrency 1. The evidence records the immutable image and
model hashes. This uses the September 23 native runtime with exact
baseline/candidate preprocessing source, not an exact-current native build.

Start separate baseline and candidate Dynamo/SGLang servers using those
settings, then run against each server:

```bash
python components/src/dynamo/frontend/tests/reproduce_sglang_parallel_generation.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B --corpus components/src/dynamo/frontend/tests/sglang_parallel_generation_corpus.json --arm candidate --pair 1 --repetitions 1 --output candidate-generation.json
```

Use `--arm baseline` and a separate output file for the baseline; substitute
`sglang_parallel_two_city_corpus.json` for the second prompt. Both bounded
Jobs and their Pods were cleaned up. Cancellation, multi-GPU and
disaggregation were not tested. Worker signal diagnostics occurred during
intentional teardown; graceful engine shutdown and full CI remain unqualified.
