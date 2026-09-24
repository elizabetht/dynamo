<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Complete forced-tool JSON arrays

With SGLang required tool calls, a model can generate two complete calls while
Dynamo returns only the first. The default 20-token response batching can leave
`JsonArrayParser` with one fully populated call, causing the previous finish-time
fallback to skip the rest of the array. Tokenwise and whole-response delivery do
not necessarily reproduce this loss.

At completion, Dynamo now reparses forced-tool JSON arrays using the existing
JSON fallback. It retains every complete call, filters known tool names, and
assigns sequential call indexes. Tool events are already buffered until completion,
so reconstruction does not replace IDs already sent to the client. Model-specific
parsers retain their existing behavior. This adds one terminal JSON parse; it does
not add a per-token scan. Fixing the adapter avoids requiring an engine upgrade or
changing every model-specific detector.

CPU reproduction, with Dynamo bindings, SGLang, and the cached Qwen3 tokenizer:

```bash
PYTHONPATH=components/src python -m pytest -q components/src/dynamo/frontend/tests/test_sglang_processor_unit.py -k json_array_recovers_later_complete_calls
```

Retain the regression while reverting the condition in `sglang_prepost.py` to
reproduce the baseline: the 20-token case loses Rome; tokenwise and whole-response
controls pass. The candidate passes all three. The broader CPU suite passed 294
tests; one TinyLlama byte-fallback test could not initialize because its tokenizer
was absent from the offline cache.

The [generation corpus](tests/sglang_json_array_corpus.json) and
[client](tests/reproduce_sglang_json_array.py) exercise explicit true/false parallel
calls with unary and SSE responses. Start baseline and candidate Dynamo/SGLang
servers separately with `--dyn-chat-processor sglang --tool-call-parser qwen25
--reasoning-parser qwen3`, the default stream interval of 20, and the pinned
Qwen/Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`.

```bash
python components/src/dynamo/frontend/tests/reproduce_sglang_json_array.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B --corpus components/src/dynamo/frontend/tests/sglang_json_array_corpus.json --arm candidate --pair 1 --repetitions 1 --output candidate.json
```

Use `--arm baseline` for the baseline server: it asserts the known one-call behavior.
The candidate expects two calls only for explicit true; explicit false stays at one.
This fix is independent of the omitted/null parallel-default correction.
No performance improvement has been measured.

The [saved GPU evidence](tests/sglang_json_array_evidence.json) records eight
requests on one GB10 using SGLang 0.5.19 and the immutable image in that file.
For explicit true, both arms generated a two-call array: baseline returned only
Paris, while the candidate returned Paris and Rome in both unary and SSE modes.
All four explicit-false controls retained one call. The engine-input schemas,
raw engine text/token IDs, and client events are included for comparison.
This used September 23 native bindings with the frozen Python source, not an
exact-current native build. Only one correctness pair was run; cancellation,
multi-GPU behavior, performance, and graceful engine shutdown remain unqualified.
