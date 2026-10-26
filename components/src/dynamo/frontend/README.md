<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.


## Truncated required-tool arrays

When required-tool JSON generation ends at the token limit, a requested stop string, or a canonical worker filter before
closing its array, the frontend returns the original generated text with the
original `length`, `stop`, or `content_filter` finish reason and no tool calls. Previously, identical output exposed zero, one or
more calls depending on token batching. Terminal handling now matches native
SGLang's unary JSON-array failure fallback. No missing argument delimiters are
invented. Complete arrays and model-specific marker parsing retain their existing
handling. Native SGLang streaming emits incremental calls; Dynamo buffers calls
until finish and therefore can apply this fallback without retracting events.

Reproduce the CPU parser regression with installed SGLang dependencies and the
Qwen3-0.6B tokenizer cached locally:

```bash
PYTHONPATH=components/src python -m pytest -q components/src/dynamo/frontend/tests/test_sglang_tool_calls.py -k truncated_json_array
PYTHONPATH=components/src python components/src/dynamo/frontend/tests/reproduce_sglang_truncated_array.py --model-path /path/to/cached/tokenizer --output /tmp/truncated-array-20 --interval 20
```

Repeat the HTTP replay at intervals 1 and 1024. It uses real native HTTP admission,
SGLang preprocessing/postprocessing and serialization with a task-local synthetic
token worker; it does not load model weights. The baseline fails all six parser
regressions and 12 HTTP truncation checks. The candidate passes those checks, and
both arms pass six HTTP complete-array controls. Saved sanitized HTTP results are
in `tests/sglang_truncated_array_evidence.json`. Each length-limited response preserves `length`.

Add `--termination stop` to replay complete worker JSON cut by requested `]` or
`CUT` stop strings, with a complete-array control stopped after the array. Before
the stop correction (commit `4502a9683ee9f4a058ca85ad7bd53d806ac978b1`), the
18-request replay passes only the six controls; truncated arrays expose different
call counts across batching intervals and may report `tool_calls`. The corrected
candidate passes all 18 stop cases and all 18 length cases. Stop-truncated text
retains `stop`; complete arrays still report `tool_calls`. Sanitized before/after
results are in `tests/sglang_stop_array_evidence.json`.

Add `--termination content_filter` for the canonical worker filter case. The
mapping-only correction proposed in PR #28 is included here: filtering must not
be reported as `stop` or `tool_calls`. With that mapping alone, identical
incomplete arrays still expose two, one, or zero calls across batching intervals.
The combined correction preserves exactly the text emitted by the worker, emits
no calls from an incomplete array, and retains `content_filter`. It cannot recover
or disclose tokens withheld by the worker. Complete-array controls preserve their
calls and the non-success filter status. All 18 filtered HTTP cases and six new
parser regressions pass; sanitized comparisons are in
`tests/sglang_filtered_array_evidence.json`. Stock SGLang has no demonstrated
filter producer here: this validates Dynamo's supported canonical worker boundary
using synthetic tokens, not real model filtering.

Validation used Qwen3-0.6B tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca` and SGLang source revision
`303def2253ae36fad89eb34dc66a7245003743f8`. The broader CPU suites passed
335 tests; one TinyLlama byte-fallback fixture could not initialize because its
tokenizer was absent from the offline cache. The local native binding is older than this Python source;
matching-image cluster validation remains pending. Native shutdown logs report
residual tasks even though the bounded local harness exits successfully; graceful
shutdown is not qualified here. No performance improvement has been measured.
