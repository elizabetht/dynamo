<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.


## Truncated required-tool arrays

When required-tool JSON generation ends at the token limit before closing its
array, the frontend returns the original generated text with `finish_reason:
"length"` and no tool calls. Previously, identical output exposed zero, one or
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
in `tests/sglang_truncated_array_evidence.json`. Each response preserves `length`.

Validation used Qwen3-0.6B tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca` and SGLang source revision
`303def2253ae36fad89eb34dc66a7245003743f8`. The broader CPU suites passed
323 tests; one TinyLlama byte-fallback fixture could not initialize because its
tokenizer was absent from the offline cache. The local native binding is older than this Python source;
matching-image cluster validation remains pending. Native shutdown logs report
residual tasks even though the bounded local harness exits successfully; graceful
shutdown is not qualified here. No performance improvement has been measured.
