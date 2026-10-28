<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Filtered worker results in SGLang processing

The SGLang Python frontend preserves the canonical worker `content_filter`
finish reason. Previously it mapped that reason to `stop`; tool processing
then changed it to `tool_calls` when a tool had been parsed. Clients therefore
saw a successful terminal status for filtered output. The mapping now retains
`content_filter`, so the existing tool finalizer preserves it too.

This fixes the supported token-worker protocol boundary. It does not introduce
a filtering policy: the inspected stock SGLang scheduler emits stop, length
and abort reasons, and this check deliberately supplies a canonical filtered
worker result. Normal stop, length and tool-call behavior remain unchanged.

With Dynamo bindings, SGLang and the Qwen3-0.6B tokenizer available:

```bash
PYTHONPATH=components/src python -m pytest -q \
  --confcutdir=components/src/dynamo/frontend/tests \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'preserves_worker_content_filter or TestMapFinishReason'
```

On the baseline, three assertions fail (mapping, plain generator and tool
generator); fourteen controls pass. The candidate passes all seventeen. The
regression uses the real SGLang parser and tokenizer with deterministic worker
responses, including empty terminal chunks and argument/content preservation.

A local native HTTP check used the same canonical token-worker boundary and
real SGLang processor: twelve requests per revision, twenty-four observations
in total, with streaming and unary requests for plain and tool output.

| Worker finish | Baseline client finish | Candidate client finish |
| --- | --- | --- |
| `content_filter`, plain | `stop` | `content_filter` |
| `content_filter`, tool | `tool_calls` | `content_filter` |
| `stop`, plain / tool | `stop` / `tool_calls` | Unchanged |
| `length`, plain / tool | `length` | Unchanged |

The available processor suite passed 292 tests. One additional test could not
load the uncached TinyLlama tokenizer offline. CPU dependency overlays and
synthetic worker tokens do not qualify model generation; cluster validation
is pending. Native shutdown reported outstanding runtime tasks, so these
checks do not establish whole-runtime leak freedom.
No performance improvement has been measured.
