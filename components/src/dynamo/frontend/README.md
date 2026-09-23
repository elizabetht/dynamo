<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## Cache namespaces with the vLLM chat processor

With `--dyn-chat-processor vllm`, `nvext.cache_salt` and the `x-tenant-id`
header must reach both KV routing and the worker's vLLM prompt. Previously,
these accepted values were lost when the Python processor built its token
request. Identical prompts with different requested namespaces consequently
arrived at the worker without a salt.

The processor now resolves the salt once and forwards it to local input
processing, `routing.cache_salt`, and `extra_args.nvext.cache_salt`. The worker
uses its existing `dynamo-cache-salt:` prefix. No wire fields or engine changes
are introduced. Header normalization remains in HTTP admission: the last
non-empty, trimmed `x-tenant-id` overrides the body. A non-empty canonical
`nvext.cache_salt` takes precedence over a legacy top-level value when present
at the Python boundary. Empty values are absent. Other routing hints are
preserved without mutating the caller's dictionary.

CPU validation against base `70e094f650ab89eb81d359825adcb78fc9256aba`:

| Observation | Baseline | Candidate |
| --- | --- | --- |
| Salt reaches both router and worker prompt | 0/20 | 20/20 |
| Unsalted and empty-salt controls remain unsalted | 8/8 | 8/8 |
| HTTP requests complete successfully | 28/28 | 28/28 |

The 56 local HTTP requests exercise real Rust HTTP admission, TCP routing,
native vLLM input/output processing and the worker salt helper with synthetic
worker tokens. Cases cover canonical body salt, body precedence, header
precedence, JSON guidance, tools, and unsalted controls; each uses unary/SSE
and native stream intervals 1/20. This does not exercise model inference,
actual KV-cache reuse, or cache-event publication.

Run the checked-in CPU regression in a Dynamo/vLLM development environment:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/frontend/tests/test_vllm_processor_unit.py \
  -k cache_namespace -q
```

The regression checks canonical precedence, empty-canonical fallback,
unsalted behavior, local prompt metadata, both outgoing envelopes, and
preservation of existing routing hints. It fails twice on the baseline and
passes all three cases with the fix. The broader frontend CPU suite passes
with the pinned vLLM dependencies.

For engine validation, send identical chat requests with `nvext.cache_salt`
set to different values (or use `x-tenant-id`), and observe the worker's
original prompt salt and cache events. Repeat each salt to check within-salt
reuse; a different salt must not reuse the earlier namespace. Include guided,
tool and ordinary-chat controls. **Cluster validation is pending.**

Legacy top-level `cache_salt` is independently dropped by base-main Rust
serialization before reaching Python; its HTTP transport depends on the
separate accepted-extension serialization fix in PR #22. This change fixes
the canonical body/header path independently. Mixed-version engine/cache-event
behavior and multimodal serving have not been qualified.

No performance improvement has been measured.
