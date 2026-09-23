<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Dynamo Frontend

The API gateway for serving LLM inference requests with OpenAI-compatible HTTP and KServe gRPC endpoints.

See [docs/components/frontend/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/overview.md) for documentation.

## SGLang cache namespaces

The Python SGLang frontend preserves the resolved request cache salt in both
`routing.cache_salt` and `extra_args.nvext.cache_salt`. Previously, a request with
`nvext.cache_salt` reached the SGLang worker without that namespace. This affected
ordinary chat, guided JSON, and tool requests using the Python processor.

The shared inline/pool preprocessor follows the Rust preprocessor's precedence:
nonempty `nvext.cache_salt`, then the legacy top-level `cache_salt`. The native
HTTP policy resolves `x-tenant-id` into the canonical field before preprocessing.
Routing priorities and other passthrough metadata are retained; the input request
is not mutated. SGLang's existing worker resolver and engine compatibility check
consume the forwarded field. Engines without cache-salt support retain their
existing explicit rejection behavior.

### CPU evidence and reproduction

At base `70e094f650ab89eb81d359825adcb78fc9256aba`, the focused regression has two
failures and one passing control. The candidate passes 289 processor tests; one
byte-fallback test was excluded because its separate tokenizer was not cached.
The native HTTP comparison used SGLang v0.5.19 parser sources and the cached
Qwen3-0.6B tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`.

| Native HTTP observations | Baseline | Candidate |
| --- | ---: | ---: |
| Requested salt preserved at worker | 0/20 | 20/20 |
| Unsalted controls remain unsalted | 8/8 | 8/8 |

Seven inputs cover canonical salt, legacy/canonical precedence, tenant-header
precedence, JSON guidance, auto tools, omitted salt, and empty salt. Each runs as
SSE and unary HTTP at processor stream intervals 1 and 20. A synthetic token
worker returns `OK`; this verifies admission, routing, and the real worker salt
resolver, not constrained generation or physical KV-cache isolation.

With matching Dynamo bindings and SGLang frontend dependencies installed, and
that tokenizer cached, run from the repository root:

```bash
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python -m pytest -q \
  --confcutdir=components/src/dynamo/frontend/tests \
  components/src/dynamo/frontend/tests/test_sglang_processor_unit.py \
  -k 'not byte_fallback'
PYTHONPATH=components/src HF_HUB_OFFLINE=1 python \
  components/src/dynamo/frontend/tests/sglang_cache_salt_probe.py \
  --output /tmp/sglang-cache-salt-evidence
```

The standalone probe uses local native HTTP/RPC and ephemeral discovery; it loads
no model weights. It asserts all 28 cases on the candidate and fails on the first
salted case on the baseline. To reproduce the baseline, retain the probe and
regression test while restoring only `sglang_processor.py` from the base revision
in a disposable checkout. Local validation used CPU dependency overlays; it is
not an installed GPU runtime qualification. Native shutdown reported remaining
runtime tasks, so these results do not establish whole-runtime leak freedom.

Cluster validation is pending: use a matched SGLang runtime to check repeated
same-namespace prefixes and separation across namespaces, including guided/tool
requests. No performance improvement has been measured.
