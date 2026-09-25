<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## Multimodal request cancellation

The dedicated multimodal handler monitors the Dynamo request context in aggregated
mode and in decode mode after prefill bootstrap. Previously a disconnected client
could leave the handler waiting for its engine stream without requesting an abort.
The handler now tracks active IDs from engine output, removes completed IDs, and
uses the existing bounded cancellation monitor. Explicitly closing the Python
handler also closes its owned cancellation stream before returning.

Only exact observed request IDs are aborted; `abort_all` is never requested.
Before output, ordered cancellation uses the existing engine capability and
stable-ID checks. Parallel children that never emit an ID remain outside this
fix: complete group cancellation requires engine-owned child tracking. Prefill
bootstrap cancellation is also outside this change. Normal completion retains
its terminal output; shutdown does not invalidate already delivered terminals.

This includes the shared task-result cleanup from [draft PR #39](https://github.com/elizabetht/dynamo/pull/39),
with its two regression cases, to consume simultaneous stream completion and
shutdown. That dependency is not assumed merged.

### CPU validation

With the supported SGLang dependencies installed:

```bash
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/sglang/tests/test_sglang_multimodal_cancellation.py \
  components/src/dynamo/sglang/tests/test_sglang_decode_cancellation.py
```

On main `a76e12e6584f6029aeaa551c7cc8967671f67345`, the focused public-handler
regression has eight failures (stop, kill, shutdown, and close in both modes)
and two passing normal-completion controls. The candidate passes all 102
affected CPU tests. Engines and prefill responses in these tests are scripted.

A separate replay uses real native localhost HTTP admission, context cancellation,
tokenizer and response serialization, then invokes the real multimodal handler.
It bypasses the encoder and scripts engine output and prefill bootstrap:

```bash
PYTHONPATH=components/src python components/src/dynamo/sglang/tests/reproduce_multimodal_cancellation.py \
  --model-metadata /path/to/local/model-metadata \
  --output cancellation.json --expect-cancellation
```

Add `--disaggregated` for the decode handler. Use the same script on baseline
source without `--expect-cancellation`. Both arms must receive the disconnect;
the candidate must abort exactly `actual-rid` and finish before manual stream
release. The baseline requires manual release and records no abort. Each replay
also checks a direct context-cancellation control.

See `tests/multimodal_cancellation_evidence.json` for sanitized paired results
and dependency identities. The native binary used for this CPU replay differs
from current source; native interpreter-exit task warnings remain in both arms.
Full native shutdown is not qualified. Cluster validation is pending: these
results do not establish vision encoding, KV transfer, scheduler resource release,
or real model generation. No performance improvement has been measured.
