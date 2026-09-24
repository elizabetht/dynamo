<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## Cancellation before parallel output

For a parallel request (`n > 1`) whose engine has not emitted a child request ID,
stop and shutdown previously waited for the first output before the cancellation
monitor could run. A disconnected client could therefore leave Dynamo's handler
waiting indefinitely. The monitor now observes cancellation immediately when its
caller supplies the set of observed child IDs. Stop starts the existing one-second
drain; shutdown retains the existing `EngineShutdown` error. Late child IDs remain
eligible for exact-ID abort during the drain. Stable submitted-ID ordering and the
legacy single-ID monitor are unchanged.

Reproduce the handler regression with the supported SGLang dependencies installed:

```bash
PYTHONPATH=components/src python -m pytest -q \
  components/src/dynamo/sglang/tests/test_sglang_decode_cancellation.py
```

The new `test_parallel_cancellation_before_any_response` cases cover stop and
shutdown using the real text handler with a scripted engine. Both time out on the
baseline. The candidate finishes without inventing a child ID or requesting a
global abort. Existing cancellation tests cover ordered abort, late responses,
normal completion, and bounded drains.

A CPU localhost HTTP replay also submits a guided chat request with `n=2`, waits
for the text worker to start, then disconnects before output. Native admission and
context propagation execute normally; generation is scripted. Baseline and
candidate both receive cancellation, but only the candidate returns before manual
engine-stream release. Sanitized observations are in
[preoutput_cancellation_evidence.json](tests/preoutput_cancellation_evidence.json).

This bounds Dynamo's wait; it does not establish scheduler or GPU resource
reclamation. Stock SGLang can leave an unseen parallel prefix or child running;
engine-owned cancellation is separate work. No engine patch is included or
required for this handler behavior. Cluster validation is pending. The local
native runtime reports an existing interpreter-exit task warning, so full runtime
shutdown is not qualified. No performance improvement has been measured.

The selected change reuses the existing drain rather than adding an ID-discovery
scan or a second timer. Waiting for output retains the demonstrated hang, and
aborting a guessed parent ID cannot safely identify engine-generated children.
