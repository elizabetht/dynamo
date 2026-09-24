<!-- # SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# SGLang

See [docs/backends/sglang/](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/sglang/overview.md) for documentation.

## Simultaneous stream completion and shutdown

The shared SGLang stream adapter gives shutdown precedence when its cancellation
monitor and next-item task finish together. Previously, that could leave a
completed EOF or engine-error task unconsumed, producing `Task exception was
never retrieved`. Cleanup now retrieves completed results that were not handled
by the normal stream path. EOF stays quiet; an engine failure is logged once.
Shutdown still propagates, and pending tasks retain bounded cancellation cleanup.

The two race regressions fail on base
`70e094f650ab89eb81d359825adcb78fc9256aba`; all 87 decode cancellation tests pass
with the change:

```bash
PYTHONPATH=components/src python -m pytest \
  components/src/dynamo/sglang/tests/test_sglang_decode_cancellation.py -q
```

A [portable reproduction](tests/reproduce_stream_shutdown.py) exercises local
native HTTP admission, token preprocessing, `DecodeWorkerHandler.generate`,
sampling conversion, cancellation and SSE serialization. Only the engine stream
is scripted. Run it in a Dynamo/SGLang development environment with Python 3.11,
SGLang 0.5.19 and locally cached model metadata (tokenizer, config and chat template;
weights are not loaded):

```bash
git show 70e094f650ab89eb81d359825adcb78fc9256aba:components/src/dynamo/sglang/request_handlers/cancellation.py > /tmp/baseline-cancellation.py
# MODEL_METADATA points to a local tokenizer/config directory.
for mode in normal eof error; do
  PYTHONPATH=components/src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python \
    components/src/dynamo/sglang/tests/reproduce_stream_shutdown.py \
    --model-metadata "$MODEL_METADATA" --mode "$mode" \
    --baseline-helper /tmp/baseline-cancellation.py --output "/tmp/baseline-$mode.json"
  PYTHONPATH=components/src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python \
    components/src/dynamo/sglang/tests/reproduce_stream_shutdown.py \
    --model-metadata "$MODEL_METADATA" --mode "$mode" \
    --output "/tmp/candidate-$mode.json"
done
```

[Recorded CPU evidence](tests/stream_shutdown_evidence.json) includes source and
metadata hashes. Across three paired scenarios (six HTTP requests), baseline has
one unconsumed exception for EOF and one for engine failure; candidate has zero.
Both arms emit `hello`, followed by `stop` for normal completion or one SSE 503
for shutdown, with no successful finish after shutdown. Sampling parameters,
engine abort IDs and normalized SSE events match. This demonstrates diagnostic
cleanup, not a change in the HTTP response contract.

Alternatives considered: leaving the race alone retains misleading EOF errors;
always consuming results during cleanup can log an already-propagated engine
failure twice. Tracking whether the result was consumed reuses existing cleanup
without awaiting a cancellation-resistant iterator. In-context self-review of
callers, error precedence, tests and the final diff found no outstanding defects.

Actual engine scheduler behavior and cluster validation remain pending. The
local native runtime also reports a pre-existing interpreter-exit task warning;
this change does not claim to fix full runtime shutdown. No performance
improvement has been measured.
