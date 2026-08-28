<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SSE single-event adapter benchmark

`sse_single_event.rs` isolates the zero-or-one iterator adapter used by the
OpenAI video SSE response path. The legacy case materializes a temporary `Vec`;
the reused case iterates the transposed `Option` directly. Serialization, Axum
event construction, SSE cadence, and wire bytes are outside this benchmark and
are unchanged by the source patch.

Run from the repository root:

```bash
rustc --edition=2024 -D warnings -C opt-level=3 \
  lib/llm/benches/sse_single_event.rs \
  -o /tmp/dynamo-sse-single-event-bench
DYN_SSE_BENCH_ITERATIONS=5000000 \
  /tmp/dynamo-sse-single-event-bench
```

The benchmark performs 10,000 warm-up operations per case before measuring.
Run it at least five times. Promotion requires:

- identical `Some`, `None`, and `Err` stream semantics;
- strictly fewer allocations and allocated bytes for emitted data and errors;
- no new allocation for filtered `None`; and
- no median latency regression.

## Backup 4 result

Controller run on 2026-08-28 from base
`b3358bd7207d84103dcaef9cfcf72eb0a1bdc41d`, with five trials of 5,000,000
measured operations per case.

| Case | Legacy allocations/op | New allocations/op | Legacy bytes/op | New bytes/op | Legacy median ns/op | New median ns/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Data (`Some`) | 1.000 | 0.000 | 16.0 | 0.0 | 5.24 | 0.37 |
| Filtered (`None`) | 0.000 | 0.000 | 0.0 | 0.0 | 0.00 | 0.00 |
| Error (`Err`) | 1.000 | 0.000 | 16.0 | 0.0 | 5.14 | 0.37 |

The measured payload is a fixed-size stand-in so results isolate iterator
materialization. The allocation count is the production-relevant result; the
16-byte measurement is a benchmark-payload lower bound, not the size of an
Axum `Event`. The source promotion gates passed. Cluster testing was not used
because the running Spark workloads are unrelated and this source benchmark
directly exercises the old and new adapter expressions.
