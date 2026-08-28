<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router load-snapshot allocation benchmark

`load_snapshot.rs` measures the production worker-load projection operation used by
router admission. It compares constructing a fresh `FxHashMap` with clearing and
refilling an already-sized map. The counting allocator reports allocation count and
requested bytes; elapsed wall time reports throughput and latency.

Run the benchmark from the repository root:

```bash
DYN_LOAD_SNAPSHOT_WORKERS=32 \
DYN_LOAD_SNAPSHOT_ITERATIONS=500000 \
cargo bench -p dynamo-kv-router --bench load_snapshot
```

The benchmark performs 1,000 unmeasured warm-up operations per case. Allocation
counters and timing start afterward. Run on an otherwise quiet host and repeat the
command at least five times. Promotion requires:

- identical focused and crate test results;
- zero steady-state allocations and bytes for the reused-map case;
- strictly fewer allocations and bytes than the fresh-map case; and
- no latency regression across the five-run median.

## Experiment 3 result

Controller environment on 2026-08-28: AMD Ryzen 7 6800H, 8 cores / 16 threads,
Rust 1.96.1. Base commit: `b3358bd7207d84103dcaef9cfcf72eb0a1bdc41d`.
Each trial used 32 workers and 500,000 measured iterations.

| Trial | Fresh allocations/op | Fresh bytes/op | Fresh ns/op | Reused allocations/op | Reused bytes/op | Reused ns/op |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 3152.0 | 434.5 | 0.000 | 0.0 | 307.7 |
| 2 | 1.000 | 3152.0 | 391.2 | 0.000 | 0.0 | 366.2 |
| 3 | 1.000 | 3152.0 | 343.8 | 0.000 | 0.0 | 314.3 |
| 4 | 1.000 | 3152.0 | 345.6 | 0.000 | 0.0 | 318.0 |
| 5 | 1.000 | 3152.0 | 347.4 | 0.000 | 0.0 | 327.1 |
| Median | 1.000 | 3152.0 | 347.4 | 0.000 | 0.0 | 318.0 |

The reused-map path removed one allocation and 3,152 allocated bytes per admission
after warm-up. Its median latency was 8.5% lower, and it was lower in all five trials.
The source promotion gates passed; no cluster benchmark was needed because this
benchmark invokes the same projection methods used by production admission.
