// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant as StdInstant;

use dynamo_kv_router::{ActiveSequencesMultiWorker, NoopSequencePublisher};
use rustc_hash::FxHashMap;
use tokio::time::Instant;

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: Forward the allocator contract and unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from the matching System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: Forward the allocator contract to System with the requested size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn counters() -> (u64, u64) {
    (
        ALLOCATION_COUNT.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

fn run_case(name: &str, iterations: u64, mut operation: impl FnMut()) {
    for _ in 0..1_000 {
        operation();
    }

    let before = counters();
    let start = StdInstant::now();
    for _ in 0..iterations {
        operation();
    }
    let elapsed = start.elapsed();
    let after = counters();
    let allocations = after.0 - before.0;
    let bytes = after.1 - before.1;
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
    let ops_per_second = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "{name}: iterations={iterations} allocations={allocations} allocated_bytes={bytes} \
         allocations_per_op={:.3} bytes_per_op={:.1} ns_per_op={ns_per_op:.1} \
         ops_per_second={ops_per_second:.0}",
        allocations as f64 / iterations as f64,
        bytes as f64 / iterations as f64,
    );
}

fn main() {
    let worker_count = std::env::var("DYN_LOAD_SNAPSHOT_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32_u64);
    let iterations = std::env::var("DYN_LOAD_SNAPSHOT_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200_000_u64);
    let workers: HashMap<_, _> = (0..worker_count)
        .map(|worker_id| (worker_id, (0_u32, 1_u32)))
        .collect();
    let sequences =
        ActiveSequencesMultiWorker::new(NoopSequencePublisher, 16, workers, false, 0, "benchmark");
    let now = Instant::now();

    run_case("fresh_map", iterations, || {
        black_box(sequences.project_worker_loads(None, now));
    });

    let mut projections = FxHashMap::default();
    sequences.project_worker_loads_into(None, now, &mut projections);
    run_case("reused_map", iterations, || {
        sequences.project_worker_loads_into(None, now, &mut projections);
        black_box(&projections);
    });
}
