// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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

fn legacy_adapter(value: Result<Option<u64>, u64>) -> std::vec::IntoIter<Result<u64, u64>> {
    match value {
        Ok(Some(event)) => vec![Ok(event)].into_iter(),
        Ok(None) => vec![].into_iter(),
        Err(error) => vec![Err(error)].into_iter(),
    }
}

fn reused_adapter(value: Result<Option<u64>, u64>) -> std::option::IntoIter<Result<u64, u64>> {
    value.transpose().into_iter()
}

fn run_case<I>(name: &str, iterations: u64, mut operation: impl FnMut() -> I)
where
    I: Iterator<Item = Result<u64, u64>>,
{
    for _ in 0..10_000 {
        for item in operation() {
            let _ = black_box(item);
        }
    }

    let before = counters();
    let start = Instant::now();
    for _ in 0..iterations {
        for item in operation() {
            let _ = black_box(item);
        }
    }
    let elapsed = start.elapsed();
    let after = counters();
    let allocations = after.0 - before.0;
    let bytes = after.1 - before.1;
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

    println!(
        "{name}: iterations={iterations} allocations={allocations} allocated_bytes={bytes} \
         allocations_per_op={:.3} bytes_per_op={:.1} ns_per_op={ns_per_op:.2}",
        allocations as f64 / iterations as f64,
        bytes as f64 / iterations as f64,
    );
}

fn main() {
    let iterations = std::env::var("DYN_SSE_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000_000_u64);

    run_case("legacy_some", iterations, || legacy_adapter(Ok(Some(7))));
    run_case("reused_some", iterations, || reused_adapter(Ok(Some(7))));
    run_case("legacy_none", iterations, || legacy_adapter(Ok(None)));
    run_case("reused_none", iterations, || reused_adapter(Ok(None)));
    run_case("legacy_error", iterations, || legacy_adapter(Err(9)));
    run_case("reused_error", iterations, || reused_adapter(Err(9)));
}
