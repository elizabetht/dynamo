// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_protocols::types::ChatCompletionRequestMessage;
use minijinja::value::Value;
use serde_json::json;

// Isolates the message conversion used by OAIChatLikeRequest::messages; excludes
// HTTP parsing, template execution, tokenization, routing, and GPU generation.
fn prompt_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_messages");
    group.sample_size(50);
    for (turns, bytes_per_message) in [(2, 128), (32, 4096), (128, 4096)] {
        let messages: Vec<ChatCompletionRequestMessage> = (0..turns)
            .map(|turn| {
                serde_json::from_value(json!({
                    "role": if turn % 2 == 0 { "user" } else { "assistant" },
                    "content": "context ".repeat(bytes_per_message / 8),
                }))
                .unwrap()
            })
            .collect();
        let legacy = || Value::from_serialize(serde_json::to_value(&messages).unwrap());
        let direct = || Value::from_serialize(&messages);
        assert_eq!(
            serde_json::to_string(&legacy()).unwrap(),
            serde_json::to_string(&direct()).unwrap()
        );
        group.throughput(Throughput::Bytes((turns * bytes_per_message) as u64));
        group.bench_with_input(
            BenchmarkId::new("via_json", turns),
            &messages,
            |b, messages| {
                b.iter(|| {
                    Value::from_serialize(serde_json::to_value(black_box(messages)).unwrap())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("direct", turns),
            &messages,
            |b, messages| {
                b.iter(|| Value::from_serialize(black_box(messages)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, prompt_messages);
criterion_main!(benches);
