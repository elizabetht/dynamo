# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for a separately authorized, wrapper-owned real-engine experiment."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import jsonschema


async def main(args):
    corpus = json.loads(args.corpus.read_text())
    rows = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=90)
    ) as session:
        for name, payload in corpus:
            for iteration in range(args.repetitions + 1):
                payload = {
                    **payload,
                    "model": args.model,
                    "stream": True,
                    "temperature": 0,
                    "seed": 17,
                    "max_tokens": 128,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream_options": {"include_usage": True},
                }
                start = time.perf_counter()
                first = None
                event_times = []
                events = []
                pending = b""
                async with session.post(
                    args.url + "/v1/chat/completions", json=payload
                ) as response:
                    if response.status != 200:
                        raise RuntimeError((response.status, await response.text()))
                    async for data in response.content.iter_any():
                        pending += data
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            line = line.strip()
                            if (
                                not line.startswith(b"data: ")
                                or line == b"data: [DONE]"
                            ):
                                continue
                            event = json.loads(line[6:])
                            events.append(event)
                            if any(
                                c.get("delta", {}).get("content")
                                or c.get("delta", {}).get("tool_calls")
                                for c in event.get("choices", [])
                            ):
                                now = time.perf_counter()
                                event_times.append(now - start)
                                if first is None:
                                    first = now - start
                elapsed = time.perf_counter() - start
                content = ""
                calls = {}
                finish = None
                usage = None
                for event in events:
                    usage = event.get("usage") or usage
                    for c in event.get("choices", []):
                        delta = c.get("delta", {})
                        content += delta.get("content") or ""
                        finish = c.get("finish_reason") or finish
                        for call in delta.get("tool_calls", []):
                            entry = calls.setdefault(
                                call["index"], {"name": "", "arguments": ""}
                            )
                            for field in entry:
                                entry[field] += (
                                    call.get("function", {}).get(field) or ""
                                )
                assert finish in ("stop", "tool_calls"), (name, finish)
                if "response_format" in payload:
                    jsonschema.validate(
                        json.loads(content),
                        payload["response_format"]["json_schema"]["schema"],
                    )
                schemas = {
                    t["function"]["name"]: t["function"]["parameters"]
                    for t in payload.get("tools", [])
                }
                for call in calls.values():
                    jsonschema.validate(
                        json.loads(call["arguments"]), schemas[call["name"]]
                    )
                choice = payload.get("tool_choice")
                if choice == "required" or isinstance(choice, dict):
                    assert calls, (name, "missing forced call")
                if isinstance(choice, dict):
                    assert all(
                        c["name"] == choice["function"]["name"] for c in calls.values()
                    )
                assert content or calls, (name, "empty output")
                rows.append(
                    {
                        "case": name,
                        "warmup": iteration == 0,
                        "ttft_seconds": first,
                        "duration_seconds": elapsed,
                        "content_event_times_seconds": event_times,
                        "usage": usage,
                        "events": events,
                        "valid": True,
                    }
                )
                args.output.write_text(
                    json.dumps(
                        {"arm": args.arm, "pair": args.pair, "rows": rows}, indent=2
                    )
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["url", "model", "arm"]:
        p.add_argument("--" + n, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pair", type=int, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    asyncio.run(main(p.parse_args()))
