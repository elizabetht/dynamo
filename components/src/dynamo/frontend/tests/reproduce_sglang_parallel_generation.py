# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay parallel-tool defaults against a separately managed real server."""
import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import jsonschema


def validate(payload, events):
    calls = {}
    finishes = []
    for event in events:
        assert not event.get("error"), event
        for choice in event.get("choices", []):
            assert choice["index"] == 0
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
            message = choice.get("delta", choice.get("message", {}))
            for index, call in enumerate(message.get("tool_calls") or []):
                index = call.get("index", index)
                entry = calls.setdefault(
                    index, {"id": None, "name": "", "arguments": ""}
                )
                if call.get("id"):
                    assert entry["id"] in (None, call["id"])
                    entry["id"] = call["id"]
                for field in ("name", "arguments"):
                    entry[field] += call.get("function", {}).get(field) or ""
    assert finishes and all(f in ("stop", "tool_calls") for f in finishes), finishes
    assert calls, "missing forced call"
    assert sorted(calls) == list(range(len(calls))), calls
    assert len({c["id"] for c in calls.values()}) == len(calls)
    schemas = {
        t["function"]["name"]: t["function"]["parameters"] for t in payload["tools"]
    }
    for call in calls.values():
        assert call["id"]
        jsonschema.validate(json.loads(call["arguments"]), schemas[call["name"]])
        if isinstance(payload["tool_choice"], dict):
            assert call["name"] == payload["tool_choice"]["function"]["name"]
    if payload.get("parallel_tool_calls") is False:
        assert len(calls) == 1
    return {"calls": list(calls.values()), "finish_reasons": finishes}


async def main(args):
    rows = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=90)
    ) as session:
        for case in json.loads(args.corpus.read_text()):
            for iteration in range(args.repetitions):
                payload = {**case["request"], "model": args.model}
                events = []
                row = {"case": case["id"], "iteration": iteration, "events": events}
                try:
                    async with session.post(
                        args.url + "/v1/chat/completions", json=payload
                    ) as response:
                        row["http_status"] = response.status
                        if response.status != 200:
                            raise RuntimeError(await response.text())
                        if payload["stream"]:
                            pending = b""
                            done = False
                            async for data in response.content.iter_any():
                                pending += data
                                while b"\n" in pending:
                                    line, pending = pending.split(b"\n", 1)
                                    line = line.strip()
                                    if line == b"data: [DONE]":
                                        done = True
                                    elif line.startswith(b"data: "):
                                        events.append(json.loads(line[6:]))
                            assert done, "missing SSE DONE"
                        else:
                            events.append(await response.json())
                    row.update(validate(payload, events), valid=True)
                except Exception as error:
                    row.update(valid=False, error=repr(error))
                rows.append(row)
                args.output.write_text(
                    json.dumps(
                        {"arm": args.arm, "pair": args.pair, "rows": rows}, indent=2
                    )
                )
    assert all(r["valid"] for r in rows), "see saved invalid rows"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("url", "model", "arm"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    asyncio.run(main(parser.parse_args()))
