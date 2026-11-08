import asyncio
import json

import pytest


def test_seeded_trace_and_offered_denominators(tmp_path):
    pytest.importorskip("httpx")
    from waste_assistant.benchmark.generate import build_trace
    from waste_assistant.benchmark.report import summarize

    source = tmp_path / "test.jsonl"
    source.write_text(
        "\n".join(json.dumps({"path": str(i)}) for i in range(20)), encoding="utf-8"
    )
    assert build_trace(source, "burst", 9) == build_trace(source, "burst", 9)
    assert build_trace(source, "burst", 9) != build_trace(source, "burst", 10)
    rows = [
        {
            "trace_id": "1",
            "outcome": "success",
            "sent_at": 0,
            "latency_ms": 10,
            "finished_at": 0.01,
        },
        {
            "trace_id": "2",
            "outcome": "success",
            "sent_at": 0.9,
            "latency_ms": 200,
            "finished_at": 1.1,
        },
        {"trace_id": "3", "outcome": "deadline", "sent_at": 0.9, "latency_ms": 200},
        {"trace_id": "4", "outcome": "rejected", "sent_at": 0.9, "latency_ms": 2},
        {"trace_id": "5", "outcome": "generator_drop"},
    ]
    for row in rows:
        row["lateness_ms"] = 0
    terminals = [{"trace_id": str(i), "admitted": i < 4} for i in range(1, 5)]
    result = summarize(rows, terminals, [], [], 1)
    assert result["offered"] == 5
    assert result["admitted"] == 3
    assert result["on_time_fraction"] == 0.4
    assert result["successful_throughput_per_s"] == 1
    assert result["generator_drop"] == 1
    assert result["p95_ms"] == pytest.approx(190.5)


def test_open_loop_records_client_capacity_drops():
    httpx = pytest.importorskip("httpx")
    from waste_assistant.benchmark.traffic import replay

    async def scenario():
        trace = {"arrivals": [{"at": 0, "image": {"path": "image.jpg"}}] * 2}

        def handler(request):
            assert request.headers["x-benchmark-id"]
            return httpx.Response(429, json={"request_id": "r", "code": "gateway_busy"})

        async with httpx.AsyncClient(
            base_url="http://localhost", transport=httpx.MockTransport(handler)
        ) as client:
            return await replay(client, trace, {"image.jpg": b"bytes"}, "demo", 1)

    sent, dropped = asyncio.run(scenario())
    assert sent["outcome"] == "rejected"
    assert dropped["outcome"] == "generator_drop" and "sent_at" not in dropped
