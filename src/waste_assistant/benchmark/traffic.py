"""Bounded open-loop replay with a terminal record for every offered arrival."""

import asyncio
from time import perf_counter
from uuid import uuid4

import httpx


async def replay(client, trace, images, locality, limit, *, max_lateness_ms=100):
    start = perf_counter()
    pending = set()
    rows = []

    async def send(row, image):
        sent = perf_counter()
        row["sent_at"] = sent - start
        try:
            response = await client.post(
                "/v1/classify",
                headers={"x-benchmark-id": row["trace_id"]},
                data={"locality_id": locality},
                files={"image": ("sample.jpg", image, "image/jpeg")},
            )
            row["http_status"] = response.status_code
            try:
                reply = response.json()
            except ValueError:
                reply = {}
            if not isinstance(reply, dict):
                reply = {}
            row["request_id"] = reply.get("request_id")
            row["code"] = reply.get("code")
            row["outcome"] = {200: "success", 429: "rejected", 504: "deadline"}.get(
                response.status_code, "failure"
            )
            if response.status_code == 200 and not row["request_id"]:
                row["outcome"] = "failure"
        except httpx.RequestError as error:
            row["outcome"] = "client_error"
            row["code"] = type(error).__name__
        finally:
            finished = perf_counter()
            row["finished_at"] = finished - start
            row["latency_ms"] = (finished - sent) * 1000

    for arrival in trace["arrivals"]:
        delay = start + arrival["at"] - perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        lateness = (perf_counter() - start - arrival["at"]) * 1000
        row = {
            "trace_id": uuid4().hex,
            "scheduled_at": arrival["at"],
            "lateness_ms": lateness,
            "image": arrival["image"]["path"],
        }
        rows.append(row)
        if len(pending) >= limit or lateness > max_lateness_ms:
            row["outcome"] = "generator_drop"
            row["code"] = "in_flight_limit" if len(pending) >= limit else "late_arrival"
            continue
        task = asyncio.create_task(send(row, images[row["image"]]))
        pending.add(task)
        task.add_done_callback(pending.discard)
    if pending:
        await asyncio.gather(*pending)
    return rows
