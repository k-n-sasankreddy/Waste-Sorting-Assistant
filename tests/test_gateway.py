import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gateway(monkeypatch):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from waste_assistant.gateway.app import create_app
    from waste_assistant.shared.config import load_settings

    monkeypatch.chdir(ROOT)
    settings = load_settings("config/dev.yaml")
    settings.gateway.worker_id = "worker-1"
    settings.limits.max_outstanding = 1
    upstream = {"score": 0.8, "label": "plastic", "status": 200, "calls": 0}

    async def worker(request):
        if request.url.path != "/ready":
            upstream["calls"] += 1
        if upstream["status"] != 200:
            return httpx.Response(upstream["status"])
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={
                    "worker_id": "worker-1",
                    "boot_id": "boot",
                    "ready": True,
                    "busy": False,
                    "model_version": "example-model",
                    "preprocessing_version": "prep",
                    "label_mapping_version": "labels",
                    "warmup_ms": 5,
                },
            )
        if "started" in upstream:
            upstream["started"].set()
            assert await asyncio.to_thread(upstream["release"].wait, 5)
        body = request.read()
        fields = {
            key: re.search(rb'name="' + key.encode() + rb'"\r\n\r\n([^\r]+)', body)[
                1
            ].decode()
            for key in ("request_id", "attempt_id", "remaining_budget_ms")
        }
        assert 0 < int(fields["remaining_budget_ms"]) <= 10000
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "request_id": fields["request_id"],
                    "attempt_id": upstream.get("attempt_id", fields["attempt_id"]),
                    "prediction": {
                        "candidate_label": upstream["label"],
                        "score": upstream["score"],
                        "model_version": "example-model",
                        "score_type": "uncalibrated_softmax",
                    },
                    "processing_ms": 3,
                    "preprocessing_version": "prep",
                    "label_mapping_version": "labels",
                    "review_threshold": 0.6,
                }
            ),
        )

    with TestClient(
        create_app(settings, transport=httpx.MockTransport(worker))
    ) as client:
        yield client, upstream


def classify(client, locality="demo-locality", **context):
    return client.post(
        "/v1/classify",
        data={"locality_id": locality, **context},
        files={"image": ("photo.jpg", b"test transport bytes", "image/jpeg")},
    )


def test_review_clarification_and_missing_guidance(gateway):
    client, upstream = gateway
    assert client.get("/ready").json() == {"ready": True}
    missing = classify(client).json()
    assert missing["status"] == "classified"
    assert missing["guidance"]["status"] == "needs_clarification"
    assert missing["guidance"]["ruleset_status"] == "demonstration_only"
    complete = classify(client, item_type="Bottle", contamination="false").json()
    assert complete["guidance"]["status"] == "available"
    upstream["score"] = 0.59
    review = classify(client, item_type="Bottle", contamination="false").json()
    assert review["status"] == "needs_review"
    assert review["guidance"]["status"] == "needs_clarification"
    upstream.update(score=0.8, label="trash")
    assert classify(client).json()["guidance"]["status"] == "unavailable"


def test_gateway_rejects_locality_and_maps_upstream_failures(gateway):
    client, upstream = gateway
    reply = classify(client, locality="unsupported")
    assert reply.status_code == 422 and reply.json()["code"] == "unsupported_locality"
    assert upstream["calls"] == 0
    upstream["status"] = 429
    assert classify(client).status_code == 429
    upstream["status"] = 504
    assert classify(client).status_code == 504
    upstream["status"] = 503
    assert classify(client).status_code == 503
    assert client.get("/health").status_code == 200
    upstream.update(status=200, attempt_id="wrong")
    assert classify(client).status_code == 503


def test_gateway_rejects_second_upload_without_forwarding_it(gateway):
    client, upstream = gateway
    started, release = Event(), Event()
    upstream.update(started=started, release=release)
    with ThreadPoolExecutor(1) as requests:
        first = requests.submit(classify, client)
        try:
            assert started.wait(3)
            reply = classify(client)
            assert reply.status_code == 429 and reply.json()["code"] == "gateway_busy"
            assert upstream["calls"] == 1
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
