import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest
from PIL import Image

CONFIG = Path(__file__).resolve().parents[1] / "config/dev.yaml"


@pytest.fixture
def worker_setup(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from fastapi.testclient import TestClient

    from waste_assistant.ml.artifacts import (
        create_model,
        export_model,
        preprocessing_from_weights,
    )
    from waste_assistant.ml.dataset import CLASSES
    from waste_assistant.shared.config import load_settings
    from waste_assistant.worker.app import create_app

    settings = load_settings(CONFIG)
    settings.model.cpu_threads = 1
    settings.limits.max_upload_bytes = 4096
    settings.limits.max_decoded_pixels = 100
    worker = settings.workers[0].model_copy(
        update={"model_path": tmp_path / "artifact"}
    )
    export_model(
        worker.model_path,
        create_model(len(CLASSES)),
        labels=list(CLASSES),
        preprocessing=preprocessing_from_weights(100),
        threshold=0.5,
        training={},
    )
    return TestClient, create_app, settings, worker


def png(size=(4, 4)):
    stream = io.BytesIO()
    Image.new("RGB", size, "red").save(stream, format="PNG")
    return stream.getvalue()


def infer(client, image, *, attempt="attempt-1", budget=5000):
    return client.post(
        "/v1/infer",
        files={"image": ("sample.png", image, "image/png")},
        data={
            "request_id": "request-1",
            "attempt_id": attempt,
            "remaining_budget_ms": str(budget),
        },
    )


def test_worker_matches_offline_and_rejects_bad_inputs(worker_setup):
    from waste_assistant.ml.predict import Predictor

    test_client, create_app, settings, worker = worker_setup
    with test_client(create_app(settings, worker)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["warmup_ms"] > 0
        prediction = Predictor(worker.model_path).predict(png())
        response = infer(client, png())
        assert response.status_code == 200
        result = response.json()
        assert result["prediction"] == prediction.model_dump()
        assert result["processing_ms"] > 0
        assert result["attempt_id"] == "attempt-1"
        assert result["label_mapping_version"] == ready.json()["label_mapping_version"]
        invalid = infer(client, b"bad image")
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "invalid_image"
        assert invalid.json()["request_id"] == "request-1"
        assert infer(client, png((11, 10))).status_code == 422
        assert infer(client, b"x" * 4097).status_code == 413
        assert client.post("/v1/infer", content=b"bad form").status_code == 400
        chunked = client.post(
            "/v1/infer",
            content=iter([b"x" * 70_000]),
            headers={"content-type": "multipart/form-data; boundary=test"},
        )
        assert chunked.status_code == 413
        status = client.get("/status").json()
        assert status["completed_inferences"] == 1
        assert status["failed_inferences"] == 2
        assert status["busy"] is False


def test_worker_remains_busy_after_deadline_and_health_responds(
    worker_setup, monkeypatch
):
    from waste_assistant.ml.predict import Predictor

    test_client, create_app, settings, worker = worker_setup
    original = Predictor.predict
    started = Event()
    unblock = Event()
    image = png()

    def blocked_predict(self, data):
        if data == image:
            started.set()
            assert unblock.wait(5)
        return original(self, data)

    monkeypatch.setattr(Predictor, "predict", blocked_predict)
    with (
        test_client(create_app(settings, worker)) as client,
        ThreadPoolExecutor(1) as calls,
    ):
        try:
            first = calls.submit(infer, client, image, budget=1000)
            assert started.wait(3)
            assert client.get("/health").status_code == 200
            assert not first.done()
            assert client.get("/status").json()["current_attempt_id"] == "attempt-1"
            assert infer(client, image, attempt="attempt-2").status_code == 429
            assert first.result(timeout=3).status_code == 504
            assert client.get("/status").json()["busy"] is True
            assert infer(client, image, attempt="attempt-3").status_code == 429
        finally:
            unblock.set()
        deadline = monotonic() + 3
        while client.get("/status").json()["busy"] and monotonic() < deadline:
            sleep(0.01)
        assert infer(client, image, attempt="attempt-4").status_code == 200


def test_missing_weights_keep_worker_unready(worker_setup):
    test_client, create_app, settings, worker = worker_setup
    (worker.model_path / "weights.pt").unlink()
    with test_client(create_app(settings, worker)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        status = client.get("/status").json()
        assert status["ready"] is False and status["model_version"] is None
        assert infer(client, png()).status_code == 503
