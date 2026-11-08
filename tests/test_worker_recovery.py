import asyncio
from pathlib import Path

import pytest


def test_retry_quarantine_reconciliation_and_queued_reassignment():
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from waste_assistant.gateway.pool import WorkerPool
    from waste_assistant.shared.config import load_settings
    from waste_assistant.shared.http import RequestProblem
    from waste_assistant.shared.schemas import InferenceResponse, Prediction

    async def scenario():
        config = Path(__file__).resolve().parents[1] / "config/dev.yaml"
        settings = load_settings(config)
        statuses = {
            worker.id: {
                "worker_id": worker.id,
                "boot_id": "original",
                "ready": True,
                "busy": False,
                "model_version": "model",
                "preprocessing_version": "prep",
                "label_mapping_version": "labels",
                "warmup_ms": 10,
            }
            for worker in settings.workers
        }
        attempts = {}

        async def status(request):
            worker = f"worker-{request.url.port - 8100}"
            return (
                httpx.Response(200, json=statuses[worker])
                if worker in statuses
                else httpx.Response(503)
            )

        async def infer(client, worker, job):
            calls = attempts.setdefault(job.request_id, [])
            calls.append((worker.id, job.attempt_id, job.deadline))
            if job.request_id in ("ambiguous", "restart") and len(calls) == 1:
                raise httpx.ReadTimeout("response lost")
            if job.request_id == "always-fail":
                raise httpx.ConnectError("connection refused")
            if job.request_id == "invalid":
                raise RequestProblem(422, "invalid_image", "invalid")
            return InferenceResponse(
                request_id=job.request_id,
                attempt_id=job.attempt_id,
                review_threshold=0.5,
                prediction=Prediction(
                    candidate_label="glass",
                    score=0.9,
                    score_type="uncalibrated_softmax",
                    model_version="model",
                ),
                preprocessing_version="prep",
                label_mapping_version="labels",
                processing_ms=1,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(status)) as client:
            pool = WorkerPool(settings, client, infer)
            await pool.refresh()
            tasks = [
                asyncio.create_task(pool.dispatch(worker)) for worker in pool.workers
            ]

            def enqueue(name):
                pool.reserve_upload()
                return pool.enqueue(name, b"image", pool.loop.time() + 3)

            try:
                worker, first = enqueue("ambiguous")
                await first.result
                original, retry = attempts["ambiguous"]
                assert original[0] != retry[0] and original[1] != retry[1]
                assert original[2] == retry[2] == first.deadline
                assert pool.retries == 1 and pool.outstanding() == 1 and worker.active
                await pool.refresh()
                assert not pool.available(worker), (
                    "An idle report alone must not clear quarantine"
                )
                statuses[worker.config.id]["busy"] = True
                await pool.refresh()
                assert not pool.available(worker)
                statuses[worker.config.id].update(
                    busy=False, last_finished_attempt_id=original[1]
                )
                await pool.refresh()
                assert pool.available(worker) and pool.outstanding() == 0

                worker, restarted = enqueue("restart")
                await restarted.result
                assert worker.uncertain_attempt
                statuses[worker.config.id].update(boot_id="new", warmup_ms=25)
                await pool.refresh()
                assert pool.available(worker) and worker.estimated_service_ms == 25

                worker, waiting = enqueue("queued")
                del statuses[worker.config.id]
                # Apply a deterministic failed probe before any dispatch begins.
                worker.status = None
                pool.reassign_waiting(worker)
                await waiting.result
                assert len(attempts["queued"]) == 1
                assert attempts["queued"][0][0] != worker.config.id
                assert attempts["queued"][0][2] == waiting.deadline

                _, invalid = enqueue("invalid")
                with pytest.raises(RequestProblem, match="invalid"):
                    await invalid.result
                assert len(attempts["invalid"]) == 1

                _, failed = enqueue("always-fail")
                with pytest.raises(httpx.ConnectError):
                    await failed.result
                assert len(attempts["always-fail"]) == 2
                assert pool.outstanding() == 0 and pool.queued_bytes == 0
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
