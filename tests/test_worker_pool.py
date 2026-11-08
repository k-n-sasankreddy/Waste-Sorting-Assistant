import asyncio
from pathlib import Path

import pytest

from waste_assistant.gateway.scheduling import choose_worker, estimated_completion_ms


def test_policies_skip_ineligible_workers_and_rotate_ties():
    assert choose_worker("round_robin", [5, None, 1], 0) == 0
    assert choose_worker("least_outstanding", [5, None, 1], 0) == 2
    assert choose_worker("least_outstanding", [2, None, 2], 1) == 2
    assert choose_worker("round_robin", [None, None], 0) is None


def test_estimated_completion_matches_hand_calculation():
    scores = [
        estimated_completion_ms(100, 2, 60, 0.1),
        estimated_completion_ms(200, 1, 50, 0.1),
        estimated_completion_ms(140, 0, None, 0.1),
    ]
    assert scores == [340, 550, 140]
    assert choose_worker("adaptive", scores, 0) == 2
    assert estimated_completion_ms(100, 0, 200, 0.1) == 110
    assert choose_worker("adaptive", [140, None, 140], 1) == 2
    assert choose_worker("adaptive", [None, None], 0) is None


@pytest.mark.parametrize("policy", ["round_robin", "least_outstanding", "adaptive"])
def test_pool_bounds_fifo_and_checks_model_compatibility(policy, monkeypatch):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from waste_assistant.gateway import pool as pool_module
    from waste_assistant.shared.config import load_settings
    from waste_assistant.shared.http import RequestProblem
    from waste_assistant.shared.schemas import InferenceResponse, Prediction

    clock = [0.0]
    monkeypatch.setattr(pool_module, "perf_counter", lambda: clock[0])

    async def scenario():
        settings = load_settings(
            Path(__file__).resolve().parents[1] / "config/dev.yaml"
        )
        settings.policy = policy
        settings.limits.max_waiting_per_worker = 2
        settings.limits.max_outstanding = 5
        settings.limits.max_queued_image_bytes = 4
        versions = {8101: "model", 8102: "model", 8103: "different"}
        calls = {worker.id: [] for worker in settings.workers}
        release = asyncio.Event()

        async def status(request):
            port = request.url.port
            return httpx.Response(
                200,
                json={
                    "worker_id": f"worker-{port - 8100}",
                    "boot_id": "test",
                    "ready": True,
                    "busy": False,
                    "model_version": versions[port],
                    "preprocessing_version": "prep",
                    "label_mapping_version": "labels",
                    "warmup_ms": 10,
                },
            )

        async def infer(client, worker, job):
            calls[worker.id].append(job.request_id)
            await release.wait()
            if job.request_id == "slow":
                clock[0] += 0.5
            if job.request_id == "failed":
                clock[0] += 0.5
                raise TimeoutError
            return InferenceResponse(
                request_id=job.request_id,
                attempt_id="attempt",
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
            pool = pool_module.WorkerPool(settings, client, infer)
            await pool.refresh()
            assert [pool.available(worker) for worker in pool.workers] == [
                True,
                True,
                False,
            ]
            tasks = [
                asyncio.create_task(pool.dispatch(worker)) for worker in pool.workers
            ]
            try:

                def add(name, image=b"ab", budget=3):
                    pool.reserve_upload()
                    return pool.enqueue(name, image, pool.loop.time() + budget)

                first = [add("first"), add("second")]
                await asyncio.sleep(0)
                assert [len(entries) for entries in calls.values()] == [1, 1, 0]
                waiting = [add("third"), add("fourth", budget=0.03)]
                assert pool.queued_bytes == 4 and pool.outstanding() == 4
                pool.reserve_upload()
                with pytest.raises(RequestProblem, match="queued image budget"):
                    pool.enqueue("too-many-bytes", b"x", pool.loop.time() + 3)
                with pytest.raises(RequestProblem, match="at capacity"):
                    pool.reserve_upload()
                pool.uploading -= 1
                await asyncio.sleep(0.04)
                _, expired = waiting.pop()
                pool.withdraw(expired)
                assert pool.queued_bytes == 2 and pool.outstanding() == 3
                waiting.append(add("fifth"))
                release.set()
                await asyncio.gather(*(job.result for _, job in first + waiting))
                expected = {
                    "worker-1": ["first", "third"],
                    "worker-2": ["second"],
                    "worker-3": [],
                }
                expected["worker-1" if policy == "round_robin" else "worker-2"].append(
                    "fifth"
                )
                assert calls == expected
                assert pool.queued_bytes == 0 and pool.outstanding() == 0
                pool.limits.max_waiting_per_worker = 1
                pool.limits.max_queued_image_bytes = 100
                capped = [add("cap-1"), add("cap-2")]
                pool.reserve_upload()
                with pytest.raises(RequestProblem, match="Worker queues are full"):
                    pool.enqueue("cap-3", b"x", pool.loop.time() + 3)
                pool.uploading -= 1
                for _, job in capped:
                    pool.withdraw(job)
                assert pool.queued_bytes == 0 and pool.outstanding() == 0
                if policy == "adaptive":
                    slow_worker, slow_job = add("slow")
                    prior = slow_worker.estimated_service_ms
                    await slow_job.result
                    assert slow_worker.estimated_service_ms == pytest.approx(
                        0.2 * 500 + 0.8 * prior
                    )
                    preferred, next_job = add("after-slowdown")
                    assert preferred is not slow_worker
                    await next_job.result
                    failed_worker, failed_job = add("failed")
                    estimate = failed_worker.estimated_service_ms
                    with pytest.raises(TimeoutError):
                        await failed_job.result
                    assert failed_worker.estimated_service_ms == estimate
                    preferred.seen_at -= settings.limits.status_stale_seconds + 1
                    assert not pool.available(preferred)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
