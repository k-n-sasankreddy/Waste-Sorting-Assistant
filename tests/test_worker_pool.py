import asyncio
from pathlib import Path

import pytest

from waste_assistant.gateway.scheduling import choose_worker


def test_policies_skip_ineligible_workers_and_rotate_ties():
    assert choose_worker("round_robin", [5, None, 1], 0) == 0
    assert choose_worker("least_outstanding", [5, None, 1], 0) == 2
    assert choose_worker("least_outstanding", [2, None, 2], 1) == 2
    assert choose_worker("round_robin", [None, None], 0) is None


@pytest.mark.parametrize("policy", ["round_robin", "least_outstanding"])
def test_pool_bounds_fifo_and_checks_model_compatibility(policy):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    from waste_assistant.gateway.pool import WorkerPool
    from waste_assistant.shared.config import load_settings
    from waste_assistant.shared.http import RequestProblem
    from waste_assistant.shared.schemas import InferenceResponse, Prediction

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
                },
            )

        async def infer(client, worker, job):
            calls[worker.id].append(job.request_id)
            await release.wait()
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
            pool = WorkerPool(settings, client, infer)
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
                expired_worker, expired = waiting.pop()
                pool.withdraw(expired_worker, expired)
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
                for worker, job in capped:
                    pool.withdraw(worker, job)
                assert pool.queued_bytes == 0 and pool.outstanding() == 0
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
