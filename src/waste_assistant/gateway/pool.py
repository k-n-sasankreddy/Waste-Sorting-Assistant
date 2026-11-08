"""Event-loop-owned worker registry, FIFO queues, and dispatch tasks."""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import httpx
from pydantic import ValidationError

from waste_assistant.gateway.scheduling import choose_worker, estimated_completion_ms
from waste_assistant.shared.config import Settings, WorkerSettings
from waste_assistant.shared.http import RequestProblem
from waste_assistant.shared.schemas import InferenceResponse, WorkerStatus

LOGGER = logging.getLogger("uvicorn.error")


@dataclass(eq=False)
class Job:
    request_id: str
    image: bytes
    deadline: float
    result: asyncio.Future[InferenceResponse]


@dataclass
class Worker:
    config: WorkerSettings
    queue: deque[Job] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    status: WorkerStatus | None = None
    seen_at: float = 0.0
    active_since: float | None = None
    estimated_service_ms: float | None = None
    estimate_boot_id: str | None = None
    assigned: int = 0
    completed: int = 0

    @property
    def active(self) -> bool:
        return self.active_since is not None


def fingerprint(status: WorkerStatus) -> tuple | None:
    versions = (
        status.model_version,
        status.preprocessing_version,
        status.label_mapping_version,
    )
    return versions if status.ready and all(versions) else None


class WorkerPool:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, infer):
        self.loop = asyncio.get_running_loop()
        self.limits = settings.limits
        self.policy = settings.policy
        self.adaptive = settings.adaptive
        self.client = client
        self.infer = infer
        selected = settings.gateway.worker_id
        self.workers = [
            Worker(config)
            for config in settings.workers
            if selected in (None, config.id)
        ]
        self.expected = None
        self.uploading = 0
        self.queued_bytes = 0
        self.cursor = 0

    def available(self, worker: Worker) -> bool:
        return (
            worker.status is not None
            and self.expected is not None
            and fingerprint(worker.status) == self.expected
            and worker.estimated_service_ms is not None
            and self.loop.time() - worker.seen_at <= self.limits.status_stale_seconds
        )

    def outstanding(self) -> int:
        return self.uploading + sum(len(w.queue) + w.active for w in self.workers)

    def reserve_upload(self) -> None:
        if self.outstanding() >= self.limits.max_outstanding:
            raise RequestProblem(
                429, "gateway_busy", "The gateway is at capacity. Try again shortly."
            )
        self.uploading += 1

    def enqueue(
        self, request_id: str, image: bytes, deadline: float
    ) -> tuple[Worker, Job]:
        # No await between selection, reservation and accounting.
        if self.loop.time() >= deadline:
            raise TimeoutError
        decision_start = perf_counter()
        ready = [self.available(worker) for worker in self.workers]
        if not any(ready):
            raise RequestProblem(
                503, "worker_unavailable", "No compatible worker is available."
            )
        if self.queued_bytes + len(image) > self.limits.max_queued_image_bytes:
            raise RequestProblem(
                429, "gateway_busy", "The queued image budget is full."
            )
        scores = [
            self.score(worker, decision_start)
            if available and len(worker.queue) < self.limits.max_waiting_per_worker
            else None
            for worker, available in zip(self.workers, ready, strict=True)
        ]
        index = choose_worker(self.policy, scores, self.cursor)
        if index is None:
            raise RequestProblem(
                429, "gateway_busy", "Worker queues are full. Try again shortly."
            )
        worker = self.workers[index]
        job = Job(request_id, image, deadline, self.loop.create_future())
        worker.queue.append(job)
        self.queued_bytes += len(image)
        self.uploading -= 1
        worker.assigned += 1
        self.cursor = (index + 1) % len(self.workers)
        worker.wake.set()
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "scheduler_decision %s",
                json.dumps(
                    {
                        "request_id": request_id,
                        "policy": self.policy,
                        "worker_id": worker.config.id,
                        "image_bytes": len(image),
                        "scores": {
                            w.config.id: score
                            for w, score in zip(self.workers, scores, strict=True)
                        },
                        "decision_ms": (perf_counter() - decision_start) * 1000,
                    }
                ),
            )
        return worker, job

    def score(self, worker: Worker, now: float) -> float:
        if self.policy != "adaptive":
            return len(worker.queue) + worker.active
        assert worker.estimated_service_ms is not None
        elapsed = (
            (now - worker.active_since) * 1000
            if worker.active_since is not None
            else None
        )
        return estimated_completion_ms(
            worker.estimated_service_ms,
            len(worker.queue),
            elapsed,
            self.adaptive.residual_floor_fraction,
        )

    def withdraw(self, worker: Worker, job: Job) -> None:
        if job in worker.queue:
            worker.queue.remove(job)
            self.queued_bytes -= len(job.image)
        if not job.result.done():
            job.result.cancel()
        elif not job.result.cancelled():
            job.result.exception()

    async def probe(self, worker: Worker) -> WorkerStatus | None:
        try:
            async with asyncio.timeout(min(self.limits.request_budget_seconds, 1)):
                reply = await self.client.get(
                    httpx.URL(str(worker.config.url)).join("/ready")
                )
                if reply.status_code != 200:
                    return None
                status = WorkerStatus.model_validate_json(reply.content)
                return status if status.worker_id == worker.config.id else None
        except (httpx.RequestError, ValidationError, TimeoutError):
            return None

    async def refresh(self) -> None:
        statuses = await asyncio.gather(
            *(self.probe(worker) for worker in self.workers)
        )
        for worker, status in zip(self.workers, statuses, strict=True):
            worker.status, worker.seen_at = status, self.loop.time()
            if status is not None and status.boot_id != worker.estimate_boot_id:
                worker.estimated_service_ms = None
                worker.estimate_boot_id = None
            if (
                worker.estimated_service_ms is None
                and status is not None
                and status.warmup_ms is not None
                and status.warmup_ms > 0
            ):
                versions = fingerprint(status)
                if versions is not None and self.expected in (None, versions):
                    self.expected = versions
                    worker.estimated_service_ms = status.warmup_ms
                    worker.estimate_boot_id = status.boot_id

    async def monitor(self) -> None:
        while True:
            await asyncio.sleep(self.limits.health_poll_seconds)
            await self.refresh()

    async def dispatch(self, worker: Worker) -> None:
        while True:
            await worker.wake.wait()
            while worker.queue:
                job = worker.queue.popleft()
                self.queued_bytes -= len(job.image)
                try:
                    if job.deadline <= self.loop.time():
                        raise TimeoutError
                    if not self.available(worker):
                        raise RequestProblem(
                            503,
                            "worker_unavailable",
                            "The assigned worker is unavailable.",
                        )
                    dispatch_start = worker.active_since = perf_counter()
                    boot_id = worker.estimate_boot_id
                    async with asyncio.timeout_at(job.deadline):
                        result = await self.infer(self.client, worker.config, job)
                    dispatch_ms = (perf_counter() - dispatch_start) * 1000
                    actual = (
                        result.prediction.model_version,
                        result.preprocessing_version,
                        result.label_mapping_version,
                    )
                    if actual != self.expected:
                        worker.status = None
                        raise RequestProblem(
                            503,
                            "worker_unavailable",
                            "The worker model is incompatible.",
                        )
                    worker.completed += 1
                    if (
                        boot_id == worker.estimate_boot_id
                        and worker.estimated_service_ms is not None
                    ):
                        alpha = self.adaptive.alpha
                        worker.estimated_service_ms = (
                            alpha * dispatch_ms
                            + (1 - alpha) * worker.estimated_service_ms
                        )
                    if LOGGER.isEnabledFor(logging.INFO):
                        LOGGER.info(
                            "scheduler_sample %s",
                            json.dumps(
                                {
                                    "request_id": job.request_id,
                                    "worker_id": worker.config.id,
                                    "dispatch_ms": dispatch_ms,
                                    "worker_processing_ms": result.processing_ms,
                                    "estimated_service_ms": worker.estimated_service_ms,
                                }
                            ),
                        )
                    if not job.result.done():
                        job.result.set_result(result)
                except Exception as error:
                    if not job.result.done():
                        job.result.set_exception(error)
                finally:
                    worker.active_since = None
            worker.wake.clear()

    def snapshot(self) -> dict:
        return {
            "policy": self.policy,
            "outstanding": self.outstanding(),
            "max_outstanding": self.limits.max_outstanding,
            "queued_image_bytes": self.queued_bytes,
            "max_queued_image_bytes": self.limits.max_queued_image_bytes,
            "workers": [
                {
                    "id": worker.config.id,
                    "available": self.available(worker),
                    "model_version": worker.status.model_version
                    if worker.status
                    else None,
                    "waiting": len(worker.queue),
                    "active": worker.active,
                    "assigned": worker.assigned,
                    "completed": worker.completed,
                    "estimated_service_ms": worker.estimated_service_ms,
                }
                for worker in self.workers
            ],
        }
