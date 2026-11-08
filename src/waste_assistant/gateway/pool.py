"""Event-loop-owned worker registry, FIFO queues, and dispatch tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

import httpx
from pydantic import ValidationError

from waste_assistant.gateway.scheduling import choose_worker, estimated_completion_ms
from waste_assistant.shared.config import Settings, WorkerSettings
from waste_assistant.shared.http import RequestProblem
from waste_assistant.shared.schemas import InferenceResponse, WorkerStatus

LOGGER = logging.getLogger("uvicorn.error")


class TransientWorkerError(RequestProblem):
    """The worker responded with a retryable service error."""


@dataclass(eq=False)
class Job:
    request_id: str
    image: bytes
    deadline: float
    result: asyncio.Future[InferenceResponse]
    queued_worker: Worker | None = None
    attempt_id: str | None = None
    attempts: int = 0
    queued_at: float = 0.0


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
    uncertain_attempt: tuple[str, str] | None = None
    externally_busy: bool = False

    @property
    def active(self) -> bool:
        return self.active_since is not None or self.uncertain_attempt is not None


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
        self.retries = 0

    def available(self, worker: Worker) -> bool:
        return (
            worker.status is not None
            and self.expected is not None
            and fingerprint(worker.status) == self.expected
            and worker.estimated_service_ms is not None
            and worker.uncertain_attempt is None
            and not worker.externally_busy
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
        job = Job(request_id, image, deadline, self.loop.create_future())
        worker = self.assign(job)
        self.uploading -= 1
        return worker, job

    def assign(self, job: Job, *, exclude: Worker | None = None) -> Worker:
        # No await between selection, reservation and accounting.
        if self.loop.time() >= job.deadline:
            raise TimeoutError
        decision_start = perf_counter()
        ready = [
            worker is not exclude and self.available(worker) for worker in self.workers
        ]
        if not any(ready):
            raise RequestProblem(
                503, "worker_unavailable", "No compatible worker is available."
            )
        if self.queued_bytes + len(job.image) > self.limits.max_queued_image_bytes:
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
        job.queued_worker = worker
        job.queued_at = perf_counter()
        worker.queue.append(job)
        self.queued_bytes += len(job.image)
        worker.assigned += 1
        self.cursor = (index + 1) % len(self.workers)
        worker.wake.set()
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "scheduler_decision %s",
                json.dumps(
                    {
                        "request_id": job.request_id,
                        "policy": self.policy,
                        "worker_id": worker.config.id,
                        "image_bytes": len(job.image),
                        "prior_attempts": job.attempts,
                        "scores": {
                            w.config.id: score
                            for w, score in zip(self.workers, scores, strict=True)
                        },
                        "decision_ms": (perf_counter() - decision_start) * 1000,
                    }
                ),
            )
        return worker

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

    def withdraw(self, job: Job) -> None:
        if worker := job.queued_worker:
            worker.queue.remove(job)
            self.queued_bytes -= len(job.image)
            job.queued_worker = None
        if not job.result.done():
            job.result.cancel()
        elif not job.result.cancelled():
            job.result.exception()

    async def probe(self, worker: Worker) -> WorkerStatus | None:
        try:
            async with asyncio.timeout(min(self.limits.request_budget_seconds, 1)):
                reply = await self.client.get(
                    httpx.URL(str(worker.config.url)).join("/status")
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
            if status is not None and worker.uncertain_attempt is not None:
                boot, attempt = worker.uncertain_attempt
                if status.boot_id != boot or status.last_finished_attempt_id == attempt:
                    worker.uncertain_attempt = None
            worker.externally_busy = bool(
                status and status.busy and worker.active_since is None
            )
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
        for worker in self.workers:
            if not self.available(worker):
                self.reassign_waiting(worker)

    def take_next(self, worker: Worker) -> Job:
        job = worker.queue.popleft()
        self.queued_bytes -= len(job.image)
        job.queued_worker = None
        return job

    def reassign_waiting(self, worker: Worker) -> None:
        while worker.queue:
            job = self.take_next(worker)
            if not job.result.done():
                try:
                    self.assign(job, exclude=worker)
                except (RequestProblem, TimeoutError) as error:
                    job.result.set_exception(error)

    def retry(self, worker: Worker, job: Job) -> bool:
        remaining_ms = (job.deadline - self.loop.time()) * 1000
        if (
            job.result.done()
            or job.attempts > self.limits.max_retries
            or remaining_ms < self.limits.min_retry_budget_ms
            or self.outstanding() >= self.limits.max_outstanding
        ):
            return False
        try:
            self.assign(job, exclude=worker)
        except (RequestProblem, TimeoutError):
            return False
        self.retries += 1
        return True

    async def monitor(self) -> None:
        while True:
            await asyncio.sleep(self.limits.health_poll_seconds)
            await self.refresh()

    async def dispatch(self, worker: Worker) -> None:
        while True:
            await worker.wake.wait()
            while worker.queue:
                if not self.available(worker):
                    self.reassign_waiting(worker)
                    break
                await self.run_attempt(worker, self.take_next(worker))
            worker.wake.clear()

    async def run_attempt(self, worker: Worker, job: Job) -> None:
        if job.result.done():
            return
        if job.deadline <= self.loop.time():
            job.result.set_exception(TimeoutError())
            return
        boot_id = worker.estimate_boot_id
        assert boot_id is not None
        attempt_id = job.attempt_id = uuid4().hex
        job.attempts += 1
        dispatch_start = worker.active_since = perf_counter()
        error = None
        retryable = False
        outcome = "failed"
        try:
            deadline = min(
                job.deadline, self.loop.time() + self.limits.attempt_timeout_seconds
            )
            async with asyncio.timeout_at(deadline):
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
                    503, "worker_unavailable", "The worker model is incompatible."
                )
            worker.completed += 1
            if (
                boot_id == worker.estimate_boot_id
                and worker.estimated_service_ms is not None
            ):
                alpha = self.adaptive.alpha
                worker.estimated_service_ms = (
                    alpha * dispatch_ms + (1 - alpha) * worker.estimated_service_ms
                )
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.info(
                    "scheduler_sample %s",
                    json.dumps(
                        {
                            "request_id": job.request_id,
                            "attempt_id": attempt_id,
                            "worker_id": worker.config.id,
                            "dispatch_ms": dispatch_ms,
                            "worker_processing_ms": result.processing_ms,
                            "estimated_service_ms": worker.estimated_service_ms,
                        }
                    ),
                )
            outcome = "completed"
            if not job.result.done():
                job.result.set_result(result)
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            TransientWorkerError,
        ) as failure:
            worker.status = None
            retryable = True
            error = failure
        except (httpx.TransportError, TimeoutError) as failure:
            # The request may have reached the worker. Keep its slot occupied
            # until a status probe confirms this attempt ended or the boot changed.
            worker.uncertain_attempt = (boot_id, attempt_id)
            outcome = "uncertain"
            retryable = True
            error = failure
        except Exception as failure:
            error = failure
        finally:
            worker.active_since = None
        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info(
                "scheduler_attempt %s",
                json.dumps(
                    {
                        "request_id": job.request_id,
                        "attempt_id": attempt_id,
                        "worker_id": worker.config.id,
                        "boot_id": boot_id,
                        "outcome": outcome,
                        "attempt_number": job.attempts,
                        "queue_ms": (dispatch_start - job.queued_at) * 1000,
                        "dispatch_ms": (perf_counter() - dispatch_start) * 1000,
                    }
                ),
            )
        if error is not None and not job.result.done():
            if not (retryable and self.retry(worker, job)):
                job.result.set_exception(error)

    def snapshot(self) -> dict:
        return {
            "policy": self.policy,
            "outstanding": self.outstanding(),
            "retries": self.retries,
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
                    "boot_id": worker.status.boot_id if worker.status else None,
                    "quarantined": worker.uncertain_attempt is not None,
                    "active": worker.active,
                    "assigned": worker.assigned,
                    "completed": worker.completed,
                    "estimated_service_ms": worker.estimated_service_ms,
                }
                for worker in self.workers
            ],
        }
