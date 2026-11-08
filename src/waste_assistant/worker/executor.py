"""One event-loop-owned admission slot and one background inference thread."""

import asyncio
import hashlib
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from uuid import uuid4

import torch
from PIL import Image

from waste_assistant.ml.predict import Predictor
from waste_assistant.shared.config import Settings, WorkerSettings
from waste_assistant.shared.schemas import (
    InferenceFields,
    InferenceResponse,
    WorkerStatus,
)

LOGGER = logging.getLogger(__name__)


class WorkerExecutor:
    def __init__(self, settings: Settings, worker: WorkerSettings):
        self.worker = worker
        self.boot_id = uuid4().hex
        self.threads = settings.model.cpu_threads if settings.model else 1
        self.max_pixels = settings.limits.max_decoded_pixels
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.predictor: Predictor | None = None
        self.warmup_ms: float | None = None
        self.label_mapping_version: str | None = None
        self.busy = False
        self.attempt_id: str | None = None
        self.completed = 0
        self.failed = 0
        self.last_processing_ms: float | None = None

    def _load_and_warm(self) -> tuple[Predictor, float]:
        torch.set_num_threads(self.threads)
        predictor = Predictor(
            self.worker.model_path,
            max_pixels=self.max_pixels,
            formats=("JPEG", "PNG", "WEBP"),
        )
        sample = io.BytesIO()
        Image.new("RGB", (1, 1), "gray").save(sample, format="PNG")
        start = perf_counter()
        predictor.predict(sample.getvalue())
        return predictor, (perf_counter() - start) * 1000

    async def start(self) -> None:
        try:
            (
                self.predictor,
                self.warmup_ms,
            ) = await asyncio.get_running_loop().run_in_executor(
                self.pool, self._load_and_warm
            )
        except Exception:
            LOGGER.exception(
                "Worker %s failed model loading or warm-up", self.worker.id
            )
            return
        mapping = json.dumps(self.predictor.artifact.labels, separators=(",", ":"))
        self.label_mapping_version = hashlib.sha256(mapping.encode()).hexdigest()

    async def close(self) -> None:
        await asyncio.to_thread(self.pool.shutdown, wait=True)

    def reserve(self) -> bool:
        if self.busy:
            return False
        self.busy = True
        return True

    def release(self) -> None:
        self.busy = False
        self.attempt_id = None

    def submit(
        self, fields: InferenceFields, image: bytes
    ) -> asyncio.Future[InferenceResponse]:
        assert self.busy and self.attempt_id is None and self.predictor is not None
        future = asyncio.get_running_loop().run_in_executor(
            self.pool, self._infer, self.predictor, fields, image
        )
        self.attempt_id = fields.attempt_id
        future.add_done_callback(self._finish)
        return future

    def _infer(
        self, predictor: Predictor, fields: InferenceFields, image: bytes
    ) -> InferenceResponse:
        start = perf_counter()
        prediction = predictor.predict(image)
        processing_ms = (perf_counter() - start) * 1000
        return InferenceResponse(
            request_id=fields.request_id,
            attempt_id=fields.attempt_id,
            prediction=prediction,
            preprocessing_version=predictor.artifact.preprocessing.version,
            label_mapping_version=self.label_mapping_version,
            processing_ms=processing_ms,
        )

    def _finish(self, future: asyncio.Future[InferenceResponse]) -> None:
        try:
            result = future.result()
        except (Exception, asyncio.CancelledError):
            self.failed += 1
        else:
            self.completed += 1
            self.last_processing_ms = result.processing_ms
        finally:
            self.release()

    def status(self) -> WorkerStatus:
        artifact = self.predictor.artifact if self.predictor else None
        return WorkerStatus(
            worker_id=self.worker.id,
            boot_id=self.boot_id,
            ready=artifact is not None,
            busy=self.busy,
            current_attempt_id=self.attempt_id,
            model_version=artifact.model_version if artifact else None,
            preprocessing_version=artifact.preprocessing.version if artifact else None,
            label_mapping_version=self.label_mapping_version,
            warmup_ms=self.warmup_ms,
            completed_inferences=self.completed,
            failed_inferences=self.failed,
            last_processing_ms=self.last_processing_ms,
        )
