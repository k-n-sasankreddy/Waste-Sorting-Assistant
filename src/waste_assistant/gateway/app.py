"""Classify uploads through a bounded pool of compatible workers."""

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import ValidationError
from starlette.responses import JSONResponse

from waste_assistant.gateway.pool import Job, TransientWorkerError, WorkerPool
from waste_assistant.guidance.rules import classify, load_rules
from waste_assistant.shared.config import Settings, WorkerSettings, load_settings
from waste_assistant.shared.http import (
    RequestProblem,
    error_response,
    multipart_openapi,
    read_upload,
)
from waste_assistant.shared.schemas import (
    ClassificationFields,
    ClassificationResponse,
    ErrorResponse,
    InferenceResponse,
)

LOGGER = logging.getLogger(__name__)


async def infer_worker(
    client: httpx.AsyncClient, worker: WorkerSettings, job: Job
) -> InferenceResponse:
    request_id = job.request_id
    attempt_id = job.attempt_id
    assert attempt_id is not None
    remaining_ms = int((job.deadline - asyncio.get_running_loop().time()) * 1000)
    if remaining_ms < 1:
        raise TimeoutError
    reply = await client.post(
        httpx.URL(str(worker.url)).join("/v1/infer"),
        data={
            "request_id": request_id,
            "attempt_id": attempt_id,
            "remaining_budget_ms": str(remaining_ms),
        },
        files={"image": ("upload", job.image, "application/octet-stream")},
    )
    if reply.status_code == 504:
        raise TimeoutError
    if reply.status_code == 429:
        raise TransientWorkerError(429, "worker_busy", "Try again in a moment.")
    if reply.status_code in (500, 503):
        raise TransientWorkerError(
            503, "worker_unavailable", "The classifier is unavailable."
        )
    if reply.status_code == 413:
        raise RequestProblem(413, "upload_too_large", "Image exceeds the upload limit.")
    if reply.status_code == 422:
        upstream_error = ErrorResponse.model_validate_json(reply.content)
        if (
            upstream_error.code == "invalid_image"
            and upstream_error.request_id == request_id
        ):
            raise RequestProblem(
                422,
                "invalid_image",
                "Upload a valid JPEG, PNG, or WebP within the pixel limit.",
            )
    if reply.status_code != 200:
        raise RequestProblem(
            503, "worker_unavailable", "The classifier is unavailable."
        )
    result = InferenceResponse.model_validate_json(reply.content)
    if result.request_id != request_id or result.attempt_id != attempt_id:
        raise RequestProblem(
            503,
            "worker_unavailable",
            "The classifier returned an invalid response.",
        )
    return result


def create_app(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    if settings.gateway is None:
        raise ValueError("Gateway configuration is missing")
    rules = load_rules(settings.gateway.rules_path)
    if rules.locality_id != settings.locality_id:
        raise ValueError("Gateway locality must match the demo ruleset")
    budget = settings.limits.request_budget_seconds

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=budget, trust_env=False, transport=transport
        ) as client:
            pool = WorkerPool(settings, client, infer_worker)
            app.state.pool = pool
            await pool.refresh()
            tasks = [
                asyncio.create_task(pool.dispatch(worker)) for worker in pool.workers
            ]
            tasks.append(asyncio.create_task(pool.monitor()))
            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="Waste Sorting Assistant gateway", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request, response: Response) -> dict[str, bool]:
        pool = request.app.state.pool
        is_ready = any(pool.available(worker) for worker in pool.workers)
        if not is_ready:
            response.status_code = 503
        return {"ready": is_ready}

    @app.get("/v1/system/status")
    async def system_status(request: Request) -> dict:
        return request.app.state.pool.snapshot()

    @app.post(
        "/v1/classify",
        response_model=ClassificationResponse,
        responses={
            code: {"model": ErrorResponse} for code in (400, 413, 422, 429, 503, 504)
        },
        openapi_extra=multipart_openapi(ClassificationFields),
    )
    async def classify_upload(
        request: Request,
    ) -> ClassificationResponse | JSONResponse:
        request_id = uuid4().hex
        pool = request.app.state.pool
        deadline = pool.loop.time() + budget
        reserved = False
        queued = None
        try:
            pool.reserve_upload()
            reserved = True
            async with asyncio.timeout_at(deadline):
                fields, image = await read_upload(
                    request, settings.limits.max_upload_bytes, ClassificationFields
                )
                if fields.locality_id != rules.locality_id:
                    raise RequestProblem(
                        422,
                        "unsupported_locality",
                        "Select the supported demo locality.",
                    )
                queued = pool.enqueue(request_id, image, deadline)
                result = await asyncio.shield(queued[1].result)
                return classify(fields, result, rules)
        except RequestProblem as error:
            return error_response(request_id, error.status, error.code, str(error))
        except (httpx.TimeoutException, TimeoutError):
            return error_response(
                request_id, 504, "deadline_exceeded", "Classification took too long."
            )
        except (httpx.RequestError, ValidationError):
            LOGGER.warning("Worker communication failed for request %s", request_id)
            return error_response(
                request_id, 503, "worker_unavailable", "The classifier is unavailable."
            )
        finally:
            if queued is not None:
                pool.withdraw(queued[1])
            elif reserved:
                pool.uploading -= 1

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        app = create_app(settings)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Gateway configuration failed: {error}\n")
    uvicorn.run(app, host="127.0.0.1", port=settings.gateway_port, workers=1)


if __name__ == "__main__":
    main()
