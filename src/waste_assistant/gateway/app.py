"""Classify uploads through one worker and apply locality guidance."""

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

from waste_assistant.guidance.rules import classify, load_rules
from waste_assistant.shared.config import Settings, load_settings
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
    WorkerStatus,
)

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    if settings.gateway is None:
        raise ValueError("Gateway configuration is missing")
    worker = next(
        worker for worker in settings.workers if worker.id == settings.gateway.worker_id
    )
    rules = load_rules(settings.gateway.rules_path)
    if rules.locality_id != settings.locality_id:
        raise ValueError("Gateway locality must match the demo ruleset")
    budget = settings.limits.request_budget_seconds
    busy = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            base_url=str(worker.url),
            timeout=budget,
            trust_env=False,
            transport=transport,
        ) as client:
            app.state.worker = client
            yield

    app = FastAPI(title="Waste Sorting Assistant gateway", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request, response: Response) -> dict[str, bool]:
        is_ready = False
        try:
            async with asyncio.timeout(min(budget, 1)):
                reply = await request.app.state.worker.get("/ready")
                if reply.status_code == 200:
                    status = WorkerStatus.model_validate_json(reply.content)
                    is_ready = (
                        status.ready
                        and status.worker_id == worker.id
                        and bool(
                            status.model_version
                            and status.preprocessing_version
                            and status.label_mapping_version
                        )
                    )
        except (httpx.RequestError, ValidationError, TimeoutError):
            pass
        if not is_ready:
            response.status_code = 503
        return {"ready": is_ready}

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
        nonlocal busy
        request_id = uuid4().hex
        if busy:
            return error_response(
                request_id, 429, "gateway_busy", "Try again in a moment."
            )
        busy = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        try:
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
                attempt_id = uuid4().hex
                remaining_ms = int((deadline - loop.time()) * 1000)
                if remaining_ms < 1:
                    raise TimeoutError
                reply = await request.app.state.worker.post(
                    "/v1/infer",
                    data={
                        "request_id": request_id,
                        "attempt_id": attempt_id,
                        "remaining_budget_ms": str(remaining_ms),
                    },
                    files={"image": ("upload", image, "application/octet-stream")},
                )
                if reply.status_code == 504:
                    raise TimeoutError
                if reply.status_code == 429:
                    raise RequestProblem(429, "worker_busy", "Try again in a moment.")
                if reply.status_code == 413:
                    raise RequestProblem(
                        413, "upload_too_large", "Image exceeds the upload limit."
                    )
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
            busy = False

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
