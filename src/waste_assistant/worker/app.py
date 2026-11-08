"""FastAPI endpoints for one local inference worker."""

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import ValidationError
from python_multipart.multipart import parse_options_header
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from waste_assistant.ml.artifacts import InvalidImage
from waste_assistant.shared.config import Settings, WorkerSettings, load_settings
from waste_assistant.shared.schemas import (
    ErrorResponse,
    InferenceFields,
    InferenceResponse,
    WorkerStatus,
)
from waste_assistant.worker.executor import WorkerExecutor

LOGGER = logging.getLogger(__name__)
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class RequestProblem(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def error_response(
    request_id: str, status: int, code: str, message: str
) -> JSONResponse:
    error = ErrorResponse(request_id=request_id, code=code, message=message)
    return JSONResponse(error.model_dump(), status_code=status)


async def read_inference(
    request: Request, max_bytes: int
) -> tuple[InferenceFields, bytes]:
    content_type, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary", b"")
    if content_type != b"multipart/form-data" or not 1 <= len(boundary) <= 70:
        raise RequestProblem(
            400, "invalid_multipart", "Expected a multipart image upload"
        )
    max_body = max_bytes + MULTIPART_OVERHEAD_BYTES
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared_bytes = int(length)
        except ValueError as error:
            raise RequestProblem(
                400, "invalid_multipart", "Invalid Content-Length"
            ) from error
        if declared_bytes > max_body:
            raise RequestProblem(
                413, "upload_too_large", "Multipart body exceeds the upload limit"
            )

    async def bounded_stream():
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_body:
                raise RequestProblem(
                    413, "upload_too_large", "Multipart body exceeds the upload limit"
                )
            yield chunk

    parser = MultiPartParser(
        request.headers, bounded_stream(), max_files=1, max_fields=3, max_part_size=4096
    )
    try:
        form = await parser.parse()
    except (MultiPartException, ClientDisconnect) as error:
        raise RequestProblem(
            400, "invalid_multipart", "Could not read the multipart upload"
        ) from error
    try:
        items = form.multi_items()
        values = dict(items)
        image = values.pop("image", None)
        if len(items) != 4 or not isinstance(image, UploadFile) or len(values) != 3:
            raise RequestProblem(
                422, "invalid_fields", "Expected one image and three inference fields"
            )
        try:
            fields = InferenceFields.model_validate(values)
        except ValidationError as error:
            raise RequestProblem(
                422, "invalid_fields", "Invalid request ID, attempt ID, or budget"
            ) from error
        data = await image.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RequestProblem(
                413, "upload_too_large", "Image exceeds the upload limit"
            )
        if not data:
            raise RequestProblem(422, "invalid_image", "The uploaded image is empty")
        return fields, data
    finally:
        await form.close()


def create_app(settings: Settings, worker: WorkerSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = WorkerExecutor(settings, worker)
        app.state.executor = executor
        try:
            await executor.start()
            yield
        finally:
            await executor.close()

    app = FastAPI(
        title=f"Waste Sorting Assistant worker ({worker.id})", lifespan=lifespan
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": WorkerStatus}})
    async def ready(request: Request, response: Response) -> WorkerStatus:
        status = request.app.state.executor.status()
        if not status.ready:
            response.status_code = 503
        return status

    @app.get("/status")
    async def status(request: Request) -> WorkerStatus:
        return request.app.state.executor.status()

    @app.post(
        "/v1/infer",
        response_model=InferenceResponse,
        responses={
            code: {"model": ErrorResponse}
            for code in (400, 413, 422, 429, 500, 503, 504)
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image", *InferenceFields.model_fields],
                            "properties": {
                                "image": {"type": "string", "format": "binary"},
                                **InferenceFields.model_json_schema()["properties"],
                            },
                        }
                    }
                },
            }
        },
    )
    async def infer(request: Request) -> InferenceResponse | JSONResponse:
        executor: WorkerExecutor = request.app.state.executor
        request_id = uuid4().hex
        if executor.predictor is None:
            return error_response(
                request_id, 503, "worker_not_ready", "Worker model is unavailable"
            )
        if not executor.reserve():
            return error_response(
                request_id, 429, "worker_busy", "Worker inference slot is occupied"
            )

        submitted = False
        loop = asyncio.get_running_loop()
        start = loop.time()
        budget = settings.limits.request_budget_seconds
        try:
            async with asyncio.timeout_at(start + budget):
                fields, image = await read_inference(
                    request, settings.limits.max_upload_bytes
                )
            request_id = fields.request_id
            deadline = start + min(budget * 1000, fields.remaining_budget_ms) / 1000
            if loop.time() >= deadline:
                raise TimeoutError
            async with asyncio.timeout_at(deadline):
                future = executor.submit(fields, image)
                submitted = True
                return await asyncio.shield(future)
        except RequestProblem as error:
            return error_response(request_id, error.status, error.code, str(error))
        except InvalidImage as error:
            return error_response(request_id, 422, "invalid_image", str(error))
        except TimeoutError:
            return error_response(
                request_id, 504, "deadline_exceeded", "Worker deadline exhausted"
            )
        except Exception:
            LOGGER.exception("Inference failed for request %s", request_id)
            return error_response(
                request_id, 500, "inference_failed", "Worker inference failed"
            )
        finally:
            if not submitted:
                executor.release()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        worker = next(
            (worker for worker in settings.workers if worker.id == args.worker_id), None
        )
        if worker is None:
            raise ValueError(f"Unknown worker ID: {args.worker_id}")
        url = worker.url
        if (
            url.scheme != "http"
            or url.path != "/"
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Worker URL must be a plain HTTP origin")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Worker configuration failed: {error}\n")
    uvicorn.run(
        create_app(settings, worker),
        host=url.host.strip("[]"),
        port=url.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
