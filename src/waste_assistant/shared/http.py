"""Shared bounded multipart and structured HTTP errors for local services."""

from fastapi import Request
from pydantic import ValidationError
from python_multipart.multipart import parse_options_header
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from waste_assistant.shared.schemas import Contract, ErrorResponse

MULTIPART_OVERHEAD_BYTES = 64 * 1024


def multipart_openapi(fields: type[Contract]) -> dict:
    schema = fields.model_json_schema()
    schema["properties"]["image"] = {"type": "string", "format": "binary"}
    schema["required"] = ["image", *schema.get("required", [])]
    return {
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": schema}},
        }
    }


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


async def read_upload[T: Contract](
    request: Request, max_bytes: int, schema: type[T]
) -> tuple[T, bytes]:
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
        request.headers,
        bounded_stream(),
        max_files=1,
        max_fields=len(schema.model_fields),
        max_part_size=4096,
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
        if len(items) != len(values) + 1 or not isinstance(image, UploadFile):
            raise RequestProblem(
                422, "invalid_fields", "Expected one image and unique text fields"
            )
        try:
            fields = schema.model_validate(values)
        except ValidationError as error:
            raise RequestProblem(
                422, "invalid_fields", "Invalid or missing multipart fields"
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
