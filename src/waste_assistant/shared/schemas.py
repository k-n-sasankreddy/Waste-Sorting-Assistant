"""Shared payloads for the gateway and worker boundaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
RequestIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]
Score = Annotated[float, Field(ge=0, le=1)]
DurationMs = Annotated[float, Field(ge=0)]
Material = Literal["glass", "paper", "cardboard", "plastic", "metal", "trash"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ClassificationFields(Contract):
    """Text fields alongside the image in a multipart upload."""

    locality_id: NonEmpty
    item_type: NonEmpty | None = None
    contamination: bool | None = None


class InferenceFields(Contract):
    """Metadata sent with an image to a worker."""

    request_id: RequestIdentifier
    attempt_id: RequestIdentifier
    remaining_budget_ms: PositiveInt


class Prediction(Contract):
    candidate_label: Material
    score: Score
    score_type: Literal["uncalibrated_softmax"]
    model_version: NonEmpty


class InferenceResponse(Contract):
    request_id: NonEmpty
    attempt_id: NonEmpty
    prediction: Prediction
    preprocessing_version: NonEmpty
    label_mapping_version: NonEmpty
    processing_ms: DurationMs
    review_threshold: Score


class Guidance(Contract):
    status: Literal["available", "needs_clarification", "unavailable"]
    locality_id: NonEmpty
    ruleset_id: NonEmpty | None = None
    ruleset_status: Literal["demonstration_only"] | None = None
    message: NonEmpty


class ClassificationResponse(Contract):
    request_id: NonEmpty
    status: Literal["classified", "needs_review"]
    prediction: Prediction
    guidance: Guidance


class ErrorResponse(Contract):
    request_id: NonEmpty
    code: NonEmpty
    message: NonEmpty


class WorkerStatus(Contract):
    worker_id: NonEmpty
    boot_id: NonEmpty
    ready: bool
    busy: bool
    current_attempt_id: NonEmpty | None = None
    last_finished_attempt_id: NonEmpty | None = None
    model_version: NonEmpty | None = None
    preprocessing_version: NonEmpty | None = None
    label_mapping_version: NonEmpty | None = None
    warmup_ms: DurationMs | None = None
    completed_inferences: Annotated[int, Field(ge=0)] = 0
    failed_inferences: Annotated[int, Field(ge=0)] = 0
    last_processing_ms: DurationMs | None = None
