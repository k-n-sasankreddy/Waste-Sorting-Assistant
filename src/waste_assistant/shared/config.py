"""Load and validate the local development configuration."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from waste_assistant.shared.schemas import Contract

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
Fraction = Annotated[float, Field(gt=0, le=1)]


class Limits(Contract):
    max_upload_bytes: PositiveInt
    max_decoded_pixels: PositiveInt
    max_outstanding: PositiveInt
    max_waiting_per_worker: PositiveInt
    max_queued_image_bytes: PositiveInt
    request_budget_seconds: PositiveFloat
    health_poll_seconds: PositiveFloat
    status_stale_seconds: PositiveFloat
    max_retries: Literal[0, 1]

    @model_validator(mode="after")
    def check_relationships(self) -> Self:
        if self.max_queued_image_bytes < self.max_upload_bytes:
            raise ValueError(
                "The queued-image budget must hold at least one maximum upload"
            )
        if self.status_stale_seconds <= self.health_poll_seconds:
            raise ValueError("The stale threshold must exceed the polling interval")
        return self


class AdaptiveSettings(Contract):
    alpha: Fraction
    residual_floor_fraction: Fraction


class WorkerSettings(Contract):
    id: Identifier
    url: HttpUrl
    model_path: Path


class DatasetSettings(Contract):
    image_root: Path
    output_dir: Path
    source_url: HttpUrl
    seed: Annotated[int, Field(ge=0)]
    near_duplicate_distance: Annotated[int, Field(ge=0, le=128)] = 4
    same_object_groups: Path | None = None


class ModelSettings(Contract):
    artifact_dir: Path
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    batch_size: PositiveInt
    head_epochs: PositiveInt
    learning_rate: PositiveFloat
    cpu_threads: PositiveInt


class GatewaySettings(Contract):
    worker_id: Identifier
    rules_path: Path


class Settings(Contract):
    gateway_port: Annotated[int, Field(ge=1, le=65535)]
    locality_id: Identifier
    policy: Literal["round_robin", "least_outstanding", "adaptive"]
    limits: Limits
    adaptive: AdaptiveSettings
    workers: Annotated[list[WorkerSettings], Field(min_length=1)]
    dataset: DatasetSettings | None = None
    model: ModelSettings | None = None
    gateway: GatewaySettings | None = None

    @model_validator(mode="after")
    def check_worker_uniqueness(self) -> Self:
        if len({worker.id for worker in self.workers}) != len(self.workers):
            raise ValueError("Worker IDs must be unique")
        if len({worker.url for worker in self.workers}) != len(self.workers):
            raise ValueError("Worker URLs must be unique")
        if self.gateway and self.gateway.worker_id not in {
            worker.id for worker in self.workers
        }:
            raise ValueError("Gateway worker ID must match a configured worker")
        return self


def load_settings(path: str | Path) -> Settings:
    with Path(path).open(encoding="utf-8") as source:
        return Settings.model_validate(yaml.safe_load(source))
