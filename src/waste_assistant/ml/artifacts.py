"""Versioned classifier weights, labels and deterministic preprocessing."""

import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self

import torch
from PIL import Image, ImageOps
from pydantic import PositiveFloat, PositiveInt, StringConstraints, model_validator
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from waste_assistant.ml.dataset import CLASSES
from waste_assistant.shared.schemas import Contract, Score

PRETRAINED = MobileNet_V3_Small_Weights.IMAGENET1K_V1
Checksum = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Preprocessing(Contract):
    version: Literal["rgb-exif-resize-crop-v1"] = "rgb-exif-resize-crop-v1"
    resize_short_edge: PositiveInt
    crop_size: PositiveInt
    mean: tuple[float, float, float]
    std: tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    interpolation: Literal["bilinear"] = "bilinear"
    max_pixels: PositiveInt

    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    self.resize_short_edge,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )


class Artifact(Contract):
    format_version: Literal[1] = 1
    architecture: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    labels: list[str]
    preprocessing: Preprocessing
    threshold: Score
    weights_sha256: Checksum
    training: dict

    @property
    def model_version(self) -> str:
        identity = {
            "architecture": self.architecture,
            "labels": self.labels,
            "preprocessing": self.preprocessing.model_dump(),
            "threshold": self.threshold,
            "weights_sha256": self.weights_sha256,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        return f"{self.architecture}-{digest[:16]}"

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        if sorted(self.labels) != list(CLASSES):
            raise ValueError("Artifact must map each supported material exactly once")
        return self


def preprocessing_from_weights(max_pixels: int) -> Preprocessing:
    preset = PRETRAINED.transforms()
    if preset.interpolation != transforms.InterpolationMode.BILINEAR:
        raise ValueError("Unexpected pretrained interpolation")
    return Preprocessing(
        resize_short_edge=preset.resize_size[0],
        crop_size=preset.crop_size[0],
        mean=preset.mean,
        std=preset.std,
        max_pixels=max_pixels,
    )


def image_tensor(data: bytes, max_pixels: int, transform, *, flip: bool = False):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError("Image exceeds the decoded pixel limit")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return transform(ImageOps.mirror(rgb) if flip else rgb)
    except Image.DecompressionBombError as error:
        raise ValueError("Image exceeds the decoded pixel limit") from error


def create_model(class_count: int, *, pretrained: bool = False) -> nn.Module:
    model = mobilenet_v3_small(weights=PRETRAINED if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def export_model(output: Path, model: nn.Module, **metadata) -> Artifact:
    if output.exists():
        raise ValueError(
            f"Artifact directory already exists: {output}. Choose a new path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        weights = staged / "weights.pt"
        torch.save(model.state_dict(), weights)
        artifact = Artifact(weights_sha256=sha256_file(weights), **metadata)
        (staged / "manifest.json").write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        staged.rename(output)
    return artifact


def load_model(directory: Path) -> tuple[Artifact, nn.Module]:
    artifact = Artifact.model_validate_json((directory / "manifest.json").read_bytes())
    weights = (directory / "weights.pt").read_bytes()
    if hashlib.sha256(weights).hexdigest() != artifact.weights_sha256:
        raise ValueError("Model weight checksum does not match manifest")
    model = create_model(len(artifact.labels))
    model.load_state_dict(
        torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True)
    )
    return artifact, model.eval()
