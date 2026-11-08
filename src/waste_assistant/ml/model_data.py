"""Read frozen split manifests and verify image bytes before preprocessing."""

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset

from waste_assistant.ml.artifacts import Checksum, Preprocessing, image_tensor
from waste_assistant.shared.schemas import Contract, Material


class SplitRow(Contract):
    path: str
    label: Material
    sha256: Checksum


@dataclass(frozen=True)
class Split:
    rows: list[SplitRow]
    sha256: str

    @classmethod
    def load(cls, manifest: Path) -> "Split":
        data = manifest.read_bytes()
        rows = [SplitRow.model_validate_json(line) for line in data.splitlines()]
        if not rows or len({row.path for row in rows}) != len(rows):
            raise ValueError(f"Split must contain unique image paths: {manifest}")
        return cls(rows, hashlib.sha256(data).hexdigest())


class SplitImages(Dataset):
    def __init__(
        self,
        root: Path,
        split: Split,
        labels,
        preprocessing: Preprocessing,
        *,
        flip=False,
    ):
        self.root = root.resolve()
        self.split = split
        self.labels = {label: index for index, label in enumerate(labels)}
        self.preprocessing = preprocessing
        self.transform = preprocessing.transform()
        self.flip = flip

    def __len__(self) -> int:
        return len(self.split.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.split.rows[index]
        relative = PurePosixPath(row.path)
        path = (self.root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(self.root)
        ):
            raise ValueError(f"Invalid image path in manifest: {row.path}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row.sha256:
            raise ValueError(f"Image checksum does not match manifest: {row.path}")
        tensor = image_tensor(
            data, self.preprocessing.max_pixels, self.transform, flip=self.flip
        )
        return tensor, self.labels[row.label]
