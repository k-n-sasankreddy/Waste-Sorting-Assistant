"""Load a trained artifact and classify a single local image."""

import argparse
import json
from pathlib import Path

import torch

from waste_assistant.ml.artifacts import image_tensor, load_model
from waste_assistant.shared.config import load_settings
from waste_assistant.shared.schemas import Prediction


class Predictor:
    def __init__(
        self,
        directory: Path,
        *,
        max_pixels: int | None = None,
        formats: tuple[str, ...] | None = None,
    ):
        self.artifact, self.model = load_model(directory)
        self.transform = self.artifact.preprocessing.transform()
        artifact_limit = self.artifact.preprocessing.max_pixels
        self.max_pixels = (
            artifact_limit if max_pixels is None else min(max_pixels, artifact_limit)
        )
        self.formats = formats

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> Prediction:
        tensor = image_tensor(
            image_bytes, self.max_pixels, self.transform, formats=self.formats
        )
        score, index = self.model(tensor.unsqueeze(0)).softmax(dim=1).max(dim=1)
        return Prediction(
            candidate_label=self.artifact.labels[index.item()],
            score=score.item(),
            score_type="uncalibrated_softmax",
            model_version=self.artifact.model_version,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if settings.model is None:
            raise ValueError("Prediction requires model configuration")
        torch.set_num_threads(settings.model.cpu_threads)
        predictor = Predictor(
            settings.model.artifact_dir, max_pixels=settings.limits.max_decoded_pixels
        )
        with args.image.open("rb") as source:
            image = source.read(settings.limits.max_upload_bytes + 1)
        if len(image) > settings.limits.max_upload_bytes:
            raise ValueError("Image exceeds the configured file-size limit")
        prediction = predictor.predict(image)
        status = (
            "classified"
            if prediction.score >= predictor.artifact.threshold
            else "needs_review"
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(
        json.dumps({"status": status, "prediction": prediction.model_dump()}, indent=2)
    )


if __name__ == "__main__":
    main()
