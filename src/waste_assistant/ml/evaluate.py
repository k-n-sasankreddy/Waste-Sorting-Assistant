"""Evaluate an exported classifier on the frozen TrashNet test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from waste_assistant.ml.artifacts import load_model
from waste_assistant.ml.metrics import abstention_metrics, classification_metrics
from waste_assistant.ml.model_data import Split, SplitImages
from waste_assistant.shared.config import Settings, load_settings


@torch.inference_mode()
def evaluate(settings: Settings) -> dict:
    dataset, config = settings.dataset, settings.model
    if dataset is None or config is None:
        raise ValueError("Evaluation requires dataset and model configuration")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    artifact, model = load_model(config.artifact_dir)
    training_checksums = set()
    for name in ("train", "validation"):
        split = Split.load(dataset.output_dir / f"{name}.jsonl")
        if split.sha256 != artifact.training.get(f"{name}_manifest_sha256"):
            raise ValueError(
                f"The {name} manifest differs from the one used during training"
            )
        training_checksums.update(row.sha256 for row in split.rows)
    test = Split.load(dataset.output_dir / "test.jsonl")
    if any(row.sha256 in training_checksums for row in test.rows):
        raise ValueError("Test split overlaps training or validation by image checksum")
    images = SplitImages(
        dataset.image_root, test, artifact.labels, artifact.preprocessing
    )
    batches = DataLoader(images, batch_size=config.batch_size)
    batch_scores, batch_predictions, batch_targets = [], [], []
    for inputs, targets in batches:
        scores, predictions = model(inputs).softmax(dim=1).max(dim=1)
        batch_scores.append(scores)
        batch_predictions.append(predictions)
        batch_targets.append(targets)
    scores = torch.cat(batch_scores)
    predictions = torch.cat(batch_predictions)
    targets = torch.cat(batch_targets)
    correct = predictions == targets
    mistake_indices = (~correct).nonzero(as_tuple=True)[0].tolist()
    examples = sorted(
        mistake_indices, key=lambda i: (-scores[i].item(), test.rows[i].path)
    )[:12]
    return {
        "split": "trashnet_test",
        "test_manifest_sha256": test.sha256,
        "sample_count": len(test.rows),
        "model_version": artifact.model_version,
        "weights_sha256": artifact.weights_sha256,
        "classification": classification_metrics(targets, predictions, artifact.labels),
        "abstention": abstention_metrics(scores, correct, artifact.threshold),
        "high_score_mistakes": [
            {
                "path": test.rows[index].path,
                "expected": test.rows[index].label,
                "predicted": artifact.labels[predictions[index].item()],
                "score": scores[index].item(),
            }
            for index in examples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = (
            json.dumps(evaluate(load_settings(args.config)), indent=2, sort_keys=True)
            + "\n"
        )
        if args.output is None:
            print(report, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8", newline="\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
