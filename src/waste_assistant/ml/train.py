"""Train a linear TrashNet head on frozen ImageNet MobileNetV3-Small features."""

import argparse
import platform
from copy import deepcopy

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from waste_assistant.ml.artifacts import (
    PRETRAINED,
    create_model,
    export_model,
    preprocessing_from_weights,
)
from waste_assistant.ml.dataset import CLASSES
from waste_assistant.ml.metrics import classification_metrics, select_threshold
from waste_assistant.ml.model_data import Split, SplitImages
from waste_assistant.shared.config import Settings, load_settings


@torch.no_grad()
def extract_features(model, loader):
    features, labels = [], []
    for images, targets in loader:
        values = model.avgpool(model.features(images)).flatten(1)
        features.append(model.classifier[:-1](values))
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def train(settings: Settings) -> None:
    dataset, config = settings.dataset, settings.model
    if dataset is None or config is None:
        raise ValueError("Training requires dataset and model configuration")
    if config.artifact_dir.exists():
        raise ValueError(f"Artifact directory already exists: {config.artifact_dir}")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)

    train_split = Split.load(dataset.output_dir / "train.jsonl")
    validation_split = Split.load(dataset.output_dir / "validation.jsonl")
    if {row.sha256 for row in train_split.rows} & {
        row.sha256 for row in validation_split.rows
    }:
        raise ValueError("Training and validation contain the same image checksum")
    preprocessing = preprocessing_from_weights(settings.limits.max_decoded_pixels)
    model = create_model(len(CLASSES), pretrained=True).eval()
    model.requires_grad_(False)
    head = model.classifier[-1].requires_grad_(True)

    def features(split, flip=False):
        images = SplitImages(
            dataset.image_root, split, CLASSES, preprocessing, flip=flip
        )
        return extract_features(model, DataLoader(images, batch_size=config.batch_size))

    print(
        "Caching training features from original and horizontally flipped images",
        flush=True,
    )
    original, targets = features(train_split)
    flipped, flipped_targets = features(train_split, flip=True)
    print("Caching validation features", flush=True)
    validation, validation_targets = features(validation_split)
    loader = DataLoader(
        TensorDataset(
            torch.cat((original, flipped)), torch.cat((targets, flipped_targets))
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate)
    best_f1 = -1.0
    history = []
    for epoch in range(1, config.head_epochs + 1):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(head(inputs), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = head(validation).argmax(dim=1)
            metrics = classification_metrics(validation_targets, predictions, CLASSES)
        f1 = metrics["macro_f1"]
        history.append(
            {"epoch": epoch, "accuracy": metrics["accuracy"], "macro_f1": f1}
        )
        print(f"Epoch {epoch:02}: validation macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch, best_state = f1, epoch, deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    with torch.no_grad():
        scores, predictions = head(validation).softmax(dim=1).max(dim=1)
    abstention = select_threshold(scores, predictions == validation_targets)
    artifact = export_model(
        config.artifact_dir,
        model,
        labels=list(CLASSES),
        preprocessing=preprocessing,
        threshold=abstention["threshold"],
        training={
            "pretrained": PRETRAINED.name,
            "pretrained_url": PRETRAINED.url,
            "method": "frozen_features_linear_head_with_horizontal_flip",
            "settings": config.model_dump(mode="json", exclude={"artifact_dir"}),
            "train_manifest_sha256": train_split.sha256,
            "validation_manifest_sha256": validation_split.sha256,
            "selected_epoch": best_epoch,
            "validation_history": history,
            "validation_abstention": abstention,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
            },
        },
    )
    print(f"Exported {artifact.model_version} to {config.artifact_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    try:
        train(load_settings(args.config))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
