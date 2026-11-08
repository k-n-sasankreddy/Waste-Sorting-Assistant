"""Classification and score-based abstention metrics."""

import torch


def classification_metrics(targets, predictions, labels) -> dict:
    size = len(labels)
    matrix = torch.bincount(targets * size + predictions, minlength=size**2).reshape(
        size, size
    )
    by_class = {}
    for index, label in enumerate(labels):
        hits = matrix[index, index].item()
        support = matrix[index].sum().item()
        predicted = matrix[:, index].sum().item()
        by_class[label] = {
            "precision": hits / predicted if predicted else 0.0,
            "recall": hits / support if support else 0.0,
            "f1": 2 * hits / (support + predicted) if support + predicted else 0.0,
            "support": support,
        }
    return {
        "accuracy": matrix.trace().item() / len(targets),
        "macro_f1": sum(row["f1"] for row in by_class.values()) / size,
        "label_order": list(labels),
        "confusion_matrix": matrix.tolist(),
        "per_class": by_class,
    }


def abstention_metrics(scores, correct, threshold: float) -> dict:
    accepted = scores >= threshold
    count = accepted.sum().item()
    return {
        "threshold": threshold,
        "accepted": count,
        "coverage": count / len(scores),
        "accepted_accuracy": (accepted & correct).sum().item() / count
        if count
        else None,
    }


def select_threshold(scores, correct) -> dict:
    """Maximize validation coverage at 90% accuracy, with a 30% coverage floor."""
    candidates = [
        abstention_metrics(scores, correct, threshold)
        for threshold in [0.0, *scores.unique().tolist()]
    ]
    eligible = [row for row in candidates if row["coverage"] >= 0.30]
    qualifying = [row for row in eligible if row["accepted_accuracy"] >= 0.90]
    if qualifying:
        return max(qualifying, key=lambda row: (row["coverage"], -row["threshold"]))
    return max(
        eligible,
        key=lambda row: (row["accepted_accuracy"], row["coverage"], -row["threshold"]),
    )
