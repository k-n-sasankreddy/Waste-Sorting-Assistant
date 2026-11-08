"""Audit class-folder images and publish deterministic, grouped splits."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from PIL import Image, ImageOps
from pydantic import TypeAdapter

from waste_assistant.shared.config import DatasetSettings, load_settings
from waste_assistant.shared.schemas import Material

CLASSES = tuple(sorted(get_args(Material)))
TARGETS = {"train": 70, "validation": 15, "test": 15}


@dataclass
class ImageRecord:
    path: str
    label: str
    sha256: str
    width: int | None = None
    height: int | None = None
    decoded: bool = False
    rejection: str | None = None
    group: str | None = None
    split: str | None = None


def difference_hash(image: Image.Image) -> int:
    """Encode horizontal and vertical luminance gradients as 128 bits."""
    gray = ImageOps.exif_transpose(image).convert("L")
    horizontal = list(
        gray.resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data()
    )
    vertical = list(gray.resize((8, 9), Image.Resampling.LANCZOS).get_flattened_data())
    signature = 0
    for row in range(8):
        for col in range(8):
            h, v = row * 9 + col, row * 8 + col
            signature = (signature << 1) | (horizontal[h] > horizontal[h + 1])
            signature = (signature << 1) | (vertical[v] > vertical[v + 8])
    return signature


def scan_images(
    root: Path, max_pixels: int
) -> tuple[list[ImageRecord], dict[int, int]]:
    records = []
    signatures = {}
    for path in sorted(root.rglob("*"), key=Path.as_posix):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Image path escapes the dataset root: {relative}")
        label = relative.parts[0] if len(relative.parts) > 1 else ""
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
            source.seek(0)
            record = ImageRecord(relative.as_posix(), label, checksum)
            records.append(record)
            try:
                with Image.open(source) as image:
                    record.width, record.height = image.size
                    if image.width * image.height > max_pixels:
                        record.rejection = "pixel_limit"
                        continue
                    image.load()
                    record.decoded = True
                    if label not in CLASSES:
                        record.rejection = "unknown_label"
                    else:
                        signatures[len(records) - 1] = difference_hash(image)
            except Image.DecompressionBombError:
                record.rejection = "pixel_limit"
            except (OSError, SyntaxError, ValueError):
                record.rejection = "decode_error"
    return records, signatures


def group_images(
    records: list[ImageRecord],
    signatures: dict[int, int],
    distance: int,
    known_groups: list[list[str]],
) -> tuple[list[list[ImageRecord]], list[dict]]:
    parent = {index: index for index in signatures}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        parent[find(right)] = find(left)

    first_by_hash = {}
    first_by_path = {}
    links = []
    unique = []
    for index, signature in signatures.items():
        record = records[index]
        first_by_path[record.path] = index
        first = first_by_hash.setdefault(record.sha256, index)
        if first != index:
            join(first, index)
            links.append(
                {"left": records[first].path, "right": record.path, "kind": "exact"}
            )
            continue
        for other, previous in unique:
            gap = (signature ^ previous).bit_count()
            if gap <= distance:
                join(other, index)
                links.append(
                    {
                        "left": records[other].path,
                        "right": record.path,
                        "kind": "near_candidate",
                        "distance": gap,
                    }
                )
        unique.append((index, signature))

    for paths in known_groups:
        if len(set(paths)) < 2 or any(path not in first_by_path for path in paths):
            raise ValueError(
                f"Same-object group must reference at least two valid images: {paths}"
            )
        first, *rest = sorted(set(paths))
        for path in rest:
            join(first_by_path[first], first_by_path[path])
            links.append({"left": first, "right": path, "kind": "known_object"})

    groups = defaultdict(list)
    for index in signatures:
        groups[find(index)].append(records[index])
    for group in groups.values():
        conflict = len({record.label for record in group}) > 1
        for record in group:
            record.group = group[0].path
            if conflict:
                record.rejection = "conflicting_group_labels"
    return list(groups.values()), links


def assign_splits(groups: list[list[ImageRecord]], seed: int) -> None:
    by_label = defaultdict(list)
    for group in groups:
        if group[0].rejection is None:
            by_label[group[0].label].append(group)

    for label in CLASSES:
        class_groups = by_label[label]
        total = sum(map(len, class_groups))
        if total == 0:
            raise ValueError(f"No usable images for class {label!r}")
        assigned = Counter()
        ordered = sorted(
            class_groups,
            key=lambda group: (
                -len(group),
                hashlib.sha256(f"{seed}:{group[0].path}".encode("utf-8")).digest(),
            ),
        )
        for group in ordered:
            split = max(
                TARGETS, key=lambda name: total * TARGETS[name] - 100 * assigned[name]
            )
            for record in group:
                record.split = split
            assigned[split] += len(group)


def json_lines(rows) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_immutable(output: Path, files: dict[str, str]) -> None:
    def matches() -> bool:
        return (
            output.is_dir()
            and set(files) == {path.name for path in output.iterdir()}
            and all(
                (output / name).is_file()
                and (output / name).read_bytes() == content.encode("utf-8")
                for name, content in files.items()
            )
        )

    if output.exists():
        if matches():
            return
        raise ValueError(
            f"Manifest directory differs; choose a new output directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / "manifests"
        staged.mkdir()
        for name, content in files.items():
            (staged / name).write_bytes(content.encode("utf-8"))
        try:
            staged.rename(output)
        except OSError:
            if not matches():
                raise


def prepare_dataset(config: DatasetSettings, max_pixels: int) -> dict:
    root = config.image_root.resolve()
    output = config.output_dir.resolve()
    if not root.is_dir():
        raise ValueError(
            f"Dataset directory is missing: {root}. See the README import steps."
        )
    if root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Image root and manifest output must be separate directories")
    known_groups = []
    if config.same_object_groups is not None:
        known_groups = TypeAdapter(list[list[str]]).validate_json(
            config.same_object_groups.read_bytes()
        )
    records, signatures = scan_images(root, max_pixels)
    groups, links = group_images(
        records, signatures, config.near_duplicate_distance, known_groups
    )
    assign_splits(groups, config.seed)

    distribution = {name: Counter() for name in TARGETS}
    for record in records:
        if record.split is not None:
            distribution[record.split][record.label] += 1
    report = {
        "manifest_version": 1,
        "source_url": str(config.source_url),
        "seed": config.seed,
        "max_pixels": max_pixels,
        "near_duplicate_distance": config.near_duplicate_distance,
        "target_percentages": TARGETS,
        "scanned": len(records),
        "decoded": sum(record.decoded for record in records),
        "accepted": sum(sum(counts.values()) for counts in distribution.values()),
        "rejected_by_reason": dict(
            Counter(r.rejection for r in records if r.rejection)
        ),
        "duplicate_groups": sum(len(group) > 1 for group in groups),
        "links_by_kind": dict(Counter(link["kind"] for link in links)),
        "split_counts": {
            split: {label: counts[label] for label in CLASSES}
            for split, counts in distribution.items()
        },
    }
    files = {
        "audit.jsonl": json_lines(asdict(record) for record in records),
        "duplicate-links.jsonl": json_lines(links),
        "report.json": json.dumps(report, indent=2, sort_keys=True) + "\n",
    }
    for split in TARGETS:
        files[f"{split}.jsonl"] = json_lines(
            {"path": record.path, "label": record.label, "sha256": record.sha256}
            for record in records
            if record.split == split
        )
    write_immutable(output, files)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if settings.dataset is None:
            raise ValueError("The configuration is missing its dataset section")
        report = prepare_dataset(settings.dataset, settings.limits.max_decoded_pixels)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
