import json
import random
from pathlib import Path
from shutil import copyfile

import pytest
from PIL import Image, PngImagePlugin

from waste_assistant.ml.dataset import CLASSES, prepare_dataset
from waste_assistant.shared.config import DatasetSettings


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_images(root: Path) -> None:
    rng = random.Random(17)
    for label in CLASSES:
        (root / label).mkdir(parents=True)
        for index in range(10):
            image = Image.frombytes("L", (24, 24), rng.randbytes(24 * 24))
            image.save(root / label / f"{index}.png")

    copyfile(root / "glass/0.png", root / "metal/conflict.png")
    copyfile(root / "paper/0.png", root / "paper/copy.png")
    with Image.open(root / "paper/1.png") as image:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("test", "different file bytes")
        image.save(root / "paper/reencoded.png", pnginfo=metadata)
    (root / "trash/bad.jpg").write_bytes(b"invalid image")


def test_dataset_audit_groups_and_reproducible_splits(tmp_path):
    root = tmp_path / "images"
    output = tmp_path / "snapshot"
    known_groups = tmp_path / "same-objects.json"
    known_groups.write_text('[["paper/1.png", "paper/2.png"]]', encoding="utf-8")
    make_images(root)
    config = DatasetSettings(
        image_root=root,
        output_dir=output,
        source_url="https://example.com/dataset",
        seed=42,
        same_object_groups=known_groups,
    )

    report = prepare_dataset(config, max_pixels=10_000)

    assert report["scanned"] == 64
    assert report["accepted"] == 61
    assert report["rejected_by_reason"] == {
        "conflicting_group_labels": 2,
        "decode_error": 1,
    }
    assert report["links_by_kind"] == {
        "exact": 2,
        "near_candidate": 1,
        "known_object": 1,
    }
    audit = {row["path"]: row for row in read_rows(output / "audit.jsonl")}
    assert audit["trash/bad.jpg"]["decoded"] is False
    assert audit["glass/0.png"]["split"] is None
    assert audit["metal/conflict.png"]["split"] is None
    assert audit["paper/0.png"]["group"] == audit["paper/copy.png"]["group"]
    known = [audit[f"paper/{file}.png"] for file in ("1", "2", "reencoded")]
    assert len({row["group"] for row in known}) == 1
    assert len({row["split"] for row in known}) == 1
    paths = []
    for split in ("train", "validation", "test"):
        for row in read_rows(output / f"{split}.jsonl"):
            assert audit[row["path"]]["split"] == split
            assert audit[row["path"]]["sha256"] == row["sha256"]
            paths.append(row["path"])
    assert len(paths) == len(set(paths)) == report["accepted"]

    second = tmp_path / "second"
    assert (
        prepare_dataset(config.model_copy(update={"output_dir": second}), 10_000)
        == report
    )
    assert prepare_dataset(config, 10_000) == report
    assert all(
        file.read_bytes() == (second / file.name).read_bytes()
        for file in output.iterdir()
    )
    with pytest.raises(ValueError, match="choose a new output directory"):
        prepare_dataset(config.model_copy(update={"seed": 43}), 10_000)
