"""Generate seeded open-loop arrivals from the frozen TrashNet test manifest."""

import argparse
import json
import random
from pathlib import Path

from waste_assistant.shared.config import load_settings

# Each segment specifies seconds and offered requests per second.
SCENARIOS = {
    "steady": [(1.2, 12)],
    "ramp": [(0.8, 20), (0.8, 80), (0.8, 160)],
    "burst": [(0.4, 10), (0.15, 240), (0.45, 10), (0.15, 240), (0.55, 10)],
    "heterogeneous": [(1.6, 65)],
    "slowdown": [(2.0, 65)],
    "failure": [(4.0, 45)],
    "recovery": [(8.0, 45)],
}


def build_trace(manifest: Path, scenario: str, seed: int) -> dict:
    rng = random.Random(seed)
    images = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    candidates = rng.sample(images, min(16, len(images)))
    arrivals = []
    start = 0.0
    for duration, rate in SCENARIOS[scenario]:
        for index in range(round(duration * rate)):
            arrivals.append(
                {"at": round(start + index / rate, 6), "image": rng.choice(candidates)}
            )
        start += duration
    return {"scenario": scenario, "seed": seed, "duration": start, "arrivals": arrivals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dev.yaml")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_settings(args.config).dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build_trace(dataset.output_dir / "test.jsonl", args.scenario, args.seed),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
