"""Rehearse the upload, traffic, recovery, and reporting steps locally."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import httpx
import yaml
from streamlit.testing.v1 import AppTest

from waste_assistant.benchmark.run import LocalSuite, service, wait_for

UI_SCRIPT = Path(__file__).resolve().parents[1] / "src/waste_assistant/ui/app.py"


async def check_uploads(config_path: Path, output: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["dataset"]["image_root"])
    output.mkdir()
    with ExitStack() as stack:
        suite = LocalSuite(output, config, stack)
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            await suite.start_workers(client)
            gateway = stack.enter_context(
                service(
                    "waste_assistant.gateway.app", config_path, output / "gateway.log"
                )
            )
            await wait_for(
                client,
                suite.base_url + "/ready",
                gateway,
                lambda status: status["ready"],
            )
            previous = os.environ.get("WASTE_CONFIG")
            os.environ["WASTE_CONFIG"] = str(config_path)
            try:
                for relative in (
                    "paper/paper1.jpg",
                    "cardboard/cardboard125.jpg",
                    "trash/trash10.jpg",
                ):
                    image = root / relative
                    app = AppTest.from_file(str(UI_SCRIPT), default_timeout=20).run()
                    app.file_uploader[0].set_value(
                        (image.name, image.read_bytes(), "image/jpeg")
                    )
                    app.button[0].click().run()
                    if app.exception or app.error or not app.metric:
                        raise RuntimeError(
                            f"Upload rehearsal failed for {image}: {list(app.error)}"
                        )
                    review = "review prompted" if app.warning else "accepted"
                    guidance = app.markdown[-1].value
                    print(
                        f"{image.name} | {app.metric[0].value} | {review} | {guidance}",
                        flush=True,
                    )
            finally:
                if previous is None:
                    os.environ.pop("WASTE_CONFIG", None)
                else:
                    os.environ["WASTE_CONFIG"] = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/dev.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-model", default="not recorded")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True)
    asyncio.run(check_uploads(args.config, args.run_dir / "uploads"))
    traffic = args.run_dir / "traffic"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "waste_assistant.benchmark.run",
            "--config",
            str(args.config),
            "--run-dir",
            str(traffic),
            "--scenarios",
            "burst",
            "slowdown",
            "recovery",
            "--policy",
            "adaptive",
            "--repetitions",
            "1",
            "--cpu-model",
            args.cpu_model,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "waste_assistant.benchmark.report",
            "--run-dir",
            str(traffic),
        ],
        check=True,
    )
    for trial in sorted(traffic.glob("*/run.json")):
        result = json.loads(trial.read_text(encoding="utf-8"))
        for fault in result["faults"]:
            print(
                f"{result['scenario']} at {fault['at']:.2f}s | {fault['event']}",
                flush=True,
            )
    print(f"Rehearsal report: {traffic / 'report' / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
