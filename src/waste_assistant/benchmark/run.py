"""Run randomized localhost comparisons on real model replicas."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
import yaml

from waste_assistant.benchmark.generate import SCENARIOS, build_trace
from waste_assistant.benchmark.traffic import replay

POLICIES = ("single", "round_robin", "least_outstanding", "adaptive")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


@contextmanager
def service(module, config, log_path, *args, env=None):
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--config", str(config), *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


async def wait_for(client, url, process, valid):
    for _ in range(150):
        if process.poll() is not None:
            raise RuntimeError(f"Service exited while waiting for {url}")
        try:
            reply = await client.get(url, timeout=1)
            if reply.status_code == 200 and valid(reply.json()):
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError(f"Service never became ready: {url}")


def load_images(root, trace):
    images = {}
    for arrival in trace["arrivals"]:
        row = arrival["image"]
        if row["path"] not in images:
            image = (root / row["path"]).read_bytes()
            if hashlib.sha256(image).hexdigest() != row["sha256"]:
                raise ValueError(f"Image checksum mismatch: {row['path']}")
            images[row["path"]] = image
    return images


class LocalSuite:
    def __init__(self, output, config, stack):
        self.output = output
        self.config = config
        self.workers = config["workers"]
        self.base_url = f"http://127.0.0.1:{config['gateway_port']}"
        self.base_config = output / "workers.yaml"
        self.base_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.delay = output / "worker-delay.txt"
        self.first = stack.enter_context(ExitStack())
        self.other = stack.enter_context(ExitStack())

    def start_first(self, log_path):
        return self.first.enter_context(
            service(
                "waste_assistant.benchmark.worker",
                self.base_config,
                log_path,
                "--worker-id",
                self.workers[0]["id"],
                env={**os.environ, "WASTE_BENCHMARK_DELAY": str(self.delay)},
            )
        )

    async def start_workers(self, client):
        first = self.start_first(self.output / "worker-1.log")
        for index, worker in enumerate(self.workers):
            process = (
                first
                if index == 0
                else self.other.enter_context(
                    service(
                        "waste_assistant.worker.app",
                        self.base_config,
                        self.output / f"{worker['id']}.log",
                        "--worker-id",
                        worker["id"],
                    )
                )
            )
            await wait_for(
                client, worker["url"] + "/ready", process, lambda row: row["ready"]
            )

    async def trial(self, client, directory, policy, trace, images):
        trial = {**self.config, "gateway": dict(self.config["gateway"])}
        trial["policy"] = "round_robin" if policy == "single" else policy
        selected = self.workers[:1] if policy == "single" else self.workers
        if policy == "single":
            trial["gateway"]["worker_id"] = self.workers[0]["id"]
        gateway_config = directory / "gateway.yaml"
        gateway_config.write_text(yaml.safe_dump(trial), encoding="utf-8")
        scenario = trace["scenario"]
        self.delay.write_text(
            "35" if scenario == "heterogeneous" else "0", encoding="ascii"
        )
        for worker in selected:
            for _ in range(50):
                reply = await client.post(
                    worker["url"] + "/v1/infer",
                    data={
                        "request_id": "warm",
                        "attempt_id": "warm",
                        "remaining_budget_ms": "1500",
                    },
                    files={
                        "image": (
                            "sample.jpg",
                            next(iter(images.values())),
                            "image/jpeg",
                        )
                    },
                )
                if reply.status_code != 429:
                    break
                await asyncio.sleep(0.1)
            reply.raise_for_status()
        status_url = self.base_url + "/v1/system/status"
        faults = []
        with service(
            "waste_assistant.gateway.app", gateway_config, directory / "gateway.log"
        ) as gateway:
            await wait_for(
                client,
                status_url,
                gateway,
                lambda row: (
                    len(row["workers"]) == len(selected)
                    and all(worker["available"] for worker in row["workers"])
                ),
            )
            initial = (await client.get(status_url)).json()
            start = perf_counter()

            async def inject():
                if scenario == "slowdown":
                    await asyncio.sleep(0.65)
                    self.delay.write_text("75", encoding="ascii")
                    faults.append(
                        {
                            "at": perf_counter() - start,
                            "event": "worker-1 delay set to 75 ms",
                        }
                    )
                if scenario in ("failure", "recovery"):
                    await asyncio.sleep(0.55)
                    self.first.close()
                    faults.append(
                        {"at": perf_counter() - start, "event": "worker-1 stopped"}
                    )
                    await asyncio.sleep(0.35)
                    new = self.start_first(directory / "worker-1-restarted.log")
                    faults.append(
                        {
                            "at": perf_counter() - start,
                            "event": "worker-1 restart launched",
                        }
                    )
                    await wait_for(
                        client,
                        self.workers[0]["url"] + "/ready",
                        new,
                        lambda row: row["ready"],
                    )
                    faults.append(
                        {"at": perf_counter() - start, "event": "worker-1 ready"}
                    )

            async with httpx.AsyncClient(
                base_url=self.base_url,
                trust_env=False,
                timeout=3,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=64),
            ) as traffic:
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(inject())
                    load = tasks.create_task(
                        replay(traffic, trace, images, self.config["locality_id"], 64)
                    )
                rows = load.result()
            remaining = trace["duration"] - (perf_counter() - start)
            if remaining > 0:
                await asyncio.sleep(remaining)
            final = (await client.get(status_url)).json()
        self.delay.write_text("0", encoding="ascii")
        return rows, {"initial": initial, "final": final, "faults": faults}


async def run_suite(args):
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["model"]["cpu_threads"] = 1
    config["limits"].update(
        request_budget_seconds=1.5,
        attempt_timeout_seconds=0.65,
        health_poll_seconds=0.15,
        status_stale_seconds=0.8,
        min_retry_budget_ms=100,
    )
    config["gateway"].pop("worker_id", None)
    model = Path(config["model"]["artifact_dir"]) / "manifest.json"
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": args.cpu_model,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "threads_per_worker": 1,
        "deployment": "same-machine multi-process",
        "packages": {
            name: version(name) for name in ("torch", "torchvision", "httpx", "uvicorn")
        },
    }
    experiment = {
        "seed": args.seed,
        "repetitions": args.repetitions,
        "scenarios": args.scenarios,
        "hardware": hardware,
        "model": json.loads(model.read_text(encoding="utf-8")),
        "generator_in_flight_limit": 64,
        "generator_max_lateness_ms": 100,
        "runs": [],
        "policies": list(args.policy or POLICIES),
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    metadata = output / "experiment.json"
    if args.resume:
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        saved.setdefault("policies", list(POLICIES))
        if (
            any(
                saved[key] != value
                for key, value in experiment.items()
                if key != "runs"
            )
            or yaml.safe_load((output / "workers.yaml").read_text(encoding="utf-8"))
            != config
        ):
            raise ValueError(
                "Resume requires the original experiment settings and model"
            )
        experiment = saved
    write_json(metadata, experiment)
    manifest = Path(config["dataset"]["output_dir"]) / "test.jsonl"
    root = Path(config["dataset"]["image_root"])
    with ExitStack() as stack:
        suite = LocalSuite(output, config, stack)
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
            await suite.start_workers(client)
            for repetition in range(args.repetitions):
                for scenario_index, scenario in enumerate(args.scenarios):
                    seed = args.seed + 100 * repetition + scenario_index
                    trace = build_trace(manifest, scenario, seed)
                    images = load_images(root, trace)
                    trace_path = output / f"trace-{repetition + 1}-{scenario}.json"
                    write_json(trace_path, trace)
                    order = list(args.policy or POLICIES)
                    random.Random(seed + 10_000).shuffle(order)
                    for policy in order:
                        name = f"{repetition + 1}-{scenario}-{policy}"
                        if name in experiment["runs"]:
                            continue
                        directory = output / name
                        directory.mkdir(exist_ok=args.resume)
                        rows, status = await suite.trial(
                            client, directory, policy, trace, images
                        )
                        with (directory / "requests.jsonl").open(
                            "w", encoding="utf-8", newline="\n"
                        ) as log:
                            for row in rows:
                                log.write(json.dumps(row) + "\n")
                        write_json(
                            directory / "run.json",
                            {
                                "scenario": scenario,
                                "policy": policy,
                                "repetition": repetition + 1,
                                "seed": seed,
                                "trace": trace_path.name,
                                "duration": trace["duration"],
                                "trace_sha256": hashlib.sha256(
                                    trace_path.read_bytes()
                                ).hexdigest(),
                                **status,
                            },
                        )
                        experiment["runs"].append(name)
                        write_json(output / "experiment.json", experiment)
                        print(f"{len(experiment['runs'])}: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("config/dev.yaml"), type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument(
        "--scenarios",
        choices=SCENARIOS,
        nargs="+",
        default=[name for name in SCENARIOS if name != "recovery"],
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-model", default="not recorded")
    args = parser.parse_args()
    if args.repetitions < 1 or len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Use at least one repetition and unique scenarios")
    if args.policy and len(set(args.policy)) != len(args.policy):
        parser.error("Select each policy at most once")
    asyncio.run(run_suite(args))


if __name__ == "__main__":
    main()
