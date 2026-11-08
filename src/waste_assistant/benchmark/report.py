"""Report per-run outcomes, latency and selected comparisons from raw logs."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from waste_assistant.benchmark.run import POLICIES, write_json


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize(rows, terminals, attempts, decisions, duration, budget_ms=1500):
    outcomes = Counter(row["outcome"] for row in rows)
    successes = [row for row in rows if row["outcome"] == "success"]
    latencies = [row["latency_ms"] for row in successes]
    on_time = [row for row in successes if row["latency_ms"] <= budget_ms]
    attempt_counts = Counter(row["request_id"] for row in attempts)
    terminal_by_id = {row["trace_id"]: row for row in terminals}
    admitted = sum(
        bool(terminal_by_id.get(row["trace_id"], {}).get("admitted")) for row in rows
    )
    sent = sum("sent_at" in row for row in rows)
    return {
        "offered": len(rows),
        "offered_per_s": len(rows) / duration,
        "sent": sent,
        "sent_per_s": sent / duration,
        "admitted": admitted,
        "admitted_fraction": admitted / len(rows),
        "sent_without_gateway_terminal": sent
        - len(terminal_by_id.keys() & {row["trace_id"] for row in rows}),
        "success": outcomes["success"],
        "rejected": outcomes["rejected"],
        "deadline": outcomes["deadline"],
        "failure": outcomes["failure"],
        "client_error": outcomes["client_error"],
        "generator_drop": outcomes["generator_drop"],
        "generator_drop_fraction": outcomes["generator_drop"] / len(rows),
        "rejected_fraction": outcomes["rejected"] / len(rows),
        "deadline_fraction": outcomes["deadline"] / len(rows),
        "failure_fraction": (outcomes["failure"] + outcomes["client_error"])
        / len(rows),
        "success_fraction": len(successes) / len(rows),
        "late_success": len(successes) - len(on_time),
        "on_time_fraction": len(on_time) / len(rows),
        "successful_throughput_per_s": sum(
            row["finished_at"] <= duration for row in on_time
        )
        / duration,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "generator_lateness_p95_ms": percentile(
            [
                (row["sent_at"] - row["scheduled_at"]) * 1000
                if "sent_at" in row and "scheduled_at" in row
                else row["lateness_ms"]
                for row in rows
            ],
            0.95,
        ),
        "queue_p95_ms": percentile([row["queue_ms"] for row in attempts], 0.95),
        "dispatch_p95_ms": percentile([row["dispatch_ms"] for row in attempts], 0.95),
        "decision_p95_ms": percentile([row["decision_ms"] for row in decisions], 0.95),
        "retry_attempts": sum(row["attempt_number"] > 1 for row in attempts),
        "retried_requests": sum(count > 1 for count in attempt_counts.values()),
        "retry_fraction": sum(count > 1 for count in attempt_counts.values())
        / len(rows),
    }


def read_trial(directory):
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    requests = [
        json.loads(line)
        for line in (directory / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events = defaultdict(list)
    markers = (
        "benchmark_terminal",
        "scheduler_attempt",
        "scheduler_decision",
        "scheduler_sample",
    )
    for line in (directory / "gateway.log").read_text(encoding="utf-8").splitlines():
        for marker in markers:
            if f" {marker} " in line:
                events[marker].append(json.loads(line.split(marker + " ", 1)[1]))
                break
    with (directory / "attempts.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as raw:
        for event in events["scheduler_attempt"]:
            raw.write(json.dumps(event) + "\n")
    return meta, requests, events


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def plot(output, metrics, timelines):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "waste-assistant"
    from matplotlib import pyplot as plt

    fig, (latency, throughput) = plt.subplots(
        2, 1, figsize=(9, 7), layout="constrained"
    )
    for policy in POLICIES:
        runs = [run for run in metrics if run["policy"] == policy]
        if not runs:
            continue
        latency.scatter(
            [run["offered_per_s"] for run in runs if run["p95_ms"] is not None],
            [run["p95_ms"] for run in runs if run["p95_ms"] is not None],
            label=policy,
            alpha=0.72,
        )
        throughput.scatter(
            [run["offered_per_s"] for run in runs],
            [run["successful_throughput_per_s"] for run in runs],
            label=policy,
            alpha=0.72,
        )
    for axis in (latency, throughput):
        axis.set_xlabel("Offered requests per second (whole run)")
        axis.grid(alpha=0.2)
    latency.set_ylabel("Successful-request p95 (ms)")
    latency.set_yscale("log")
    latency.legend(ncol=2)
    throughput.set_ylabel("On-time completions in window / second")
    save_svg(fig, output / "load-comparison.svg")
    plt.close(fig)

    selected = [
        (meta, rows, events)
        for meta, rows, events in timelines
        if meta["policy"] == "adaptive"
        and meta["repetition"] == 1
        and meta["scenario"] in ("burst", "failure", "recovery")
    ]
    if not selected:
        return
    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(10, 3.4 * len(selected)),
        squeeze=False,
        layout="constrained",
    )
    for (outcomes, assignments), (meta, rows, events) in zip(
        axes, selected, strict=True
    ):
        by_id = {row["request_id"]: row for row in events["benchmark_terminal"]}
        scheduled = {row["trace_id"]: row["scheduled_at"] for row in rows}
        binned = Counter(
            (int(row["scheduled_at"] / 0.2), row["outcome"]) for row in rows
        )
        bins = range(math.ceil(meta["duration"] / 0.2))
        for label in sorted({row["outcome"] for row in rows}):
            outcomes.step(
                [i * 0.2 for i in bins],
                [binned[i, label] for i in bins],
                where="post",
                label=label,
            )
        for worker in meta["initial"]["workers"]:
            xs = [
                scheduled[by_id[event["request_id"]]["trace_id"]]
                for event in events["scheduler_decision"]
                if event["worker_id"] == worker["id"] and event["request_id"] in by_id
            ]
            assignments.hist(
                xs,
                bins=[i * 0.2 for i in range(len(bins) + 1)],
                histtype="step",
                label=worker["id"],
            )
        for axis in (outcomes, assignments):
            for fault in meta["faults"]:
                axis.axvline(fault["at"], color="black", linestyle=":", alpha=0.5)
            axis.set_xlabel("Scheduled arrival time (s)")
            axis.legend(fontsize=8)
        outcomes.set_title(f"{meta['scenario'].title()} outcomes per 200 ms")
        assignments.set_title("Assignments by originating arrival")
    save_svg(fig, output / "burst-and-recovery.svg")
    plt.close(fig)


def report(root, output):
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    metrics, evidence, timelines = [], [], []
    for name in experiment["runs"]:
        directory = root / name
        meta, rows, events = read_trial(directory)
        timelines.append((meta, rows, events))
        metrics.append(
            {
                "run": name,
                **{
                    key: meta[key]
                    for key in ("scenario", "policy", "repetition", "seed", "duration")
                },
                **summarize(
                    rows,
                    events["benchmark_terminal"],
                    events["scheduler_attempt"],
                    events["scheduler_decision"],
                    meta["duration"],
                ),
            }
        )
        paths = [
            directory / filename
            for filename in (
                "gateway.yaml",
                "run.json",
                "requests.jsonl",
                "gateway.log",
                "attempts.jsonl",
            )
        ] + [root / meta["trace"]]
        restart_log = directory / "worker-1-restarted.log"
        if restart_log.exists():
            paths.append(restart_log)
        evidence.append(
            {
                "run": name,
                "faults": meta["faults"],
                "files": {
                    path.relative_to(root).as_posix(): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in paths
                },
            }
        )
    if not metrics:
        raise ValueError("No completed benchmark runs found")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    groups = defaultdict(list)
    for row in metrics:
        groups[row["scenario"], row["policy"]].append(row)
    lines = [
        "# Measured scheduling results",
        "",
        "| Workload | Policy | Successful / offered | Median run p95 (ms) | Rejected | Deadline | Other failures | Generator drops |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (scenario, policy), runs in sorted(groups.items()):
        p95 = [row["p95_ms"] for row in runs if row["p95_ms"] is not None]
        latency = f"{median(p95):.1f}" if p95 else "—"

        def total(key):
            return sum(row[key] for row in runs)

        lines.append(
            f"| {scenario} | {policy} | {total('success')} / {total('offered')} | {latency} | "
            f"{total('rejected')} | {total('deadline')} | {total('failure') + total('client_error')} | "
            f"{total('generator_drop')} |"
        )
    lines.extend(
        [
            "",
            "Latency uses successful requests. The table shows medians of per-run p95 values.",
            "See [per-run metrics](metrics.csv) and the [evidence index](evidence.json).",
            "",
        ]
    )
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    experiment["model"] = {
        key: experiment["model"][key]
        for key in ("architecture", "weights_sha256", "labels", "preprocessing")
    }
    write_json(
        output / "evidence.json",
        {**experiment, "raw_run_directory": root.name, "artifacts": evidence},
    )
    plot(output, metrics, timelines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report(args.run_dir, args.output or args.run_dir / "report")


if __name__ == "__main__":
    main()
