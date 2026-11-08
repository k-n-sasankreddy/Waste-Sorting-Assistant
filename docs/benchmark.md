# Scheduling benchmark

Phase 8 compares a single-worker gateway with round-robin, least-outstanding,
and adaptive scheduling across three replicas. The comparisons use the
Phase 2 model and the same machine for workers, gateway, and load generator.
Each worker uses one PyTorch CPU thread. These are synthetic workloads in
local multi-process emulation.

The comparison uses on-time completion and multiple arrival patterns,
following the evaluation objectives in He et al. The policies here route
independent requests among full model replicas. Their study evaluated
batching and client-edge collaboration.
[Adaptive Scheduling for Edge-Assisted DNN Serving](https://zhaoyuan-he.github.io/assets/pdf/Adaptive-Scheduling-for-Edge-Assisted-DNN-Serving.pdf)

## Reproduce

Prepare the dataset, model and environment using the [local setup
guide](demo.md). Stop any existing gateway or workers that use the benchmark
ports. From the repository root in PowerShell, run the following commands.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.lock
.\.venv\Scripts\python.exe -m waste_assistant.benchmark.run --run-dir runs/my-comparison --seed 42 --repetitions 3 --cpu-model 'Your CPU model'
.\.venv\Scripts\python.exe -m waste_assistant.benchmark.report --run-dir runs/my-comparison
```

Use a fresh output directory for each suite. On Linux, substitute
`.venv/bin/python`. Select a smaller run with
`--scenarios steady burst --policy adaptive --repetitions 1`.
If a suite is interrupted, repeat its command with `--resume` and the
original settings to retain completed trials.
The runner creates and saves all traces before replaying each matched group.
Use the standalone generator to inspect a trace.

```powershell
.\.venv\Scripts\python.exe -m waste_assistant.benchmark.generate --scenario burst --seed 44 --output runs/example-trace.json
```

## Workloads

| Workload | Offered arrivals | Controlled change |
|---|---|---|
| Steady | 12/s for 1.2 s | None |
| Ramp | 20, 80, and 160/s for 0.8 s each | Rate steps upward |
| Burst | Two 150 ms bursts at 240/s, with 10/s between them | Saved burst trace |
| Heterogeneous | 65/s for 1.6 s | Worker 1 adds 35 ms of processing delay |
| Slowdown | 65/s for 2 s | Worker 1 adds 75 ms after about 0.65 s |
| Failure | 45/s for 4 s | Stop worker 1 near 0.55 s and launch its replacement after 0.35 s |

Select `--scenarios recovery` for the same fault with an eight-second
arrival window. The longer window captures traffic after the replacement
worker becomes ready on this Windows host.

Processing delays are injected before prediction and leave the trained
model intact. Fault logs record the observed injection, restart launch,
and readiness times. Separate-host LAN and network impairment measurements
were unavailable for this run.

For each scenario and repetition, a seeded trace selects held-out images
and replays the same scheduled arrivals against all four policies in a
seeded random order. The generator verifies file checksums and bounds
in-flight requests at 64. A scheduled arrival more than 100 ms late or
beyond the generator limit is recorded as a drop. Each trial starts a
fresh gateway and warms the selected workers directly before measurement.
The same request budget (1.5 s), per-attempt budget (0.65 s), queue limits
and retry rules apply to every policy. The single-worker run uses worker 1,
so the controlled worker-1 faults affect its only serving replica.

## Read the results

- `offered` counts every scheduled arrival, including client-side drops.
  `admitted` counts requests the gateway accepted into a worker queue.
  The report separately records sent arrivals without a matching gateway
  terminal record.
- Successful p50, p95, and p99 use send-to-response latency. The summary
  reports the median of each repetition's p95. See the per-run sample
  counts before interpreting the tail, especially for steady traffic.
- On-time success uses the gateway's configured end-to-end budget as a
  client latency limit and divides on-time successes by offered arrivals.
  Window throughput counts on-time completions before the scheduled
  arrival window closes. Completion counts can also include the drain.
- Gateway queue and dispatch times are per attempt. Retry fractions use
  offered arrivals as the denominator. Rejections, deadlines, other HTTP
  and client failures, and generator drops are reported separately.

The generator shares the machine with the services, including worker
restart work. Short runs and Windows timer resolution can affect send
timing. Compare policies within the same workload and consult the recorded
generator lateness and per-run sample counts.

Each trial retains its gateway configuration, raw request rows, gateway
events, and fault timestamps. The report extracts attempt rows from the
gateway log. The evidence index records relative paths and SHA-256
checksums so the outputs can be checked against the local raw suite.
Generated raw logs stay under the ignored `runs/` directory.

## Recorded comparison

The six-workload comparison completed 72 trials and scheduled 8,664
arrivals. The extended recovery suite completed another 12 trials and
scheduled 4,320 arrivals.
[Main comparison](experiments/phase8/comparison/results.md) ·
[Load and latency plots](experiments/phase8/comparison/load-comparison.svg) ·
[Burst and failure plots](experiments/phase8/comparison/burst-and-recovery.svg)

Adaptive had the lowest median run p95 in the heterogeneous and slowdown
workloads. In the ramp, least-outstanding served 590 of 624 arrivals at
368 ms median run p95. Adaptive served 545 of 624 at 576 ms. In the burst
workload, round-robin had the lowest median run p95 and all three
multi-worker policies served every offered request.

During extended recovery, round-robin served the most arrivals. Adaptive
had the lowest median run p95 among successful responses. The replacement
worker became ready between 5.36 and 8.72 seconds after the start of traffic.
In two trials it became ready after the eight-second arrival window.
[Extended recovery results](experiments/phase8/recovery/results.md) ·
[Recovery timeline](experiments/phase8/recovery/burst-and-recovery.svg)
