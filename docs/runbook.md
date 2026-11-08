# Demo runbook

## Rehearse before presenting

Complete the [installation guide](install.md) and stop any running gateway
or workers. From the repository root, run the rehearsal with a fresh output
directory.

```powershell
..venvScriptspython.exe scripts/demo.py --run-dir runs/demo-rehearsal
```

On Linux use `.venv/bin/python`. The script submits three real images through
Streamlit's form test harness and a live gateway and worker pool. It then runs
one adaptive trial each for burst, slowdown and worker recovery, prints the
recorded fault times, and builds the report under
`runs/demo-rehearsal/traffic/report/`. Its services stop on exit. The
rehearsal is a fresh observation of the system. Use the matched Phase 8
trials for policy comparisons.

## Present the demo

1. Show the [architecture diagram](architecture.md). Explain that a replica
   predicts a material and the gateway applies the review and guidance rules.
2. Start the [local services](demo.md), open Streamlit, and upload
   `paper/paper1.jpg` from the extracted dataset. With the recorded model,
   expect paper and a demo guidance message. Try `cardboard/cardboard125.jpg`
   for a review prompt and `trash/trash10.jpg` for unavailable guidance.
3. Open **Worker status** and identify the queue, active slot, service
   estimate, and quarantine columns. Close the interactive services before
   running the rehearsal command because it uses the same ports.
4. Run the rehearsal. Use its burst chart to show the arrival spike and
   assignments. Use the recorded fault events and gateway log to show the
   75 ms processing slowdown, worker stop, and restart. A replacement can
   take longer than the offered window. The script records actual readiness
   time even when that happens.
5. Open the [matched benchmark results](experiments/phase8/comparison/results.md)
   for the comparison with the baselines. Show successful and failed
   outcomes alongside latency.

## Regenerate a report

Rebuild the rehearsal report from the saved trial logs without sending more
requests.

```powershell
..venvScriptspython.exe -m waste_assistant.benchmark.report --run-dir runs/demo-rehearsal/traffic --output runs/demo-regenerated
```

The same command accepts a saved Phase 8 run directory such as
`runs/phase8-full` when its raw files are available. The
[benchmark guide](benchmark.md) describes how to generate a new matched suite.
