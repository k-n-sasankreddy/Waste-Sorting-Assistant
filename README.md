# Waste Sorting Assistant

The project pairs single-item waste image classification with an experiment in
distributed inference. The planned app sends a photo to a gateway. The gateway
assigns it to a worker running a full copy of the classifier, interprets the
prediction, and looks up guidance for the selected locality. A material
prediction alone cannot determine how an item should be discarded.

## Development setup

Use Python 3.12 and run the commands from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

On Linux, create the environment with `python3.12 -m venv .venv` and replace
`.\.venv\Scripts\python.exe` in the remaining commands with `.venv/bin/python`.
Setup was checked on Windows using a CPU-only Python environment.

## What runs today

- `shared.config.load_settings()` reads `config/dev.yaml` and validates worker
  addresses, scheduling settings, and request and queue limits. Model paths are
  relative to the repository root and are reserved for a later phase.
- `shared.schemas` defines the multipart metadata, predictions, worker status,
  guidance, and errors that the future services will exchange.
- The pytest fixture returns a fixed, explicitly synthetic score. It is outside
  the installed package, and the API prediction schema rejects its score type.
- The offline dataset command audits TrashNet and produces reproducible,
  group-aware splits. See the [dataset instructions](docs/dataset.md).
- The offline model commands train a six-class MobileNetV3-Small classifier,
  evaluate the saved model, and classify local photos. See the
  [model instructions and results](docs/model.md).
- The local inference worker loads that classifier once and accepts one image
  at a time over HTTP. See the [worker setup and API](docs/worker.md).

The configuration describes one gateway and three loopback workers. Each
future worker will have one inference slot. The gateway will own its queues
in memory, so it will run as one process and remain a single point of failure.
Proposed admission limits include a 5 MiB upload cap, 64 outstanding requests,
16 waiting jobs per worker, and a 64 MiB combined queued-image budget.
The byte budget covers waiting image payloads. In-flight uploads, active
inference, and decoded images will require their own bounds when serving is
implemented. The configured locality is for demonstration.

## Next milestones

Phase 4 will connect a single worker to a gateway and deliver the first
upload-to-guidance flow.

Later experiments will compare round-robin, least-outstanding, and estimated
completion time with a fixed model and matched workloads. See the
[phase tracker](docs/progress.md) for the current checkpoint.
