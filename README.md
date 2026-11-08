# Waste Sorting Assistant

The project pairs single-item waste image classification with an experiment in
distributed inference. The app sends a photo to a gateway. The gateway
assigns it to a worker running a full copy of the classifier, interprets the
prediction, and looks up guidance for the selected locality. A material
prediction alone cannot determine how an item should be discarded.

[Install the full stack](docs/install.md) · [Rehearse the demo](docs/runbook.md)
· [Architecture](docs/architecture.md) · [Submission report](docs/submission.md)

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
  relative to the repository root.
- `shared.schemas` defines the multipart metadata, predictions, worker status,
  guidance, and errors exchanged by the services.
- The pytest fixture returns a fixed, explicitly synthetic score. It is outside
  the installed package, and the API prediction schema rejects its score type.
- The offline dataset command audits TrashNet and produces reproducible,
  group-aware splits. See the [dataset instructions](docs/dataset.md).
- The offline model commands train a six-class MobileNetV3-Small classifier,
  evaluate the saved model, and classify local photos. See the
  [model instructions and results](docs/model.md).
- The local inference worker loads that classifier once and accepts one image
  at a time over HTTP. See the [worker setup and API](docs/worker.md).
- The Streamlit demo lets you upload a photo and see the predicted material,
  review prompts, and demonstration guidance. See the
  [local startup guide](docs/demo.md).
- The gateway distributes requests across three local model replicas using
  round-robin, least-outstanding, or estimated completion time. The Worker
  status page shows queue depth, assignments, and service estimates.
- The gateway can retry transient worker failures and reconcile uncertain
  attempts. See the [recovery and LAN guide](docs/recovery.md).
- The benchmark replays matched workloads and reports latency, throughput,
  and request outcomes. See the [benchmark guide](docs/benchmark.md).

The default setup uses one gateway and three loopback workers. Each worker
has one inference slot. The gateway stores queues in memory and remains a
single point of failure. Defaults cap uploads at 5 MiB, outstanding
requests at 64, waiting jobs at 16 per worker, and queued image payloads
at 64 MiB. Uploads and active requests count toward the outstanding limit.
The configured locality is for demonstration.

The [phase tracker](docs/progress.md) records verification and remaining
evaluation limitations.
