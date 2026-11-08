# Installation and verification

Use Python 3.12 on Windows or Linux x64, Git, and a local checkout of the
project. Initial package, dataset, and pretrained-weight downloads require
internet access. The application runs locally after setup.

## Install from the lockfiles

Run from the repository root. These commands use PowerShell.

```powershell
py -3.12 -m venv .venv
..venvScriptspython.exe -m pip install -r requirements-dev.lock
..venvScriptspython.exe -m pip install --no-build-isolation --no-deps -e .
..venvScriptspython.exe -m pip install -r requirements-ml-cpu.lock
..venvScriptspython.exe -m pip install -r requirements-worker.lock -r requirements-demo.lock -r requirements-benchmark.lock
..venvScriptspython.exe -m pip check
..venvScriptspython.exe -m pytest
```

On Linux use `python3.12 -m venv .venv`, then substitute `.venv/bin/python`
for the PowerShell Python path.

## Prepare the data and model

1. Follow the [TrashNet import and checksum instructions](dataset.md#import-and-reproduce).
   The audit command verifies the extracted files against the committed split
   snapshot.
2. Train and evaluate the model using the [model instructions](model.md#reproduce).
   If a teammate already has the recorded artifact, copy its `weights.pt` and
   `manifest.json` together into `artifacts/models/mobilenet-v3-small/`.
   The loader checks the weights against the manifest. Use the same artifact
   on every worker when comparing policies.
3. Run the [upload and recovery rehearsal](runbook.md). It uses ports 8000
   and 8101–8103 by default. The interactive Streamlit UI uses port 8501.
   Stop other local services on those ports before starting the demo.

The dataset and weights are local files. The recorded evaluation reports and
evidence indices are included in the repository.
