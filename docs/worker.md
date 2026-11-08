# Local inference worker

Complete the [development setup](../README.md#development-setup) and create
the [Phase 2 model artifact](model.md#reproduce). From the repository root
in PowerShell, install the pinned HTTP dependencies and start one worker.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-worker.lock
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/dev.yaml --worker-id worker-1
```

The worker binds to the selected worker's HTTP origin in `config/dev.yaml`.
The default for `worker-1` is `http://127.0.0.1:8101`. On Linux, use
`.venv/bin/python` and replace `curl.exe` below with `curl`.

From a second terminal, check readiness and upload a real image.

```powershell
curl.exe --fail-with-body http://127.0.0.1:8101/ready
curl.exe --fail-with-body -F "image=@data/raw/dataset-resized/paper/paper1.jpg" -F "request_id=local-demo" -F "attempt_id=local-demo-1" -F "remaining_budget_ms=5000" http://127.0.0.1:8101/v1/infer
curl.exe --fail-with-body http://127.0.0.1:8101/status
```

| Endpoint | Behavior |
|---|---|
| `GET /health` | Reports that the HTTP process responds. |
| `GET /ready` | Returns 200 after successful model loading and warm-up, or 503 if either fails. |
| `GET /status` | Reports worker and boot IDs, readiness, slot state, active attempt, model versions, warm-up time, and inference counters. |
| `POST /v1/infer` | Returns request and attempt IDs, a material prediction, preprocessing and label-map versions, and processing time. |

The worker reserves its only slot before reading the multipart body. It
accepts one JPEG, PNG, or WebP image, bounds both uploaded bytes and decoded
pixels, and rejects additional requests with 429 while the slot is occupied.
There is no inference queue inside the worker. Early errors may carry a
worker-generated request ID because the multipart metadata has yet to be
read. Inference runs on a dedicated thread so health and status can respond
while the classifier runs. The status includes the last finished attempt
ID so the gateway can reconcile a lost response.
[FastAPI concurrency model](https://fastapi.tiangolo.com/async/)

The request budget is measured from worker admission, includes upload time,
and is capped by the configured worker budget. A 504 can arrive while the
model is still computing. The slot stays occupied until that computation
finishes, and the status counters reflect execution outcomes even when an
HTTP caller already timed out. `processing_ms` measures decode, preprocessing,
and prediction, excluding multipart parsing and HTTP response delivery.

Malformed multipart bodies return 400. Invalid fields or undecodable images
return 422, oversized uploads return 413, and an unavailable model returns
503. The worker returns the raw model score. The gateway will apply
abstention and disposal rules in a later phase.
