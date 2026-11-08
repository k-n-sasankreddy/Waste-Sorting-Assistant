# Local upload demo

Complete the [development setup](../README.md#development-setup) and
[train the Phase 2 classifier](model.md#reproduce). From the repository root,
install the worker and demo dependencies in the same virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-worker.lock
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.lock
.\.venv\Scripts\python.exe -m pip check
```

Start these commands in separate terminals, each at the repository root.
On Linux, use `.venv/bin/python` in place of the PowerShell Python path.

| Terminal | Command |
|---|---|
| Worker 1 | `.\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/dev.yaml --worker-id worker-1` |
| Worker 2 | `.\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/dev.yaml --worker-id worker-2` |
| Worker 3 | `.\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/dev.yaml --worker-id worker-3` |
| Gateway | `.\.venv\Scripts\python.exe -m waste_assistant.gateway.app --config config/dev.yaml` |
| Interface | `.\.venv\Scripts\python.exe -m streamlit run src/waste_assistant/ui/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false` |

Open [the local interface](http://127.0.0.1:8501) and upload
`data/raw/dataset-resized/paper/paper1.jpg` to see a material prediction
and demo guidance. Try `data/raw/dataset-resized/cardboard/cardboard125.jpg`
to see the review prompt with the saved Phase 2 model.

## Public gateway

The interface sends a multipart request to `POST /v1/classify` on port 8000.
The form accepts one `image`, the required `locality_id`, and optional
`item_type` and `contamination` fields. The gateway bounds the upload and
forwards it to the configured worker, which checks image decoding and
dimensions. The gateway assigns each image to one eligible worker's FIFO
queue. Each worker executes one request at a time. Admission beyond the
request, queue, or queued-image byte limit receives 429.

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/ready
curl.exe --fail-with-body -F "image=@data/raw/dataset-resized/paper/paper1.jpg" -F "locality_id=demo-locality" http://127.0.0.1:8000/v1/classify
```

`GET /health` checks the gateway process. `GET /ready` checks for an eligible
worker based on the registry's most recent probes. A failed connection or incompatible
worker response returns 503. An expired request returns 504. The API returns
structured errors with a request ID.

## Scheduling and dashboard

The default `policy: round_robin` rotates across eligible workers.
Set `policy: least_outstanding` and restart the gateway to prefer the smallest
sum of active and waiting requests. Set `policy: adaptive` to prefer the
earliest estimated completion. Ties rotate among eligible workers.
To run one worker
only, set `gateway.worker_id` to its configured ID. The gateway runs as one
process. Its admission and queue updates are synchronous on that process's
event loop.

Open **Worker status** in the Streamlit sidebar to see worker eligibility,
model versions, waiting and active counts, the estimated service time, and
cumulative assignments and completions. The same data is available at
`GET /v1/system/status`.
This setup is local multi-process emulation. The workers share one machine
and each loads a full copy of the model.

The gateway pins the first ready model fingerprint in configuration order
and checks each worker's model, preprocessing, and label-map versions.
It refreshes readiness periodically and excludes incompatible or stale
workers. Restart the gateway when deliberately changing the model across
all replicas. A request's deadline includes upload, waiting and dispatch.
Expired queued requests release capacity without starting inference.

Defaults allow up to 64 outstanding requests including uploads, 16 waiting
jobs per worker, and 64 MiB of queued image payloads. Each upload is capped
at 5 MiB plus bounded multipart overhead. Active and uploading images count
against the global request limit and can use memory beyond the queued-byte
limit, along with multipart buffers and HTTP transport copies.

### Adaptive estimate

For each eligible worker, the scheduler adds the estimated remaining time
of the active request to the service estimate multiplied by the number of
waiting requests plus one. For an active worker, remaining time is the
larger of estimated service minus elapsed time and
`residual_floor_fraction × estimated service`. The remaining time is zero
for an idle worker. The default floor is 10%.

The worker's measured model warm-up seeds the estimate for its current boot.
Warm-up measures local processing, so it serves as a starting proxy. After
each successful compatible dispatch, the gateway updates the estimate with
an exponentially weighted moving average. The default `alpha: 0.2` gives the
latest sample 20% weight. Samples include HTTP transfer and worker
processing and exclude time spent waiting at the gateway. Failed or timed-out
dispatches do not update the estimate.

Gateway logs include `scheduler_decision` events with a JSON payload for
each assignment. The payload records the request ID, policy, image size,
selected worker, and candidate scores. Adaptive scores use milliseconds.
`scheduler_sample` records dispatch and worker processing durations separately
for successful results. Sudden slowdowns take time to show up in the moving
average.

The worker includes the review threshold from its saved model artifact.
The gateway applies that threshold before consulting the versioned
[`demo-v1` rules](../config/rules/demo.yaml). Below-threshold predictions
prompt the user for a clearer photo. Plastic guidance asks for item type and
contamination when those details are missing. If there is no rule for an
accepted material, the response says guidance is unavailable.

The demonstration locality does not represent a municipality. Its messages
are prompts for checking local rules. The UI presents scores as uncalibrated.
[Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
