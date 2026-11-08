# Local upload demo

Complete the [development setup](../README.md#development-setup) and
[train the Phase 2 classifier](model.md#reproduce). From the repository root,
install the worker and demo dependencies in the same virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-worker.lock
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.lock
.\.venv\Scripts\python.exe -m pip check
```

Start these commands in three terminals, each at the repository root.
On Linux, use `.venv/bin/python` in place of the PowerShell Python path.

| Terminal | Command |
|---|---|
| Worker | `.\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/dev.yaml --worker-id worker-1` |
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
dimensions. One gateway request can be active at a time. A second request
receives 429. The gateway has no waiting queue in this phase.

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/ready
curl.exe --fail-with-body -F "image=@data/raw/dataset-resized/paper/paper1.jpg" -F "locality_id=demo-locality" http://127.0.0.1:8000/v1/classify
```

`GET /health` checks the gateway process. `GET /ready` checks that the
configured worker reports a ready model. A failed connection or incompatible
worker response returns 503. An expired request returns 504. The API returns
structured errors with a request ID.

The worker includes the review threshold from its saved model artifact.
The gateway applies that threshold before consulting the versioned
[`demo-v1` rules](../config/rules/demo.yaml). Below-threshold predictions
prompt the user for a clearer photo. Plastic guidance asks for item type and
contamination when those details are missing. If there is no rule for an
accepted material, the response says guidance is unavailable.

The demonstration locality does not represent a municipality. Its messages
are prompts for checking local rules. The UI presents scores as uncalibrated.
[Streamlit file uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
