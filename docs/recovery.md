# Worker recovery and LAN setup

The gateway polls `GET /status` on each worker. A failed or stale probe,
an incompatible model fingerprint, or an uncertain attempt makes that
worker ineligible. Waiting jobs can move to other workers while retaining
their original deadline. Each worker reports a new boot ID after restart and
must finish model warm-up before it can rejoin.

## Deadlines and retries

The default request budget is 10 seconds including upload, queueing, and
inference. The gateway waits up to four seconds per attempt. It allows one
retry on a different eligible worker for a transient failure if at least
200 ms remain and capacity allows. Configure these values under `limits`
in `config/dev.yaml`. Invalid inputs and incompatible responses are terminal.

A transport failure or timeout after a request may have reached a worker
puts that slot in quarantine. The slot counts against the gateway's capacity
until a status probe identifies the exact finished attempt or a new worker
boot. An idle report alone cannot establish that a delayed request never
arrived. If the worker cannot confirm the attempt, restarting it clears
quarantine after it reports its new boot ID.

Retrying can duplicate computation. Each attempt gets its own ID and each
request has one terminal response. Late work on the original worker can
finish and release quarantine without replacing the retry result. The
gateway status and Streamlit dashboard show quarantined workers and the
retry count. Gateway `scheduler_attempt` events record request, attempt,
worker and boot IDs with the attempt outcome.

## Start workers on a private LAN

1. Install the same pinned Python and worker dependencies on each machine.
   Copy the same trained `weights.pt` and `manifest.json` to each configured
   model directory. Each worker verifies the weight checksum at startup.
2. Copy `config/dev.yaml` to `config/lan.yaml`. Replace the worker URL
   hosts with the actual LAN addresses of the worker machines. Keep one
   unique address and port per worker. For example, a worker on a machine
   with LAN IP `192.168.1.21` might use `http://192.168.1.21:8101`.
   Use this configuration on the gateway and worker machines.
3. On each worker machine, run the following with its assigned worker ID.

   ```powershell
   .\.venv\Scripts\python.exe -m waste_assistant.worker.app --config config/lan.yaml --worker-id worker-1
   ```

4. From the gateway machine, check `/status` for each worker using its
   configured LAN URL. Ensure network and host firewall rules permit
   gateway-to-worker traffic. Worker endpoints use plain HTTP and are
   intended for a trusted private network.
5. On the gateway machine, run the gateway with `--config config/lan.yaml`.
   Launch Streamlit there using the [local startup guide](demo.md) and
   set `$env:WASTE_CONFIG = 'config/lan.yaml'` before starting it. The
   gateway and UI bind locally on that machine.

On Linux, use `.venv/bin/python` and
`export WASTE_CONFIG=config/lan.yaml`. Recovery was verified using multiple
processes on one Windows machine. A separate LAN host was unavailable for
verification.
