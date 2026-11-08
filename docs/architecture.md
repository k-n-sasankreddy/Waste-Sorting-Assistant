# Architecture

```mermaid
flowchart LR
    UI[Streamlit upload form] --> GW[FastAPI gateway]
    GW --> A[Upload admission and deadline]
    A --> S[Scheduler and worker registry]
    S --> Q1[Worker 1 FIFO]
    S --> Q2[Worker 2 FIFO]
    S --> Q3[Worker 3 FIFO]
    Q1 --> W1[Worker 1 model replica]
    Q2 --> W2[Worker 2 model replica]
    Q3 --> W3[Worker 3 model replica]
    W1 --> G[Review threshold and demo rules]
    W2 --> G
    W3 --> G
    G --> UI
    W1 -. Status and timing .-> S
    W2 -. Status and timing .-> S
    W3 -. Status and timing .-> S
    S --> T[Worker status dashboard and logs]
```

The gateway owns every waiting queue in memory. Admission reserves capacity
before reading a multipart upload. The scheduler selects a compatible worker,
and one dispatcher per worker sends requests in FIFO order. Each worker holds
a full copy of the trained model and runs one inference at a time.

The gateway applies the model's saved review threshold before looking up demo
guidance. Deadlines cover the original request, including any retry. Worker
status supplies the boot ID and last finished attempt so the gateway can
reconcile uncertain work. The gateway is a single process and a single point
of failure. [Scheduling and recovery details](demo.md)

Offline, the dataset audit builds checksummed train, validation, and test
manifests. Training selects the checkpoint and review threshold on validation
data, then exports weights, preprocessing, and label order. The held-out model
evaluation and serving benchmarks answer separate questions.
[Model instructions and results](model.md)
