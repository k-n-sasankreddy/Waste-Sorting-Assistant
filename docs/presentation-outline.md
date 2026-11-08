# Presentation outline

| Time | Topic | Show or explain |
|---|---|---|
| 0:00–0:45 | User problem and project goal | A single-item upload, candidate material, and demo guidance. Separate the sorting problem from the serving research question. |
| 0:45–1:45 | Architecture | Follow one request through the gateway, queue, worker, review threshold, and guidance. Explain full model replicas and the single gateway. |
| 1:45–2:45 | Data and classifier | Audit and grouped splits, validation checkpoint and review threshold, held-out accuracy and macro-F1. Show one high-score mistake from the model report. |
| 2:45–4:15 | Live demo | Upload a clear image and an uncertain image. Switch to Worker status, then show the burst and the recorded slowdown and restart events. |
| 4:15–5:45 | Scheduling comparison | Describe round-robin, least-outstanding, and estimated completion time. Compare heterogeneous latency and ramp success counts. Include failures alongside latency. |
| 5:45–6:30 | Scope and questions | Same-host measurements, demonstration locality, external validation, and single-gateway limits. Be ready to explain why a retry can repeat computation without replacing the terminal result. |

[Demo runbook](runbook.md) · [Architecture](architecture.md) ·
[Submission report and related work](submission.md) · [Measured plots](benchmark.md#recorded-comparison)
