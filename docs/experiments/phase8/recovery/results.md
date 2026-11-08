# Measured scheduling results

| Workload | Policy | Successful / offered | Median run p95 (ms) | Rejected | Deadline | Other failures | Generator drops |
|---|---|---:|---:|---:|---:|---:|---:|
| recovery | adaptive | 827 / 1080 | 59.4 | 0 | 1 | 252 | 0 |
| recovery | least_outstanding | 843 / 1080 | 64.1 | 0 | 0 | 237 | 0 |
| recovery | round_robin | 859 / 1080 | 65.3 | 0 | 0 | 221 | 0 |
| recovery | single | 322 / 1080 | 109.7 | 13 | 1 | 744 | 0 |

Latency uses successful requests. The table shows medians of per-run p95 values.
See [per-run metrics](metrics.csv) and the [evidence index](evidence.json).
