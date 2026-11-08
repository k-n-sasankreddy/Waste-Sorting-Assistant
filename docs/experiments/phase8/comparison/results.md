# Measured scheduling results

| Workload | Policy | Successful / offered | Median run p95 (ms) | Rejected | Deadline | Other failures | Generator drops |
|---|---|---:|---:|---:|---:|---:|---:|
| burst | adaptive | 258 / 258 | 429.6 | 0 | 0 | 0 | 0 |
| burst | least_outstanding | 258 / 258 | 476.9 | 0 | 0 | 0 | 0 |
| burst | round_robin | 258 / 258 | 384.3 | 0 | 0 | 0 | 0 |
| burst | single | 169 / 258 | 598.4 | 89 | 0 | 0 | 0 |
| failure | adaptive | 417 / 540 | 56.2 | 0 | 0 | 123 | 0 |
| failure | least_outstanding | 410 / 540 | 56.5 | 0 | 0 | 130 | 0 |
| failure | round_robin | 372 / 540 | 76.8 | 0 | 0 | 168 | 0 |
| failure | single | 63 / 540 | 103.0 | 0 | 0 | 477 | 0 |
| heterogeneous | adaptive | 312 / 312 | 94.1 | 0 | 0 | 0 | 0 |
| heterogeneous | least_outstanding | 312 / 312 | 178.6 | 0 | 0 | 0 | 0 |
| heterogeneous | round_robin | 312 / 312 | 756.7 | 0 | 0 | 0 | 0 |
| heterogeneous | single | 124 / 312 | 1067.5 | 188 | 0 | 0 | 0 |
| ramp | adaptive | 545 / 624 | 576.2 | 75 | 0 | 0 | 4 |
| ramp | least_outstanding | 590 / 624 | 368.2 | 32 | 0 | 0 | 2 |
| ramp | round_robin | 560 / 624 | 497.7 | 62 | 0 | 0 | 2 |
| ramp | single | 312 / 624 | 547.4 | 312 | 0 | 0 | 0 |
| slowdown | adaptive | 390 / 390 | 116.1 | 0 | 0 | 0 | 0 |
| slowdown | least_outstanding | 390 / 390 | 201.1 | 0 | 0 | 0 | 0 |
| slowdown | round_robin | 384 / 390 | 1280.4 | 0 | 6 | 0 | 0 |
| slowdown | single | 131 / 390 | 1279.9 | 213 | 5 | 41 | 0 |
| steady | adaptive | 42 / 42 | 45.5 | 0 | 0 | 0 | 0 |
| steady | least_outstanding | 42 / 42 | 59.7 | 0 | 0 | 0 | 0 |
| steady | round_robin | 42 / 42 | 49.1 | 0 | 0 | 0 | 0 |
| steady | single | 42 / 42 | 53.7 | 0 | 0 | 0 | 0 |

Latency uses successful requests. The table shows medians of per-run p95 values.
See [per-run metrics](metrics.csv) and the [evidence index](evidence.json).
