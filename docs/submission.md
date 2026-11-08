# Waste Sorting Assistant

## Project summary

Waste Sorting Assistant accepts a photo of one item, predicts one of six
TrashNet material classes, and displays a review prompt or demonstration
guidance. It also provides an experimental platform for comparing request
scheduling across model replicas. The project separates classifier quality
from serving performance. [Architecture](architecture.md) ·
[Installation](install.md) · [Demo runbook](runbook.md)

## Classifier evaluation

The dataset audit decoded 2,527 images and removed six files from exact
duplicate pairs with conflicting labels. It retained 2,521 images in
reproducible train, validation, and test splits. A MobileNetV3-Small classifier
trained with frozen ImageNet features and a new six-class head. Checkpoint
selection and the review threshold used validation data.

On 377 held-out TrashNet images, top-1 accuracy was **85.7%** and macro-F1
was **0.848**. The review threshold accepted 339 images (**89.9% coverage**),
with **89.1% accuracy** among accepted predictions. The model report includes
the confusion matrix and high-score mistakes. [Dataset audit](dataset.md) ·
[Model report](model.md#measured-results)

## Scheduling experiment

The gateway compares one worker with round-robin, least-outstanding, and
estimated-completion-time scheduling across three full model replicas.
The adaptive policy uses a moving average of successful dispatch times,
queue depth, and an estimate of the active request's remaining time.

The controlled evaluation used 84 matched trials and 12,984 offered
arrivals on one Windows host. Each workload and repetition used a shared
seeded trace across policies. The values below are medians of each run's
p95 latency among successful responses.

| Workload | Round-robin | Least-outstanding | Adaptive | Successful / offered |
|---|---:|---:|---:|---|
| Burst | 384.3 ms | 476.9 ms | 429.6 ms | 258/258 for all three policies |
| Heterogeneous workers | 756.7 ms | 178.6 ms | 94.1 ms | 312/312 for all three policies |
| Load ramp | 497.7 ms | 368.2 ms | 576.2 ms | 560, 590, and 545 successes respectively out of 624 per policy |
| Extended recovery | 65.3 ms | 64.1 ms | 59.4 ms | 859, 843, and 827 successes respectively out of 1,080 per policy |

Adaptive scheduling reduced successful-request tail latency under
heterogeneous workers and the injected slowdown. Least-outstanding
completed more arrivals during the ramp. During extended recovery,
round-robin completed the most arrivals while adaptive had the lowest
successful-response p95. The full tables include the single-worker
baseline, rejections, deadlines, and generator drops. [Benchmark method and
plots](benchmark.md) · [Main results](experiments/phase8/comparison/results.md)
· [Extended recovery](experiments/phase8/recovery/results.md)

## Related work

| Work | Approach studied | Connection to this project |
|---|---|---|
| He et al., IEEE MASS 2023 | Batch-aware scheduling and client-edge collaboration | Motivates measuring on-time completion under several arrival patterns. The project schedules individual image requests across identical replicas. [Author-hosted paper](https://zhaoyuan-he.github.io/assets/pdf/Adaptive-Scheduling-for-Edge-Assisted-DNN-Serving.pdf) |
| Tan et al., IEEE Transactions on Mobile Computing 2024 | Reinforcement learning for variable workloads and model selection | Motivates testing changing demand. The project uses an interpretable moving average and a fixed model. [University publication record](https://digitalcommons.njit.edu/fac_pubs/945/) |
| Zhang et al., IEEE Transactions on Network and Service Management 2024 | Joint adaptive batching and concurrent edge inference | Motivates reporting latency and throughput together. Each replica in this project has one inference slot and handles one image per dispatch. [Author-hosted paper](https://yangece.github.io/papers/journal/TNSM-2024-BCEdge.pdf) |
| Crankshaw et al., USENIX NSDI 2017, Clipper | Low-latency serving with batching, caching, and model selection | Provides serving-system context. The project compares only replica assignment policies. [USENIX paper page](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw) |
| Gujarati et al., USENIX OSDI 2020, Clockwork | Predictable DNN execution and centralized scheduling | Motivates separating processing time from end-to-end latency. The project's estimate incorporates observed worker and transport delays. [USENIX paper page](https://www.usenix.org/conference/osdi20/presentation/gujarati) |

## Limitations

- TrashNet mostly shows single items on a controlled background. The audit
  can miss separate photos of the same object. External photo evaluation,
  score calibration, and unknown-object detection are outside the
  measured results.
- Disposal guidance is demonstration-only and reflects no verified
  municipality. The project measured classifier and serving behavior.
  Environmental impact and usability remain unevaluated.
- All serving measurements use synthetic traffic and shared CPU resources
  on one machine. A separate LAN host and network impairment were
  unavailable for testing. Some recovery trials outlasted the offered
  arrival window. Short trials limit interpretation of p99.
- The gateway stores queues in memory and remains a single point of
  failure. Retried work can execute twice. Each request still receives
  one terminal outcome, and quarantine protects uncertain worker slots.

[Presentation outline](presentation-outline.md)
