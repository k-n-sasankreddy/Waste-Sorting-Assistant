# Offline classifier

Phase 2 trains a six-class linear head on frozen MobileNetV3-Small features.
The backbone and the pretrained classifier's intermediate layer stay in
evaluation mode. Training uses both the original image and its horizontal
flip. Validation and test use one deterministic view per image. The model
uses the ImageNet weight preset's resize, center crop and normalization,
with EXIF orientation and RGB conversion applied first.
[MobileNetV3-Small weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) ·
[PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Reproduce

Complete [development setup](../README.md#development-setup) and the
[TrashNet import](dataset.md#import-and-reproduce). From the repository root
in PowerShell, install the pinned CPU packages and run the offline commands.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-cpu.lock
.\.venv\Scripts\python.exe -m pip check
$env:TORCH_HOME = Join-Path (Get-Location).Path 'artifacts/models/.torch'
.\.venv\Scripts\python.exe -m waste_assistant.ml.train --config config/dev.yaml
.\.venv\Scripts\python.exe -m waste_assistant.ml.evaluate --config config/dev.yaml --output runs/phase2/test-report.json
.\.venv\Scripts\python.exe -m waste_assistant.ml.predict data/raw/dataset-resized/paper/paper1.jpg --config config/dev.yaml
```

On Linux x64 use `.venv/bin/python` and
`export TORCH_HOME="$PWD/artifacts/models/.torch"`. The commands were
verified on Windows x64 with Python 3.12 and four CPU threads.
Training downloads the official ImageNet initialization on first use.
For offline training, place
[`mobilenet_v3_small-047dcff4.pth`](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth)
under `$TORCH_HOME/hub/checkpoints/`. Evaluation and prediction load only
the exported classifier.
The initialization file SHA-256 is
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

The local artifact directory `artifacts/models/mobilenet-v3-small/` contains
`weights.pt` and `manifest.json`. The loader verifies the weights' SHA-256 and
uses the exported label order and preprocessing. The model binaries stay
outside Git. To retrain alongside an existing artifact, copy `config/dev.yaml`
and change `model.artifact_dir` before running the commands with that config.

## Selection and abstention

The run uses AdamW, seed 42, and 30 head-training epochs. It selects the first
checkpoint with the highest validation macro-F1. Only after checkpoint
selection does it choose a score threshold on validation. It chooses the
highest coverage with at least 90% accepted accuracy and at least 30%
coverage. If no threshold meets both targets, it maximizes accepted accuracy
among thresholds with at least 30% coverage. A score below the selected
threshold returns `needs_review`.

Scores are uncalibrated softmax outputs. Coverage is the fraction of images
that meet the threshold. Accepted accuracy measures correctness within that
fraction. The test split is used only after both checkpoint and threshold are
fixed.

## Measured results

Epoch 7 was selected with validation macro-F1 **0.838**. Its validation
threshold was **0.6280**, accepting 340 of 380 images (89.5% coverage) at
90.0% accepted accuracy.

| Held-out TrashNet test metric | Result |
|---|---:|
| Images | 377 |
| Top-1 accuracy | 85.7% (323/377) |
| Macro-F1 | 0.848 |
| Coverage at the validation threshold | 89.9% (339/377) |
| Accuracy among accepted predictions | 89.1% (302/339) |

The test threshold results use the validation threshold as exported. Glass
was predicted as plastic nine times, the most frequent off-diagonal error.
The model labeled one glass image as metal with a softmax score above 0.999,
so high scores still warrant caution. The `trash` class had the lowest F1
(0.769) and only 20 test images.

The [test report](experiments/phase2/test-report.json) records class precision,
recall, F1, support, the confusion matrix, and 12 high-score mistakes.
Confusion matrix rows are ground truth and columns are predictions, in the
recorded label order. The [model manifest](experiments/phase2/model-manifest.json)
records the selected epoch, validation history, split checksums, and weights
checksum. Two independent evaluation processes on the same host produced
byte-identical reports. There is no external real-world evaluation set yet.
These results measure the held-out TrashNet split, whose images were largely
taken against a white background.
