# TrashNet audit and splits

TrashNet is by Gary Thung and Mindy Yang. The audited input is the authors'
resized archive at Git commit `6fa2b878c6c1b4304b91109070ce0edf9279bb31`.
The upstream repository carries an MIT license, and the authors' Hugging Face
dataset card also lists MIT. Retain the upstream notice if redistributing the
archive. The authors request a repository citation.
[Upstream dataset](https://github.com/garythung/trashnet) ·
[MIT notice](https://github.com/garythung/trashnet/blob/6fa2b878c6c1b4304b91109070ce0edf9279bb31/LICENSE) ·
[Dataset card](https://huggingface.co/datasets/garythung/trashnet)

The Hugging Face viewer lists about 5,050 rows. This snapshot uses the
authors' 2,527-image GitHub archive to match the published class counts.
The raw images and archive are stored under the Git-ignored `data/raw/`.

## Import and reproduce

Run the [development setup](../README.md#development-setup), then run these
commands in PowerShell from the repository root.

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
git clone https://github.com/garythung/trashnet.git data/raw/upstream
git -C data/raw/upstream checkout --detach 6fa2b878c6c1b4304b91109070ce0edf9279bb31
$archive = 'data/raw/upstream/data/dataset-resized.zip'
if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne '0BF472790F8B20E5C950D5B5012A9D38AF0D3392EFD65F8CE171334FC16B07C2') {
    throw 'TrashNet archive checksum mismatch'
}
.\.venv\Scripts\python.exe -m zipfile -e $archive data/raw
.\.venv\Scripts\python.exe -m waste_assistant.ml.dataset --config config/dev.yaml
```

For an offline import, use a local copy of the same archive, verify its
SHA-256 above, and extract it into `data/raw/`. The expected image root is
`data/raw/dataset-resized/` with the six material subdirectories. Run the
last command after extraction. On Linux, use `mkdir -p data/raw`,
`sha256sum` to check the archive, and `.venv/bin/python` for both Python
commands. Config paths are relative to the repository root.

## Audit method and results

The audit ignores hidden metadata files, records file SHA-256 and image
dimensions, and fully decodes each image with a pixel limit. It connects
byte-identical images and near-duplicate candidates. Candidate comparison uses
a 128-bit difference hash of the EXIF-oriented image and a Hamming distance of
at most four. The four flagged near pairs were visually reviewed and retained
as same-split groups. Three byte-identical pairs appeared under different
labels. All six files in those groups were excluded pending label resolution.

The source supplies no object IDs used in this split. The similarity check can
miss separate views of the same object. If known object IDs are available,
set `dataset.same_object_groups` to a JSON file containing arrays of at least
two dataset-relative paths per group. Links are transitive. If a group
contains different labels, all its members are excluded for review.

Groups are assigned intact within each class. Larger groups are assigned
first, with ties ordered by a hash of the seed and group path. Each group goes
to the split furthest below its 70/15/15 class target. The committed snapshot
uses seed 42.

| Class | Scanned | Rejected | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Cardboard | 403 | 0 | 282 | 61 | 60 |
| Glass | 501 | 3 | 348 | 75 | 75 |
| Metal | 410 | 1 | 286 | 62 | 61 |
| Paper | 594 | 0 | 416 | 89 | 89 |
| Plastic | 482 | 2 | 336 | 72 | 72 |
| Trash | 137 | 0 | 96 | 21 | 20 |
| **Total** | **2,527** | **6** | **1,764** | **380** | **377** |

All 2,527 images decoded. The [committed snapshot](../data/manifests/trashnet-v1/)
contains the full audit, duplicate links, class counts, and one JSONL manifest
per split. Each split row contains the image path relative to the extracted
root, its label, and its SHA-256. A rerun with identical results leaves the
snapshot intact. Use a new output directory when changing the dataset,
grouping, or seed. Training should use the fixed train and validation
manifests and reserve the test manifest for final evaluation.
