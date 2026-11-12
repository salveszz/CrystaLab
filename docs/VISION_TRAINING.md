# First YOLOv8-seg training pipeline

The local image-to-model pipeline now works: audited images and polygon labels
can train YOLOv8n-seg, save weights and produce inspectable prediction masks.
The first experiments use synthetic images only. The latest
[grouped pilot](VISION_PILOT.md) detects seeds/secondary objects in its small
validation set, with bubble false positives and missed inclusions. Real-image
outlines still need review. This is a development checkpoint, not an evaluated KDP detector.
The later [expanded synthetic experiment](VISION_REFINEMENT.md) compares standard
and full-resolution separate-mask supervision on 84 training / 44 validation
images, preserving misses, per-family results and a separate video comparison.

## Inputs and safeguards

`src/vision/training_data.py` accepts two source types:

- An audited optical dataset with recorded development RGB/mask inspection.
- An export from the explicit real-annotation review workflow. An unreviewed
  proposal bundle is rejected. Image/annotation hashes must match the receipt,
  and YOLO labels must reconstruct the receipted polygon coordinates.

The adapter preserves original experiment/scene families, source material,
real/synthetic identity, reviewed negatives and source manifests. It rejects
cross-split families and duplicate images across splits. Auditing compares
exported pixels/labels with the original source; rewriting export hashes alone
cannot hide a changed polygon. Original source directories are required for this
reconstruction audit. Outputs are local and never overwritten.

Synthetic binary masks can have small interior holes. YOLO's external polygon
representation fills those holes. Export records the changed pixels and resulting
mask IoU, requiring at least 0.95 for each eligible instance. Large representation
loss or disconnected masks fails export and requires inspection. This threshold
controls a format conversion, not detector accuracy. The masks remain optical
emission proxies, as described in [optical synthetic data](OPTICAL_SYNTHETIC.md).

Images below a source's instance-visibility gate do not acquire a positive polygon.
Retain their source objects and visibility counts in the manifest. Real uncertain
objects stay excluded from accepted review export rather than becoming negatives.

## Runtime and initialization

The characterized local CPU environment uses Python 3.14.8, Ultralytics 8.4.210,
PyTorch 2.12.1 and torchvision 0.27.1. The optional training extra pins the model
packages and OpenCV; each run records actual package versions and thread count.
Install within the project environment:

```bash
.venv/bin/python -m pip install -e '.[vision-training]'
.venv/bin/python scripts/fetch_vision_weights.py
```

[Runtime identity](../configs/yolov8n_seg_runtime.json) pins the official
YOLOv8n-seg COCO initialization URL and SHA256. The fetch command verifies an
existing cache or downloads to a separate partial file before accepting it.
The project classes require fine-tuning. Upstream Ultralytics software/weights
carry their stated AGPL terms; source-dataset permissions remain attached to data.

Training uses local weights and isolated settings, with automatic package/model
downloads, synchronization and external experiment integrations disabled. It
loads optional training packages only when explicitly called. The first runtime
is characterized for CPU; GPU training needs a separately declared configuration.

## Reproduce the software checks

Prepare the accepted synthetic smoke batch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_segmentation.py \
  --source data/synthetic_vision_smoke_v2 --output data/yolo_optical_smoke_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_segmentation.py \
  --output data/yolo_optical_smoke_v1 --audit
```

These output paths already exist on the development machine. Use a new path for
another export. A fresh checkout also needs the source images/render described in
[optical synthetic data](OPTICAL_SYNTHETIC.md). Generated images and weights are
Git-ignored; configurations and compact evidence reports are tracked.

Train three epochs and inspect development predictions:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/train_vision.py \
  --dataset data/yolo_optical_smoke_v1 --config configs/vision_training_smoke.json \
  --output results/vision_training_smoke_v2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/evaluate_vision.py \
  --run results/vision_training_smoke_v2 --dataset data/yolo_optical_smoke_v1 \
  --output results/vision_training_smoke_eval_v2
```

Each training run retains settings, source snapshots, exact input manifest/weight
identity, actual trainer options, loss CSV, best/last weights and a run manifest.
A new run also retains resolved mask-supervision settings, the hash of the
trainer's argument file, and hashes of the installed segmentation implementation.
Historical manifests without these newer fields still load; newer argument-file
changes are rejected. Defaults remain mask ratio four with merged overlap masks.
A failure retains its manifest and error. The development evaluator preserves
prediction masks, overlays, evidence IDs and explicit confidence/matching gates.
It supports training/validation images; final test scoring is excluded at this
stage. Green outlines are dataset polygons and red outlines are predictions.

## What the initial experiments establish

The smoke export contains 18 images from six scene families: 12 training, three
validation and three reserved test images. Eligible instances are 18 seeds,
10 secondary crystals and four inclusion candidates. The test split is not scored.

The first three-epoch fit (`vision_training_smoke_v1`) completed in 8.83 seconds
on the named local CPU runtime. At confidence 0.25 and same-class mask IoU gate
0.5, its three validation images from one scene yielded:

| Class | Matched objects | Missed objects | False positives |
|---|---:|---:|---:|
| Seed | 2 | 1 | 0 |
| Secondary crystal | 0 | 2 | 9 |
| Inclusion candidate | 0 | 1 | 2 |

The validation overlay shows secondary false positives around a bubble/glare
region and image borders. These failures remain available for inspection.
Three correlated frames from one validation scene provide very weak generalization
evidence. Report these counts as a diagnostic, not real-world detection accuracy.

A separate 60-epoch overfit check used two training images and reused those same
images for validation. It completed in 31.68 seconds and reached mask mAP50 0.995
on those images. It checks that the model can learn the small supplied example;
it supplies no holdout result and does not cover all classes.

The follow-up smoke run verifies the current source after additional provenance
checks. Its source snapshots and manifest are authoritative for that revision;
prior runs remain separate evidence. The
[training checkpoint](VISION_TRAINING_CHECKPOINT.json) records the 264-test /
136-subtest pass, exact model hashes and development errors. The follow-up
three-epoch run completed in 7.97 seconds and reproduced the first fit's metrics.

## Grouped pilot and unfinished gates

[The bounded optical pilot](../configs/synthetic_vision_pilot.json) declares
20 base scenes, three frames each, with 12/4/4 scene families by split. Generation,
audit, development review and the
[30-epoch experiment](../configs/vision_training_pilot.json) are now complete.
The [pilot report](VISION_PILOT.md) records actual data/model identities, seed
baseline comparison and bubble/inclusion errors. Use family learning curves and
error coverage before larger generation.

Real-only/mixed training, the real OpenCV accuracy comparison, relevant real holdout
performance, face-specific interpretation, temporal alerts and first-hour claims
remain open. The real review path is in [annotations](ANNOTATIONS.md); the full
perception/control/HPC/Qiskit sequence remains in the [roadmap](../roadmap.md).
Additional [source triage](REAL_VISION_SOURCE_CANDIDATES.json) records a KDP
surface-inspection lead and a calcium-carbonate microscopy deposit separately.
Neither is counted as acquired, relevant KDP growth-event training truth.

The trained model now runs through [timestamped learned observation](LEARNED_VIDEO.md).
That replay retains raw labels/masks, seeded identity across visual class changes
and inspection requests for compatible competing outlines. Real replay estimates
remain unreviewed; they are not accepted training labels or accuracy evidence.

Official implementation references checked on 2026-11-11:
[YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/),
[segmentation task](https://docs.ultralytics.com/tasks/segment/) and
[the pinned upstream package](https://pypi.org/project/ultralytics/8.4.210/).
Explicit YOLOv8 weights preserve the pitch's model choice even when current
upstream examples use another model generation.
