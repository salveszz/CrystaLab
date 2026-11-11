# Grouped synthetic vision pilot

The first grouped YOLOv8n-seg experiment is complete. It uses 60 synthetic images
from 20 base scenes, with 12/4/4 scene families split before rendering. Seed and
secondary-crystal segmentation now work on the small synthetic validation set,
but two bubbles cause repeated secondary false positives and all inclusion
candidates are missed. This model has no evaluated real KDP accuracy.

## Data and source evidence

| Item | Observed value |
|---|---|
| Renderer/data | `data/synthetic_vision_pilot_v1`, Blender 4.5.16 LTS / Cycles CPU |
| Render time | 2,181.51 seconds, approximately 36.36 minutes, four rendering threads |
| Whole-family split | 36 training / 12 validation / 12 reserved test images |
| Eligible optical-proxy masks | 60 seeds / 38 secondary crystals / 21 inclusion candidates |
| Development visual review | First/last processed frames from all 12 training scenes, 24 images |
| Training export | `data/yolo_optical_pilot_v1`, source reconstruction audit passes |
| Reviewed real labels added | 0 |
| Physical quality/action-response labels | 0 |

The source protocol is [synthetic vision pilot](../configs/synthetic_vision_pilot.json).
The generator preserves its actual producer sources, resolved scene parameters,
Blender identity, RGB, raw optical/geometric passes and postprocessing versions.
The current auditor reconstructs the masks from those retained passes. Producer
and auditor hashes are recorded separately where they differ.

[Optical previews](OPTICAL_SYNTHETIC.md) select training scenes only. Generating
previews is separate from actual inspection. Recorded acceptance here is a
synthetic software/proxy-label review, not a human real-image annotation receipt.
All 119 eligible masks passed the declared external-polygon conversion gate.
The optics, tiny inclusions and geometric progression remain illustrative.

## Frozen first experiment

[The training configuration](../configs/vision_training_pilot.json) declares
30 epochs, 256-pixel input, batch four, AdamW at initial learning rate 0.001,
seed 20261111 and two CPU training threads. It uses the pinned COCO-initialized
YOLOv8n-seg weights. Full actual trainer arguments, environment, source snapshots,
loss CSV and best/last model files remain in `results/vision_training_pilot_v1`.
Training completed in 59.41 seconds.

The best model is local at:

```text
results/vision_training_pilot_v1/ultralytics/train/weights/best.pt
```

Its SHA256 is
`579b43ebd39e0a287192c877cd86a303cfba64d27521487242f6cc62cc8764bb`.
Generated images/models remain Git-ignored. The tracked configuration and
[compact checkpoint](VISION_PILOT_CHECKPOINT.json) record their identities.

## Development results and failures

The diagnostic used confidence 0.25 and same-class mask IoU matching gate 0.5 on
12 validation images from four scene families. It did not score the test split.

| Class | Matched | Missed | False-positive instances | Mean IoU per truth, misses zero |
|---|---:|---:|---:|---:|
| Seed | 12 | 0 | 0 | 0.9055 |
| Secondary crystal | 6 | 0 | 6 | 0.7832 |
| Inclusion candidate | 0 | 6 | 0 | 0.0000 |

The six secondary false-positive instances are two bubbles in scene 013 across
its three frames. Their scores exceed the fixed threshold, and they persist.
Temporal debounce by itself cannot resolve this example. Inclusion masks are
small and remain unlearned; a missing prediction cannot establish a healthy
crystal. All 12 validation overlays were inspected and the failures retained.
These are image-instance results, not six independent physical events or an
alert-rate measurement over real observation hours.

Upstream validation reports aggregate mask mAP50 0.6132 and mAP50–95 0.4472 under
its own evaluation procedure. Those metrics differ from the fixed-confidence
counts above. Avoid comparing them directly with the earlier smoke scores as a
controlled improvement experiment: the data and training duration both changed.

Measured CPU prediction time was 1.354 seconds for the first call and a median
0.0183 seconds across the remaining eleven calls. This excludes source audit,
image loading, artifact writing and the future tracking/controller/UI loop.
It does not establish end-to-end real-time performance.

## OpenCV comparison

The [seed benchmark](VISION_BENCHMARK.md) uses identical images and truth polygons.
The OpenCV baseline receives a fixed seed point and central observation ROI.
A 20-candidate threshold/polarity grid selected dark threshold 176 using only
36 training images from 12 families. Training mean IoU including misses was
0.0951, with 9/36 valid measurements. The selected settings were frozen for the
validation comparison.

| Method | Valid seed measurements | Mean IoU including misses | Area MAE on valid measurements |
|---|---:|---:|---:|
| Training-calibrated OpenCV threshold/ROI | 0/12 | 0.0000 | unavailable |
| Grouped-pilot YOLOv8n-seg | 12/12 | 0.9055 | 199 pixels, n=12 |

The seed point is inside every validation truth seed. The threshold baseline
fails its identification/clipping gate for these transparent/reflected views.
This result describes this baseline and declared synthetic camera, with its
manual seed-point prior. It does not establish superiority over every classical
method or measurement accuracy on real material.

## Reproduce on a machine with the source data

Use unused output paths for reruns. The following paths already exist locally.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_segmentation.py \
  --source data/synthetic_vision_pilot_v1 --output data/yolo_optical_pilot_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/train_vision.py \
  --dataset data/yolo_optical_pilot_v1 --config configs/vision_training_pilot.json \
  --output results/vision_training_pilot_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/evaluate_vision.py \
  --run results/vision_training_pilot_v1 --dataset data/yolo_optical_pilot_v1 \
  --output results/vision_training_pilot_eval_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_vision.py \
  --dataset data/yolo_optical_pilot_v1 --config configs/optical_seed_roi.json \
  --calibration configs/optical_seed_calibration.json \
  --output results/vision_seed_pilot_calibration_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_vision.py \
  --dataset data/yolo_optical_pilot_v1 \
  --config results/vision_seed_pilot_calibration_v1/selected_config.json \
  --learned results/vision_training_pilot_eval_v1 \
  --output results/vision_seed_pilot_benchmark_v1
```

## Next gates in the pitch sequence

- Review the real outlines and observable event definitions. Public images are
  acquired, but alpha-LGA/ADP development examples and phase-specific KDP clips
  do not yet constitute labeled KDP tank-growth validation.
- Connect the learned model to timestamped video observation and explicit
  identity/uncertainty handling. Add temporal event evidence and evaluation;
  frame detections alone do not complete the pitch's early-warning behavior.
- Investigate bubble false positives and inclusion visibility/scale using
  development data. Keep the reserved final tests outside tuning. Use family
  learning curves before larger synthetic generation or architecture changes.
- Complete real-only/mixed comparisons when reviewed labels support them, then
  freeze real acceptance budgets and evaluate independent runs.

PID/adaptive supervision, the operator demo, physical action-response evidence,
HPC/GPU and the justified Qiskit impurity study remain open in the
[roadmap](../roadmap.md). This pilot completes a bounded synthetic model experiment,
not the full pitch or the project's real-data validation.
