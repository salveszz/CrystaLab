# Seed measurement benchmark

`benchmark_vision.py` compares the existing OpenCV seed method with saved YOLO
predictions on the same audited development images and projected seed polygons.
It reports missing measurements, mask IoU/Dice and projected-area error. It does
not score temporal alerts, 3D face rates, physical quality or the final test split.

## Comparison rules

The OpenCV method receives a declared seed point and optional observation ROI.
Each image starts a new tracker, so this benchmark measures single-frame
segmentation with a fixed view rather than temporal identity tracking. The point
and ROI do not come from each image's truth mask. A live camera would need its
own initialization and pose/calibration procedure.

Learned seed measurements require exactly one seed-class prediction. No seed,
multiple seeds, an empty mask or a clipped mask produces an invalid measurement.
The evaluator never chooses a seed by its overlap with the answer. All truth-seed
images remain in mean mask IoU/Dice, with missing/invalid measurements scored as
zero overlap. Area MAE/bias uses valid measurements only and reports that smaller
denominator alongside valid coverage. Negative images have a separate false-valid
measurement count.

The benchmark checks input dataset identity, saved prediction/mask hashes, image
IDs and family membership. It retains overlays, per-image errors, per-family
summaries, configuration and source snapshots. Source labels are the YOLO-visible
polygons; synthetic labels retain their optical-proxy limitations. The same
nearest-pixel rasterization is used by the detector diagnostic and comparison.

## Default and calibrated baseline

[The default optical config](../configs/optical_seed_baseline.json) uses dark
Otsu thresholding and image-center seed initialization on the declared 256×256
synthetic view. It is separate from the real alpha-LGA pilot configuration.

A bounded calibration can select threshold/polarity on training images only.
[The central ROI](../configs/optical_seed_roi.json) comes from inspected training
views, while [the grid](../configs/optical_seed_calibration.json) declares two
polarities and ten thresholds, including Otsu. The primary objective is mean seed
IoU over all training truth images, including misses as zero. Fixed grid order
resolves ties. Calibration retains all candidates and the selected configuration;
it does not read validation/test pixels or labels for selection. The separate
source integrity audit still verifies all stored artifacts.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_vision.py \
  --dataset data/yolo_optical_smoke_v1 --config configs/optical_seed_roi.json \
  --calibration configs/optical_seed_calibration.json \
  --output results/vision_seed_smoke_calibration_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_vision.py \
  --dataset data/yolo_optical_smoke_v1 \
  --config results/vision_seed_smoke_calibration_v1/selected_config.json \
  --learned results/vision_training_smoke_eval_v2 \
  --output results/vision_seed_smoke_calibrated_benchmark_v2
```

These local output paths already exist. Choose unused paths for another run.
Without `--learned`, the same command measures the OpenCV baseline alone.

## Observed smoke result

The 20-candidate calibration used 12 training images from four scene families.
It selected bright threshold 192 with the declared central ROI. Even on training
images its valid coverage was only 3/12 and mean IoU over all truth images was
0.0409. Keep this failure visible rather than assuming calibration makes the
threshold baseline adequate for these transparent/reflected images.

On the three validation images from one scene:

| Method | Valid measurements | Mean IoU including misses | Area MAE on valid measurements |
|---|---:|---:|---:|
| Default Otsu | 0/3 | 0.0000 | unavailable |
| Training-calibrated threshold/ROI | 0/3 | 0.0000 | unavailable |
| Three-epoch YOLOv8n-seg | 2/3 | 0.5477 | 143 pixels, n=2 |

The declared seed point lies inside the truth seed on all three validation
images; the calibrated ROI contains those seeds. Both baselines fail the seed
identification/clipping gate. The result describes this small fixed-view synthetic
case. It does not establish that YOLO outperforms a stronger classical method,
that the mask proxies are exact real visibility, or that the model generalizes to
real KDP. Secondary/inclusion diagnostics remain in [training](VISION_TRAINING.md).

The [larger grouped pilot](VISION_PILOT.md) comparison is now complete. Its
training-only calibration selected dark threshold 176. On 12 validation images
from four families it yielded 0/12 valid measurements, while the learned model
yielded 12/12, mean IoU 0.9055 and conditional area MAE 199 pixels. These remain
synthetic proxy measurements. Real measurement accuracy needs actual reviewed
masks. Detection confidence is not calibrated uncertainty or proof of a healthy crystal.
