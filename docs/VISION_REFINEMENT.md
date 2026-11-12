# Expanded synthetic vision experiment

Two new 30-epoch YOLOv8n-seg fits reduce some synthetic errors but leave a
tradeoff: standard mask supervision gives fewer false video alerts, while
full-resolution separate masks give better image-weighted seed measurements.
Neither detects the validation inclusions or beats area persistence with the
existing forecast on its MJPG replay. No default model or control policy was promoted.
The subsequent [paired encoding experiment](VIDEO_ENCODING.md) reproduces those
MJPG results, verifies exact FFV1 input and finds lower-than-persistence rate
errors for the two expanded detectors on the same two validation families.

The [frozen experiment](../configs/vision_refinement_experiment.json) was written
before fitting. The [checkpoint](VISION_REFINEMENT_CHECKPOINT.json) retains actual
model identities, per-family counts, measurements, event coverage, independent
audits and verification. Historical pilot and temporal checkpoints are unchanged.

## Data and comparison

`data/yolo_optical_refinement_v1` combines the 60-image pilot with the 96-frame
longer optical source. It preserves whole scene-family membership:

| Split | Images | Families | Seed / secondary / inclusion masks |
|---|---:|---:|---|
| Training | 84 | 15 | 84 / 52 / 22 |
| Validation | 44 | 6 | 44 / 17 / 6 |
| Reserved test | 28 | 5 | 28 / 20 / 4 |

All 277 masks are synthetic polygon proxies. Twelve processed frames from the
three temporal training families received a separate recorded synthetic QA
acceptance. This creates no human real-image review. Real labels used: zero.
Reserved tests were neither visually inspected for tuning nor scored.

Both new fits start from the same pinned COCO initialization, use seed 20261111,
256-pixel images, batch four, two CPU threads and AdamW at learning rate 0.001.
The standard fit uses `mask_ratio=4, overlap_mask=true`; the fullmask fit uses
`mask_ratio=1, overlap_mask=false`. Actual trainer arguments and installed
segmentation implementation files are hashed. Training took 99.09 and 108.59
seconds respectively, with other local verification running concurrently.

This changes two supervision settings together. It does not isolate their
individual effects. Compared with the original 36-image fit, the 84-image fits
also perform more optimization work. This is not a data-only comparison.
Each trainer selects its best checkpoint on the same development images;
standalone scoring uses common native polygon masks instead of comparing trainer
mAP values across different mask representations.

Ultralytics documents `mask_ratio` as training-mask downsampling and
`overlap_mask` as merging versus separate masks, with smaller masks overwriting
larger masks when merged. Changing `mask_ratio` does not change predicted-mask
resolution. See the [official configuration reference](https://docs.ultralytics.com/usage/cfg/).
The pinned local 8.4.210 implementation upsamples prototype predictions to the
supervision shape in the loss; this experiment does not add prototype capacity.

## Common native-image validation

Confidence is 0.25, same-class matching IoU is 0.5 and all predicted masks use
native image dimensions. All three models have 44 seed TP, zero seed FP/FN and
100% valid single-seed coverage on these images.

| Metric | Original pilot | Expanded standard | Expanded fullmask |
|---|---:|---:|---:|
| Mean seed IoU, image weighted | 0.9187 | 0.9281 | 0.9407 |
| Mean seed IoU, family macro | 0.9116 | 0.9243 | 0.9302 |
| Seed area MAE, image weighted, px² | 129.70 | 83.27 | 65.16 |
| Seed area MAE, family macro, px² | 167.24 | 92.42 | 92.88 |
| Seed area signed bias, px² | +67.66 | −8.32 | −36.84 |
| Secondary TP / FP / FN | 17 / 10 / 0 | 15 / 2 / 2 | 15 / 5 / 2 |
| Inclusion TP / FP / FN | 0 / 0 / 6 | 0 / 0 / 6 | 0 / 0 / 6 |

The two new secondary misses are in pilot family 014. Standard's two remaining
false secondary instances are in family 013. Fullmask adds two false instances
in family 012 and retains three in 013. Closely spaced frames are not independent
experiments: two temporal families supply 32 of the 44 images. Family macro
results therefore accompany image-weighted results. The frozen OpenCV baseline
has 9/44 valid measurements and mean IoU 0.0807 with invalid cases counted as zero.

## Video alerts and forecasts

Each model is replayed on the same five development clips. Three clips belong
to training and two to validation. The runner now records separate split metrics
so training-clip events cannot be reported as validation success.

| Validation evidence, two clips | Original pilot | Standard | Fullmask |
|---|---:|---:|---:|
| Crystal events TP / FP / FN | 1 / 1 / 0 | 1 / 0 / 0 | 1 / 1 / 0 |
| Complete usable crystal deadlines matched / eligible | 1 / 1 | 1 / 1 | 1 / 1 |
| Annotated usable crystal-negative source seconds | 3.50 | 4.75 | 4.75 |
| Valid seed observations / 32 frames | 32 | 31 | 32 |

There is only one positive validation crystal event and no eligible validation
inclusion event. The inclusion constructed in the validation source has zero
eligible proxy pixels. Inclusion event recall remains unevaluated there. All
three models miss the one eligible inclusion event in the training clips.
Standard loses seed identification at the last frame of validation family 003;
the preceding forecast cut frames remain available. Both new fits remove the
earlier matched crystal-origin error. These are synthetic source seconds, with
no physical first-hour or false-alert-hours interpretation.

The old area-forecast parameters remain frozen, without recalibration to the
new detector. All models retain 30 validation windows and the same 28 windows
with sufficient rate history. On those 28 shared windows:

| Future-area MAE, px² | Original pilot | Standard | Fullmask |
|---|---:|---:|---:|
| Persistence | 248.04 | 233.36 | 201.00 |
| Recent rate | 350.82 | 349.64 | 283.25 |
| Training mean fraction | 563.81 | 512.91 | 487.71 |
| Calibrated recent rate | 339.71 | 334.14 | 273.37 |

Better still-image masks do not guarantee stable video measurements. Native PNG
inference and MJPG replay use different input pixels, and observation retains an
associated seed component rather than the whole raw instance. A paired saved-mask
check uses the same original proxy reference on the 15 forecast cut frames per
validation family. For standard, native-instance area bias is −32.07/+67.40 px²
in families 003/004, while encoded seed-component bias is +319.87/+282.27 px².
The encoded raw instances are also oversized; component extraction removes
63.07/0.20 pixels on average. Decoder differences average about two intensity
levels per channel. This supports a measurement-path investigation before another
forecast fit, rather than interpreting the static measurement gains as a solved
video gate. No codec-only retraining ablation was performed.

## Reproduction and verification

Existing outputs refuse overwrite; use fresh paths for reruns. With both accepted
synthetic sources available, prepare the cohort with `scripts/prepare_segmentation.py`
using both `--source` paths. Run `scripts/train_vision.py` with the frozen standard
and fullmask configurations **sequentially**, since the trainer shares label caches.
Use `scripts/evaluate_vision.py` and `scripts/benchmark_vision.py` with the common
dataset and frozen `results/vision_seed_pilot_calibration_v1/selected_config.json`.
Create a new source-bound temporal experiment for each resulting model and use
`scripts/run_temporal_diagnostic.py`; preserve the original frozen experiment.

Actual outputs are `results/vision_refinement_{standard,fullmask}_v1`, their
`_eval_v1`, `_seed_v1` and `_temporal_v1` siblings, and the common original-model
`results/vision_refinement_pilot_{eval,seed}_v1` comparison. Weights and raw data
remain Git-ignored. A fresh clone needs source generation and training.

Independent scripts under `results/` reconstruct 132 image evaluations from native
polygons and saved masks, plus 240 temporal observations, 225 causal windows and
900 forecast records. They verify assignments, class/family counts, seed errors,
source event labels, negative/deadline coverage and forecast formulas. All pass;
1,548 saved artifact hashes were also checked. Full regression passed **378 tests
and 269 subtests** in 129.36 seconds. The checkpoint binds evidence to the exact
source, configuration, dataset and model hashes.

Next compare lossless input with MJPG using frozen weights, review the six real
training proposals, and freeze the supported measurement/alert task. Build the
video/process/proposal/override demonstration alongside this work. Inclusion,
face/asymmetry measurements, mixed/real evaluation, adaptive state/policy,
equipment response, HPC and Qiskit remain explicit pitch requirements.
