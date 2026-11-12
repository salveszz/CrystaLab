# Inclusion learnability and input-resolution experiment

The pitch's visible-inclusion task is still unfinished. A completed larger-input
fit detects **one of six** synthetic validation inclusions, but increases false
secondary detections and worsens seed measurement. Keep the existing model choice;
no candidate is promoted. The matched inclusion alert in a training clip now has
an inspectable operator replay. This supplies no new real labels or KDP accuracy.

The [frozen protocol](../configs/inclusion_resolution_experiment.json) keeps the
existing 84 training / 44 validation / 28 reserved images and whole-family splits.
No mask or confidence threshold is revised after results. All RGB images remain
256 pixels wide; using a 512-pixel input adds no captured detail.

## Training-only diagnostic

There are 22 eligible inclusion masks in training. Their median area is 76 pixels
and median shorter bounding-box side is eight pixels. The pinned model's observed
prototype tensors are 32 × 64 × 64 at input 256 and 32 × 128 × 128 at input 512.
Increasing input size changes network sampling and computation; the original
image information remains the same.

Both small fits use the first 12 existing training rows, including three inclusion
examples from one family. They start from the same pinned COCO weights, use 100
epochs, batch four, constant learning rate 0.001, no warmup/augmentation and full
separate masks. Each has 300 planned optimizer updates. The final checkpoint is
scored, without selecting the best epoch. Trainer validation reuses these same
images and is explicitly memorization, not performance on unseen data.

Both inputs match all three inclusion targets with zero inclusion false positives
on these selected images. They also match all 12 seed and six secondary targets.
The native inclusion IoU sum is 2.36397 at 256 and 2.38785 at 512. This establishes
that the 256 model can represent these examples; it does not establish that
resolution alone caused the previous grouped failures.

The independent auditor reconstructs native masks and verifies maximum same-class
match counts with a separate augmenting-path algorithm. An adversarial check
moves a prediction, recomputes its hash and retains false perfect metrics; the
auditor correctly rejects that claim. Hashes alone do not establish accuracy.

## Grouped comparison

The learnability gate permits the predeclared 30-epoch full 512 fit, using the
same three pitch classes, training images, initialization, seed, batch and mask
settings as the previous 256 fullmask run. Each plans 630 optimizer updates.
The larger input uses more computation; this is not a fixed-compute comparison.
The historical and new models use the same best-checkpoint selection procedure.

Standalone evaluation uses common native masks, confidence 0.25 and same-class
mask IoU 0.5. Fresh 256/512 image evaluations and the new model's FFV1 development
replay preserve missed targets, false alerts, seed failures, family membership,
negative/deadline coverage and original clocks. The old forecast remains frozen.
Reserved test families are not scored. Actual results and exact identities are
recorded in the [checkpoint](INCLUSION_RESOLUTION_CHECKPOINT.json).

| Native validation result | Existing 256 | New 512 |
|---|---:|---:|
| Seed TP / FP / FN | 44 / 0 / 0 | 44 / 0 / 0 |
| Mean seed IoU, image weighted | 0.94066 | 0.92971 |
| Seed area MAE, px² | 65.16 | 182.84 |
| Secondary TP / FP / FN | 15 / 5 / 2 | 17 / 10 / 0 |
| Inclusion TP / FP / FN | 0 / 0 / 6 | 1 / 0 / 5 |
| Subsequent median CPU prediction call, ms | 9.97 | 21.84 |

The fresh 256 evaluation reproduces all 64 historical predicted masks, classes,
scores and assignments exactly. The new inclusion match is `optical_pilot_013:2`,
with IoU 0.60256. The other five inclusions are missed. All ten new secondary false
instances are in `optical_temporal_004`; these are repeated frames of a related
scene, not ten independent experiments. Per-family counts remain in the audit.
The 512 fit took 374.10 seconds; memorization fits took 74.87/228.96 seconds.
Other verification ran concurrently. These are recorded execution times, not
isolated hardware or fixed-compute performance measurements.

## Video and operator evidence

All five FFV1 clips preserve the 80 original frames exactly. The new model matches
three constructed secondary events and adds one false crystal event in validation
family 004. The old lossless 256 run had three matches and no false crystal events.
Usable annotated crystal-negative coverage decreases from 9.25 to 8.75 source
seconds; validation coverage is 4.25 seconds with the new false event inside it.

The single eligible inclusion event is matched at IoU 0.74359, 0.25 source seconds
after its first eligible proxy frame. **That clip is training data.** Validation
still has zero eligible inclusion events, so no validation inclusion-event recall
can be estimated. Negatives and visibility here follow the existing emission-proxy
eligibility contract, including its limitations; they do not establish physical
absence or exact visibility of constructed objects. Physical run-start and
first-hour coverage remain unknown.

An explicit training-only console session preserves the learned inclusion
localization and an unreviewed `inspect_and_mark_inclusion` proposal at source
time 1.75 s. It has no simulated process attached, numerical commands or human
review receipts. Open it without writes:

```bash
.venv/bin/python scripts/operator_console.py \
  --session results/operator_inclusion_resolution_v1 --read-only
```

Read-only verification checked all 16 frames, the localized proposal and unchanged
session files. This is a software-path demonstration, not independent detection
validation. The original model defaults and sessions are unchanged.

The unchanged area forecast also retains its failures: on 28 shared validation
windows, persistence/recent-rate/calibrated-rate future-area MAE is
228.32/140.57/149.90 px². The old 256 recent-rate result is 77.00 px². The frozen
training-mean method happens to give 76.71 px² for the new masks; this is not a
refit or evidence for an action-response model. No forecast or controller is
promoted from this comparison.

Independent checks reconstruct 112 selected/native image evaluations, 80 temporal
observations, 75 causal windows and 300 forecast records, including masks, event
matches, negative periods and deadline coverage. The source, models and reserved
groups retain their identities. Full regression passes **417 tests and 305
subtests** in 183.32 s. Standalone figures include class counts and every one of
the six validation inclusion examples; the display crop never changes scoring.

## Reproduction

Use fresh outputs and the existing local pinned runtime/weights:

```bash
.venv/bin/python scripts/diagnose_inclusion_resolution.py --prepare \
  --output results/inclusion_resolution_preparation_new

# Run each prepared overfit_256.json / overfit_512.json sequentially.
.venv/bin/python scripts/train_vision.py \
  --dataset data/yolo_optical_refinement_v1 \
  --config results/inclusion_resolution_preparation_new/overfit_256.json \
  --output results/inclusion_overfit_256_new

.venv/bin/python scripts/diagnose_inclusion_resolution.py \
  --preparation results/inclusion_resolution_preparation_new \
  --run results/inclusion_overfit_256_new --output results/inclusion_scores_new

.venv/bin/python scripts/audit_inclusion_resolution.py \
  --evaluation results/inclusion_scores_new --output results/inclusion_audit_new.json
```

After the explicit gate passes, use
`configs/vision_training_inclusion_resolution_512.json` for the full fit,
`scripts/evaluate_vision.py` for native validation and
`scripts/run_temporal_diagnostic.py` with a new FFV1 experiment bound to the
completed model's hashes. Retain original experiments and all failed outputs.
Neither memorization nor synthetic validation authorizes physical control or
automatic model promotion. Relevant real pixel/event review, faces/asymmetry,
first-hour clocks, action response and the remaining HPC/Qiskit work stay open.

Next prioritize varied inclusion/negative examples and independent real pixel/event
review. Increasing input resolution alone did not produce a dependable detector.
Use another declared grouped experiment for any further model change; retain
the original baselines, failures and untouched final groups.
