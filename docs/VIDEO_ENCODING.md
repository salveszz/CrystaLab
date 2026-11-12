# Video encoding and measurement consistency

Lossy encoding was a major source of the earlier still-image/video measurement
discrepancy. A paired experiment replays the same three frozen detectors with
MJPG and lossless FFV1, keeping source frames, role association, candidate rules
and the old forecast parameters unchanged. All MJPG metrics reproduce their
historical results. FFV1 reproduces every original image pixel and the saved
native-image raw predictions on all validation frames checked.

The [experiment declaration](../configs/vision_codec_experiment.json) was frozen
before replay. The [checkpoint](VIDEO_ENCODING_CHECKPOINT.json) records all six
runs, source/model identities, failures, independent reconstruction and regression.
No detector or forecast was retrained, and no final test or real accuracy was scored.

## Implemented behavior

`export_optical_sequence` accepts `codec="MJPG"` or `codec="FFV1"`. It decodes
every written frame, checks count/dimensions and records pixel differences against
the original BGR image. FFV1 must reproduce those pixels exactly. An unavailable codec,
wrong stream type, missing/extra frame or changed FFV1 pixel fails the export;
there is no silent fallback. Failed outputs retain their failure manifest.

The export manifest and source information retain the requested codec, reported
FourCC, decoded frame count and pixel verification. OpenCV's local FFmpeg decoder
reports `ffv1`; matching normalizes case while preserving that original tag.
Frame mapping and the four-FPS display clock keep their earlier semantics.
Synthetic geometry steps still do not become physical growth seconds.

Forecast preparation accepts `sequence_codec`, and temporal experiments accept
the same explicit field. The historical omitted-field behavior remains MJPG.
For subsequent synthetic measurement experiments, explicitly choose verified
FFV1. Preview overlays may remain MJPG: forecast values come from native saved
measurement masks, rather than pixels read from an overlay.

Example export using a new output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/export_optical_sequence.py \
  --source data/synthetic_vision_temporal_v1 --scene optical_temporal_003 \
  --codec FFV1 --output results/optical_temporal_003_lossless_export
```

For `scripts/prepare_area_forecasts.py`, pass `--sequence-codec FFV1`. For
`scripts/run_temporal_diagnostic.py`, use a new source/model-bound experiment
containing `"sequence_codec": "FFV1"`. Preserve historical frozen experiments.
The paired study's six experiment paths and hashes are in its declaration;
their outputs are `results/vision_codec_{pilot,standard,fullmask}_{mjpg,ffv1}_v1`.
Use fresh output paths for reruns. Source images and trained weights remain local
Git-ignored artifacts; a fresh checkout needs their generation/training steps.

## Paired evidence

Each run processes the same five development families, 16 frames each. Three
families belong to detector training, and two to validation. Reserved family 005
is excluded. Two hundred forty decoded FFV1 frames across the three models match
original pixels exactly. MJPG changes all 240 corresponding frames, averaging
1.6848 intensity levels of absolute difference per channel over this cohort.

On the 32 validation frames per model, FFV1 raw instance counts, classes, masks
and scores exactly match saved native-PNG inference. That is **96/96 frames**
across the models. Seed-component extraction remains a distinct measurement
operation, and its removed pixels/validity are retained separately.

The two expanded detectors have 32/32 valid validation seed observations with
FFV1. Standard's former final-frame seed failure disappears. Fullmask's mean
seed-component area bias changes from +209.63/+338.25 px² with MJPG to
−38.06/+25.56 px² with FFV1 in validation families 003/004. Its mean absolute
errors on those families are 48.81/40.44 px² with FFV1. These are errors against
original optical emission proxies, rather than independently reviewed real masks.

## Forecast result and its limits

Every run retains 30 labeled validation windows and the same 28 windows with
history available for all methods. The independent comparison intersects window
IDs across **all six runs and four methods**; no difficult windows are discarded
to make encodings or models comparable. Family sizes are equal on this subset,
so its image/window-weighted and family-macro future-area MAE agree.

| Detector / encoding | Persistence | Recent rate | Frozen calibrated rate | Training mean fraction |
|---|---:|---:|---:|---:|
| Original pilot / MJPG | 248.04 | 350.82 | 339.71 | 563.81 |
| Original pilot / FFV1 | 87.04 | 165.43 | 153.79 | 287.27 |
| Expanded standard / MJPG | 233.36 | 349.64 | 334.14 | 512.91 |
| Expanded standard / FFV1 | 123.43 | 104.82 | 105.03 | 221.90 |
| Expanded fullmask / MJPG | 201.00 | 283.25 | 273.37 | 487.71 |
| Expanded fullmask / FFV1 | 116.18 | 77.00 | 78.55 | 193.80 |

Values are future-area MAE in px², with a 0.25-source-second horizon. Recent-rate
gain MAE with FFV1 is 60.00 px² for standard and 44.71 px² for fullmask, compared
with persistence's 98.61 px². The old calibration was fitted using an earlier
detector and MJPG observations; it is deliberately not refitted here.

The expanded detectors' rates now beat persistence on these **two synthetic
validation families**. The original detector's rates still lose. This changes the
earlier negative finding for this declared lossless input path; it does not
establish independent KDP forecasting, an action-response model or adaptive control.

## Events and retained failures

Both expanded detectors keep the one validation crystal event, without false
validation inspections under FFV1. Standard already had no false validation event
with MJPG; fullmask previously had one. There are 4.75 usable annotated-negative
source seconds for each expanded detector. These small synthetic clips do not
support a physical false-alert-per-hour or first-hour detection claim.

The original detector's FFV1 replay removes one validation false inspection but
misses the later crystal event in **training family 001**. Its training crystal
result is 1 TP / 1 FP / 1 FN, compared with 2 TP / 2 FP / 0 FN under MJPG.
The missed event lacks full usable deadline coverage; conditional deadline recall
does not erase the ordinary missed event. This failure is retained.

Every detector still misses the one eligible training inclusion event. Validation
contains no eligible inclusion event. The earlier six still-image validation
inclusion misses remain unchanged because the model weights and native-image
predictions are unchanged. Face/asymmetry interpretation is also unfinished.

## Verification and next use

Independent scripts import no CrystaLab scorer. They decode all 480 source frames,
reconstruct native event mask matching, usable negative/deadline coverage and
1,800 forecast records from 450 causal windows, and compare the 96 FFV1/native
prediction frames. All audits pass, including 3,202 artifact hashes. Full
regression passes **381 tests / 274 subtests** in 139.41 seconds. Tests also check
textured color preservation, loss recording, bad-codec rejection and missing or
changed lossless frames. Replay jobs ran concurrently; these times are not a
codec-throughput or live inference-latency acceptance benchmark.

Use verified FFV1 for subsequent synthetic measurement/control-demo inputs and
keep this paired MJPG evidence. Characterize the actual camera's encoding before
real capture/model acceptance. Continue the six-image real-label review and
video/process/proposal/override demonstration. Real mixed-training/evaluation,
inclusion and face coverage, adaptive state/policy, physical response, HPC and
Qiskit remain separate pitch deliverables. No detector or action policy is
automatically promoted by this diagnostic.
