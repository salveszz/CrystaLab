# Timestamped learned video observation

The trained YOLOv8-seg model now runs through the existing file/camera observation
path. It retains source timestamps, raw candidate masks, projected seed estimates,
role/uncertainty evidence and original-frame mappings. File replays were executed
on an optical development scene and a real alpha-LGA microscope sequence.
Real segmentation/event accuracy remains unreviewed, and this path issues no
actuator commands.

## Seed identity and uncertainty

An explicit seed point initializes the known seed. Subsequent observations use
mask overlap and a bounded source-video-clock gap. A larger unrelated object
cannot replace it. Clipped, weak, missing, split or competing distinct candidates
produce invalid measurements; a long gap requires a new explicit initialization.
This observer batch gives the seed a persistent role and retains other raw objects
by per-frame index. The subsequent [temporal analysis](TEMPORAL_CANDIDATES.md)
adds causal candidate identities and inspection events to saved replays. These
remain experimental visible candidates rather than confirmed parasitic nucleation.

Raw model classes `seed` and `secondary_crystal` are visual predictions, while
seed identity comes from the supplied point and history. The real pilot exposed
a class-role failure: the same small crystal received the model's secondary label,
then its seed label as it grew. A seed role can therefore use either configured
crystal class, preserving the original label and score in every raw observation.
Inclusion-class masks cannot initialize the seed.

Nearly identical cross-class seed masks can be compatible boundary hypotheses.
The declared mask-IoU gate is 0.9. The tracker retains their raw masks, selects the
best prior-mask overlap (highest score during initialization), records the range
of their pixel areas and requests inspection. A raw-class transition also requests
inspection. Other near-tied candidates remain ambiguous. This policy addresses
duplicate role labels; it does not certify the object, mask boundary or physical
meaning of either prediction.

`projected_area_range_px` is a range of retained model estimates, not a calibrated
confidence interval. Detector scores are uncalibrated. A valid tracking estimate
can still require inspection, and even an unflagged estimate lacks real accuracy
validation. No absence of a candidate establishes healthy optical quality.

## Run the optical development replay

Use unused output paths for reruns. These paths already exist locally.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/export_optical_sequence.py \
  --source data/synthetic_vision_pilot_v1 --scene optical_pilot_013 \
  --output results/optical_pilot13_sequence_v1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/observe_video.py \
  --video results/optical_pilot13_sequence_v1/sequence.avi \
  --config configs/optical_learned_observation.json \
  --model-run results/vision_training_pilot_v1 \
  --source-info results/optical_pilot13_sequence_v1/source_info.json \
  --frame-map results/optical_pilot13_sequence_v1/frame_map.jsonl \
  --run-id optical_pilot13_learned_hypotheses_v3 \
  --output results/optical_pilot13_learned_v3
```

The exporter accepts one training/validation scene and excludes final test scenes.
It preserves source IDs, family/split, image hashes and illustrative geometry
steps. MJPG encoding changes image pixels; it does not create additional samples
or a physical-time conversion. Three source frames remain three replay frames.

## Run the real development replay

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-training-mpl \
  .venv/bin/python scripts/observe_video.py \
  --video results/leeds_pilot12_v1/sequence.avi \
  --config configs/leeds_pilot12_learned.json \
  --model-run results/vision_training_pilot_v1 \
  --source-info results/leeds_pilot12_v1/source_info.json \
  --frame-map results/leeds_pilot12_v1/frame_map.jsonl \
  --run-id leeds12_learned_hypotheses_v3 \
  --output results/leeds_pilot12_learned_v3
```

The sequence is the first 30 source images of alpha-LGA training run 12, resized
to 960×720 and displayed at 4 FPS. Its original JPEG identities and resize factors
remain in the source frame map. It is a different material and imaging setup from
the illustrative training scenes and the intended KDP tank application.

`--model-run` loads a completed local model after checking weight/source identity.
The optional runtime uses isolated local settings and disabled cloud integrations.
Omitting it retains the existing OpenCV baseline interface. A missing or changed
model fails explicitly rather than silently substituting the baseline.
`--camera INDEX` still requires an explicit frame limit and arrival clock. No
physical camera was opened in these executions.

## Output and timing contract

| Artifact or field | Meaning |
|---|---|
| `manifest.json` | Input/model/config identity, actual observer sources, versions, counts, completion/failure and output hashes. |
| `measurements.jsonl` | Original source-video index/time, seed role/component, pixel area/dimensions, overlap, raw class, uncertainty and source-frame reference. |
| `objects` / `instances/` | Every retained raw post-NMS instance's original-frame binary mask, label, uncalibrated score and per-frame role. Compatible seed hypotheses stay visible. |
| `masks/` | Selected seed connected component. Extra islands in the raw instance remain in the raw mask, with excluded pixel count recorded. |
| `events.jsonl` / `evidence/` | Observation state and inspection transitions, including alternative seed outlines and class changes. These are not evaluated defect alerts. |
| `frame_map.jsonl` | Exact copied derivative-to-original map, bound to the hash in source metadata. |
| `source_snapshot/` | Observer, association, decoder and rendering source files used by the replay. |
| `overlay.avi` / `final_observation.png` | Inspectable estimates/candidates; amber indicates an estimate requiring inspection, red an invalid observation. |

The frame-map hash must match `source_info.frame_map_sha256`. Indices must be
contiguous and display timestamps must match decoded video timing within 1 ms.
Mapped records preserve original sample indices, image hashes and source timing
semantics. `nominal_elapsed_s` remains a nominal source-paper cadence, and
`synthetic_time` remains an illustrative geometry step.

Experiment time and actual acquisition time stay null in these replays.
The 4 FPS display clock does not establish elapsed growth time, actual camera
cadence or first-hour coverage. Association gap limits use source video seconds,
not physical growth seconds. A physical-time map requires separately supported
origin/scale evidence under the existing [video contract](VISION_BASELINE.md).

## Observed development behavior

| Replay | Processed | Seed estimates / invalid | Inspection frames | Raw instances |
|---|---:|---:|---:|---:|
| Optical scene 013, current policy | 3 | 3 / 0 | 0 | 11 |
| Alpha-LGA run 12, current policy | 30 | 30 / 0 | 11 | 40 |

The optical replay retains three seed instances and eight unassigned crystal
candidates. These include the persistent bubble false positives already exposed
by the image pilot; replay has not solved them. Its selected seed components have
mean IoU 0.9006 against original optical proxy masks across these three frames.
This is a synthetic proxy check, not real accuracy or a controlled comparison
with the earlier image benchmark.

The real replay retains ten compatible alternative seed hypotheses across frames
15–24 and a raw-label transition at frame 25. Each requests inspection; the seed
role continues. The overlays show smooth/inflated boundaries requiring review.
Thirty exported estimates do not mean thirty correct masks. There are still zero
accepted real pixel/event annotations and no measured real accuracy.

Two earlier runs remain as failure evidence. The class-0-only prototype exported
15 estimates and 15 invalid observations. Allowing both crystal classes without
handling duplicate role labels exported the first 15 estimates, then eight
ambiguities and seven long-gap invalid states. The current policy preserves the
competing evidence and flags it rather than treating raw label changes as proof
of a new crystal or silently dropping the disagreement.

The checkpoint records the 284-test / 147-subtest run, replay source/model hashes,
codec/frame counts, source mapping and numeric mask/component reconstruction.
[Verification record](LEARNED_VIDEO_CHECKPOINT.json).

## Remaining pitch gates

This batch completes a learned observation interface, seed-role tracking and
inspectable uncertainty behavior. Reviewed real masks/events, non-seed identities,
persistent event decisions, pose-aware face/asymmetry measurements and calibrated
budgets remain open. Inclusions remain unlearned in the synthetic pilot. Operator
inspection requests do not establish early defect detection or prevention.

Next: review the real pilot and add traceable candidate tracking/event evidence,
including negative periods and bubble confusion. PID/adaptive supervision, physical
action response, the integrated operator demo and HPC/Qiskit extensions remain in
the [roadmap](../roadmap.md).
