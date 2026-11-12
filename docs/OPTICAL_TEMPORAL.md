# Longer optical temporal diagnostic

The first optical pilot has three frames per scene: only 0.5 display seconds from
first to last frame at 4 FPS. This cannot provide a full one-second alert window
after its later events. The new protocol supplies **16 frames per scene**, ending
at 3.75 source-video seconds, with explicitly scheduled objects and varied growth
profiles. These are illustrative geometry steps, not physical growth seconds.

The protocol and [frozen experiment](../configs/optical_temporal_experiment.json)
were saved before rendering or scoring. The existing detector, forecast parameters,
confidence, candidate policy and matching convention remain fixed. No new model
is fitted or promoted by this diagnostic.

## Cohort and construction

[`synthetic_vision_temporal.json`](../configs/synthetic_vision_temporal.json)
declares six new base families, each with 16 frames: **48 training / 32 validation /
16 reserved-test images**. Seeds and family IDs differ from the earlier pilot.
The reserved family is rendered and checked for file/mask integrity, but is excluded
from visual tuning, development replay and scoring.

| Scene suffix | Split | Constructed secondary presence | Constructed inclusion presence | Other variation |
|---|---|---|---|---|
| 000 | Train | None | None | Bubble enters at frame 2; steady outward offsets. |
| 001 | Train | Frames 4–15 | None | Bubble from startup; outward offsets slow after frame 7. |
| 002 | Train | From startup | Frames 6–15 | Bubble from startup; outward offsets accelerate. |
| 003 | Validation | Frames 5–15 | Frames 8–15 | Two bubble distractors; outward offsets accelerate after frame 7. |
| 004 | Validation | None | None | Bubble enters and exits; constant geometry offsets during frames 5–10. |
| 005 | Reserved test | Not used for development | Not used for development | Separate base family; no test result reported. |

Each camera and seed orientation stays fixed within its scene. The offset schedule
moves the existing convex solid's face planes; it is not calibrated kinetics or
an action-response law. Randomized Cycles sampling and postprocessing may change
pixel masks even while geometry offsets are constant. No temperature, rotation or
pump command is inferred from these profiles. Face-specific growth/asymmetry
measurement remains a separate pitch requirement.

The optional `crystalab_optical_temporal_v1` contract requires a finite offset for
every frame, beginning at zero, and an explicit presence interval for every
configured non-seed object. Intervals use inclusive start and exclusive end;
`null` means never present. Each object currently has one contiguous interval.
This does not yet model disappearance followed by reappearance or camera occlusion.
Legacy protocols keep their original offset and presence behavior.

Presence is distinct from pixel eligibility. Event labels come from original
optical emission-proxy masks passing the existing area threshold, not predictions
or a claim of exact visibility through transparent material. Geometric masks,
proxy passes and RGB remain separate under the [optical mask contract](OPTICAL_SYNTHETIC.md).
Reflections, bubbles and source masks are not relabeled after seeing detector errors.

## Generate, audit and run

Using the existing pinned Blender 4.5.16 CPU runtime:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/generate_vision_images.py \
  --config configs/synthetic_vision_temporal.json \
  --blender /tmp/crystalab-blender-runtime/blender-4.5.16-linux-x64/blender \
  --output data/synthetic_vision_temporal_v1 --timeout 7200
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/generate_vision_images.py \
  --output data/synthetic_vision_temporal_v1 --audit
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_temporal_diagnostic.py \
  --source data/synthetic_vision_temporal_v1 \
  --experiment configs/optical_temporal_experiment.json \
  --output results/optical_temporal_diagnostic_v1
```

Use unused output paths for reruns. The audit checks declared presence and target
identity, native RGB/mask artifacts and reconstruction from the original proxy
passes. Existing 60-image pilot integrity still passes with the new consumer.
The diagnostic refuses incomplete rendering or changed protocol/model/config
identities before creating an experiment output.

The workflow prepares five development sequences and their causal learned
observations, retains original proxy labels separately, scores the frozen
projected-area parameters and runs candidate/event evaluation for each clip.
The new families must be disjoint from forecast-calibration families. The reusable
frozen scoring function reports any overlap explicitly when used on another cohort;
source split names alone do not prove independence. It writes an evaluation
artifact, not a newly trained model.

## What the metrics mean

The experiment uses a 0.25-source-second area horizon and an event matching
convention of mask IoU 0.5 and maximum delay one source second. These are declared
diagnostic settings, not the real accuracy/latency budgets still awaiting pilot
review. Area reports retain startup coverage, unavailable predictions, per-family
errors and the shared windows on which all methods can predict.

Event reports now distinguish two follow-up definitions:

- `complete_followup` retains the earlier elapsed-clip definition: the clip ends
  after the target's first eligible appearance plus its matching deadline.
- `observable_deadline_followup` additionally requires uninterrupted intervals
  whose adjacent observations are declared usable for that event class. Unknown
  views and gaps are excluded with reasons, even when the clip is long enough.

Usable intervals are reconstructed from adjacent observation times, class flags
and the configured gap limit; rehashed interval claims cannot bridge an unknown
observation. These are the algorithm's declared observation states, not independently
validated optical visibility. Ordinary event matches, misses, false inspections
and origin errors remain reported even when a target lacks full usable follow-up.
Negative coverage is intersected with independent construction labels and stays
in source seconds. Physical negative hours and first-hour claims remain unscored.

## Actual completed run

The render completed in 898.20 seconds on eight CPU threads. All 96 images and
158 eligible proxy masks pass reconstruction: 96 seed, 51 secondary-crystal and
11 inclusion masks. The frozen diagnostic processed 80 development frames and
75 within-clip forecast windows; the reserved 16-frame scene was not replayed.
The [checkpoint](OPTICAL_TEMPORAL_CHECKPOINT.json) records exact artifacts,
independent execution reconstruction and the 377-test / 262-subtest regression.

Across the five development clips, secondary-crystal event matching gives
**3 TP / 3 FP / 0 FN**, with 6.0 usable annotated-negative source seconds.
One match is a startup object and two are later appearances. Each match occurs
0.25 source seconds after its first eligible frame. One later appearance retains
an origin mismatch: the source says clear-view appearance, while the tracker
reports appearance after an unknown interval. The three false crystal inspections
remain in the report. All three target windows have full class-usable follow-up.

The inclusion result is **0 TP / 0 FP / 1 FN**, with full usable follow-up for that
one target and 16.25 negative source seconds across clips. This eligible target
is in a training scene. The validation scene's constructed inclusion has zero
proxy-mask pixels at frames 8–15, so validation has **no eligible inclusion target**.
No validation inclusion recall is claimed, and the missing class coverage is not
fixed by treating a constructed object as a visible positive.

The two validation families provide 28 shared rate-history windows:

| Frozen method | Future-area MAE (px²) | Gain MAE (px²) |
|---|---:|---:|
| Persistence | 248.04 | 98.61 |
| Recent rate | 350.82 | 92.11 |
| Earlier training-mean fraction | 563.81 | 221.52 |
| Earlier calibrated rate | 339.71 | 82.97 |

Persistence/mean predictions cover all 30 validation windows; rate methods cover
28, with two startup windows lacking history. Although calibrated rate has lower
gain error, it has higher future-area error than persistence. Native current-area
bias remains positive: mean +359.33 and +322.20 px² over the 15 windows in the two
validation families. Zero-growth persistence partly cancels this state error.
The old coefficient and prior remain byte-identical; no forecast is promoted.

The independent audit reconstructs all 80 native observations, 75 causal windows,
300 forecast records, source event/negative labels, matched native-mask IoUs,
usable-deadline counts and forecast metrics without calling the production
forecast or event scoring functions. Training-only raw previews (12 cells) and
processed first/last overlays (six cells) were inspected. No validation/test RGB
was visually inspected or used for generator tuning in this batch.

The new data do not establish real KDP accuracy, quality, action benefit or a
validated adaptive policy. Human Leeds/KDP annotation, relevant longer real
trajectories, measured action responses, operator integration, HPC and Qiskit
remain open in the [pitch-led roadmap](../roadmap.md).
