# Video observation baseline

Implemented: file decoding, an explicitly selected camera adapter, seed silhouette
tracking, projected pixel measurements, invalid-state events and auditable replay
outputs. This is the first software batch in the [roadmap](../roadmap.md), not a
trained crystal/defect detector. It issues no actuator commands.

The same CLI now also accepts a verified model via `--model-run`; see
[learned video observation](LEARNED_VIDEO.md) for that configuration, seed-role
uncertainty and source-frame mapping. This document describes the baseline mode.

## Install and smoke replay

Install the optional OpenCV dependency in the project environment:

```bash
.venv/bin/python -m pip install -e '.[vision]'
```

The optional extra changes `pyproject.toml`, not the growth equations or saved
dataset. Existing growth data still passes its audit. Its historical source
snapshot is retained; the strict generator resume check will reject a changed
packaging/source fingerprint instead of silently reusing it.

Create the independent non-optical integration fixture in an unused directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/make_vision_fixture.py \
  --output results/vision_fixture_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/observe_video.py \
  --video results/vision_fixture_v1/fixture.avi \
  --config results/vision_fixture_v1/baseline.json \
  --source-info results/vision_fixture_v1/source_info.json \
  --run-id analytic_seed_fixture_v1 --output results/vision_baseline_v1
```

Both commands refuse existing output directories. The 24-frame fixture has a
known seed, a larger unrelated object and a four-frame seed-visibility gap. The
baseline recovers 20 valid observations and records four invalid observations.
Fixture masks are pixel-geometry ground truth; they do not represent vessel
optics or a measured crystal-growth/defect process. The separate
[Blender optical pipeline](OPTICAL_SYNTHETIC.md) and [grouped model pilot](VISION_PILOT.md)
are now implemented; their evidence does not turn this analytic fixture into real data.

## Observation configuration

`--config` is a JSON object accepted by `vision.baseline.BaselineConfig`:

```json
{
  "seed_point": [85, 110],
  "roi": null,
  "polarity": "dark",
  "threshold": 110,
  "minimum_area_px": 32,
  "maximum_area_fraction": 0.8,
  "morphology_size": 1,
  "minimum_contrast": 5.0,
  "minimum_focus": 0.0,
  "minimum_iou": 0.1,
  "ambiguity_margin": 0.05,
  "maximum_gap_s": 2.0
}
```

These values describe the fixture, not a calibrated laboratory default. Select
the seed point and observation ROI from development clips. ROI is `[x,y,width,
height]`; coordinates remain in the original frame. Threshold null uses Otsu;
dark/bright specifies foreground polarity. Contrast is the ROI's gray dynamic
range, and focus is Laplacian variance; neither is a physical visibility measure.
An overlap IoU is a geometric association score, not model confidence.

The first seed must contain the supplied point. Later observations must overlap
the previous valid seed mask. Clipped, ambiguous, low-contrast and missing seeds
produce invalid records with null measurements. After the maximum source-clock
gap, identity requires explicit reinitialization in a new reviewed run. A larger
unrelated object cannot silently replace the seed. Moving/rotating cameras,
touching objects, transparent boundaries and reflection changes remain limitations.
Even a valid record means the configured baseline found a silhouette, not that
a human has validated its identity or measurement accuracy.

## Clocks and source metadata

Files default to OpenCV's container timestamps. Nonfinite, negative, duplicate or
backward times fail explicitly. When a backend cannot supply usable timestamps,
select `--clock nominal_fps` and optionally `--fps`; that estimate remains labeled.
There is no automatic clock fallback. Variable-frame-rate timing is preserved in
JSONL; the convenience MJPG overlay uses constant display FPS and source-time text.

`--sample-every N` decodes sequentially and retains original frame indices/times.
`--max-frames N` limits decoded frames, including unsampled ones. Camera input uses
`--camera INDEX`, monotonic frame-arrival times and a required frame limit; the
command opens only the explicitly selected device. Arrival is not exposure time.

`--source-info` supplies JSON provenance. Record source/material/experiment IDs,
source URL, license and `evidence_kind` (`real`, `synthetic`, `fixture` or `unknown`).
Publications, montages and result overlays require content review before they
become raw training data. Keep source clips and derivatives in the same split.

Experiment time is null by default. Mapping video time to an experiment requires
both `--experiment-origin-s` and `--experiment-seconds-per-video-second`, plus
`experiment_clock_provenance` in source metadata. Do not infer acquisition time,
time-lapse factor or first-hour growth coverage from playback FPS.

## Output contract

| Artifact | Meaning |
|---|---|
| `manifest.json` | Source file hash/device, timestamp basis, configuration, source metadata, versions, completion/failure state, frame counts and output hashes. |
| `measurements.jsonl` | Source index/time, validity/reason, projected area, bounding dimensions, centroid and mask path; experiment time only if explicitly mapped. |
| `masks/` | Binary PNG for each valid processed seed observation. |
| `events.jsonl` | Seed acquisition/recovery and invalid-state transitions, with evidence paths. These are observation-validity events, not crystal-defect labels. |
| `evidence/` | Timestamped overlays at validity transitions. |
| `overlay.avi` | Constant-display-FPS MJPG replay; JSONL holds original timing. |
| `final_observation.png` | Inspectable final processed frame. |

Failed runs retain a failed manifest and partial evidence. Source files are
hashed before and after file replay; changing the input fails the run. Each
accepted output artifact has a hash. This supports provenance, not experimental
accuracy or adversarial tamper protection.

## Validation and next integration

`tests/test_vision.py` exercises independent analytic silhouettes, distractors,
ambiguous splits, clipping, bounded recovery, actual codec decoding, container
and nominal clocks, preserved sampling indices, mask agreement, artifacts,
failure retention and guarded output creation. Run with the vision extra installed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tests -p test_vision.py -v
```

The full first-batch checkpoint passed 244 tests and 130 subtests. See the
[versioned verification record](VISION_CHECKPOINT.json), including source hashes,
fixture/development counts and the explicitly unfinished model/control gates.

Real microscope import, synthetic optical images, YOLOv8-seg fitting and learned
seed replay are implemented in their linked reports. Next: review actual visible
masks/events and compare methods on real development data. Event-class evaluation,
predictive supervision and physical validation remain open.

The clock implementation follows the backend-dependent contract documented by
[OpenCV VideoCapture properties](https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html).
