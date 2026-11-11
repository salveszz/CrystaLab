# Vision and control MVP contract

Status: pilot tooling on 2026-11-11. The file/camera adapter, OpenCV seed baseline,
real-image importer, annotation workflow and optical renderer exist. Real review
remains open. The 60-image grouped synthetic cohort, 30-epoch YOLOv8-seg fit and
development comparison are complete. Learned replay now preserves seed role,
uncertainty and source-frame mapping; real defect evaluation and control remain
open. See [Learned video](LEARNED_VIDEO.md), [Grouped pilot](VISION_PILOT.md), [Training](VISION_TRAINING.md), [Video baseline](VISION_BASELINE.md)
and [Real vision data](REAL_VISION_DATA.md) for actual evidence.
The [roadmap](../roadmap.md) defines the sequence. This document turns the
[pitch](../Pitch_KDP.pptx), especially slides 5–9, into reviewable deliverables.

## Product path and evidence levels

The intended path is video/camera -> segmentation and tracking -> visible-state
measurements and events -> operator/PID/supervisor decision -> action adapter ->
logged response. Process sensors, including temperature, have their own inputs.
Perception and control share timestamps and run IDs, not hidden simulator truth.

Start with recorded video and a simulated actuator. Add camera streaming and real
video shadow operation using the same interfaces. Physical actuation follows
equipment characterization and action-response evidence. A replay cannot show
what would have happened under a different physical action.

| Evidence level | What completion establishes |
|---|---|
| Synthetic imagery and scripted events | Label construction, perception experiments and software decision behavior under declared assumptions. |
| Independent real-video replay/shadow operation | Visible-event and measurement performance for the recorded material, optics and conditions. |
| Bench actuator/feedback tests | Timing, PID behavior, limits and command/acknowledgment handling on the tested equipment. |
| Independent physical growth experiments | The measured growth/visible-quality effect of the tested actions within the tested regime. |

Every report and demo states its evidence level and material. Alum/ADP can test
capture/software when available, as proposed in the pitch; their results cannot
be labeled validated KDP performance.

## First data contract

Inventory source footage and hardware first. Camera, thermostat, rotation stage,
pump, GPU and experiment access are unconfirmed dependencies. Without real footage,
implement synthetic examples and ingestion while keeping real evaluation open.

Each capture retains:

- Source, material, experiment/run/crystal IDs, usage permissions, original-file
  hash and relationships between originals, clips and annotations.
- Clock origin, frame timestamps, dropped/invalid frames, playback factor and
  synchronization with sensor/command records.
- Camera view/pose, image size, exposure/lighting, vessel/liquid conditions and
  scale/calibration where available. Keep unknowns explicit.
- Seed/secondary-object identities, masks/regions, visibility, occlusion,
  annotation confidence, event-onset interval and supporting evidence.
- Measured temperature and separately logged temperature commands; rotation and
  flow measurements/commands if available, with units and timing.

Begin with a fixed view and comparable poses. Calibration must account for the
intended measurement through vessel/solution; retain pixel measurements where a
physical conversion is unsupported. Rotation requires pose-aware comparison or a
fixed observation pose. Apparent width changes alone cannot identify asymmetry.

## Observable targets and pitch responses

| Target | Ground truth and initial behavior | Interpretation limit |
|---|---|---|
| Seed silhouette | Reviewed visible mask; track projected area/width/height and change over comparable views. | A single projection does not directly measure 3D volume, all face rates or internal quality. |
| Visible secondary crystal | Reviewed object identity, location and first-visible interval outside the seed; persistent candidate alert and ramp review/pause proposal. | Distinguish bubbles, debris, glare and fragments. First visibility is not microscopic nucleation time. |
| Projected growth asymmetry | Annotated comparable views and a defined projected metric; operator review and rotation/circulation proposal when supported. | Pose changes are not automatically a physical face-growth imbalance. |
| Visible inclusion candidate | Annotated region and visibility/confidence; preserve evidence, alert operator and mark location. | Visibility does not quantify optical performance or prove an internal inclusion. |
| Invalid/uncertain view | Occlusion, lost track, glare or poor visibility produces an inspection request and invalid measurement flag. | An unreadable image cannot establish a healthy crystal. |

Prioritize the seed and visible secondary objects. Pilot review freezes classes
that can be labeled reliably. Classes lacking real evidence stay unfinished pitch
requirements; synthetic demonstrations do not close their real-data gates. Fine
striations remain an optical-instrument extension, consistent with the pitch.

"Defect in the first hour" is a target to test. Record both elapsed growth-run
time and independently annotated first-visible event time. Report first-hour
coverage, missed events and alert delay. Accelerated video or clips without a
trustworthy run origin cannot establish that claim.

## Real and synthetic image datasets

Use Blender for the declared crystal/vessel/camera scene and OpenCV for image
processing/augmentation and the classical baseline. Reuse existing crystal
geometry where useful. The Matplotlib simulator demo is not an annotated optical
training dataset.

Generate masks, event metadata and temporal scene identity. Distinguish projected
geometric masks from visible annotation boundaries through transparent/refracting
materials; inspect images and labels together. Vary lighting, reflections,
background, blur, bubbles, occlusion, scale and pose under a recorded design.
Synthetic defect placement is a rendering assumption, not a measured defect
response to temperature or agitation.

Split whole real experiments and synthetic base scenes/trajectories before
sampling frames or generating derivatives. Keep clips, augmentations and related
views from a source run/scene in one split. Exclude final real test runs from
visual tuning, annotation-guideline development, threshold selection, early
stopping and render matching. Pilot runs remain development data. One real run
can support a pilot, not independent-run training/validation/test claims.

Audit source identity, masks, classes, run/scene membership and split overlap.
Retain negative observation time and distractors. Dataset size follows independent
run/event coverage, rather than a target count of correlated frames. Image data
needs its own declared split; the tabular dataset does not supply image labels.

## Model and evaluation contract

Establish an OpenCV measurement/event baseline, then compare the pitch's
YOLOv8-seg. Choose the small-model size on development data and available hardware.
Pin package versions, pretrained-weight identity, preprocessing, training settings
and seeds. Pretraining supplies initialization; project labels still need review.

Track identity and temporal persistence after segmentation. Evaluate event alerts
as temporal decisions, not just segmentation confidence. Compare real-only,
synthetic-only and mixed training when data supports it; report real and synthetic
results separately. Use development data for fitting and model selection.

Pilot review must specify and then freeze:

- Mask IoU/Dice and projected measurement error against independent annotations,
  with uncertainty and physical units only where calibration supports them.
- Event onset, persistence/debounce, matching and missed/duplicate-alert rules.
  Report per-class precision/recall, delay from visible onset and false alerts
  per observed negative hour.
- Per-run/view/material/lighting results and independent run/event counts.
  Estimate uncertainty at the independent-run level when enough runs exist.
- Observation cadence, end-to-end latency and resource use on named hardware,
  including deliberate frame sampling and wall-clock processing time.
- Numeric budgets for measurement error, event recall, false alerts, alert delay
  and latency. Set these after pilot visibility/annotation review and before
  validation model selection and final test. They remain open until that review.

Freeze model, preprocessing and decision thresholds before final test scoring.
Missing real events or negative hours limit the evaluated claim. A high mask score
alone does not establish reliable early warnings or improved crystal yield.

## Controller and operator contract

Temperature PID uses measured temperature, a setpoint/ramp, an update clock,
saturation, anti-windup and characterized sensor/actuator timing. The supervisory
layer consumes valid visible-state events and supported process measurements.
Start with bounded event rules as a baseline. The adaptive agent must then have
an explicit changing state/policy: use recent observation history and a declared
short-horizon forecast to choose among permitted setpoint/ramp changes. Specify
the forecast target/horizon, decision objective, state updates and action bounds
before evaluation. Compare fixed recipe, PID-only, rule-supervised and adaptive
variants. A fixed alert-to-command rule alone does not complete this deliverable.

Forecast targets must be observable and labeled: for example projected growth
over a pilot-defined horizon. Compare with persistence using whole held-out runs.
An event-risk target additionally requires independently labeled future events,
causal input windows and enough negative/event coverage. Predicting an event
before visibility is a separate claim from detecting its first visible evidence.
A learned predictor can supply the forecast when its data supports evaluation.
Action-conditioned prediction requires supported action histories and responses.
Do not insert the current tabular model's idealized 3D inputs into a camera path
without an independently validated measurement adapter.

Log proposal time, evidence/event IDs, command, units, limits, approval/override
state, dispatch, acknowledgment and measured response. Keep recommended, applied
and observed values distinct. Exercise missing feedback/frames, saturation,
uncertain events and operator override. Define fallback behavior against the
actual process/equipment before physical use; do not invent corrective setpoints.

The current engine's temperature/aggregate-agitation interface retains its model
units. It does not distinguish physical rotation from pump flow, calibrate a
hardware PID or supply a validated quality response. Scripted visual anomalies
can test a software loop with logged commands, but cannot establish defect
prevention. Real replay supports shadow evaluation; physical action benefit
requires separate experiments and supported limits.

## Current implementation sequence

The adapters, baseline and importer are implemented. The annotation workflow and
Blender/OpenCV renderer are local additions with a smoke dataset, not a completed
real-training/evaluation dataset. Follow the current batches in the
[roadmap](../roadmap.md):

1. Checkpoint those additions, review the six proposed real outlines, expand a
   bounded pilot and freeze annotation/evaluation rules. Declare a synthetic
   cohort and identify missing physical capture dependencies.
2. The YOLOv8-seg training/evaluation entry points and first synthetic software
   checks in `src/models/` are implemented, including the grouped synthetic pilot
   and synthetic seed comparison. Compare on reviewed real data when available.
3. Learned seed-role tracking and inspection transitions are implemented. Extend
   identity to other candidates, add supported face/edge measurements and temporal
   event decisions with evidence and held-out evaluation.
4. Fill `src/controllers/` with temperature PID, rule supervision and bounded
   adaptation. Integrate an operator view in `src/dashboard/` and traceable
   simulated/shadow actions before physical use.

Preserve the simulator/data contracts and introduce optional training dependencies
reproducibly. Synthetic-only fitting can test the pipeline while real review is
pending. It does not close real accuracy, first-hour or physical-control gates.

## Implementation references

The [original pitch](../Pitch_KDP.pptx) defines scope. Official references checked
on 2026-11-11:

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and
  [instance segmentation](https://docs.ultralytics.com/tasks/segment/) describe
  the first planned segmentation implementation. Explicitly select YOLOv8 weights;
  generic current examples may use a different model generation.
- [Qiskit Nature ground-state solvers](https://qiskit-community.github.io/qiskit-nature/tutorials/03_ground_state_solvers.html)
  describes VQE and classical-reference workflows for the separate extension.
  It does not supply KDP impurity-tolerance calibration.

These references supply implementation guidance, not project-specific labels,
controller-response evidence or completed validation.
