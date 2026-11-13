# CrystaLab roadmap — 2026-11-13

## Goal and order of work

Deliver the system in [Pitch_KDP.pptx](Pitch_KDP.pptx), especially slides 5–9:
camera observations, YOLOv8-seg perception, visible-event interpretation,
temperature PID plus an adaptive supervisor, and traceable actions with an
operator involved. The intended actions are temperature-ramp changes, rotation
and circulation. The demonstration must identify its material and evidence level.

**The next milestone is reviewed real-image/event evidence and validation of the
adaptive objective/action response.** Candidate decisions, temperature PID,
event-rule proposals, operator replay and bounded adaptive software pauses are
implemented. The first projected-area forecast diagnostic exposed current-mask
bias and failed to beat future-area persistence on four shared development
windows. Reviewed measurements and action-responsive trajectories remain the
gates for a physically useful adaptive objective. Real review and equipment
characterization can progress alongside software integration.
The original MJPG diagnostic confirms the persistence advantage on 28 shared
validation windows. The later paired encoding experiment verifies lossless input
and finds that the expanded detectors' rate forecasts beat persistence on those
two synthetic validation families. Real accuracy, action response and the
live operator/equipment validation remain open.
The [adaptive software diagnostic](docs/ADAPTIVE_SUPERVISION.md) now implements
causal parameter updates and forecast-scaled bounded pauses, with a four-way PID
comparison. Its 80.71 px² shared validation forecast error is worse than plain
recent rate's 77.00 px². Shorter simulated pauses establish recipe-delay behavior;
the fixed video cannot establish defect-prevention benefit. No model is promoted.
Preserve the simulation and growth dataset as supporting assets. New CFD work or
an XGBoost benchmark does not close the perception/control requirements.

The [Vision MVP contract](docs/VISION_MVP.md) defines measurements, evaluation,
controller interfaces and interpretation limits. This roadmap defines execution
order. HPC and Qiskit remain explicit pitch deliverables with bounded experiments.

## What we do next

The [plain-language next steps](docs/NEXT_STEPS.md) summarize this sequence for
the current public-data-only setup.

The pitch's next useful demonstration is a video of the growing crystal with
measured outlines, supported alerts and a traceable proposed control response.
The foundations run; dependable real detection and adaptive control are unfinished.
The user currently has access to **public datasets only**. Prioritize documented
public KDP images, labels, clocks and process/action records; live equipment and
physical outcome validation remain later gates until suitable access/evidence exists.
The [additional public KDP review](docs/PUBLIC_KDP_SOURCE_REVIEW.md) retains surface
crack/crater leads without treating them as acquired growth-event training data.

**Current next action:** check one native KDP image in the existing review pack,
then finish that small development pack. In plain language, the outline is the
example answer showing the model where the crystal ends. An event label says
what appeared and when. Public images can supply inputs; unchecked computer
outlines cannot establish real accuracy. If no reviewer is available, retain
these as uncertain drafts and keep real supervised acceptance pending. Continue
independent software/supporting experiments within their declared limits.

1. **Use consistent synthetic measurements and characterize real capture.** The
   [paired encoding experiment](docs/VIDEO_ENCODING.md) now verifies exact FFV1
   input and matches native-image predictions. Use explicitly declared FFV1 for
   subsequent synthetic inputs. Expanded-detector rates beat persistence on two
   synthetic validation families; retain persistence and the original detector's
   negative result. Characterize the real camera's encoding before real acceptance.
2. **Check one real KDP outline, then the small pack.** The
   [KDP review pack](docs/KDP_VIDEO_DATA.md) now preserves six native frames from
   the acquired public movies, with phase/timing context, two uncertain outlines
   and four manual sheets. All 881 frames are indexed in one development family.
   The original six Leeds proposals remain available. A person corrects the outline and marks
   uncertainty. An event label adds what appeared and when. This small review
   establishes the example answers used to train and check the model. It needs
   no specialist job title; ambiguous defects remain unresolved until someone
   with relevant knowledge can assess them. No reviews have been invented.
3. **Complete the pitch-class comparison with reviewed labels.** A separate
   [source-fit silhouette study](docs/SOURCE_SHAPE_TRAINING.md) now trains synthetic,
   real-projection and mixed models on explicit weak targets. Mixed reaches 0.9255
   IoU on 12 real proxies; real-only misses all synthetic targets. This does not
   validate the pitch's defect classes. Expand training examples across preserved
   development runs, independently review evaluation labels, and compare the
   corresponding synthetic-only, real-only and mixed pitch-class fits.
   Report missed crystals, false alerts, outline error and supported event delay.
   Visible inclusions and face asymmetry remain open tasks with their own labels.
   The [inclusion-resolution diagnostic](docs/INCLUSION_RESOLUTION.md) now confirms
   memorization of three training inclusions and detects 1/6 validation inclusions
   at larger input, but doubles false secondary instances and worsens seed area
   error. Its matched video inclusion is in a training clip, with zero eligible
   validation inclusion events. Keep the candidate unpromoted.
4. **Use the operator demonstration while continuing data review.** The
   [replay console](docs/OPERATOR_REPLAY.md) now shows video, outlines, process
   values and proposal evidence, with explicit review, bounded simulated pause
   and manual resume. Receipts and each counterfactual response remain separate;
   real sources stay in shadow mode. The adaptive model/policy has a separate
   four-way software comparison. The [forecast view](docs/OPERATOR_FORECAST.md)
   now displays its source-bound forecast, state and recommendation and preserves
   that context with suggested/manual pause reviews. Next validate real capture,
   reviewed measurements and action response.
5. **Complete the remaining pitch evidence in order.** Evaluate an explicitly
   defined adaptive objective against persistence/PID/rules, gather relevant KDP
   runs with synchronized process/action logs and first-hour coverage, and complete
   remaining transport/chemistry validation. The first CPU/hybrid GPU comparison
   and a [hydrated Al VQE/classical solver diagnostic](docs/QUANTUM_IMPURITY.md)
   are complete within their bounded assumptions. Physical quality,
   yield and energy claims need measured outcome comparisons.

The [expanded vision experiment](docs/VISION_REFINEMENT.md) is complete as a
synthetic development comparison. Standard supervision cuts validation secondary
FP from 10 to 2 but introduces two misses; fullmask improves seed overlap further
while retaining more false alerts. Both miss all six validation inclusions. Neither
is automatically promoted. Its [checkpoint](docs/VISION_REFINEMENT_CHECKPOINT.json)
records 378 tests / 269 subtests and independent mask/event/forecast audits.

## Current evidence

| Component | Actual position | Remaining work |
|---|---|---|
| Simulation and numerical methods | Illustrative faceted growth and verified flow/transport components. Separate fixed-domain flow kernels are not coupled into current growth labels. | Physical calibration and any coupling needed by a specified experiment. |
| Tabular growth data | 360 episodes / 1,080 audited samples, frozen grouped splits and no quality labels. | Optional supporting forecast benchmark. These are not camera-training examples. |
| Video observation | Baseline/learned file/camera adapters, explicit clocks, seed role/uncertainty, raw masks, mapped source frames and candidate identities/inspection events during observation or on saved masks. | Reviewed real event accuracy and live equipment verification. Integrated decisions were tested on files; candidates are not confirmed defects. |
| Real vision sources | 2,123 alpha-LGA images from nine runs, 79 source 3D shape fits, three phase-specific KDP movies, nine small panels from a separate [KDP gel-growth study](docs/KDP_GEL_DATA.md), a separate ADP movie and 587 protein-crystal images with 4,986 published crystal/drop masks. | Reviewed Leeds/KDP pixel/event labels and representative independent KDP evaluation data. Source fits are not masks; publication panels retain their marks/domain/resolution limits; protein labels need a compatible task. |
| Annotation tooling | Six Leeds proposals, six native KDP video sheets and nine uncertain KDP publication sheets, source-linked context, previews, guarded review/export and a local polygon editor. All 881 public KDP movie frames have a native-pixel index. | Zero actual human reviews. Resolve clipping, co-seeded identities, publication marks and uncertain boundaries before accepted real training export; obtain independent real evaluation groups. |
| Optical synthetic images | Audited 18-frame smoke, 60-image/20-scene pilot and 96-frame/six-family scheduled temporal source. Combined export has 84 training / 44 validation / 28 reserved images. Two new fits and development replays are complete; reserved scenes remain unscored. | Relevant real diversity and inclusion coverage: the new validation inclusion has zero eligible proxy pixels. Refracted masks remain annotation proxies. |
| Real source-fit silhouette study | 66 author-shape proxies: 54 training / 12 validation. Three fixed-budget one-class fits and a static baseline; mixed real proxy IoU 0.9255 and synthetic 0.9078. Independent target/score reconstruction passes; real-only misses all synthetic targets. | Reviewed real pixel/event labels, relevant KDP coverage and the corresponding pitch-class comparison. Auxiliary weights are refused by the pitch observer; no model is promoted. |
| Inclusion diagnostic | Both small fits learn three training examples. A grouped 512 fit matches 1/6 synthetic validation inclusions and the one eligible training-clip inclusion event; it also doubles secondary false instances and worsens seed measurements. A training-only operator replay preserves the localized cue without reviews or commands. | Relevant reviewed inclusion diversity/negatives, grouped real evaluation and an acceptable overall error tradeoff. No validation inclusion-event coverage or model promotion. |
| Projected edge advance | A [reference/edge diagnostic](docs/PROJECTED_EDGES.md) audits 96 constructed RGB frames, two naive baselines, pose/reference refusals and an identical-image growth/translation ambiguity. Camera-motion imbalance is at most 0.2401 px; six real sheets produce no qualified measurements. | Source-supported attachment/pose/visibility, calibrated optics, reviewed edges/faces, learned-mask evaluation and operator integration. External fixture assertions are not automatic pose recognition or real accuracy. |
| Observable forecast | Lossless fullmask recent-rate MAE is 77.00 px² versus persistence 116.18 px² on 28 shared validation windows. Online adaptation gives 80.71 px² and does not beat recent rate. Parameters update only from matured observed masks; real replay remains unscored shadow output. | Reviewed real/action-labelled trajectories, capture characterization, acceptance budgets and supported action-response objective. No action-response growth model exists. |
| Learned perception, controllers and UI | Local training/evaluation, candidate analysis, PID/rules, operator replay and bounded adaptive pause comparison. The console now displays causal forecasts/state and preserves explicit suggested/manual duration reviews. Independent audits retain secondary misses and inclusion failures. | Review/evaluate real data, support face measurements, validate the adaptive action-response objective and characterize equipment/operator response. |
| HPC and Qiskit | A [CPU/hybrid Vulkan comparison](docs/VULKAN_FLOW_BENCHMARK.md) passes fixed-domain NS/solute checks but is about five times slower on GPU. Separate [heat/solute/buoyancy verification](docs/THERMAL_TRANSPORT.md) passes numerical convergence and budgets with declared test properties. A [hydrated Al precursor experiment](docs/QUANTUM_IMPURITY.md) independently audits six 8-qubit VQE starts against the same-space classical solution. HF already meets its frozen error target. | KDP thermal calibration, independent coupled-flow evidence, coupled CPU/GPU comparison and physical risk interpretation; KDP surface/reference chemistry, geometry/basis/active-space convergence and any impurity-tolerance claim. Neither extension establishes physical control or a speedup. |

The historical [vision checkpoint](docs/VISION_CHECKPOINT.json) records 244 tests /
130 subtests at its source revision. The new
[optical/annotation checkpoint](docs/OPTICAL_CHECKPOINT.json) records the 254-test
run, a targeted follow-up after an environment-metadata addition, and fresh optical
and real-source audits. Training work has a separate verification gate and
[experiment report](docs/VISION_TRAINING.md). The
[20-scene/60-image pilot](docs/VISION_PILOT.md) is complete and trained. Its
[checkpoint](docs/VISION_PILOT_CHECKPOINT.json) records the 270-test result,
source/model identities and failures. Human real-label review remains open.
The [learned replay checkpoint](docs/LEARNED_VIDEO_CHECKPOINT.json) adds 284-test
verification, original-frame mapping and actual seed-role/uncertainty replays.
The [temporal checkpoint](docs/TEMPORAL_CANDIDATES_CHECKPOINT.json) records 300
tests, audited candidate artifacts and independent optical/fixture event scores.
The [streaming checkpoint](docs/STREAMING_CANDIDATES_CHECKPOINT.json) adds
single-pass delivery, identical saved-analysis decisions, processing timing and
304-test verification. No physical camera or actuator was used.
The [temperature checkpoint](docs/TEMPERATURE_CONTROL_CHECKPOINT.json) adds the
first PID baseline, four declared thermal scenarios, 4,808 logged rows and an
independent sensor/command/plant/metric audit. This is software evidence only.
The [event-supervision checkpoint](docs/EVENT_SUPERVISION_CHECKPOINT.json) adds
actual optical/real shadow proposals and an audited three-variant software
comparison. Its scripted pause delays recipe phase; no quality benefit is modeled.
The [area-forecast checkpoint](docs/AREA_FORECAST_CHECKPOINT.json) adds the causal
cohort/model/replay experiment, its measurement-bias finding and independent metric
reconstruction. Final test families remain unscored.
The [image-review checkpoint](docs/ANNOTATION_REVIEW_CHECKPOINT.json) adds the
local still-image editor, explicit review/revision/export checks and 20 browser
interactions. Real proposals were previewed read-only and remain unreviewed.
The [published-mask checkpoint](docs/PUBLISHED_MASKS_CHECKPOINT.json) adds a
source-reconstructed auxiliary real image/mask index. It preserves author splits
and selected protein-crystal semantics without accepting them as KDP pitch labels.
The [longer optical checkpoint](docs/OPTICAL_TEMPORAL_CHECKPOINT.json) adds the
96-frame schedule, frozen-model event/forecast execution, class-specific usable
deadline coverage and independent numeric/mask reconstruction. It preserves three
false crystal alerts, a missed inclusion and the negative forecast result.

## Pitch requirements and completion evidence

| Pitch promise | Required deliverable |
|---|---|
| Camera through solution; shape and faces | Measured seed silhouette and supported visible edges/faces, with visibility/pose limits. The initial silhouette pilot does not complete face-specific interpretation. |
| Parasitic nucleation | Track a newly visible secondary-crystal candidate, retain onset evidence and propose a ramp review/pause. First visibility does not establish microscopic nucleation time. |
| Asymmetric face growth | Compare valid views/poses over time, define a measurable asymmetry criterion and propose rotation/circulation review. Pose changes must not masquerade as growth. |
| Visible inclusion | Localize supported visible evidence, mark the region and alert the operator. Optical quality remains a separate measurement. |
| PID plus adaptive agent | Temperature feedback PID and a bounded supervisor that updates its state/policy using observation history and a declared forecast. Compare with simpler baselines. |
| Real plus synthetic images | Audited image/label provenance, grouped splits, mixed-training experiment when labels support it, and separately reported real evaluation. |
| Detection in the first hour | Relevant real growth runs with trustworthy run origin, observed first-hour coverage, independently labeled visible onset, misses and alert delays. |
| HPC and Qiskit | A comparable CPU/GPU transport benchmark and a chemically justified impurity-cluster VQE/classical comparison. |
| Better yield, energy use and quality | Independent physical outcome comparisons. A software demonstration alone leaves these impact claims unverified. |

## Batch 1 — close the pilot data gate

This is the immediate work. Data review and training infrastructure can progress
alongside each other; human review remains necessary for accepted real labels.

1. **Checkpoint the existing tools — verified.** Audit the revised optical dataset and real
   source index, inspect development RGB/mask overlays, and verify the current
   annotation/rendering changes with appropriate regression tests. Save source
   hashes, results and failures. Do not rewrite historical checkpoint evidence.
2. **Show and review the six existing seed proposals.** Explain the original image
   and proposed outline in plain language before choosing a reviewer. Correct
   visible boundaries, record uncertain cases and save actual review receipts.
   The [local editor](docs/ANNOTATION_REVIEW_UI.md) now supports this directly,
   including uncertain drafts and corrections that preserve earlier receipts.
   Use this to settle annotation rules before preparing a larger bundle. The
   agent must not invent human reviews.
3. **Expand a bounded development pilot.** Start with roughly 30–50 selected images
   across the existing training families 12, 14 and 20, including difficult views.
   Keep the 20-series replicates together. This count is an annotation workload,
   not a claim of sufficient training data. Inspect KDP development clips for
   supported targets and annotate contiguous windows around any candidate event,
   including negative observation periods and distractors. Sparse stills alone
   cannot label onset, persistence or false-alert hours.
4. **Freeze the task and evaluation protocol.** Define visible boundaries,
   identities, truncation/uncertainty, event-onset intervals, comparable poses and
   clock evidence. Start with seed silhouette and secondary candidates when
   labelable. Record missing face/asymmetry/inclusion coverage explicitly. Set
   numeric measurement, event-recall, false-alert, delay and latency budgets after
   pilot review and before validation selection/final testing.
5. **Declare the first synthetic training cohort.** Freeze new base-scene IDs and
   splits before rendering. Vary illumination, reflections, bubbles, occlusion,
   camera/seed pose and target presence. Retain geometric masks, optical proxies
   and RGB for inspection. Add explicit object/onset metadata for temporal tests;
   current object-presence frames do not yet constitute an evaluated event dataset.
   Choose a small measured batch first, then scale according to failures and
   learning curves rather than generating thousands of similar images.
   The first 20-scene/60-image pilot and its 24-frame development inspection are
   complete. The three-frame scene-013 event check is implemented; it retains two
   false startup inspections and an unmatched inclusion, with insufficient
   follow-up for full-deadline recall. Longer relevant temporal coverage remains open.
6. **Write the capture and process-data checklist.** Establish available camera,
   thermostat, temperature logs, rotation/pump access and a reviewer. Plan fixed
   observation poses, scale, synchronized clocks, run origin and measured/applied
   command logging for a future available setup. Current user access is public
   datasets only; do not assume a local growth rig or camera is available.

Preserve the frozen Leeds validation/test runs. No visual tuning, annotation-rule
selection or render matching on final test images. Evaluation labels need a
separately frozen review protocol and must remain outside training/model selection.
Alpha-LGA and ADP support development; neither substitutes for KDP evidence. The
public KDP clips have specific phase/optical conditions and do not establish a
representative KDP tank-control dataset. Preserve source permissions and attribution.

**Exit:** reviewed pilot examples, a class/coverage register, frozen splits and
metrics, checked synthetic examples, source-matched software verification and a
list of missing capture/equipment dependencies. A synthetic-only first fit may
proceed as a labeled software experiment while real review is pending.

## Batch 2 — first YOLOv8-seg experiment

- Implement a causal image/label loader and training/evaluation entry points using
  `src/models/`, with explicit YOLOv8-seg weights, optional pinned dependencies,
  preprocessing, seeds, hardware identity and weight/configuration hashes.
  Implemented, with a three-epoch synthetic smoke fit and separate two-image
  overfit diagnostic. Neither closes real-data performance or temporal alerts.
- Run a tiny overfit check to verify export/label/training plumbing. Follow it with
  a bounded grouped experiment. A successful fit is distinct from generalization.
  The first grouped synthetic fit is complete; final tests remain unscored.
- Compare the seed results with the existing OpenCV baseline on the same reviewed
  inputs. Report mask IoU/Dice and projected measurement errors, visibility
  failures, per-class coverage, per-run results and inference latency.
- Compare synthetic-only, real-only and mixed training when reviewed independent
  data supports each comparison. Report real and synthetic scores separately.
  Real classes without reviewed examples remain unevaluated.
- Freeze the selected model and preprocessing before final held-out evaluation.
  Record image/label counts and independent scene/run counts in the model card.

**Exit:** a reproducible model artifact, honest baseline comparison, error examples
and one inspectable video replay. Synthetic performance or an alpha-LGA score
alone cannot close the KDP detection requirement.

## Batch 3 — tracking, faces and temporal alerts

- Seed-role association and inspection transitions now run through the learned
  video path. The real development replay exports 30 estimates with 11 inspection
  frames; masks/events remain unreviewed. See [Learned video](docs/LEARNED_VIDEO.md).
- Causal candidate identities and persistent inspection decisions now analyze
  saved learned replays, with explicit startup/new/unknown origins, unique mask
  association, missing observations and bounded retained identities. See
  [Temporal candidates](docs/TEMPORAL_CANDIDATES.md). The same recorder now runs
  during observation, with per-frame evidence flushed before the next prediction.
  Verify camera/equipment timing separately; current integrated runs use files.
- Measure projected growth from comparable views. Add visible-face/edge annotations
  or a justified measurement method before claiming face-specific asymmetry.
  The [anchored projected-edge diagnostic](docs/PROJECTED_EDGES.md) now measures
  directional silhouette supports in a registered image frame and emits one
  review-only fixture proposal. It preserves an exact growth/translation ambiguity
  and requires external attachment/pose assertions. The six real KDP sheets lack
  those inputs; visible-face identification and real/asymmetry evaluation remain open.
- Implement persistent secondary-candidate alerts, supported asymmetry decisions
  and localized inclusion alerts. Glare, bubbles and poor views must be represented
  in evaluation; uncertainty requests operator inspection.
- Save evidence frames, track/event IDs, onset intervals and decision timestamps.
  Evaluate per-class precision/recall, delay from visible onset, duplicates/misses,
  false alerts per observed negative hour and end-to-end latency.
  Saved-replay evidence and event matching are implemented. Current optical scores
  preserve bubble false inspections and an unmatched inclusion; real event scores,
  physical negative hours, first-hour coverage and numeric budgets remain open.
  The longer frozen diagnostic adds three crystal matches and three false
  inspections, plus one missed inclusion. It separates elapsed clip coverage from
  uninterrupted class-usable deadlines and exposes absent validation inclusion truth.
- Freeze thresholds and persistence settings on development/validation data before
  final test. Keep unsupported classes open rather than relabeling synthetic cases
  as real detections.

**Exit:** video, masks/tracks, supported measurements and timestamped alerts within
the pilot budgets on the relevant independent real observations. Separately report
any missing KDP material, first-hour, event or negative-time coverage.

## Batch 4 — temperature PID and adaptive supervision

1. Define separate observations and commands for temperature, rotation and flow,
   with units, clocks, bounds, actuator delays and feedback/acknowledgment fields.
   The simulator's aggregate agitation input does not represent two calibrated
   physical actuators. Establish literature-based candidate operating limits and
   their experimental applicability before physical use.
2. Implement temperature PID with saturation, anti-windup and missing/stale-feedback
   behavior. Compare fixed recipe with PID on a declared simulated/bench plant.
   **Software baseline implemented:** [temperature control](docs/TEMPERATURE_CONTROL.md)
   retains explicit unit/run/clock records, bounded PID, delayed simulated
   acceptance/application, fault/manual cases and independent trajectory audit.
   Its chosen limits and plant parameters are uncalibrated test assumptions.
3. Add event-rule supervision as the baseline, then a bounded adaptive supervisor.
   **Rule baseline implemented:** [event supervision](docs/EVENT_SUPERVISION.md)
   retains source-bound review/expiry/location evidence on actual development
   clips. A declared fixture-clock map and scripted decision connect one bounded
   pause through PID to delayed simulated actuation; real shadow inputs issue no
   numerical commands or human review receipts.
   Declare the forecast target, horizon, objective, updated state/policy and allowed
   actions. Compare its forecast with persistence using causal histories and whole
   held-out runs. Prediction before visible onset requires its own future-event
   labels; reactive alerts alone do not establish it.
   **First forecast diagnostic implemented:** [projected area](docs/AREA_FORECAST.md)
   uses original optical proxies and learned causal inputs across frozen families.
   Its calibrated rate does not beat future-area persistence; do not promote it
   as validated adaptive control. Review measurement bias and collect longer
   trajectories, including action responses when action-conditioned forecasting
   is intended. The later [adaptive diagnostic](docs/ADAPTIVE_SUPERVISION.md) now
   updates a bounded coefficient from matured observed masks and scales candidate
   pause duration using its forecast/error state. Its fixed experiment loses to
   plain recent rate. The source video has no response to thermal commands;
   an action-response objective and physical benefit remain unfinished.
4. Compare fixed recipe, PID-only, rule-supervised and adaptive variants. Existing
   growth data may support a bounded forecast study, but its idealized 3D inputs
   need a validated adapter before use with camera measurements. Scripted anomalies
   can exercise software decisions without proving defect prevention.
5. Connect perception through an actuator adapter. Log evidence, proposal, approval/
   override, dispatched command, acknowledgment and measured response separately.
   Exercise invalid frames, uncertain alerts, stale feedback and saturated commands.

**Exit:** auditable software closed loop and real-video shadow operation with a
simulated actuator clearly identified. Physical actuation requires characterized
equipment, approved operating limits and supported action-response evidence.
Real action benefit requires independent physical comparisons.

## Batch 5 — operator demonstrator and physical evidence

Build a minimal operator view alongside the control integration: video/masks,
process readings, event evidence, controller state, recommendations and override.
Show normal growth, each supported pitch event and an invalid/uncertain view.
Demonstrate ramp review/pause, rotation/circulation review and inclusion marking
at the available evidence level. Preserve original clocks and label accelerated
playback. Package the model/evaluation report, failures, commands and demo video.

The [operator replay](docs/OPERATOR_REPLAY.md) now implements video/process
display, source-linked reviews and simulated pause/manual resume, retaining each
result. Real-video shadow display has no process measurements or numerical
commands. Automated fixture checks create no human review claims. The linked
[forecast/adaptive view](docs/OPERATOR_FORECAST.md) is implemented with source-bound
state and preserved manual/suggested pause review. Supported faces/asymmetry,
rotation/flow actions, relevant real event accuracy and live equipment interaction
remain open.

A complete software demo still needs independent physical KDP runs to establish
first-hour detection and effects on growth/quality. Start capture planning in
Batch 1 because real runs and defect coverage may outlast software development.
Do not equate model training, synthetic events or shadow recommendations with
verified physical control or improved yield/energy use.

## HPC and Qiskit — included, bounded extension track

Scope these experiments during Batch 1. Execute them alongside later core batches
when their inputs and hardware are available, and include their status in the
final demonstration package.

**HPC:** choose one existing verified flow/transport reference case and a declared
risk-map output. Record hardware, source identity, accuracy, memory and wall time.
Compare a selected GPU implementation with the same CPU case. The pitch also
requires Navier–Stokes plus heat/mass transfer: identify which couplings are
implemented and validated, and keep the remainder open. A CPU timing or a visual
risk map alone does not close this deliverable. A first
[CPU/hybrid Vulkan experiment](docs/VULKAN_FLOW_BENCHMARK.md) now runs on the local
AMD Radeon 860M: 24 fresh-process runs pass numerical checks, with three measured
repeats per backend/grid. GPU stepping is about five times slower at 16³–32³.
Keep CPU as the default. The stored concentration-excess map is uncalibrated.
The separate [thermal verification](docs/THERMAL_TRANSPORT.md) now implements
heat/solute transport and Boussinesq feedback, with nine numerical cases and an
independent native-field audit. All constants are declared test designs;
compatible KDP properties and boundary/flow measurements remain to be verified.
This is not a new coupled GPU timing comparison. Interfacial growth, physical
risk interpretation and outcome validation remain open.

**Qiskit:** define one chemically justified small KDP reference/impurity cluster,
including geometry, charge, spin, basis and active space. Compare VQE with a
classical solution of the same Hamiltonian and report convergence/resources.
Fe/Cr/Al are pitch candidates requiring chemical justification. A generic molecule
can test the tools but cannot close this extension. Deriving an impurity tolerance
threshold from cluster energies requires separate evidence. Keep this study
outside the online control loop.
The first [Al aquo/phosphate precursor diagnostic](docs/QUANTUM_IMPURITY.md) now
implements this method comparison using source-motivated species and designed,
fixed geometries. Six starts pass independent fermion/matrix/circuit checks;
maximum VQE gap is 8.01e-9 Ha. Both small active spaces already meet the 1 mHa
target at HF. This completes a local solver experiment, not a pristine-KDP versus
solid-defect comparison. No solvent/lattice, geometry relaxation, active-space
convergence, adsorption energy or impurity threshold is established. Do not
subtract the two unequal-stoichiometry total energies.

## Four-week target and dependencies

| Target window from restart | Deliverable |
|---|---|
| Weeks 1–2 | Batch 1, training infrastructure and first bounded fit; capture planning and extension experiment definitions. |
| Week 3 | Evaluated perception/temporal alerts to available coverage; PID and initial adaptive integration. |
| Week 4 | Integrated operator demo, comparisons, recorded failures and bounded HPC/Qiskit reports where prerequisites permit. |

This preserves the pitch's software cadence. Dates for independent growth runs,
rare-event coverage, GPU work and chemistry validation depend on actual access
and evidence. If a prerequisite is missing, deliver the supported subset and keep
the unfulfilled pitch requirement visible. Do not weaken evaluation to fit a date.

## Preserved supporting work and working rules

Retain the [growth dataset](docs/SYNTHETIC_DATASET_V1.md), its frozen groups and
[numerical evidence](docs/SYNTHETIC_DATA_CHECKPOINT.json). Schedule the constant,
persistence/recent-rate and small XGBoost comparison only when it supports the
declared forecast. Follow the causal inputs and grouped evaluation in
[AI readiness](docs/AI_READINESS.md). Additional bulk tabular generation and CFD
expansion need a specific demonstrated need.

Preserve the existing package structure and explain changed files. Every new or
changed code file starts with a purpose summary. Work in small reproducible batches,
retain failed evidence and keep raw data out of Git. The active implementation
goal authorizes **local commits**. **Never push without the user's explicit
approval.** Current session instructions take precedence over older authorization
notes. Historical results remain in
[development history](docs/DEVELOPMENT_HISTORY.md) and their original checkpoints.
