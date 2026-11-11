# CrystaLab roadmap — 2026-11-11

## Goal and order of work

Deliver the system in [Pitch_KDP.pptx](Pitch_KDP.pptx), especially slides 5–9:
camera observations, YOLOv8-seg perception, visible-event interpretation,
temperature PID plus an adaptive supervisor, and traceable actions with an
operator involved. The intended actions are temperature-ramp changes, rotation
and circulation. The demonstration must identify its material and evidence level.

**The next milestone is a reviewed real-image/event pilot and an explicit adaptive
forecast/state policy.** Candidate decisions, temperature PID, event-rule shadow
proposals and a bounded fixture-to-PID pause are implemented. Define the observable
forecast/objective and compare adaptation with the current baselines. Real review and equipment
characterization can progress alongside this software work.
Preserve the simulation and growth dataset as supporting assets. New CFD work or
an XGBoost benchmark does not close the perception/control requirements.

The [Vision MVP contract](docs/VISION_MVP.md) defines measurements, evaluation,
controller interfaces and interpretation limits. This roadmap defines execution
order. HPC and Qiskit remain explicit pitch deliverables with bounded experiments.

## Current evidence

| Component | Actual position | Remaining work |
|---|---|---|
| Simulation and numerical methods | Illustrative faceted growth and verified flow/transport components. Separate fixed-domain flow kernels are not coupled into current growth labels. | Physical calibration and any coupling needed by a specified experiment. |
| Tabular growth data | 360 episodes / 1,080 audited samples, frozen grouped splits and no quality labels. | Optional supporting forecast benchmark. These are not camera-training examples. |
| Video observation | Baseline/learned file/camera adapters, explicit clocks, seed role/uncertainty, raw masks, mapped source frames and candidate identities/inspection events during observation or on saved masks. | Reviewed real event accuracy and live equipment verification. Integrated decisions were tested on files; candidates are not confirmed defects. |
| Real vision sources | 2,123 alpha-LGA images from nine runs, 79 source 3D shape fits, three phase-specific KDP movies and a separate ADP movie. | Reviewed pixel/event labels and representative independent KDP evaluation data. Source fits are not segmentation masks. |
| Annotation tooling | Six editable seed proposals from training run 12, previews and guarded review/export code. | Zero actual review receipts. A reviewer must inspect and correct labels before accepted real training export. |
| Optical synthetic images | Audited 18-frame smoke and 60-image/20-scene pilot, development visual inspection and independently exported construction-event labels for one three-frame scene. | Relevant real/image diversity and longer temporal coverage. Refracted masks remain annotation proxies. |
| Learned perception, controllers and UI | Local training/evaluation, grouped pilot, candidate analysis, explicit process records, PID, thermal tests, event-rule shadow proposals and scripted fixture-to-PID integration. Bubble/inclusion errors remain. Dashboard is a placeholder. | Real evaluation, equipment timing, face measurements, adaptive forecast/policy and operator demonstration. |
| HPC and Qiskit | CPU numerical reference work exists. No GPU benchmark or impurity-cluster VQE implementation. | Both bounded extension experiments and their reports. |

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
   command logging. User-owned equipment access is currently unconfirmed.

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
- Implement persistent secondary-candidate alerts, supported asymmetry decisions
  and localized inclusion alerts. Glare, bubbles and poor views must be represented
  in evaluation; uncertainty requests operator inspection.
- Save evidence frames, track/event IDs, onset intervals and decision timestamps.
  Evaluate per-class precision/recall, delay from visible onset, duplicates/misses,
  false alerts per observed negative hour and end-to-end latency.
  Saved-replay evidence and event matching are implemented. Current optical scores
  preserve bubble false inspections and an unmatched inclusion; real event scores,
  physical negative hours, first-hour coverage and numeric budgets remain open.
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
risk map alone does not close this deliverable. GPU access remains unconfirmed.

**Qiskit:** define one chemically justified small KDP reference/impurity cluster,
including geometry, charge, spin, basis and active space. Compare VQE with a
classical solution of the same Hamiltonian and report convergence/resources.
Fe/Cr/Al are pitch candidates requiring chemical justification. A generic molecule
can test the tools but cannot close this extension. Deriving an impurity tolerance
threshold from cluster energies requires separate evidence. Keep this study
outside the online control loop.

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
