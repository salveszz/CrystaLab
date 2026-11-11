# CrystaLab roadmap — 2026-11-11

## Objective and pitch alignment

Build the system proposed in [Pitch_KDP.pptx](Pitch_KDP.pptx): observe a growing
crystal through a camera, identify visible changes, decide how to respond and
record the resulting action, with an operator involved when evidence is uncertain.
Temperature ramps, rotation and circulation are the intended process controls.
The primary product milestone is a vision-led demonstrator.

The pitch's core sequence is camera -> perception -> PID/adaptive decision ->
action. Real plus synthetic **images** support perception. Detailed HPC/CFD and
the Qiskit impurity-cluster study are separate extensions. The existing tabular
growth dataset supports a later prediction experiment; its completion does not
complete the image-data or defect-detection milestones.

This plan replaces the earlier immediate priority of training XGBoost. It keeps
the completed simulation/data work and its declared limits. The implementation
contract and evidence gates are in [Vision MVP](docs/VISION_MVP.md).

## Current position

- [x] Illustrative 3D faceted growth, liquid transport, conservative local uptake,
  moving-boundary displacement and numerical verification.
- [x] Separate fixed-domain flow/solute, pressure, viscosity and momentum work.
  These operators do not generate the current growth labels.
- [x] Bounded scripted controls, causal delayed/noisy sensor proxies, exact growth
  labels, grouped splits, source snapshots and independent dataset auditing.
- [x] Accepted 32^3 focused growth-data grid/timestep protocol under the declared
  2% refinement gate, with coarser failures and limits retained.
- [x] Local working dataset: 360 episodes / 1,080 samples, 60 concentration groups,
  six recipes and 648/216/216 frozen train/validation/test rows.
- [x] Review: 227 tests pass across the full run and the nine sandbox-affected
  multiprocessing reruns; the working dataset passes a fresh reconstruction audit.
- [ ] Camera ingestion, annotated image/video data, trained perception, alert UI,
  PID/adaptive control, physical integration and Qiskit implementation.

The growth dataset has 720 fully observed rows and 360 startup rows with missing
sensor/history inputs. It uses illustrative units, one fixed growth law and no
quality targets. Keep its [card](docs/SYNTHETIC_DATASET_V1.md),
[report](docs/SYNTHETIC_DATASET_V1.json) and raw local files intact.

## Milestone 0 — camera and evidence contract

- [ ] Inventory available real footage/capture equipment, material, independent
  growth runs, temperature logs and possible actuator access. Unknown access
  remains an explicit dependency; no purchase or hardware capability is assumed.
- [ ] Define timestamp origin, camera view/pose, scale, lighting, observation
  interval and the measurements actually visible through the vessel.
- [ ] Prioritize seed silhouette and visible secondary crystals; check pilot
  footage before freezing the detectable classes. Include projected asymmetry
  and visible-inclusion annotation rules, plus uncertain/unobservable cases.
- [ ] Define run IDs, source provenance, annotation format, split ownership and
  an event-onset labeling procedure. Reserve whole independent real runs for test.
- [ ] Review pilot visibility and freeze numeric error, false-alert, detection-delay
  and latency budgets before selecting models on validation or scoring final test.

**Exit:** reviewed capture/annotation/evaluation protocol and an inventory of
usable pilot clips. If real clips are unavailable, synthetic implementation can
start, while real-data and early-defect claims remain blocked by that dependency.

## Milestone 1 — video baseline and real/synthetic image data

- [ ] Implement recorded-video ingestion and a camera adapter with common
  timestamps, calibration, frame validity and run metadata.
- [ ] Add an OpenCV seed-measurement baseline and temporal tracking; export an
  overlay with projected area/dimensions and their measurement validity.
- [ ] Label a real pilot including negative periods and optical distractors;
  establish whether candidate objects are observable and consistently labelable.
- [ ] Build the pitch's Blender/OpenCV image generator with masks and temporal
  events. Reuse existing crystal geometry where useful; add vessel optics,
  lighting, reflections, bubbles, occlusion and camera variation as declared
  rendering assumptions. Inspect rendered masks against visible image boundaries.
- [ ] Separate synthetic scenes/trajectories before frame sampling; keep all
  real frames and derivatives from a run in one split. Audit sources and labels.

**Exit:** reproducible image-data manifest, real annotation examples, synthetic
examples with checked labels, and an inspectable baseline replay. Synthetic-only
results establish software behavior until independent real clips are available.

## Milestone 2 — YOLOv8-seg and event alerts

- [ ] Fine-tune a small YOLOv8-seg model using the agreed classes and compare with
  the OpenCV baseline. Compare real-only, synthetic-only and mixed training when
  the available independent real runs support those comparisons.
- [ ] Track the seed and candidate secondary objects over time. Derive projected
  growth/asymmetry from valid comparable views; distinguish pose changes from growth.
- [ ] Implement persistent-event rules for visible secondary-crystal candidates,
  supported projected asymmetry and visible inclusions. Uncertain images produce
  an inspection request; an inclusion alert preserves its evidence and location.
- [ ] Evaluate mask/measurement errors, event precision/recall, false alerts per
  observed hour, time from annotated visible onset to alert and end-to-end latency.
- [ ] Freeze model, preprocessing, thresholds and persistence settings before
  final real-run evaluation; publish synthetic and real results separately.

**Exit:** video -> masks/tracks -> measurements -> timestamped evidence alerts on
held-out real runs within the Milestone 0 budgets. Without enough real events or
negative observation time, report the missing coverage and withhold that claim.
The pitch's first-hour claim requires real, timestamped first-hour observations.

## Milestone 3 — PID, bounded adaptation and action logging

- [ ] Define independent measured process signals and equipment commands. Bath
  temperature feedback requires a temperature sensor; camera measurements supply
  the supported visible-state signals. Record actuator units, limits and delays.
- [ ] Implement temperature PID with saturation, anti-windup, timing and explicit
  missing-feedback handling. Add an inspectable event-rule supervisor as a baseline.
- [ ] Implement a bounded adaptive supervisor using observation history and a
  declared short-horizon forecast to select permitted ramp adjustments and,
  when supported, rotation/circulation proposals. Compare its forecast with
  persistence on whole held-out runs. State exactly what adapts; fixed event
  rules alone do not complete the adaptive/predictive-control deliverable.
- [ ] Compare the fixed recipe, PID-only, rule-supervised and adaptive variants on
  declared simulation/bench cases. Scripted visual events test decision plumbing;
  they do not establish that an action fixes a physical defect.
- [ ] Connect perception, alerts, decisions and action acknowledgments in the
  local demonstrator. Log the evidence, proposal, approval state, applied command
  and measured response. Include operator override and uncertain-state behavior.
- [ ] Exercise real video in shadow mode. Advance to limited physical actuation
  only with characterized hardware, approved operating limits and independent
  assessment of the intended action response.

**Exit:** an auditable closed software loop plus real-video shadow replay. State
separately whether the actuator is simulated, on a bench or attached to a KDP run.
Report forecast and adaptive-controller performance separately. Reactive alerts
do not establish prediction before a defect becomes visible. Physical quality/
growth benefits require their own experimental comparison.

## Milestone 4 — integrated pitch demonstrator

- [ ] Deliver video/camera view, masks/tracks, projected measurements, process
  readings, event evidence, controller state and operator action controls.
- [ ] Show normal growth, an annotated visible event and an uncertain/invalid-view
  case. Preserve the original timing; label any accelerated playback explicitly.
- [ ] Demonstrate the pitch's three responses at the evidence level supported:
  secondary-crystal candidate -> ramp review/pause proposal; projected asymmetry
  -> rotation/circulation review; visible inclusion -> operator alert and marking.
- [ ] Package the evaluation report, representative failures, reproducible run
  commands and a demonstrator video. Distinguish synthetic, real replay and any
  live physical results throughout.

**Exit:** a reviewable demonstration of seeing, interpreting and responding, with
measured performance and visible operator involvement. Do not infer optical
quality, energy savings or reduced scrap from software completion.

## Pitch extensions — bounded deliverables

**HPC:** profile an existing verified flow/transport case and data-generation run;
report hardware, source identity, accuracy, memory and wall time. Establish a CPU
reference, then benchmark a selected GPU implementation against the same case if
hardware is available. GPU Navier-Stokes/heat/mass and risk maps remain an explicit
extension gate until implemented and validated; existing CPU flow kernels do not
complete it. Choose new physics/coupling only for a specified output or discrepancy.

**Qiskit:** define one chemically justified small reference/impurity cluster, with
charge, spin, geometry, basis and active space. Compare VQE with a classical
reference on the same Hamiltonian and report convergence/resources. Fe/Cr/Al are
pitch candidates, not preselected validated chemistry. A generic molecule can
test the tooling but cannot complete the KDP-impurity extension. Connecting cluster
energies to an impurity tolerance threshold needs separate evidence. This study
does not issue online control commands.

Both extensions remain part of the pitch deliverables. Scope their first
experiments after the camera/data contract; their progress does not substitute
for the core perception/control gates.

## Supporting growth model

Keep the audited v1 dataset and its frozen groups. When a short growth-forecasting
experiment supports the demonstrator, use the protocol in
[AI readiness](docs/AI_READINESS.md): causal loader, constant/persistence/recent-rate
baselines, small XGBoost comparison and group learning curves. It must use only
the saved input whitelist and report startup separately. No further bulk growth
generation or CFD expansion is required before starting vision work.

## Scheduling

The pitch's four-week outline remains a target software cadence:

| Target window | Focus | Required evidence |
|---|---|---|
| Weeks 1–2 | Milestones 0–1; scope extension experiments | Usable pilot footage, frozen evaluation contract, audited image data and baseline replay. |
| Week 3 | Milestone 2 and initial Milestone 3 | Perception/alerts on independent footage; PID/adaptive integration in software. |
| Week 4 | Milestone 4 and bounded extension reports | Integrated video, measured results, recorded limitations and extension status. |

This is a planning target, not a claim that four calendar weeks suffice for real
growth campaigns, rare defect coverage, GPU development, cluster chemistry or
physical closed-loop validation. Pilot evidence sets the actual dates. If a gate
slips, report the narrower delivered result and revise dates without weakening
labels, tests or acceptance budgets.

## Working rules

Preserve the existing package structure and explain changed files. Every new or
changed code file starts with a purpose summary. Prefer small inspectable stages
and reproducible environments. The current implementation goal authorizes local
commits. **Never push without the user's explicit approval.** Earlier task-specific
permission notes are historical; future work follows the current authorization.

The [development history](docs/DEVELOPMENT_HISTORY.md) and numerical checkpoints
retain earlier work. Current priorities are defined here; method documents and
dataset cards describe their own bounded components. This replanning changes
documentation only and starts no training, generation, deployment or hardware run.
