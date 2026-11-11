# AI readiness by task

The [pitch-led roadmap](../roadmap.md) prioritizes camera observations, real and
synthetic images, YOLOv8-seg, event alerts and PID/adaptive control. The completed
tabular growth dataset is ready for a bounded supporting experiment. It does not
make the vision system or physical controller ready. Video/camera ingestion, the
OpenCV seed baseline, real-image import, annotation tooling and the Blender
optical smoke batch are implemented. No trained model,
evaluated defect detector or automatic controller is implemented.

## Readiness gates

| Task | Current evidence | Next gate |
|---|---|---|
| Video baseline and image-data pipeline | File/camera adapter, tested seed baseline, real-image index, six editable annotation proposals and an audited 18-frame Blender optical smoke batch. | Human pixel/event review, clock/measurement checks and a declared training cohort; smoke completion does not prove training coverage. |
| YOLOv8-seg and visible-event alerts | Model task is defined in the pitch; no training or evaluated alerts. | Reviewed labels, whole-run/scene splits, frozen metrics and independent real evaluation under the [Vision MVP contract](VISION_MVP.md). |
| Synthetic growth forecast | Audited v1 data and accepted numerical regime. | Causal loader, explicit baselines, small comparison and grouped evaluation; supporting priority. |
| PID/adaptive software loop | Bounded scripted simulation commands; no feedback controller. | Sensor/command interfaces, PID and bounded supervisor, fixed-recipe comparison and traceable action logs. |
| Physical KDP growth/control | Structural references, aggregate summaries and three acquired phase-specific microscopy clips. | Reviewed trajectories/conditions, relevant calibration, equipment/response characterization and independent experiments. |
| HPC/Qiskit extensions | Separate verified CPU flow operators; no Qiskit implementation. | Bounded hardware/accuracy benchmark and a justified impurity-cluster VQE/classical comparison, as defined in the roadmap. |

## Immediate vision work

Follow Batch 1 of the [roadmap](../roadmap.md). First checkpoint the current
annotation/rendering additions and show the six existing outline proposals for
review. A reviewed outline is an image boundary that a person has checked and
corrected; it is distinct from the computer's initial suggestion. An event label
records what became visible and when, such as a secondary crystal appearing.

Then expand a small development pilot across the existing training families,
including difficult views and contiguous candidate-event/negative periods. Freeze
classes, clocks, evaluation rules and performance budgets after pilot review.
Preserve all real validation/test groups. Alpha-LGA and ADP development results
do not establish KDP performance.

Alongside review, declare a bounded synthetic scene cohort and implement the
YOLOv8-seg loader/training/evaluation path. A synthetic-only plumbing/overfit check
can proceed while real review is pending; unreviewed proposals remain excluded
from accepted real training. Compare with the OpenCV baseline on reviewed data,
then evaluate temporal decisions and keep real/synthetic results separate.

Use the [Vision MVP contract](VISION_MVP.md) for classes, evaluation, first-hour
claims and control integration. Numeric performance budgets are set after the
pilot visibility review and frozen before validation selection/final test.

## Preserved growth-data readiness

Use `data/synthetic_growth_v1/samples.jsonl` locally. The
[dataset card](SYNTHETIC_DATASET_V1.md) and [report](SYNTHETIC_DATASET_V1.json)
describe 360 episodes / 1,080 samples across 60 concentration groups. The frozen
split is 648/216/216 rows by whole base group, with 36/12/12 groups. There are 720
fully observed rows and 360 startup rows with missing sensor/history inputs.
Do not merge overlapping pilot data or reshuffle episode windows.

The recorder preserves exact future volume-gain labels, bounded held commands,
causal noisy/delayed proxies, raw events, configurations, source snapshots and
final arrays. Independent auditing reconstructs samples and provenance. The
[pipeline contract](SYNTHETIC_DATA.md) defines units and features.
The [numerical checkpoint](SYNTHETIC_DATA_CHECKPOINT.json) preserves the accepted
32^3 focused grid/timestep protocol, original 2% gates and coarser failures. Its
source checkpoint passed 227 tests. The review reran those tests and audited the
working data; numerical sources remain unchanged by replanning.

When this supporting experiment is scheduled:

1. Load only the 14 saved feature names and missing flags. Keep IDs, hidden
   configuration, exact truth, timestamps outside the whitelist and future
   outcomes out of inputs. Fit preprocessing using training data only.
2. Compare zero-growth persistence, a constant training-mean volume gain, and
   observed recent-volume-rate times horizon. For unavailable history, use the
   training-mean gain as the explicit recent-rate fallback; report its frequency.
   Do not fill inputs with future measurements or hidden initial concentration.
3. Compare a small XGBoost regressor for volume gain over 10 model time units.
   Select on training/validation, then freeze before final test scoring.
4. Report MAE/RMSE, signed error, group macro-averages and per-recipe results,
   separately for startup and observed state. Assess improvement on recent growth
   specifically when observations exist; avoid percentage errors near zero.
5. Use learning curves over training groups before increasing data or complexity.

Startup inputs have only five distinct feature vectors in each development split,
while initial concentrations and growth outcomes vary. Individual starting states
cannot be recovered from those inputs. Startup performance therefore needs its
own interpretation; missing-value support does not restore absent information.

## Limits and dependencies

The growth dataset tests concentration interpolation under one fixed illustrative
law, geometry, time range and six recipes. Its idealized 3D proxies are not camera
outputs. Quality labels remain unavailable. Several small action effects do not
consistently exceed recorded numerical sensitivity; the agitation law has no
validated quality or equipment-cost tradeoff. It establishes neither a real
recipe optimum nor physical defect response.

Collect measured video/process data alongside implementation. COD is structural
evidence, and published aggregates are not independent raw trajectories. A reduced
calibrated model can support a narrow physical task; further CFD is justified by
a specified output or discrepancy. Optical quality and action benefit require
their own evidence. See [physical data](PHYSICAL_DATA.md) and
[real-data validation](REAL_DATA_VALIDATION.md).
