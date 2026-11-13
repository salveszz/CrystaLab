# CrystaLab

For the immediate sequence under public-data-only access, read
[Next steps](docs/NEXT_STEPS.md). The detailed [roadmap](roadmap.md) records the
pitch requirements and current evidence.

CrystaLab aims to observe crystal growth through a camera, detect visible changes
and support PID/adaptive control with operator involvement, as proposed in the
[original pitch](Pitch_KDP.pptx). The implemented foundation is an illustrative 3D
growth simulator, verified numerical flow components, an audited growth-data
pipeline, an OpenCV video-observation baseline and a local YOLOv8-seg training
pipeline. Initial synthetic-only fits save working weights; real defect accuracy
and physical control remain unevaluated.

The completed growth dataset contains **360 episodes and 1,080 audited samples**.
Video ingestion, the seed baseline, editable real-image proposals and the first
Blender optical pipeline are implemented and checked. A 60-image, 20-scene
[synthetic pilot](docs/VISION_PILOT.md) is complete and trained.
[Learned video observation](docs/LEARNED_VIDEO.md) now preserves seed role,
original-frame mapping and inspection requests on synthetic/real development
replays. Bubble false positives, missed inclusions and real mask accuracy remain
open. [Temporal candidate analysis](docs/TEMPORAL_CANDIDATES.md) now retains
non-seed identities, persistent inspection evidence and independently scored
synthetic errors. The same decisions now run during single-pass observation.
The [temperature PID baseline](docs/TEMPERATURE_CONTROL.md) now runs with explicit
sensor/command records and audited simulated-plant comparisons. Real review,
equipment verification, adaptation and live operator/equipment validation
remain open. See the
[roadmap](roadmap.md) and [Vision MVP contract](docs/VISION_MVP.md). The tabular
growth benchmark is supporting work; HPC and Qiskit have separate extension gates.
The first [CPU/hybrid Vulkan flow comparison](docs/VULKAN_FLOW_BENCHMARK.md) passes
fixed-domain numerical checks but runs slower on the GPU. A separate
[heat/solute/buoyancy verification](docs/THERMAL_TRANSPORT.md) now passes numerical
convergence and budget checks with declared test properties. KDP thermal
calibration and physical risk validation remain open.
[Additional public KDP leads](docs/PUBLIC_KDP_SOURCE_REVIEW.md)
have not yet supplied verified growth-event training labels.
The first [Al aquo/phosphate VQE comparison](docs/QUANTUM_IMPURITY.md) now passes
independent checks on two small fixed-geometry Hamiltonians and all six starts.
HF already meets its energy target; KDP surface chemistry, impurity tolerance
and any quantum advantage remain unestablished.
The acquired [KDP movies now have a frame index and review pack](docs/KDP_VIDEO_DATA.md):
881 source-linked frames and six native annotation sheets, with no accepted labels
or invented acquisition/run-start clocks.
A [second KDP study](docs/KDP_GEL_DATA.md) adds nine small published microscopy
panels with a source-bound review pack. Their gel/specimen domains, publication
marks and sparse clocks remain explicit; all labels are still unreviewed.
An [anchored projected-edge diagnostic](docs/PROJECTED_EDGES.md) now verifies a
conditional measurement method on constructed images. It requires attachment
and pose evidence; the real KDP sheets currently produce no qualified edge
measurements. Visible-face identification and physical control remain open.
The [inclusion diagnostic](docs/INCLUSION_RESOLUTION.md) now detects one of six
synthetic validation inclusions and demonstrates a localized training-clip alert,
but adds false secondary detections and worsens seed measurements. It is not
promoted as the default model.

[Event-rule supervision](docs/EVENT_SUPERVISION.md) now produces source-linked
review proposals and connects a declared vision fixture through PID to simulated
actuation. Real-video shadow runs create no numerical commands; scripted fixture
decisions do not claim human review or physical action benefit.

A [projected-area forecast diagnostic](docs/AREA_FORECAST.md) now uses causal
learned observations and separate original proxy labels. Its calibrated rate
does not beat persistence on four shared development windows; measurement bias,
longer trajectories and adaptive-policy validation remain open.
A [longer optical diagnostic](docs/OPTICAL_TEMPORAL.md) now adds 96 scheduled
frames and a frozen-model comparison. It retains three false crystal alerts, a
missed inclusion and worse calibrated future-area error than persistence on 28
shared validation windows. No forecast or adaptive action is promoted.

An [expanded synthetic vision comparison](docs/VISION_REFINEMENT.md) now trains
two further fits on 84 images and checks the same 44 validation images. It improves
seed outlines and some false alerts, while retaining secondary misses, missed
inclusions and a PNG/video measurement discrepancy. The [next steps](roadmap.md#what-we-do-next)
prioritize consistent measurements, six real-image reviews and the operator demo.
A [paired encoding check](docs/VIDEO_ENCODING.md) now verifies lossless FFV1
input against original pixels. With fixed expanded detectors, recent-rate
forecasts beat persistence on the same two synthetic validation families. This
resolves a synthetic replay discrepancy; real camera encoding, labels and adaptive
control still need their own evaluation.

The [operator replay console](docs/OPERATOR_REPLAY.md) now combines video,
native measurements, proposal evidence and process records. Explicit reviews can
pause and manually resume a bounded simulated recipe, with separate saved results.
Real-video sessions retain shadow review without invented process readings or
commands. Actual human label/operator reviews remain zero; browser checks use
clearly identified automated fixture receipts.

The [adaptive supervisor diagnostic](docs/ADAPTIVE_SUPERVISION.md) now updates a
bounded forecast coefficient from newly observed mask outcomes and scales
candidate-triggered simulated pauses. Its four-way comparison retains all costs
and the negative forecast result: adaptation is worse than plain recent rate on
the two synthetic validation families. Real shadow output issues no commands;
physical action response and real accuracy remain unevaluated.
The [operator forecast view](docs/OPERATOR_FORECAST.md) now shows that forecast
and changing model state beside the video. Reviews preserve the displayed context
and distinguish suggested pause durations from manual choices.

Real observations are now local: **2,123 alpha-LGA microscope images from nine
runs, three phase-specific KDP movies and one separately identified ADP movie**.
The image index preserves 79 source manual shape fits; reviewed segmentation and
event labels are still missing. See [real vision data](docs/REAL_VISION_DATA.md)
for source licenses, frozen groups, local paths and import/replay commands.
A separate [source-shape training comparison](docs/SOURCE_SHAPE_TRAINING.md) now
uses 54 real alpha-LGA images with targets projected from the authors' manual
3D fits. Mixed training reaches 0.9255 IoU on 12 real validation proxies, while
real-only training misses all synthetic validation targets. These one-class fits
are kept separate from the pitch's defect classes and reviewed pixel accuracy.
A separate [published-mask source](docs/PUBLISHED_CRYSTAL_MASKS.md) now adds 587
real protein-crystal images with 4,986 expert-supplied crystal/drop masks. Their
selected-object definitions stay separate from the pitch's KDP training classes.
A separate [CrystalCV acquisition](docs/CRYSTALCV_DATA.md) now preserves 27,652
distinct uncontoured perovskite image arrays, 54 published rectangle measurements
and a candidate infusion/growth log. Annotated displays, duplicate source folders,
timing/scale questions and KDP task differences remain explicit.
The [local review editor](docs/ANNOTATION_REVIEW_UI.md) now lets a person correct
the six proposed outlines, save uncertain drafts and record individual reviews.
The [annotation workflow](docs/ANNOTATIONS.md) exports only explicitly reviewed
labels; actual human reviews are still zero. [Optical synthetic data](docs/OPTICAL_SYNTHETIC.md) keeps geometric masks,
refracted proxies and uncalibrated optics distinct.

## Where is the dataset?

On the machine where it was generated, open:

```text
data/synthetic_growth_v1/samples.jsonl
```

This is JSONL: one sample per line, with separate `features`, `missing`, `labels`
and split/episode metadata. The split is 648 training / 216 validation / 216 test
samples, assigned to whole base groups. There are 720 fully observed rows and
360 startup rows with explicitly missing sensor/history inputs.

Raw generated data is Git-ignored. GitHub contains its configuration and compact
verification report; a fresh clone must generate the data locally. Editors may
also hide ignored files. Start with the tracked [data guide](data/README.md),
[dataset card](docs/SYNTHETIC_DATASET_V1.md) and
[dataset report](docs/SYNTHETIC_DATASET_V1.json).

## Setup and commands

Python 3.11 or later is required. For a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Audit an existing dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_dataset.py data/synthetic_growth_v1
```

Generate it on a fresh checkout, or choose another unused output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/generate_dataset.py \
  --config configs/synthetic_growth_dataset_v1.json \
  --output data/synthetic_growth_v1 --jobs 8
```

The measured run took about 64 minutes. Generation refuses to overwrite an
existing output; `--resume` verifies matching settings, sources and environment.
See the [data pipeline guide](docs/SYNTHETIC_DATA.md) for the full contract.

The existing visual simulator remains available:

```bash
.venv/bin/python scripts/run_simulation.py --headless --output results/demo
```

Its default is still `configs/focused.json`, duration 120 model time units. The
separate dataset protocol uses 30-unit episodes and 10-unit forecasts.

## Scope and documentation

The growth dataset uses **illustrative model units and uncalibrated kinetics**.
Its inputs include idealized 3D sensor proxies; quality labels are unavailable.
Synthetic holdout performance will describe this simulator. Real KDP prediction,
vision and physical control require their own measurements and validation.

- [Documentation index](docs/README.md): methods, evidence and source references.
- [Vision MVP](docs/VISION_MVP.md): image data, perception, alerts and control gates.
- [Video baseline](docs/VISION_BASELINE.md): fixture replay, input clocks, masks and evidence outputs.
- [AI readiness](docs/AI_READINESS.md): separate vision, growth and control readiness.
- [Physical data](docs/PHYSICAL_DATA.md): structural reference and measurement limits.
- [Development history](docs/DEVELOPMENT_HISTORY.md): superseded plans and milestones.

The active dataset and original input archive are retained. Old results, pilot
datasets, duplicate extracted CIFs and project caches were removed during cleanup.
Numerical checkpoints remain versioned; historical raw-output paths are archival
references. Simulation equations, package layout and runner defaults are unchanged.
