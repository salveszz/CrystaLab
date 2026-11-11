# CrystaLab

CrystaLab aims to observe crystal growth through a camera, detect visible changes
and support PID/adaptive control with operator involvement, as proposed in the
[original pitch](Pitch_KDP.pptx). The implemented foundation is an illustrative 3D
growth simulator, verified numerical flow components, an audited growth-data
pipeline, an OpenCV video-observation baseline and a local YOLOv8-seg training
pipeline. Initial synthetic-only fits save working weights; real defect accuracy
and automatic control remain unevaluated.

The completed growth dataset contains **360 episodes and 1,080 audited samples**.
Video ingestion, the seed baseline, editable real-image proposals and the first
Blender optical pipeline are implemented and checked. A 60-image, 20-scene
[synthetic pilot](docs/VISION_PILOT.md) is complete and trained.
[Learned video observation](docs/LEARNED_VIDEO.md) now preserves seed role,
original-frame mapping and inspection requests on synthetic/real development
replays. Bubble false positives, missed inclusions and real mask accuracy remain
open, followed by temporal alerts, PID/adaptation and the operator demo. See the
[roadmap](roadmap.md) and [Vision MVP contract](docs/VISION_MVP.md). The tabular
growth benchmark is supporting work; HPC and Qiskit have separate extension gates.

Real observations are now local: **2,123 alpha-LGA microscope images from nine
runs, three phase-specific KDP movies and one separately identified ADP movie**.
The image index preserves 79 source manual shape fits; reviewed segmentation and
event labels are still missing. See [real vision data](docs/REAL_VISION_DATA.md)
for source licenses, frozen groups, local paths and import/replay commands.
The [annotation workflow](docs/ANNOTATIONS.md) exports only explicitly reviewed
labels. [Optical synthetic data](docs/OPTICAL_SYNTHETIC.md) keeps geometric masks,
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
