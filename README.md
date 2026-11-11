# CrystaLab

CrystaLab is a Python project for illustrative 3D crystal-growth simulation and
synthetic data preparation. The growth engine tracks faceted geometry, liquid
transport, local solute uptake and mass/carrier-volume conservation. Separate
verified flow operators provide a foundation for later physical extensions.

The current milestone is complete: **360 episodes and 1,080 audited growth
samples**. No model training, vision system or automatic controller is implemented.
The next step is the first synthetic growth-prediction benchmark; see the
[roadmap](roadmap.md).

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
- [AI readiness](docs/AI_READINESS.md): completed gates and the first model task.
- [Physical data](docs/PHYSICAL_DATA.md): structural reference and measurement limits.
- [Development history](docs/DEVELOPMENT_HISTORY.md): superseded plans and milestones.

The active dataset and original input archive are retained. Old results, pilot
datasets, duplicate extracted CIFs and project caches were removed during cleanup.
Numerical checkpoints remain versioned; historical raw-output paths are archival
references. Simulation equations, package layout and runner defaults are unchanged.
