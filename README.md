# CrystaLab

The complete presentation is [CrystaLab_Presentation.pptx](presentation/CrystaLab_Presentation.pptx).
It contains 20 slides covering the pitch, implemented software, complete
simulation/XGBoost results, vision/control demonstrations, HPC and Qiskit,
validation evidence and the remaining real-data milestones.

CrystaLab aims to observe crystal growth through a camera and support bounded
control with operator involvement. The [original pitch](Pitch_KDP.pptx) and
[roadmap](roadmap.md) preserve the intended system and its evidence requirements.
Real KDP defect accuracy and physical control benefit remain unevaluated.

## Completed run

The full run is versioned at
[`artifacts/full_pipeline_20261124_v1/`](artifacts/full_pipeline_20261124_v1/).
It includes the complete 120-model-time-unit simulation, all 360 synthetic
episodes and 1,080 samples, selected and candidate XGBoost models, predictions,
plots, source snapshots, logs and independent audits.

XGBoost synthetic test MAE is **0.01291**, versus **0.01898** for recent rate,
in illustrative model volume units. Startup and observed errors are reported
separately. The recorded full regression passed 523 tests and 323 subtests.
The [artifact documentation](artifacts/README.md) contains the full protocol.

## Setup and reproduction

Python 3.11 or later is required:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[growth-training]'
```

Audit the saved run without repeating simulation:

```bash
.venv/bin/python scripts/audit_pipeline.py \
  --run artifacts/full_pipeline_20261124_v1 \
  --output /tmp/crystalab_full_run_recheck.json
```

Generate another full run in a new directory:

```bash
.venv/bin/python scripts/run_pipeline.py --output artifacts/my_full_run
```

## Project files

- `src/`, `scripts/`, `configs/`, `tests/`: implementation, commands and protocols.
- `artifacts/`: the latest full run, controller benchmark and verification evidence.
- `presentation/`: the completed PowerPoint, with visuals embedded.
- [Documentation](docs/README.md), [sources](SOURCES.md) and [roadmap](roadmap.md): methods, provenance and remaining work.
- [Original data inputs](data/README.md): retained uploaded references.

Obsolete result folders, their local archive, `dazi/`, the presentation guide,
loose presentation assets and generated/downloaded working-data caches were
removed on 25 November 2026. `results/` is empty. Historical experiment paths in
the documentation must be regenerated before use; the latest full-run dataset
is available directly under `artifacts/`.
