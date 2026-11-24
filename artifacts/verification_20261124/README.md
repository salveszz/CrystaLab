# Verification evidence

`full_regression.log` records the final full suite: **523 tests and 323 subtests
passed**. It includes the new causal-input, family-isolation, test-physics
independence, saved-model reload, semantic tamper and simulation-artifact checks.
This final rerun follows the progress-count correction; the earlier passing suite
is retained as `full_regression_before_progress_fix.log`.
Run it from the repository root with:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-test-matplotlib .venv/bin/python -m pytest -q
```

Install `pytest` if it is absent. The recorded run used the locally installed
optional scientific dependencies; a minimal environment can skip optional
component checks. The XGBoost integration checks require `.[growth-training]`.

The `crystalab-growth-targeted-v*.log` files preserve development attempts. V1
failed because its metric fixture omitted `group_id`; the fixture was corrected.
V4 passes all twelve new tests. Those tests use small numerical fixtures; the
separate full pipeline executes the complete 360-episode protocol.

`simulation_audit.json` independently verifies the complete 120-unit simulation's
record count, final geometry/fields and solute/carrier conservation.
`controller_benchmark.log` records the passing audit of all eight scenario runs
and 4,808 process records under `../controller_benchmark_20261124_v1/`.
The complete simulation/data/model audit is saved inside
`../full_pipeline_20261124_v1/audit.json`.

`run_checkpoint.json` records the sealed inventory check and exact CLI reload
agreement on all 1,080 rows in `reloaded_predictions.jsonl`.
`full_pipeline_console.log` preserves the parent process's progress and terminal
completion record. The artifact guide explains its original attempt-count display
and the subsequent reporting correction.
