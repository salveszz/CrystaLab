# Full simulation and growth-model pipeline

The complete run is in [`full_pipeline_20261124_v1/`](full_pipeline_20261124_v1/).
It uses the existing full simulation and dataset configurations. Its generated
data, trained XGBoost models and independent audits are versioned, so a fresh
clone can inspect the results without generating the dataset again.

## Completed results

The run completed in 3,880 seconds (about 65 minutes). All four stages passed.
Validation selected depth 6, learning rate 0.03 and 267 boosting rounds from six
candidates. The frozen test contains 216 samples from twelve held-out groups:
72 startup samples and 144 samples with observations.

| Method | All test MAE | Startup MAE | Observed MAE |
|---|---:|---:|---:|
| Zero-growth persistence | 0.195416 | 0.213385 | 0.186432 |
| Training-mean gain | 0.036414 | 0.035607 | 0.036817 |
| Recent rate × horizon | 0.018983 | 0.035607 | 0.010670 |
| XGBoost | 0.012911 | 0.032371 | 0.003181 |

Errors are in model volume units. Startup forecasts remain limited by unavailable
sensor/history inputs; missing indicators cannot recover the initial state.
The independent auditor reconstructed all 1,080 saved model predictions exactly
from JSON trees, verified every raw episode/label and checked the complete visual
simulation. Its maximum relative mass drift was 6.17 × 10⁻¹³.
The full regression suite passed 523 tests and 323 subtests.

## Run contents

| Path inside the full run | Contents |
|---|---|
| `manifest.json`, `files.json` | Final stage status, source/configuration identities and byte counts/SHA-256 hashes for every run file. |
| `simulation/` | The full 120-model-time-unit simulation: animation, final plot, 121 recorded states, resolved settings and final 40³ fields. |
| `dataset/` | All 360 simulated episodes, final arrays and raw histories; 1,080 samples from 60 concentration groups, split 648/216/216 by whole group. |
| `training/config.json`, `selection.json`, `test_release.json` | Frozen six-candidate protocol, validation selection and the model identity saved before final test scoring. |
| `training/candidates/`, `training/model.json` | Every candidate's model/history and the selected reloadable XGBoost model. |
| `training/learning_curve/`, `training/learning_curve.json` | Models, group membership and validation errors for 6/12/24/36 training groups. |
| `training/predictions.jsonl`, `training/report.json` | Every prediction and baseline, with errors by split, startup/observed state, group and recipe. |
| `training/training_report.png`, `training/training_report.svg` | Training histories, test comparison and learning curves. |
| `audit.json`, `logs/` | Independent geometry/conservation/data/tree-prediction audits and each stage's console log. |

`manifest.json` is the authoritative completion record. `progress.json` preserves
the last periodic observation while the long-running stages were active. This
run's launch snapshot counted episode attempts, including active workers, in that
display; the committed runner corrects the display to count completed statuses.
The final dataset audit checks every episode independently.
The separate [`controller_benchmark_20261124_v1/`](controller_benchmark_20261124_v1/)
contains all eight fixed-recipe/PID scenario runs and their 4,808 state records.
[`verification_20261124/`](verification_20261124/) holds regression and additional
verification evidence.

## Generate another complete run

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[growth-training]'
.venv/bin/python scripts/run_pipeline.py --output artifacts/my_full_run
```

Choose a new output directory; the pipeline refuses to overwrite an existing
run. This executes the full numerical solver: eight dataset workers generate
360 episodes on a 32³ grid, alongside the 120-unit 40³ visual simulation.
Training and independent auditing follow generation. This takes substantially
longer than the small regression fixtures. Settings are in
[`configs/full_pipeline.json`](../configs/full_pipeline.json).

## Audit, retrain or predict without repeating simulation

Audit the committed run and save the new audit outside its sealed directory:

```bash
.venv/bin/python scripts/audit_pipeline.py \
  --run artifacts/full_pipeline_20261124_v1 \
  --output /tmp/crystalab_full_run_recheck.json
```

Train a fresh model from the committed dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/train_xgboost.py \
  --dataset artifacts/full_pipeline_20261124_v1/dataset \
  --output /tmp/crystalab_new_growth_fit
```

Reload the selected model and predict from JSONL feature/missing records:

```bash
.venv/bin/python scripts/predict_xgboost.py \
  --run artifacts/full_pipeline_20261124_v1/training \
  --input artifacts/full_pipeline_20261124_v1/dataset/samples.jsonl \
  --output /tmp/crystalab_growth_predictions.jsonl
```

The model accepts only 14 causal feature columns and their missing indicators.
Missing values use XGBoost's native handling. Training compares zero-growth
persistence, the training-mean gain and recent rate times horizon; unavailable
recent rates fall back to the training mean. Selection and learning curves use
training/validation only, with startup errors reported separately.

Repeat the separate controller comparison with:

```bash
.venv/bin/python scripts/benchmark.py --output /tmp/crystalab_control_comparison
```

## Interpretation

These results test concentration interpolation under one illustrative growth
law, fixed geometry and six recipes. The model predicts volume gain over ten
model time units from idealized process proxies. It does not establish physical
KDP accuracy, camera defect accuracy, crystal quality or equipment action benefit.
Reviewed real KDP observations and independent control-response evaluation remain
the next pitch milestones in the [roadmap](../roadmap.md).
