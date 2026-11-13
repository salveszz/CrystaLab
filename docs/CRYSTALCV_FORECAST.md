# Public observable forecast comparison

**Recent growth rate beats both small trained models on both later time blocks.**
This completes the first bounded forecast fit on the [audited public process
source](CRYSTALCV_PROCESS.md). Keep recent rate as the comparison baseline; neither
ridge model is promoted. This is one MAPbBr3 run with unreviewed automatic
contours, not KDP accuracy, an action-response policy or physical quality evidence.

The frozen [protocol](../configs/crystalcv_observable_forecast.json) has SHA256
`e8c58c8361dc6abd8b2bb5182618cb0b0cd138519809df600b2dcd6c776e3eb2`.
It was saved before any model fit or forecast scoring. Accepted results are
`results/crystalcv_observable_forecast_v1`; their [checkpoint](CRYSTALCV_FORECAST_CHECKPOINT.json)
records sources, exact outputs, independent reconstruction and verification.
All data comes from the existing public CC BY 4.0 [Zenodo V1 source](https://zenodo.org/records/19140132).
No additional source cohort or physical experiment was acquired in this step.

## Fixed experiment

The target is future square-root contour area minus current square-root area,
in pixels, at a nominal **30-minute horizon**. This length-like observation proxy
does not establish true edge advance, physical growth, mass or defects. Actual
target images can be up to 30 seconds late; their future timestamps are used for
label/provenance checks only. Forecasts always use the fixed nominal horizon.

Of the 463 source candidates, **275 train** the models at cutoffs within the first
24 recorded hours. The last training target becomes available at 24.4556 h.
Two fixed later blocks use cutoffs 26.25–32 h and 34–39.5 h, with **69 and 66
windows**. The remaining **53 windows are purged**. Actual complete image/log/time
supports are disjoint across all three cohorts, including target lateness and
the preceding held-state log row. This is not a random image/window split.

The first model uses current area, current square-root area and a past-hour
square-root-area slope. The second adds the latest available reported infusion
command and its past-hour time-weighted mean. Source commands are exposed only
at the following source row, as established by the process audit. Neither model
uses saved smoothed length/growth, source errors, estimated mass, timestamps,
future observations or physical calibration as an input.

Both models fit observable gain with ridge regression: fixed penalty 1.0 on
standardized coefficients, unpenalized intercept, prefix-only population means
and standard deviations, and no output clipping. There is no hyperparameter
search or model selection on the later blocks. Baselines are no change,
past-hour recent rate times the nominal horizon, and the mean training gain.

## Measured result

Errors below are MAE of future square-root contour area, in pixels. Gain error
is numerically the same after subtracting the shared current observation.

| Method | Later block 1: 69 windows | Later block 2: 66 windows |
|---|---:|---:|
| No change | 0.845150 | 0.809077 |
| **Recent rate** | **0.068995** | **0.103800** |
| Training mean gain | 0.144945 | 0.126786 |
| Ridge: vision only | 0.130538 | 0.343959 |
| Ridge: vision plus process | 0.126296 | 0.303517 |

Nearby windows share most of their histories. The frozen greedy diagnostic keeps
only windows with strictly disjoint full support intervals within each cohort.
It yields 16 training windows and **only four in each later block**:

| Method | Block 1 disjoint subset | Block 2 disjoint subset |
|---|---:|---:|
| No change | 0.801164 | 0.817544 |
| **Recent rate** | **0.071609** | 0.142150 |
| Training mean gain | 0.100152 | **0.116531** |
| Ridge: vision only | 0.079010 | 0.344289 |
| Ridge: vision plus process | 0.091033 | 0.325833 |

Recent rate beats both ridge models on both full blocks and both disjoint
subsets. Training mean beats recent rate on the second small disjoint subset;
do not claim a universal winner or independent-run confidence from four windows.
Both learned models fail the frozen descriptive improvement rule. Adding process
features reduces full-block error relative to vision-only ridge, but worsens its
first disjoint-subset result. This does not establish a general process-feature
benefit or a causal action effect.

The models underpredict later gains, especially block 2: signed errors are
−0.343959 px for vision-only and −0.303517 px with process, versus −0.004627 px
for recent rate. The later crystal sizes exceed the training range, so the
failure is consistent with poor extrapolation. That is an interpretation of this
fit, not an independent diagnosis of the physical process. These small proxy
errors must not be described as calibrated subpixel measurement accuracy.

## Audit and reproduction

`scripts/train_crystalcv_observable.py` checks the upstream export/audit identities,
constructs fixed cohorts and trains only the prefix. `src/models/observable_forecast.py`
contains the small fitting/prediction and support-checking functions. Saved
models use their own observable-proxy schema and are not wired into the pitch
detector, operator forecasts or physical controller. Existing defaults are unchanged.

`scripts/audit_crystalcv_observable.py` imports no producer/model code. It verifies
upstream evidence and byte identities, reassigns source supports, checks actual
image/log IDs, fits both models by augmented least squares instead of normal
equations, and reconstructs every baseline, prediction and metric independently.
All **2,050 prediction records** pass. Maximum coefficient difference is
3.78e-15 and maximum prediction difference is 1.43e-14 px. Separate unit checks
cover shared histories, availability, prefix-only preprocessing, constant
training features and rejection of target/time metadata as forecast inputs.
The saved PNG/SVG comparison was inspected visually.
Verification passed 11 new targeted tests and the full **503 tests / 323 subtests**
suite in **122.76 seconds**. Ruff selected-error and format checks passed.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_crystalcv_observable.py --output results/crystalcv_observable_new
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_crystalcv_observable.py --result results/crystalcv_observable_new --output results/crystalcv_observable_check_new.json
MPLCONFIGDIR=/tmp/crystalab-matplotlib .venv/bin/python scripts/plot_crystalcv_observable.py --result results/crystalcv_observable_new --output results/crystalcv_observable_figures_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_observable_forecast.py
```

Use fresh output paths. Results, models and source snapshots remain Git-ignored;
protocol, code, guide and checkpoint are reviewable locally. The preceding
process-preparation checkpoint's exact source state is preserved at
`results/crystalcv_process_completed_sources_v1` before these guide updates.

## Next pitch work

Prioritize reviewed real KDP boundaries/events and diverse, independently grouped
public growth/process trajectories. Keep recent rate, PID and rule baselines in
subsequent comparisons. Any action-response experiment needs a declared objective,
actual action availability/acknowledgment and observations spanning relevant
interventions; this closed-loop infusion trajectory cannot reveal counterfactual
temperature, rotation or circulation outcomes. Retain the models and negative
result; more complexity on this same run is not supported by this experiment.

The [public-data plan](NEXT_STEPS.md) keeps pitch-class training and the operator
demonstration moving, with KDP accuracy, first-hour timing and physical outcomes
still explicit open gates. Zero human reviews and zero independent evaluation
groups have been added by this forecast experiment.
