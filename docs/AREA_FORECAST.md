# Projected-area forecast diagnostic

A causal next-frame projected-area pipeline now runs on the frozen optical pilot
and saved learned observations. The fitted calibration is a diagnostic candidate;
it is **not promoted for control**. On the four shared validation windows,
persistence has lower future-area error. The [checkpoint](AREA_FORECAST_CHECKPOINT.json)
retains the actual model/data identities, comparison and regression evidence.

This supplies the first observable forecast experiment for the pitch's adaptive
supervisor. It does not implement that adaptive policy or an action-conditioned
forecast. The existing three-frame optical scenes contain no temperature/action
responses. Real masks, physical time and forecast accuracy remain unreviewed.

## Data, horizon and causal inputs

The cohort replays the existing frozen YOLOv8n-seg model on 16 development scenes:
12 training families and four validation families, with three frames each. All
48 frames are observed. The four final test families are excluded from replay,
fitting and scoring; original dataset integrity auditing remains separate.

There are 32 within-clip forecast windows: frame 0 -> 1 and frame 1 -> 2 in each
scene. The horizon is **0.25 source-video seconds**, corresponding to one
illustrative geometry step at the declared 4 FPS. It is not physical growth time.
Camera and seed orientation are fixed within each constructed scene. Every scene
uses the same declared outward-face-offset growth law, so these data do not test
different kinetics or action responses.

The seed's native predicted mask supplies observed area in px². Original optical
emission-proxy seed masks supply the current/future labels, independently of the
detector. Those proxies remain distinct from exact empirical visibility through
transparent materials. MJPG replay encoding is a declared derivative of the
original PNGs, not additional independent data.

The feature whitelist is:

- Current observed pixel area.
- Recent observed area rate in px² per source-video second.
- Duration of the two-observation history.
- Declared forecast horizon and native image pixel count.

IDs, source truth, hidden 3D volume, concentration, temperature, future observations
and action-response labels are absent from model inputs. Invalid or inspection
frames clear history. Run/clock/view/shape changes reset it, and same-clock frame
or timestamp reversal is rejected. A long observation gap also resets history.
Missing rate history remains missing; it never becomes a zero-rate measurement.

## Models and fitting

All four transparent methods remain visible in reports and replay:

| Method | Prediction |
|---|---|
| Persistence | Keep the current measured area; predict zero gain. |
| Recent rate | Extrapolate the last measured area difference over the declared horizon. |
| Training mean fraction | Apply the training mean relative proxy gain to current measured area. |
| Calibrated recent rate | Multiply recent-rate gain by a training-only fitted coefficient. |

The calibration fits relative proxy gain with one bounded ridge coefficient.
For each eligible training window, `x=observed_recent_rate*horizon/observed_area`
and `y=true_proxy_gain/current_proxy_area`. The coefficient is
`sum(x*y)/(sum(x*x)+ridge)`, then limited to the declared interval. Configuration
is fixed in `configs/area_forecast.json`; no validation labels enter fitting.

The actual fit uses 12 history-ready training windows from 12 families. Its
coefficient is 0.89373948 and mean training relative gain is 0.06917927. This
scalar calibration is not a new neural segmentation model or a physical kinetic
law. Predictions are bounded to the image area and retain a clipping flag.

## Actual development result

Persistence and the mean-fraction baseline are available on all eight labelled
validation windows. Rate methods are available on four of eight because the first
window in every scene lacks history. Comparisons below use the same four
history-ready windows/families for every method; coverage is reported separately.

| Method | Future-area MAE (px²) | Gain MAE (px²) |
|---|---:|---:|
| Persistence | **232.75** | 303.50 |
| Recent rate | 303.00 | 43.25 |
| Training mean fraction | 288.01 | **17.85** |
| Calibrated recent rate | 286.29 | 55.82 |

Future-area accuracy and growth-increment accuracy answer different questions.
The mean current-area bias on these four windows is +289.75 px². Algebraically,
`future-area error = current-state error + predicted-gain error`. Positive mask
bias can therefore partly cancel the growth missed by persistence; adding a
reasonable gain can amplify the absolute-area error. The audit reconstructs both
terms rather than treating a gain result as future-area accuracy.

The calibration does not beat persistence on future area or the simple training
prior on gain. Four windows are too little to establish a reliable ranking across
conditions. These validation families were already used during segmentation-model
development, so this is a development diagnostic rather than an independent final
test. There is no physical/real accuracy, uncertainty interval or control-benefit
claim, and no method is automatically selected for actuation.

## Replay forecasts and uncertainty

The optical scene-013 replay produces two history-based predictions across three
frames, with one startup warmup. The 30-frame real alpha-LGA development replay
produces 17 history-based predictions: 11 inspection frames block forecasting and
two usable frames need warmup. Persistence/prior predictions are available on 19
real frames. Estimates remain unreviewed and real accuracy is null.

Every replay prediction preserves source frame references, source clock, mask
identity and the original nullable experiment time. Real view comparability is
explicitly assumed/unverified. Outputs retain `requires_operator_review=true`,
`can_authorize_actuation=false` and a null physical horizon. These are projected
image-plane estimates, not face-specific growth, 3D volume, pre-visible event risk
or an equipment command.

## Reproduce and audit

Use unused paths for reruns; these outputs already exist locally.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-review-mpl .venv/bin/python scripts/prepare_area_forecasts.py \
  --source data/synthetic_vision_pilot_v1 --model-run results/vision_training_pilot_v1 \
  --observer-config configs/optical_learned_observation.json \
  --forecast-config configs/area_forecast.json --output results/area_forecast_cohort_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_area_forecasts.py \
  --output results/area_forecast_cohort_v1 --audit
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-review-mpl .venv/bin/python scripts/train_area_forecast.py \
  --cohort results/area_forecast_cohort_v1 --output results/area_forecast_fit_v3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/forecast_area.py \
  --replay results/leeds_pilot12_stream_v1 --model results/area_forecast_fit_v3/model.json \
  --output results/leeds_pilot12_area_forecast_v2
```

The cohort keeps source/model/config identities, replay artifacts, separate copied
proxy masks and labelled samples. Its audit verifies grouped source ancestry,
complete within-horizon window coverage including startup, native causal rates,
clock/horizon alignment and original proxy identity. The producer snapshot is
retained; the later consumer audit adds stricter horizon/coverage checks without
rewriting that dataset. The fit saves its coefficient/prior, source snapshots,
per-window scores, coverage, per-family metrics and comparison plot. Earlier fit
and replay versions remain intact with their own producers.

Tests check future-data prefix invariance, training-only label use, context/gap
resets, startup availability and rejection of rehashed label/horizon/coverage
changes. An independent execution audit reconstructs the training parameters,
forecast values, scores, state-bias decomposition and actual replay source links.

## Next gate for adaptation

Review current boundaries and measurement bias, then obtain longer comparable-view
trajectories with enough independent runs. If actions will be compared through a
forecast, retain synchronized observed/applied actions and actual responses;
current optical geometry steps cannot supply that relationship. Freeze the target,
horizon, acceptance budgets and grouping protocol before final evaluation.

The adaptive agent still needs an explicit changing state/policy, objective and
bounded choices, compared with persistence, PID and [event-rule supervision](EVENT_SUPERVISION.md).
Real sources remain shadow-only until their measurement, timing and action
evidence supports a stronger role. Operator review, physical first-hour evidence,
face/asymmetry measurements, HPC and Qiskit remain open in the [roadmap](../roadmap.md).
