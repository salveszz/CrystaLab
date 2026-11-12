# Causal adaptive area forecast and bounded supervision

The supervisor now has an explicit changing model and policy. It updates a
bounded gain coefficient when a newly received mask supplies the outcome of an
earlier forecast. After a fresh new-crystal ramp-review proposal, it uses the
current forecast and observed-error state to choose a bounded simulated pause.
Missing, uncertain, stale or unreliable evidence withholds that fixture action.

The [frozen experiment](../configs/adaptive_experiment.json) was written after
training-only reference calibration and before development scoring. Its
[checkpoint](ADAPTIVE_SUPERVISION_CHECKPOINT.json) retains source identities,
actual forecasts/actions, independent reconstruction and regression evidence.
No detector weights were trained or promoted by this experiment.

The adaptive forecast **does not beat plain recent rate** on the shared synthetic
validation windows: future-area MAE is 80.71 versus 77.00 px². Persistence is
116.18 px². Both validation families retain this negative comparison with recent
rate, and persistence is best on family 004. The protocol was not retuned after
seeing these results.

## Model inputs and state

The input is the existing `AreaEstimate`: native projected area, mask identity,
run/clock/view/shape, observation and receipt time, validity and inspection flag.
No source proxy truth, future frame, process-hidden state or defect label enters
the online update. Whole runs start from separate state. Context changes reset
parameters; missing/inspection/gap states clear pending forecasts and recent
history. Coefficients remain available within the same context after a temporary
gap, with new comparable history required before another forecast.

The 0.25-source-second forecast uses recent observed area rate and a scalar
coefficient initially equal to one. Only an aligned, usable target observation
can mature a pending update. Let `q` be the training reference, `A` the area at
the earlier forecast cut, `g` its recent-rate gain over the horizon, and `B` the
newly observed target area:

```text
x = g / (A × q)
y = (B − A) / (A × q)
xx ← 0.95 × xx + x²
xy ← 0.95 × xy + x × y
coefficient ← clip(xy / xx, 0, 3)
```

Initial `xx=xy=3` supplies the declared prior. Forecast area is current observed
area plus the updated coefficient times its recent-rate horizon gain, clipped
to image bounds. A late receipt cannot create a forecast whose target has
already elapsed. The class retains source/receipt order, horizon alignment and
context checks from the existing causal area contract.

The model also updates an exponential mean of the absolute *pre-update* observed
forecast residual, divided by `A × q`, with weight 0.2. This is an internal
observed-mask error ratio, not a calibrated confidence interval or real accuracy
measurement. Earlier state snapshots remain immutable when later frames arrive.
The forgetting factor advances per matured observation, not physical wall time.

## Training reference and policy

The fullmask lossless cohort supplies three training families, 48 observations,
45 positive adjacent growth increments and 42 measured recent-rate residuals.
The positive-gain 90th percentile is **0.04114072999**; the initial error ratio is
**0.1425736366**. These statistics use native *observed* mask areas only.
Validation observations and optical proxy truth do not set them.

This reference describes an illustrative image cohort. It is not an approved KDP
growth rate, physical operating limit or target optical quality.

`configs/adaptive_supervision.json` freezes the update and policy settings. A
recommendation requires at least three matured updates, observed-error ratio at
most three, a fresh future-directed forecast and positive projected gain.
For eligible evidence:

```text
requested pause = 60 model seconds × forecast relative gain / q
bounded pause   = clip(requested pause, 15, 60 model seconds)
```

The result is rounded down to the declared thermal step. Candidate supervision
additionally requires a pending unexpired ramp proposal, the same visible track,
fresh usable camera evidence, current process feedback, source coverage, no
overlapping pause and remaining total budget. The total reserved pause bound is
120 model seconds. Startup/unknown-origin candidates remain inspection requests.
Negative projected growth is a valid forecast but does not authorize this pause.

Real evidence can produce adaptive shadow forecasts but cannot enter the
simulation-authorization path. The software comparison uses explicitly scripted
fixture authorization and records `operator_reviewed=false` and
`physical_hardware_authorized=false` on its actions.

## Forecast comparison

The frozen fullmask detector and old forecast baselines remain unchanged.
Each development clip contributes 15 labeled next-frame windows. Reports retain
75 windows across five families and compare all five methods on the same
history-ready windows; no observed-error/policy gate removes a difficult forecast
from scoring. Validation has two families, 30 labeled windows and 28 shared
history-ready windows. Online validation updates are prequential: a new observed
mask can update its own run only after it arrives. Evaluation labels never update
the predictor, and final test family 005 is excluded.

| Future proxy-area MAE (px²) | Family 003 | Family 004 | Family macro / 28-window mean |
|---|---:|---:|---:|
| Persistence | 166.93 | **65.43** | 116.18 |
| Recent rate | **79.57** | 74.43 | **77.00** |
| Frozen calibrated rate | 85.16 | 71.94 | 78.55 |
| Online adaptive rate | 86.37 | 75.05 | 80.71 |
| Old training mean fraction | 129.68 | 257.92 | 193.80 |

The two families have equal numbers of shared windows; their mean and macro
result therefore agree. Adaptive gain MAE is 48.63 px² versus recent rate 44.71
and persistence 98.61. The old calibration was not refitted to this detector.
The [forecast/state figure](../results/adaptive_figures_v2/forecast_and_state.png)
shows the negative recent-rate comparison and actual parameter changes.

## Four-way temperature comparison

Each of five declared source clips runs fixed recipe, PID-only, fixed-rule
supervision and adaptive supervision on the same thermal fixture: **12,020
process records**. One source-video second is explicitly mapped to 100 thermal
model seconds. This does not convert synthetic geometry into physical growth.
All variants see the same fixed vision observations; thermal commands do not
change those observations or their defect/quality outcomes.

Only two clips have eligible new-crystal ramp proposals:

| Clip | Split | Fixed-rule pause | Adaptive pause | Adaptive decision time |
|---|---|---:|---:|---:|
| 001 | Training | 60 model s | 46 model s | 145 model s |
| 003 | Validation | 60 model s | 23 model s | 175 model s |

The other three clips have no such action. In validation clip 003 the adaptive
policy initially withholds the pause at model time 170 because projected gain is
negative. A new observed frame at 175 changes its state and yields the 23-second
recommendation. No future frame was consumed to make the earlier choice.

For that validation clip:

| Variant | MAE to its target (K) | MAE to original recipe (K) | Final recipe lag (model s) |
|---|---:|---:|---:|
| Fixed recipe | 0.18735 | 0.18735 | 0 |
| PID only | 0.01053 | 0.01053 | 0 |
| Fixed-rule supervised PID | 0.01946 | 0.13864 | 60 |
| Adaptive supervised PID | 0.01347 | 0.05809 | 23 |

The adaptive pause incurs less recipe delay than the fixed rule. PID-only still
tracks the original recipe best. Shortening a pause cannot establish defect
prevention when the video has no response to that action. This is a bounded
policy/traceability diagnostic, not an optimized physical controller. See the
[temperature/phase/state figure](../results/adaptive_figures_v2/control_validation.png).

## Real shadow and remaining gates

The existing alpha-LGA training replay supplies 30 frames: 17 forecast frames
and 15 matured observed-mask updates, with **zero numerical commands**. No
independent real accuracy labels are available. Its earlier detector and real
capture conditions differ from the fullmask synthetic reference; normalization
is not calibrated to that domain. No real masks, defect labels or human reviews
were created. The six existing annotation proposals remain byte-identical.

The later [operator forecast integration](OPERATOR_FORECAST.md) now displays this
forecast, parameter state and recommendation alongside the camera evidence.
Explicit reviews preserve that context and distinguish suggested from manual
pause durations. This original experiment and its checkpoint remain unchanged.
Action-responsive growth data, reviewed real evaluation, face/asymmetry/inclusion
coverage, real first-hour clocks, physical outcomes, HPC and Qiskit remain open.

## Reproduce and inspect

Outputs are local and refuse overwrite. Use unused paths for new runs:

```bash
.venv/bin/python scripts/adaptive_area.py fit-reference \
  --cohort results/vision_codec_fullmask_ffv1_v1/cohort \
  --config configs/adaptive_supervision.json --output results/adaptive_reference_next
.venv/bin/python scripts/adaptive_area.py evaluate \
  --cohort results/vision_codec_fullmask_ffv1_v1/cohort \
  --reference results/adaptive_area_reference_v1 \
  --baseline-model results/area_forecast_fit_v3/model.json \
  --output results/adaptive_evaluation_next
.venv/bin/python scripts/run_adaptive_control.py \
  --analysis results/vision_codec_fullmask_ffv1_v1/candidates/optical_temporal_003 \
  --policy configs/event_rule_policy.json \
  --config results/adaptive_study_inputs_v1/optical_temporal_003.json \
  --reference results/adaptive_area_reference_v1 --output results/adaptive_control_next
.venv/bin/python scripts/audit_control_baseline.py \
  --run results/adaptive_control_next --output results/adaptive_control_next_audit.json
.venv/bin/python scripts/plot_adaptive_results.py --output results/adaptive_figures_next
```

The frozen experiment lists all source/model/configuration hashes and five control
cases. State records retain parameter-before/after values, matured target mask,
pre-update residual and pending cancellation; policy assessments bind their exact
state. Decisions, commands and feedback retain separate clocks and causal IDs.
The auditors reconstruct normalization, masks, sufficient statistics, forecasts,
every eligible action opportunity, pause/recipe phase and thermal response without
executing the predictor/supervisor/controller/plant. Prediction-only and integrated
control states match exactly. Future-state, wrong-source, changed artifact and
resealed wrong-duration cases are refused.

Full regression passed **398 tests / 288 subtests** in 111.17 seconds. Independent
execution checks cover 48 training observations, 80 synthetic and 30 real states,
75 causal windows, 375 forecast records and 12,020 process records. Raw data,
weights, results and figures remain Git-ignored; configurations, code and compact
verification reports are tracked. A fresh clone must reproduce the earlier
vision datasets/models and lossless replays before running this experiment.
