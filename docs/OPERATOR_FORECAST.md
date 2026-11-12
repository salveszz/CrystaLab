# Forecast context in operator replay

The operator console now places the causal projected-area forecast beside the
camera evidence. It displays the observed and predicted area, target source time,
changing gain coefficient, completed updates and observed-error ratio. It also
shows the suggested pause scale or the reason a suggestion is withheld.

This connects the [adaptive software diagnostic](ADAPTIVE_SUPERVISION.md) to the
[operator replay](OPERATOR_REPLAY.md). It does not establish real accuracy,
action-responsive crystal growth or physical control benefit. The frozen adaptive
experiment still loses to plain recent-rate forecasting. No model is promoted.
The [checkpoint](OPERATOR_FORECAST_CHECKPOINT.json) records actual executions and
verification at this source revision; earlier checkpoints remain unchanged.

## Open the prepared examples

Use the local environment and existing ignored data:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --session results/operator_forecast_synthetic_v1 --read-only
```

Open `http://127.0.0.1:8766`. This synthetic session has 16 recorded observations
and the original three-variant thermal comparison with zero operator approvals.
Stop the server with Ctrl+C. The real example is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --session results/operator_forecast_real_v1 --read-only
```

The real alpha-LGA session has 30 observations and no temperature or actuator
records. Its detector and capture conditions differ from the synthetic reference;
forecasts remain uncalibrated, unscored shadow output. A missing or uncertain
outline withholds its forecast. Existing real annotation proposals remain
unreviewed. This console does not edit image-training labels.

## Inspect and review

1. Select a recorded frame. Its image must finish decoding before the clock,
   forecast and review controls advance together. The forecast uses only native
   observations available by that frame; retained future observations do not
   update an earlier display.
2. Inspect **Growth forecast**. The predicted area is shown only while the
   forecast is usable and future-directed. The error ratio summarizes prior
   observed-mask residuals; it is not a probability or confidence interval.
   **Forecast evidence** retains the state, reference and context identities.
3. Select a pending ramp-review proposal. In a writable simulation, choose
   **Use suggested duration** or enter a manual duration. Choosing a suggestion
   clears acknowledgment so its duration can be checked before approval.
   A suggestion does not bypass candidate visibility, expiry, feedback or budget
   requirements. It never authorizes hardware.
4. Enter a reviewer name and acknowledge the frame, then approve the simulated
   pause. The receipt preserves the exact displayed forecast context and whether
   its requested duration came from that suggestion or a manual choice. A
   changed duration cannot retain an `adaptive_suggestion` claim. Early manual
   resume remains available with a new frame acknowledgment and receipt.

The manual pause ceiling remains 60 model seconds per request and 120 reserved
model seconds in total for the declared fixture. The adaptive policy suggests
15–60 model seconds, subject to its own history/error/growth checks. Manual
requests may use a shorter positive duration within the fixture bounds. A manual
override remains distinguishable from accepting an eligible suggestion.

Real sessions hide pause/resume controls and have no simulation path. A reviewer
can record shadow inspection when opened writable, but that does not create a
numerical command or a real segmentation/event label.

## Prepare another session

Choose unused output directories:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/operator_console.py \
  --analysis results/vision_codec_fullmask_ffv1_v1/candidates/optical_temporal_003 \
  --simulation-config configs/operator_optical_demo.json \
  --adaptive-reference results/adaptive_area_reference_v1 \
  --session results/operator_forecast_synthetic_next --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --analysis results/leeds_pilot12_stream_candidates_v1 \
  --adaptive-reference results/adaptive_area_reference_v1 \
  --session results/operator_forecast_real_next --prepare-only
```

The optional `--adaptive-reference` attaches a preserved reference, native causal
state replay and frame-by-frame recommendations. It is used only when preparing
a new session. Older sessions without a reference continue to provide manual
review. A simulation configuration on real evidence is refused.

The source adapter retains its software replay receipt clock. This does not
measure actual camera delivery latency or validate physical growth time. The
synthetic thermal mapping remains one source second to 100 model seconds.
Commands do not change the recorded crystal video. Browsing a saved recording
does not establish prospective operator reaction time or adaptive-policy utility.

## Verification

Full regression passed **402 tests / 291 subtests** in 180.39 seconds.

The independent auditor reconstructs native mask areas, matured-only model
updates, every displayed state/recommendation, review/pixel/source bindings and
all preserved thermal results. It does not execute the online predictor,
recommendation implementation, operator store, PID or plant.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_operator_forecast.py \
  --session results/operator_forecast_synthetic_v1 \
  --output results/operator_forecast_synthetic_audit_next.json
```

The actual synthetic, real and generated-fixture sessions pass independent checks:
94 native states and displayed contexts, two explicitly automated fixture reviews,
four preserved thermal comparisons and 7,212 process records. The browser passes
26 checks with no JavaScript errors, including invalid durations, explicit
suggestion selection, delayed image decoding, preserved context, manual resume,
mobile layout and read-only actual sources. The generated fixture requests 33
model seconds at time 130 and resumes at 150, leaving 20 seconds of recipe lag.
Both intermediate and final results remain intact. These are software records,
not independent physical experiments or human reviews.

Real and synthetic previews leave all session/source files unchanged. Native real
images remain unscored, and no final test data or physical equipment is used.
Next follow the [roadmap](../roadmap.md): review real outlines/events, characterize
capture and action response, define a supported adaptive objective, and complete
the remaining face/inclusion, first-hour, HPC and Qiskit evidence.
