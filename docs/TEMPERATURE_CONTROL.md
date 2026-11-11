# Temperature PID software baseline

Temperature feedback now runs through explicit process records and a delayed
simulated actuator. The fixed-recipe/PID comparison covers four declared thermal
scenarios, missing/held feedback, saturation and scripted manual operation. The
[checkpoint](TEMPERATURE_CONTROL_CHECKPOINT.json) binds the configuration, source,
actual outputs and regression result. The generated comparison is
`results/temperature_control_baseline_v3/comparison.png` locally.

This completes the first software feedback baseline in roadmap Batch 4. It does
not implement the adaptive supervisor, connect vision events to actions, calibrate
KDP equipment or demonstrate defect prevention. Camera, rotation-stage, pump and
thermostat access remain unconfirmed. Existing illustrative growth controls keep
their original model-unit meaning; no new thermal constants are assigned to them.

## Records and units

`controllers.contracts` defines immutable scalar observations, command proposals
and actuator limits. Each record identifies its run, clock, source, channel and
unit. Missing observations retain missing values and an explicit reason.

| Channel | Meaning and unit |
|---|---|
| `temperature` | Measured solution temperature, K; observation only. |
| `bath_temperature` | Requested bath/jacket temperature, K; separate from solution temperature or heater power. |
| `rotation` | Signed angular velocity, rad/s. |
| `flow` | Volumetric flow, m3/s; its sign needs a declared equipment/geometry convention. |

The generic records preserve separate rotation/flow quantities; their hardware
drivers and response models do not exist. No aggregate-agitation conversion is
invented. The executed experiment commands only the simulated bath channel.

Observations have measurement and receipt times. Proposals have IDs, evidence
IDs, proposal/expiry times and the requested value. Limits specify range, command
slew and provenance. The simulated adapter distinguishes accepted, rejected and
applied receipts, including delayed application time and value. Its authorization
labels are `simulation_policy` and `scripted_simulation_override`; neither is a
human approval or a hardware authorization.

## PID behavior

`controllers.pid.TemperaturePID` computes a bath request from the target and
already received temperature feedback. It does not read plant truth. The nominal
feedforward is the target temperature, with proportional, integral and filtered
measurement-derivative corrections:

```text
error = target - measurement
raw request = target + Kp*error + I + D
limited request = range limit, then command slew limit
I prediction = bounded(I + Ki*error*controller_dt)
I next = bounded(I prediction + (limited - raw)*(1-exp(-controller_dt/Taw)))
```

`Kp` is dimensionless, `Ki` has units 1/s and `Kd` has units s for this
temperature-to-temperature interface. The derivative uses successive measurement
timestamps and a first-order exponential filter. Repeated polls of the same
packet hold the derivative estimate; setpoint changes do not enter its derivative.
This follows the measurement-derivative motivation in
[NI's PID algorithm documentation](https://www.ni.com/docs/en-US/bundle/labview/page/implementing-the-pid-algorithm-with-the-pid-vis.html).
The back-calculation correction uses the actually limited command, including
slew limiting, consistent with the saturation-feedback approach described by
[MathWorks](https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html).
The implementation is independent Python code and does not require either product.

The initial zero-dt tick holds the configured initial output. In automatic mode, stale, missing,
invalid, not-yet-received, conflicting or out-of-order feedback requests the
declared fallback. A controller update gap also triggers fallback. Integration
freezes and derivative history resets. Reacquisition/manual release aligns the
integral with the previous limited request where its declared range permits;
the first recovery tick adds no new error integral. A zero integral gain disables
integral state. Scripted manual output remains bounded, tracks integral state
when feedback is usable, and logs the external mode separately.

The configured fallback is a test assumption, not a universal safe physical
action. Back-calculation tracks the limited requested output, not unknown
downstream hardware failures. Missing/rejected hardware acknowledgments and
equipment-specific fallback/timeout behavior still need an actual adapter and
characterization before physical use.

## Thermal experiment

`controllers.thermal.ThermalPlant` integrates exactly between piecewise-constant
inputs and disturbances:

```text
dT/dt = (bath request + equilibrium disturbance - T) / time constant
T(next) = equilibrium + (T(now)-equilibrium)*exp(-elapsed/time constant)
```

The model is a declared first-order software fixture. Its time constants,
disturbance offsets, gains and limits are chosen test parameters, not KDP or
thermostat measurements. Fractional actuator delays and disturbance boundaries
are integrated at their actual times. Command slew bounds changes between accepted
requests; the simulated actuator applies those held requests after a constant delay.

`configs/temperature_control_baseline.json` declares four development scenarios,
600 simulated seconds and a one-second controller tick. Each method gets the same
recipe, plant assumptions, noise seed, sensor delay and scripted interventions.
Only the PID consumes sensor feedback. Plant truth is saved separately for scoring.
Sensor delivery selects an already sampled past value; held packets preserve
their old measurement times. The fixed recipe is open-loop at this interface,
without an implicit inner thermostat PID.

| Declared scenario | Fixed-recipe sample MAE (K) | PID sample MAE (K) | Fixed/PID command variation (K) |
|---|---:|---:|---:|
| Nominal ramp | 0.305 | 0.143 | 3.000 / 17.769 |
| Changing thermal loads | 1.006 | 0.362 | 3.000 / 24.237 |
| Unreachable load, then recovery | 1.749 | 0.796 | 3.000 / 21.180 |
| Missing/held feedback and scripted override | 0.432 | 0.285 | 6.317 / 22.263 |

PID lowers mean error in these constructed cases, with greater command movement.
It does not improve every metric: maximum above-target deviation is 0.965 versus
0.867 K in the load case and 0.464 versus 0.280 K in the unreachable-load case.
The unreachable case has 106 saturated PID ticks. The fault case has 39 fallback
ticks, 22 invalid sensor ticks and 20 scripted-manual ticks. Gaps/stale delivery
are not counted as valid fresh feedback. No physical acceptance budget or tuned
hardware performance is claimed. These are four declared scenarios, not eight
independent physical experiments or a held-out model generalization study.

## Reproduce and inspect

Use unused output paths for reruns; these named paths already exist locally.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-review-mpl .venv/bin/python scripts/run_control_baseline.py \
  --config configs/temperature_control_baseline.json \
  --output results/temperature_control_baseline_v3
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_control_baseline.py \
  --run results/temperature_control_baseline_v3 \
  --output results/temperature_control_baseline_v3_audit.json
```

Each of eight runs retains 601 observation/decision rows and a command ledger.
Every row keeps sensor data, recipe target, controller state, proposal,
authorization, dispatch receipt, applied receipts and separate plant truth.
The complete experiment has 4,808 rows. At the finite window's end, two or three
accepted commands per run are still awaiting their declared application times;
the manifest preserves them and does not claim they were applied.

The independent auditor reconstructs the artifact inventory and hashes, original
configuration/source snapshots, recipe, sensor timing/noise, command ledger,
range/slew, delayed acknowledgments, analytic thermal response and all metrics.
It calls neither the controller nor plant classes and does not prove physical
validity or every possible PID trajectory. Tests include analytic PID/plant cases,
reproducible ledgers and a rehashed artifact with future feedback that still fails
semantic audit. The earlier v1 experiment is retained with its original producer;
v2 includes the corrected derivative clock and audit sources; v3 preserves the
project's NumPy minimum by computing the trapezoidal metric with basic array
operations. The observed control trajectories and rounded results are unchanged.

## Next pitch work

Candidate inspection evidence now reaches bounded rule supervision and real-video
shadow proposals; see [Event supervision](EVENT_SUPERVISION.md). Preserve the
unknown physical clock of public display clips;
do not synthesize real sensor response from replay. Then add the pitch's adaptive
state/forecast policy and compare fixed recipe, PID-only, rule-supervised and
adaptive variants on declared evidence. Continue real label review, relevant
capture and equipment characterization alongside this software work. Face
asymmetry, the operator interface, real first-hour performance, HPC and Qiskit
remain separate open requirements in the [roadmap](../roadmap.md).
