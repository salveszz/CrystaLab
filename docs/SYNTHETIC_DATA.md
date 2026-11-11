# Synthetic growth data pipeline

The active dataset is `data/synthetic_growth_v1/samples.jsonl`: **360 episodes,
60 concentration groups and 1,080 accepted forecast samples**. See the
[dataset card](SYNTHETIC_DATASET_V1.md), [report](SYNTHETIC_DATASET_V1.json) and
[tracked data guide](../data/README.md). Generated raw files are local and
Git-ignored; the protocol and compact reports are versioned. The old pilots are
retired and must not be mixed into the working split.

The pipeline prepares tabular growth data without fitting/scoring a model or
optimizing a controller. It uses the existing illustrative 3D engine. Standalone
fixed-domain CFD operators do not currently generate its growth labels.

## Generate, resume and audit

From the repository root, audit an existing dataset:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_dataset.py data/synthetic_growth_v1
```

Generate the working design on a fresh checkout, or select a new output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/generate_dataset.py \
  --config configs/synthetic_growth_dataset_v1.json \
  --output data/synthetic_growth_v1 --jobs 8
```

Generation refuses to overwrite an existing output. After stopping an interrupted
run, add `--resume` with the identical protocol, source contents, Python/NumPy/SciPy
versions and numerical thread settings. Worker count can change. Completed
episodes are verified/reused; failed/interrupted episodes restart from time zero
in a new `attempt_N`, retaining earlier evidence. Do not launch a concurrent
resume while the original process is active.

Process-worker library callers need a real module and the usual
`if __name__ == "__main__"` guard. Interactive/stdin callers can use `jobs=1`.
The file-based CLIs already provide the guard. Bytecode writes can be disabled as
above; numerical thread settings are part of the dataset identity.

## Protocols and supported regime

| Configuration | Role |
|---|---|
| `configs/synthetic_growth_dataset_v1.json` | Working dataset: 60 distinct concentrations, 360 episodes, six recipes each. |
| `configs/synthetic_growth.json` | Small six-group verification fixture/pilot design; retained for tests and numerical studies. |
| `configs/synthetic_growth_verification.json` | Three grids, three timesteps and endpoint concentration/recipe selection for the refinement study. |

The working protocol uses a focused 32^3 grid, dt=1, 30 model-time-unit episodes
and a fixed 0.75-length-unit sampling layer. Initial concentration spans 1.25–1.45
under one fixed geometry and kinetic law. The six paired recipes are baseline,
cooling, warming, low agitation, high agitation and staged controls. The working
split is 36/12/12 base groups and 648/216/216 forecast samples. Exact group values
and assignments are frozen in the configuration and manifest.

All recipes from a base group share an initial state and split. Related parameter
families must share `family_id` and split. Copies differing only in a simulation
seed cannot cross splits. The benchmark tests concentration interpolation under
the recorded model; it does not test unseen physics or arbitrary new action
combinations. Design bounds and noise settings are assumptions, not KDP calibration.

## Actions, clocks and targets

Samples are recorded every 2 time units; decisions occur every 10 and forecasts
span 10. Sparse recipes expand to explicit held intervals. A state at a decision
timestamp is measured before its new command. The bath command changes the
thermal-relaxation setpoint, not the instantaneous temperature field. Actuator
delay is explicitly zero. Temperature bounds are 22–28 and aggregate agitation
bounds are 0–2; changes per decision are limited to 3 and 2 respectively.

Rotation and circulation enter the current mixing/growth laws through their sum.
The adapter records their equal split and exposes one aggregate agitation input.
Its saturating enhancement has no validated quality penalty or equipment cost;
it cannot justify choosing maximum stirring in a real experiment.

The primary label is exactly `V(t + H) - V(t)`, joined at recorded endpoints.
Auxiliary labels are mean volume-growth rate and density times volume gain.
There is no label interpolation. Incomplete horizons are omitted and counted;
the accepted dataset has none. Quality is always null/unavailable. The optional
engine defect counter is not a physical quality target.

## Observation and feature contract

The eight proxies are volume, width, depth, height, surface area, temperature,
concentration and liquid level. They are idealized global simulation measurements,
not an implemented camera/perception system. The working data adds 2-unit sensor
delay, independent Gaussian relative noise with sigma=0.002 for non-temperature
proxies, and absolute temperature sigma=0.02. Noise is illustrative, not calibrated.
Invalid negative positive-quantity measurements become missing rather than clipped.

Sensor RNG is isolated from simulation RNG. Related recipes share standardized
sensor draws within a group; unrelated groups receive distinct reproducible
streams. Noise cannot change growth or labels. Sensor source times and ages are
explicit. Past volume and its observed finite-difference rate use a four-unit lag.
No future observation supplies a prediction-time input.

The 14 feature names are the eight proxies, `volume_lag`, `recent_volume_rate`,
`observation_age`, `temperature_command`, `agitation_command` and `horizon`.
A `missing` dictionary accompanies them. All 360 startup rows lack delayed
sensor/history inputs; the other 720 rows are fully observed. Null inputs are
not null labels. Evaluate the two strata separately and do not impute from future
measurements. Features are not asserted to be a complete Markov state.

Group/episode/split IDs and decision/label timestamps are metadata. Exact hidden
parameters, future state, defects, budgets and final arrays are audit/provenance
data. They must not be added to the feature matrix. The manifest stores the strict
whitelist; data preparation fits no preprocessing or predictive baseline.

## Units

The mode is `illustrative_growth_3d_v1`. There is no asserted conversion from this
model's coordinates, kinetics or concentration to a calibrated physical KDP run.

| Fields | Units |
|---|---|
| `volume`, `volume_lag`, label `volume_gain` | model_length^3 |
| `width`, `depth`, `height`, `liquid_level` | model_length |
| `surface_area` | model_length^2 |
| `temperature`, `temperature_command` | model_temperature |
| `concentration` | model_solute_mass / model_carrier_volume |
| `recent_volume_rate`, label `mean_volume_growth_rate` | model_length^3 / model_time |
| `observation_age`, `horizon`, event/label timestamps | model_time |
| `agitation_command` | illustrative dimensionless command |
| Label `mass_gain` | model_solute_mass |

## Artifacts and audit

| Output | Purpose |
|---|---|
| `manifest.json` | Schema, units, complete resolved cases, feature whitelist, sources, environment, starting Git HEAD and execution history. |
| `source_snapshot/` | Exact source contents, including any uncommitted code present during generation. |
| `episodes/<id>/episode.json` | Identity, attempt/status, diagnostics, worker elapsed time and artifact hashes. |
| `episodes/<id>/attempt_N/events.jsonl` | Separate timestamped exact truth/facet data, applied controls and observations. |
| `episodes/<id>/attempt_N/final_fields.npz` | Temperature/concentration, liquid/cell volumes, edges and final crystal planes/vertices; no pickle. |
| `samples.jsonl` | Whitelisted causal features/missing flags, separate future labels and group-assigned splits. |
| `audit.json`, `independent_audit.json` | Generation audit and fresh reconstruction results. |
| `generation_run.json`, `generation.log` | Measured command, timing and output for the working run. |
| `protocol.json`, `dataset_report.json`, `README.md` | Portable configuration, compact report and guide bundled with working data. |

Truth/control records flush as integration progresses. Observations are constructed
at completion, so partial failed streams are not complete usable episodes. Every
growth substep checks finite fields, nonnegative concentration, containment,
relative tracked-mass/carrier errors <=1e-10 and continuity residual per minimum
cell volume <=1e-8. Final time must be exact. Failures cause generation failure;
no failed cases are silently filtered into a passing dataset.

The independent audit verifies snapshots and hashes, raw clocks/controls, causal
sensor construction, recorded invariants and final arrays. Arrays reproduce
solute/carrier mass, grid volumes and final geometry. It reconstructs the entire
sample table, labels and splits and compares them exactly. This checks data and
numerical consistency, not experimental accuracy or adversarial tamper security.

## Numerical evidence and cleanup

[SYNTHETIC_DATA_CHECKPOINT.json](SYNTHETIC_DATA_CHECKPOINT.json) preserves the
accepted pilot/refinement evidence and unchanged tolerances. The last full source
checkpoint passed 227 tests. All 360 working episodes passed on their first attempt;
see [SYNTHETIC_DATASET_V1.json](SYNTHETIC_DATASET_V1.json) for their fresh audit.

| Accepted comparison | Maximum label change | Maximum facet-advance change |
|---|---:|---:|
| 28^3 to 32^3, dt=1 | 0.7582% | 1.2663% |
| dt=1 to 0.5, 32^3 | 1.7310% | 0.7864% |
| dt=0.5 to 0.25, 32^3 | 0.8609% | 0.3937% |

Each must be <=2%. The coarser 24/28 facet result failed at 2.0203%; earlier
12/16/20 failures remain recorded. An isolated-axis refinement study does not
establish a formal continuous error bound or exclude all grid/time interactions.
Small early temperature/high-agitation effects do not uniformly exceed the
recorded sensitivity estimates; they cannot establish an action ranking.

To regenerate the numerical study when raw fields are needed:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/check_dataset.py \
  --config configs/synthetic_growth.json \
  --settings configs/synthetic_growth_verification.json \
  --output results/synthetic_growth_verification --jobs 8
```

Old study outputs were removed on 2026-11-10. Superseded pilots and byte-verified
extracted CIF duplicates were removed on 2026-11-11. Original source inputs,
accepted checkpoints and the complete active v1 dataset remain. Historical report
paths and commit-status notes describe the measurement date; report bytes/hashes
are preserved. The [development history](DEVELOPMENT_HISTORY.md) retains the old
roadmap. Do not regenerate a pilot merely to find the current model dataset.

## Implementation and later model evaluation

The recorder/controls live in `src/environment/kdp_env.py` and
`src/simulation/engine.py`. Dataset expansion, storage, labels and audit live in
`src/environment/dataset.py`. The three reusable CLIs are `generate_dataset.py`,
`audit_dataset.py` and `check_dataset.py`; no one-time cleanup script is added.
This documentation cleanup changes no simulation or ML code.

The next [model phase](AI_READINESS.md) compares zero-growth and recent-rate
baselines, then XGBoost, using the frozen groups and separate startup metrics.
Learning curves should determine further data needs. Synthetic test performance
alone cannot establish real KDP growth, camera/quality or physical control.
