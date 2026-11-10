# Synthetic growth data, before model development

This pipeline prepares tabular, action-conditioned growth examples from the
existing illustrative 3D engine. It implements no fitting, ML scoring, policy
search, XGBoost, vision model or ML dependency. The interactive runner and its
120-unit default are unchanged. Detailed fixed-domain CFD remains a separate
numerical track; it is not used to generate these growth labels.

## Run, resume and audit

From the repository root, using the installed project environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/generate_dataset.py \
  --config configs/synthetic_growth.json --output data/synthetic_growth_pilot_32 --jobs 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/audit_dataset.py \
  data/synthetic_growth_pilot_32
```

Choose a new output directory. Generation refuses to overwrite an existing one.
Add `--resume` to reuse verified complete episodes with exactly the same protocol,
source contents, Python/NumPy/SciPy versions and numerical thread settings.
Worker count can change: it changes execution cost, not episode identity or RNG.
A failed or interrupted episode restarts from time zero in a new `attempt_N`
directory; earlier raw records and failure metadata remain. Resume does not mean
continuing a partially integrated simulation state.

The independent audit re-reads source snapshots, episode identities, raw clocks,
commands, causal sensor construction, conservation records and final arrays. It
reconstructs the complete sample table, including labels and splits, from those
records and compares it exactly with `samples.jsonl`. Updating a file's checksum
alone cannot hide a changed feature, future sensor timestamp or incorrect label.
Audits check numerical/data consistency; they are not experimental validation.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_dataset.py \
  --config configs/synthetic_growth.json \
  --settings configs/synthetic_growth_verification.json \
  --output results/synthetic_growth_verification --jobs 6
```

The resolution study also supports an identical-settings `--resume`. It saves its
protocol before solving, exact source snapshots, every raw run and a report.
Failed numerical gates remain failures. The recorded pilot/checkpoint below is
the evidence for the supported settings; changing ranges, geometry, horizons,
gradients, sampling depth or duration requires checking the affected labels.

## Accepted pilot — 2026-11-10

The completed pilot in `data/synthetic_growth_pilot_32/` has 36 episodes and 108
samples (54 train, 18 validation, 36 test). Generation, a fresh standalone audit
and a CLI resume with three workers after six-worker generation all pass. A
full-sized serial replay matches raw truth, observations, samples and every final
field array exactly. All 227 repository tests pass, including 13 data-pipeline
and refinement-runner tests. Evidence/configuration/source hashes are preserved
in [SYNTHETIC_DATA_CHECKPOINT.json](SYNTHETIC_DATA_CHECKPOINT.json).

The accepted setting is a focused 32^3 grid, dt=1, fixed sampling depth 0.75 and
30-unit duration. The 60-case study is in
`results/synthetic_growth_verification_v4/`. Its declared acceptance gate passes:

| Comparison | Maximum forecast-increment change | Maximum final facet-advance change |
|---|---:|---:|
| 28^3 → 32^3, dt=1 | 0.7582% | 1.2663% |
| dt=1 → 0.5, 32^3 | 1.7310% | 0.7864% |
| dt=0.5 → 0.25, 32^3 | 0.8609% | 0.3937% |

Each entry must be <=2%. Coarser failures remain: the 24/28 facet change is
2.0203%, and the exploratory 12/16/20 study failed. An intervening attempt was
stopped after the audit incorrectly compared vertex order instead of the vertex
point sets; the corrected audit has a sloped-crystal regression and fresh runs.
No tolerance was increased. Not every coarse comparison passes merely because
the final acceptance gate passes.

Maximum pilot relative mass/carrier errors are 5.22e-13 / 2.38e-16; maximum
continuity residual per minimum cell volume is 1.50e-11. There are no incomplete
horizons. Volume-gain labels range from 0.09746 to 0.29880 model volume units.
All 36 startup samples explicitly lack the delayed proxies/history. Several small
early temperature/high-agitation effects do not exceed the summed grid/time
change estimates throughout all horizons. The low-agitation versus baseline
comparison does; none of this is physical calibration or an optimum claim.

Six-worker generation averaged 116.8 elapsed worker-seconds per episode under
concurrent numerical-study load. A serial replay took 84.4 worker-seconds while
eight refinement workers ran. The pilot mean extrapolates to about 32.4 summed
worker-hours for 1,000 same-duration episodes, rather than a guaranteed parallel
wall time. Cost changes with geometry, duration, resolution and machine load.
This pilot establishes reproducible plumbing and measured cost, not a sufficient
training-set size. Replay equality was checked on this machine; bitwise equality
across other numerical platforms is not promised.

Raw data/results remain under the existing Git ignore policy; the compact
checkpoint, configurations, source code and tests provide the reviewable record.
The earlier 28-cell pilot/replay remain as preliminary evidence and do not define
the final supported setting. A failed resume verification helper used Python
multiprocessing from stdin; the supported file-based CLI resumed successfully.
The earlier error remains in manifest history alongside its complete status.
Library callers using process workers must use a real module and the usual
`if __name__ == "__main__"` guard; interactive/stdin callers can use `jobs=1`.

## Fixed first contract

`configs/synthetic_growth.json` declares a 30-time-unit episode, 2-unit sampling,
10-unit decisions and a 10-unit forecast horizon. Commands are held through each
forecast; sparse recipes expand to explicit decision intervals. States at a
decision timestamp describe the instant before its new command. The bath command
changes the relaxation setpoint, not the instantaneous temperature field.
Actuator delay is explicitly zero. Command bounds are temperature 22–28 and
aggregate agitation 0–2; changes per decision are bounded by 3 and 2 respectively.
These numbers are design assumptions in model units, not measured KDP limits.

The primary label is exactly `V(t + 10) - V(t)`. Auxiliary labels are mean volume
growth rate and density times volume gain (mass gain). The join requires both
exact recorded endpoints; there is no label interpolation. A final decision with
an incomplete horizon is omitted and counted. Quality is always null/unavailable.
The optional engine defect counter is not a quality target.

The declared mode is `illustrative_growth_3d_v1`. Time, lengths, area, volume,
solute mass, temperature and concentration use the explicit `model_*` unit
strings saved in the manifest. They are not seconds, metres, kelvin or calibrated
KDP mass concentrations. Agitation is a dimensionless illustrative command.
Model diffusivities have length-squared/time units. The sampling layer is 0.75
model length, held fixed when changing the grid.

The column-to-unit mapping is:

| Fields | Units |
|---|---|
| `volume`, `volume_lag`, label `volume_gain` | model_length^3 |
| `width`, `depth`, `height`, `liquid_level` | model_length |
| `surface_area` | model_length^2 |
| `temperature`, `temperature_command` | model_temperature |
| `concentration` | model_solute_mass / model_carrier_volume |
| `recent_volume_rate`, label `mean_volume_growth_rate` | model_length^3 / model_time |
| `observation_age`, `horizon`, all event/label timestamps | model_time |
| `agitation_command` | illustrative dimensionless command |
| Label `mass_gain` | model_solute_mass |

The six base groups vary initial concentration: 1.25/1.35/1.45 for training,
1.30 for validation, and 1.28/1.40 for testing. Each group shares the same initial
state across baseline, cooling, warming, low agitation, high agitation and staged
recipes. All recipes and windows from a base group stay in one split. Repeated
parameter families must declare the same `family_id` and split. Copies differing
only in a simulation seed cannot cross splits. Seeds do not create new physical
experiments. The first pilot tests interpolation over initial concentration under
one fixed illustrative law; it does not test unseen kinetics, geometries or real
experiments. Alter the explicit group design before a different evaluation task.

## Observations, features and causality

The recorded proxy includes volume, width, depth, height, surface area,
temperature, concentration and liquid level. These are idealized synthetic sensor
proxies of global simulation measurements. In particular, a single real camera
does not directly supply exact 3D volume or surface area. This dataset does not
implement that future perception/estimation system.

The pilot uses a 2-unit sensor delay, independent Gaussian relative noise with
standard deviation 0.002 for non-temperature measurements, and absolute
standard deviation 0.02 for temperature. These are illustrative noise settings,
not sensor calibration. Sensor source timestamps and ages are explicit. Before a
measurement exists its value is null with a missing flag. Noise producing an
invalid negative positive quantity produces a missing measurement rather than a
clipped value. Observation RNG is isolated from simulation RNG; related recipes
share sensor random draws and unrelated groups receive distinct reproducible
streams. Noise cannot change growth or labels.

The 14 permitted feature columns are the eight proxies above, past observed
volume (`volume_lag`), its causal finite-difference `recent_volume_rate`, observation
age, the proposed held temperature/agitation commands and forecast horizon.
The lag spans four model time units. Missing flags accompany every feature.
At t=0 the delayed measurements are unavailable; this is explicit, and later
model work must choose how to handle that startup case without future imputation.
The features can represent partially observed dynamics; they are not asserted to
be a complete Markov state.

Split/group/episode IDs, decision time, sensor source time and label endpoint are
metadata, not extra numeric features. Exact simulator truth, hidden kinetic and
solute parameters, future state, defects, mass budgets and final arrays must not
be added to the feature matrix. The strict whitelist is saved in the manifest.
Data preparation produces no predictions and does not fit any preprocessing.

## Files and provenance

Each output contains:

- `manifest.json`: schema, simulator mode, complete resolved case/configuration
  matrix, units, feature whitelist, source hashes, numerical environment, starting
  Git HEAD and execution history. Source hashes include uncommitted code.
- `source_snapshot/`: exact source contents corresponding to those hashes, so an
  uncommitted working tree is recoverable for replay.
- `episodes/<id>/episode.json`: case identity, attempt, status, numerical diagnostics,
  measured worker elapsed time and artifact hashes.
- `episodes/<id>/attempt_N/events.jsonl`: timestamped exact truth (including stable
  facet planes/areas), applied controls and separate observations. Truth/control
  records flush as the simulation progresses. Observation records are constructed
  at episode completion; partial failed streams are not usable complete episodes.
- `episodes/<id>/attempt_N/final_fields.npz`: temperature, concentration, actual
  liquid/cell volumes, grid edges and final crystal planes/vertices; no pickle.
- `samples.jsonl`: causal feature/missingness dictionaries and separate exact
  future labels, with scenario-assigned splits.
- `audit.json`: freshly reconstructed counts, maximum invariant errors, omitted
  horizons and generation cost. The CLI recomputes this independently.

Every growth substep is checked for finite state, nonnegative concentration,
containment, tracked-solute conservation (relative error <=1e-10), carrier-volume
conservation (<=1e-10) and continuity residual per minimum cell volume (<=1e-8).
Final clocks and endpoint alignment must be exact. Final arrays independently
reproduce dissolved mass, carrier volume, grid-cell volumes and crystal geometry.
A run failing any gate is retained and excluded by a hard generation failure;
there is no silent filtering into a green dataset.

## Numerical sensitivity and action interpretation

The study uses three spatial grids and three timesteps at both endpoint
concentrations, with every recipe. Physics initialization is deterministic and
the face-neighborhood depth is fixed. The shared finest-grid/coarsest-time run
avoids a duplicate solve. The predeclared 2% gate applies to each actual forecast
volume increment and final maximum facet advance. It checks the finest adjacent
spatial pair and both adjacent timestep pairs, including the pilot timestep.
Earlier, coarser disagreements remain visible. This isolated-axis study does not
establish an asymptotic order or exclude all space/time interactions.

Action reports compare paired recipes with the same initial state, and expose
summed changes under spatial/time refinement alongside each action effect.
These finite-resolution changes are sensitivity estimates, not rigorous error
bounds. A zero difference before a staged intervention is expected. Effects
smaller than these estimates should not support an action ranking.

Rotation and circulation currently enter both mixing diffusivity and the growth
transport multiplier through their sum. The adapter records their equal split,
but exposes only aggregate agitation to avoid pretending they are independent.
Its saturating enhancement tends to favor more agitation; there is no validated
quality penalty or equipment cost. A surrogate can learn this model behavior.
It cannot establish that maximum agitation is a sensible real recipe. Likewise,
cooling/warming responses come from illustrative solubility and relaxation laws.
No controller optimization is part of this milestone.

## Evaluation protocol for the later model phase

Freeze this task, feature whitelist, horizon and scenario split before fitting.
The first task is growth prediction within the recorded simulator regime. Future
work should compare a zero-growth persistence prediction and a recent observed
volume-rate-times-horizon prediction (zero fallback when unavailable), then a
simple train-only baseline and the proposed growth model. None are computed here.
Report MAE/RMSE and mean signed error in volume-gain units, per-recipe errors,
missing/startup strata, and macro-averages over base scenarios. Avoid percentage
errors around zero growth and avoid treating correlated windows as independent
experiments. Fit/tune preprocessing and models on training/validation only; keep
the final test groups for the agreed final comparison. More base groups and
repeats may be needed once learning curves are measured in that later phase.

The pilot is a pipeline/readiness dataset, not proof of adequate sample size or
real predictive accuracy. Collect and independently evaluate measured KDP
trajectories before physical claims. Camera/quality data need their own labels.
Full CFD should resume only if a required output or measured discrepancy needs
resolved flow; it is not an automatic dependency of tabular synthetic growth.

## Code review map

- `src/simulation/engine.py`: adds `set_controls`; validates a replacement frozen
  configuration before assigning it to both engine and tank. Geometry, fields,
  coefficients and RNG are preserved. Growth/diffusion equations are unchanged.
- `src/environment/kdp_env.py`: replaces the placeholder with recording settings,
  schedule validation, synchronized sampling, isolated sensor proxies, per-step
  invariant checks and final-state capture. It is not a Gymnasium or ML agent.
- `src/environment/dataset.py`: expands scenario/recipe cases, assigns groups,
  derives observation seeds, creates exact labels/whitelisted features, saves
  durable artifacts, verifies episodes and reconstructs the whole dataset audit.
- `src/environment/__init__.py`: updates the package purpose comment only.
- `scripts/generate_dataset.py`: replaces the existing placeholder with the CLI.
- `scripts/audit_dataset.py`: audits an output independently of its old audit file.
- `scripts/check_dataset.py`: runs controlled numerical/action comparisons and
  retains raw evidence, source snapshots and unchanged acceptance thresholds.
- `configs/synthetic_growth*.json`: explicit pilot design and separate verification
  protocol; existing simulation configurations/defaults are preserved.
- `tests/test_dataset.py` and `tests/test_dataset_verification.py`: causal/clock/RNG/parallel/conservation checks, grouped
  splits, safe resume/failed attempts, and tamper detection on actual generated
  records and arrays.

Every added/changed code file starts with a purpose summary. The preceding
momentum/Stokes work included in this checkpoint is described separately in
[CUTCELL_MOMENTUM.md](CUTCELL_MOMENTUM.md); it is preserved by this change.
