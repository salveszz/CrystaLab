# Simulation readiness for AI development

The intended model predicts growth and quality under candidate temperature,
rotation and circulation actions. Training trajectories need a verified response
to those actions, and observations must represent quantities available to future
sensors/cameras. A passing passive-fluid test does not supply growth/quality
labels or a calibrated actuator response.

The current work is physical-reference and source-audit preparation only;
model training remains deferred. The existing package layout and placeholder
commands are preserved. See [PHYSICAL_DATA.md](PHYSICAL_DATA.md).

## Real-data preparation and the first XGBoost baseline

Real-data ingestion and an offline supervised baseline can proceed alongside
the remaining simulation work. They do not require a complete CFD growth model
when the training targets are measured experimental outcomes. The original
`Pitch_KDP.pptx`, slides 7-9, puts real/synthetic data before model development
and describes HPC as an extension. The roadmap retains camera-based monitoring,
human oversight and an XGBoost predictive layer; a tabular growth baseline is
one component of that system, not the complete vision/control demonstration.

1. Define experiment/source identifiers, timestamps, physical units and
   concentration basis, measurement uncertainty, available sensor/geometry
   features, action histories and an explicit prediction horizon. Preserve
   missing values and distinguish measured quantities from estimates.
2. Import and audit the first real dataset. The published records in
   `configs/benchmarks/` contain aggregate measurements, not raw trajectories.
   The ignored `data/transitions*` files are obsolete planar synthetic data;
   neither supplies a validated current training dataset.
3. Split whole experiments before constructing temporal windows. Train a simple
   reference predictor and an XGBoost growth regressor using only features
   available at prediction time; report held-out experimental errors. Predict
   the quantity actually measured, such as facet-normal advance; do not infer
   true mass gain from an uncalibrated single camera view.
4. Add quality/anomaly targets only when corresponding independent labels
   exist. Keep real evaluation separate from synthetic training data. Learning
   action effects requires suitable action variation and timing; predictive
   accuracy alone does not establish a beneficial control schedule.

`scripts/train_xgboost.py`, `scripts/generate_dataset.py` and the environment
adapter remain placeholders. No model has been trained or validated by the
flow/solute/pressure checks. The sequence below applies to simulation-generated
growth-control training; it does not block the real-data work above.

## Current position

| Dependency | Current evidence or remaining work |
|---|---|
| Convex crystal geometry and local facet sampling | Implemented and tested in the illustrative 3D growth engine. |
| SI contract and reference properties | Explicit concentration bases and a sourced constant-property point at 30 C; broader property curves remain to obtain. |
| Fixed-domain momentum and passive solute | Uniform and unequal/focused Cartesian grids have independent numerical references. |
| Sloped crystal momentum boundary | Actual cut-cell geometry, bounded topology refinement and compatible pressure projection are verified. Embedded no-slip viscosity and nonlinear momentum transport remain required; the full time-stepping runtime still rejects partial momentum cells. |
| Flow coupled to crystal growth | Pending. Reconcile interfacial solute flux, solution/solid densities, solvent accounting and rising-level volume closure. |
| KDP kinetic calibration | Published reference records exist; face laws, applicable ranges, uncertainty and independent experimental validation remain required. |
| Temperature/buoyancy and physical forcing | Source thermal properties and density derivatives; define physical ports/rotation geometry, setpoints and histories. |
| Quality target | Independent measurable quality limits and labels remain required. The synthetic defect counter is not optical quality. |
| Actuator dynamics | Define delays, slew/ramp limits and command timing before controller comparisons. |
| AI observation/action adapter | Placeholder only. Specify timestamped units, history windows, measurement confidence and missing values; keep exact simulator labels separate from observations. |
| Training/validation protocol | Define fixed-recipe/PID baselines, equal starting conditions/budgets, whole-experiment splits and numerical acceptance ranges before generating training data. |

## Next implementation order

1. Complete oblique momentum cut cells on the rectilinear mesh: add embedded
   no-slip viscosity, conservative nonlinear momentum fluxes and verified
   small-cell time stepping to the geometry/pressure foundation in
   [CUTCELL_PRESSURE.md](CUTCELL_PRESSURE.md).
2. Couple conservative interfacial uptake and boundary motion using an explicit
   density/partial-volume closure; repeat coupled growth/flow/scalar refinement.
3. Add sourced physical properties, defined forcing and applicable kinetics,
   then compare with independent published or experimental measurements.
4. Define quality/process limits and actuator/observation timing; implement the
   shared simulation-to-controller contract and deterministic episode recording.
5. Generate bounded, versioned trajectories and compare fixed-recipe, PID and
   action-conditioned XGBoost control over held-out whole experiments.

Each completed numerical gate applies to its measured regime. New seed shapes,
orientations, growth speeds, forcing ranges and grid focus bands require their
own resolution checks. The current work prepares the flow foundation; it does
not yet satisfy the roadmap's gate for growth-control training.
