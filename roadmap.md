# CrystaLab roadmap — 2026-11-11

## Current position

The simulation and data-preparation milestone is complete. We are ready for a
first **synthetic, growth-only prediction experiment**. Further CFD is conditional
on a specific modeling need. No ML fitting, vision model or controller has started.

The long-term objective is to improve growth while maintaining crystal quality
and respecting equipment/process constraints. The first growth regressor cannot
establish quality or operate a physical controller: valid quality targets,
actuator limits and experimental evaluation are separate later gates.

## Completed

- [x] Illustrative 3D faceted growth, liquid transport, conservative local uptake,
  moving-boundary displacement, reproducibility and numerical checks.
- [x] Fixed-domain pressure, viscosity and momentum verification. These operators
  remain separate from the engine used to generate growth labels.
- [x] Bounded controls, delayed/noisy sensor proxies, exact forecast labels,
  whitelisted features, grouped splits, raw records and independent dataset audit.
- [x] Accept the 32^3 focused grid and timestep protocol under the original 2%
  refinement gate. Preserve coarser failures and limits in the numerical checkpoint.
- [x] Generate 360 first-attempt episodes across 60 distinct initial concentrations
  and six recipes: 1,080 audited samples, split 648/216/216 by whole base group.
  There are 720 fully observed samples and 360 explicitly missing startup-input rows.
- [x] Remove superseded pilots, duplicate extracted inputs and generated caches;
  keep the active dataset, original source archive and numerical evidence.
- [x] Separate the current plan from archived development notes and document where
  local generated data lives and why it is absent from GitHub.

Working data: `data/synthetic_growth_v1/samples.jsonl`.
Use the [dataset card](docs/SYNTHETIC_DATASET_V1.md),
[compact report](docs/SYNTHETIC_DATASET_V1.json) and
[pipeline contract](docs/SYNTHETIC_DATA.md). The last full source-verification
checkpoint passed 227 tests; that numerical source is unchanged by this cleanup.

## Next: first growth-prediction benchmark

- [ ] Implement a dataset loader using the saved feature whitelist and existing
  group splits. Keep IDs, exact hidden parameters and future outcomes out of inputs.
- [ ] Evaluate zero-growth persistence and recent observed growth-rate baselines
  for volume gain over the next 10 model time units.
- [ ] Implement a small XGBoost comparison. Use training/validation for fitting and
  model choices; reserve the fixed test groups for the final agreed comparison.
- [ ] Report MAE/RMSE and signed error, macro-averages by base group, per-recipe
  results, and separate startup/observed-state results. Avoid future imputation.
- [ ] Use learning curves over training groups to decide whether more data is
  needed before expanding generation or model complexity.

The deliverable is a reproducible comparison with the recent-growth baseline.
It measures learning within the recorded simulator regime, not real KDP accuracy.
The current commit contains data preparation and documentation, not these models.

## Later, when justified

- Obtain suitable measured KDP growth trajectories; calibrate the relevant response
  and evaluate on whole held-out experiments. COD provides a structural reference,
  not growth histories. Published aggregates are not independent raw trajectories.
- Develop camera/vision data and valid quality labels separately. A single view
  does not directly supply every idealized 3D proxy in the synthetic table.
- Add physical control only with validated action/quality responses, process and
  equipment limits, actuator timing and independent closed-loop assessment.
- Resume detailed flow/scalar and growth coupling if a required output or measured
  discrepancy needs resolved flow. Specify the validation experiment first.
  Physical rotation/ports, thermal/buoyancy effects and higher fidelity follow
  that modeling need; they are not prerequisites for the first synthetic model.

See [AI readiness](docs/AI_READINESS.md),
[physical validation](docs/REAL_DATA_VALIDATION.md) and the
[documentation index](docs/README.md).

## Working rules and history

Preserve the existing package structure and explain changed files. Every new or
changed code file starts with a purpose summary. Commit and push only when the
user authorizes each action; the present cleanup request authorizes a local
commit **without a push**. Do not add model training as part of this cleanup.

The [archived roadmap](docs/DEVELOPMENT_HISTORY.md) preserves earlier requirements,
experiments, failures and superseded status/permission notes. Current priorities
are defined here. Generated results and obsolete pilot directories were removed;
versioned checkpoint JSON files preserve their original protocols and outcomes.
