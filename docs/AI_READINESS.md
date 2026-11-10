# Readiness for the first growth model

The simulation is intended to generate synthetic examples for model development,
pretraining and controlled experiments. A first XGBoost prototype can learn the
simulator's growth behavior before every physical extension is complete. Its
performance on synthetic holdouts describes that simulated world. Physical KDP
predictions and control recommendations need separate experimental evidence.

The current priority is a bounded data-readiness milestone, followed by the first
growth-only baseline/model. Further CFD expansion is conditional on an identified
modeling need. The existing dataset-generation entry point now records controlled episodes and
causal growth labels. The training entry point remains a placeholder; there is no
model, fitted baseline or ML dependency. See [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)
for commands, schema, provenance and a complete code review map.

## Alignment with the original plan

[Pitch_KDP.pptx](../Pitch_KDP.pptx), slides 7–9, calls for real and synthetic data,
separate real-data evaluation and literature-supported limits. Its explicit
synthetic examples are images of defects/lighting/reflections for vision. Slide 8
presents detailed Navier–Stokes/HPC as an extension. The later roadmap adds
XGBoost growth prediction from tabular trajectories; XGBoost is not named in the
deck. These are related but different datasets and model tasks.

A fluid solver can improve transport assumptions when coupled to growth. It does
not generate optical-defect annotations or calibrated quality targets. Camera
image synthesis and annotation are not prerequisites for a first tabular growth
prototype with explicitly declared observation assumptions.

## Simulator fidelity depends on the intended result

| Intended result | Required preparation |
|---|---|
| Synthetic growth software prototype | A reproducible bounded simulator, known units/assumptions, causal input/target definitions, audited episodes and independent scenario splits. Full CFD is optional. |
| Real KDP growth prediction | A response model calibrated for the intended range and evaluated on held-out experiments. A reduced or empirical model can be sufficient; choose CFD if the target or discrepancies justify it. |
| Quantitative local flow/mass-transfer prediction | Verified and appropriately resolved flow/solute coupling, actual forcing/boundaries and relevant experimental comparisons. Moving-growth labels additionally need interfacial uptake/motion and volume closure. |
| Visible-defect/segmentation model | Rendered or measured images with spatial annotations and separate real-image evaluation. A scalar synthetic defect count cannot label optical quality. |
| Physical growth/quality controller | Valid action-response and quality targets, equipment/process limits, actuator timing and independent closed-loop assessment beyond regression accuracy. |

More synthetic samples cannot independently correct a wrong growth law or an
unsupported action effect. [NIST's calibration guidance](https://www.nist.gov/publications/calibration-and-uncertainty-analysis-predictions-computational-models)
distinguishes uncertainty in measurements/inputs from inadequacy of the model,
and discusses calibrating simulations with limited physical experiments. For
this project, physical validation is a separate gate from software prototyping.

## What we have now

The existing illustrative 3D growth engine already has geometry, conservation,
reproducibility and numerical-convergence checks in declared regimes. Its length,
time, mass and growth coefficients have not been converted into a calibrated
physical KDP growth model merely by adding SI reference files.

The new fixed-domain cut-cell momentum operators pass 214 tests and 20 declared
numerical cases; see [CUTCELL_MOMENTUM.md](CUTCELL_MOMENTUM.md). They remain
separate from the growing-crystal engine and currently do not change its growth
labels. This checkpoint includes the numerical work alongside the data pipeline.
Its original coarse mesh failures, resolved focused mesh budgets and scope limits
remain explicit.

## Implemented data milestone and final gate

The recorder now supports temperature and aggregate agitation commands, exact
sample/decision clocks, isolated noisy/delayed sensor proxies, causal lagged
features, raw truth separation and per-step invariants. It records every resolved
configuration, actual source contents, final fields and failed attempts. Whole
base/family groups receive their split before samples exist. The audit reconstructs
features, labels and splits from raw records rather than trusting a saved green
report. Serial/parallel reproducibility and tamper/failure paths are tested.

The initial target is volume gain over 10 model time units; the model units are
explicit and quality remains unavailable. The pilot contains six initial
concentration groups and six paired control recipes, with fixed kinetics,
geometry and 30-unit duration. It measures a limited synthetic interpolation
problem, not generalization to new physical laws or real KDP. A single camera
cannot supply all the idealized geometric proxies in these first tabular inputs.
The original vision dataset remains a separate future task.

The data-generation gate is complete for the declared regime: 36 32^3 focused
episodes, 108 audited samples, 60 completed resolution/action cases, exact
full-sized serial/parallel replay and a changed-worker CLI resume. All 227 tests
pass. The finest spatial label/facet errors are 0.758%/1.266%; the pilot timestep
and finer timestep comparisons meet the original 2% gate. Coarser numerical
failures and the corrected vertex-order audit attempt remain recorded.

See [SYNTHETIC_DATA_CHECKPOINT.json](SYNTHETIC_DATA_CHECKPOINT.json) for the exact
configuration, source hashes, original tolerances, per-comparison results,
action sensitivity, replay and cost. The growing engine still uses illustrative
units/laws. Several small early temperature/high-agitation effects do not exceed
the summed resolution-change estimates across all horizons; those estimates
cannot justify an action ranking. There is no validated quality/cost tradeoff.

The user authorized committing and pushing the simulation/data checkpoint on
2026-11-10. We are ready to begin a separate synthetic growth-model phase using
this bounded data contract. No further CFD feature is required for that task.
This checkpoint adds no ML code, fitted baseline or ML dependency. Adequate
training-set size and real performance remain to be assessed in subsequent phases.

Persistence and recent-volume-rate baseline definitions and error metrics are
recorded for the subsequent model phase, without fitting or computing predictions
now. A small pilot establishes data plumbing, not the eventual sample count.
Further synthetic groups/repeats can use the generator after reviewing the
accepted regime. Different ranges/geometry/horizons require renewed sensitivity
checks; learning curves and training decisions belong to the later ML work.

## What can wait, and when to resume it

The next higher-fidelity physics task is canonical closed/periodic flow/scalar
integration, followed by passive-solute conservation/refinement and synchronized
clocks. The shear-free rigid lid needs its own boundary stencil and verification.
Physical uptake/moving geometry and density/partial-volume closure follow if
that CFD route is used to generate growth data. Current runtime guards remain.

Resume that work when a required output depends on resolved flow, a data
comparison identifies missing transport behavior, or a calibrated reduced model
cannot represent the intended control range. Specify the output/decision, expected
benefit and validation experiment first. Thermal/buoyancy curves, physical
stirrer/port geometry, turbulence and further solver optimization are conditional
extensions rather than automatic prerequisites for every first ML experiment.

## Real-data work and physical claims

Collect suitable measured growth trajectories and conditions in parallel. COD is
a structural reference, not growth history. Published KDP fit summaries and
aggregate tables support bounded reference checks; they are not raw independent
experiments and cannot be expanded into measured trajectories. No reviewed raw
KDP growth time series is currently imported; see [PHYSICAL_DATA.md](PHYSICAL_DATA.md).

For physical predictions, fit the relevant response on calibration experiments
and evaluate on whole held-out real experiments, with uncertainty and operating
limits. Keep real test data out of simulator calibration and model selection.
Synthetic-only holdouts cannot establish transfer to real crystals. A real-data
baseline can proceed when suitable observations exist, without waiting for full
CFD. Quality, camera monitoring, actuator behavior and physical control require
their own labels/checks. See [REAL_DATA_VALIDATION.md](REAL_DATA_VALIDATION.md).
