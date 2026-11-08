# Simulation readiness for AI development

The intended model predicts growth and quality under candidate temperature,
rotation and circulation actions. Training trajectories need a verified response
to those actions, and observations must represent quantities available to future
sensors/cameras. A passing passive-fluid test does not supply growth/quality
labels or a calibrated actuator response.

## Current position

| Dependency | Current evidence or remaining work |
|---|---|
| Convex crystal geometry and local facet sampling | Implemented and tested in the illustrative 3D growth engine. |
| SI contract and reference properties | Explicit concentration bases and a sourced constant-property point at 30 C; broader property curves remain to obtain. |
| Fixed-domain momentum and passive solute | Uniform and unequal/focused Cartesian grids have independent numerical references. |
| Sloped crystal momentum boundary | Partial momentum cells are explicitly rejected. Actual cut-face pressure/viscous/advection operators and refinement remain required. |
| Flow coupled to crystal growth | Pending. Reconcile interfacial solute flux, solution/solid densities, solvent accounting and rising-level volume closure. |
| KDP kinetic calibration | Published reference records exist; face laws, applicable ranges, uncertainty and independent experimental validation remain required. |
| Temperature/buoyancy and physical forcing | Source thermal properties and density derivatives; define physical ports/rotation geometry, setpoints and histories. |
| Quality target | Independent measurable quality limits and labels remain required. The synthetic defect counter is not optical quality. |
| Actuator dynamics | Define delays, slew/ramp limits and command timing before controller comparisons. |
| AI observation/action adapter | Placeholder only. Specify timestamped units, history windows, measurement confidence and missing values; keep exact simulator labels separate from observations. |
| Training/validation protocol | Define fixed-recipe/PID baselines, equal starting conditions/budgets, whole-experiment splits and numerical acceptance ranges before generating training data. |

## Next implementation order

1. Implement and verify actual oblique momentum cut cells on the rectilinear
   mesh, including small-cell behavior, wall conditions and pressure compatibility.
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
