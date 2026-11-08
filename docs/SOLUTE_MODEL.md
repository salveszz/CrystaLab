# Fixed-grid solute transport with resolved flow

`simulation.solute` carries concentration C in **kg KDP per m3 solution** on the
SI grid of `simulation.flow`. It solves the passive, constant-property equation
`dC/dt + div(u*C) = D*Laplacian(C)`. The default D is the sourced VTT-2006 value
5.7e-10 m2/s at the declared 30 C composition. Properties remain fixed during a
run. Verification fields with other concentrations test the numerical PDE; they
do not validate using that property point over a composition range.

This component uses uniform or rectilinear fixed geometry and the flow solver's fixed,
grid-aligned solid. There is no heat, buoyancy, surface uptake, dissolution or
growing-crystal integration. The existing illustrative `Simulation` retains its
own concentration basis and displacement model. Its empirical mixing and
agitation multipliers do not enter this resolved-flow component.

For actual unequal-cell capacities, areas, diffusion distances and reconstruction
offsets, see [RECTILINEAR_FLOW.md](RECTILINEAR_FLOW.md). `flow.cell_volumes`
supplies per-cell capacities; `flow.cell_volume` is only their minimum.

## Conservative and bounded update

`IncompressibleFlow.volume_fluxes_m3_per_s()` exports native x/y/z face velocity
times face area, with the same shapes as the staggered velocity arrays. It
returns independent arrays. Scalar transport uses these volume fluxes directly,
without interpolating a cell-centered velocity or solving another flow field.

For every independent open face between cells i and j, the numerical solute
flux F has units kg/s. A substep removes `dt*F` from i and adds exactly the same
value to j. Periodic end faces are aliases and contribute once. Blocked faces
have zero flux, including the crystal, tank walls and rigid lid. Diffusion uses
the same open connections with conductance `D*area/spacing` (m3/s).

Each Euler stage starts from donor-cell advection plus centered diffusion.
Monotonized-central linear slopes supply a less diffusive advective flux.
The difference is a signed mass correction. Summed incoming/outgoing correction
budgets and local liquid-neighbor concentration extrema bound each shared
correction in all three directions jointly. Boundary slopes fall back to zero.
Solved concentrations are not clipped or globally rescaled to repair mass.

Two bounded Euler stages form SSP RK2:
`C1 = E(C0,dt)` and `Cnew = (C0 + E(C1,dt))/2`.
Substeps satisfy
`dt * max_i(sum(outgoing_volume_flux + diffusive_conductance)/V_i) <= 0.45`.
Advection and explicit diffusion therefore share one stability restriction.
Limiting and boundary errors remain; no global second-order claim is made.
Time refinement isolates RK2 on an unlimited diffusion eigenmode.

The method follows the correction-budget framework in
[SOLUTE-FCT-2015](../SOURCES.md#solute-fct-2015) and the two-stage time integrator
in [SOLUTE-SSP-1998](../SOURCES.md#solute-ssp-1998). Our donor base, MC slopes,
local bounds and stagewise limiting are project choices; this is not the cited
paper's high-order CTU/RK4/smooth-extremum method.

## Momentum coupling and API

`SoluteTransport.advance()` advances only solute while holding its face volume
fluxes fixed. It automatically subcycles for stability, and an optional
`max_substep_s` can impose a smaller accuracy step. Analytical tests use
prescribed, divergence-free constant velocities; their momentum is not advanced.

`FlowSoluteSimulation.step()` advances momentum, then transports solute with the
average of the old and new face volume fluxes. Both endpoint fields satisfy the
same fixed-domain discrete continuity equation, so their average does too.
Scalar subcycling uses this average, including newly applied forcing. Solute
does not feed back into momentum in this passive, constant-density model.
The momentum method remains first order in time; scalar RK2 does not upgrade
the temporal order of the coupled solver.

The coupled clocks must agree. `run_until()` respects the momentum CFL and a
maximum timestep, takes the final partial step and stops at the requested time.
Invalid inputs or a failed scalar update leave the coupled physical state and
clocks unchanged. Flux validation rejects wall leakage, inconsistent periodic
endpoints, nonfinite fields, divergence and changed geometry rather than
silently repairing them. Initial liquid concentration must satisfy
`0 <= C < rho_solution`; solid cells store zero.

```python
import json
from pathlib import Path

from simulation.flow import IncompressibleFlow
from simulation.solute import FlowSoluteSimulation
from simulation.units import FluidPropertiesSI

reference = json.loads(Path("configs/physical_reference.json").read_text())
properties = FluidPropertiesSI(**reference["fluid"])
flow = IncompressibleFlow([0.01, 0.01, 0.008], [20, 20, 16], properties)
model = FlowSoluteSimulation(flow, concentration_kg_per_m3=269.576)
model.run_until(2.0, max_dt_s=0.05)  # rest unless forcing/initial velocity is supplied
print(model.measurements())
```

Full resolved input/forcing examples are in `scripts/check_solute.py`. No
illustrative `rotation` or `circulation` number is interpreted as a physical
actuator command.

## Verification protocol and evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_solute.py \
  --output results/solute_verification
```

The output directory must be fresh. Before measurement the runner records the
protocol, reference inputs, Git commit, package versions, thread settings and
SHA256 hashes, and archives the measured Python source files, including the shared
mesh implementation. Source changes
during the study invalidate the result. Outputs include initial/final scalar
fields, native velocity/pressure, fluid masks, domain dimensions, exact reference
fields where available, reports and three figures. Generated artifacts remain
ignored by Git; `SOLUTE_CHECKPOINT.json` preserves the final measured summary,
protocol and source identity in the repository.

The fifteen cases are:

- Three periodic translating/diffusing waves with nonzero x/y/z velocity,
  compared against analytically integrated cell averages. Relative error is
  normalized by the concentration fluctuation, not the much larger mean.
- Three closed-box Neumann cosine diffusion cases, compared with the continuous
  3D heat equation under spatial refinement.
- Three periodic Gaussian pulses at Pe about 17544 using the actual reference D.
  Exact Gaussian integrals define cell means. The difference of numerical and
  exact center-weighted pulse variances divided by `2*duration` defines an
  **excess-diffusivity diagnostic** on each grid. This measures broadening for
  this pulse, not a universal numerical diffusivity or interfacial resolution.
- Three diffusion timesteps compared with the exact exponential of the fixed
  centered spatial operator, isolating temporal error from spatial error.
- Rest through 120 s, and uniform/nonuniform solute in the resolved manufactured
  fixed-cube flow through 2 s. These check actual flow/scalar coupling and
  invariants; they have no independent concentration-field accuracy reference.

Every case checks total dissolved mass, initial concentration bounds, divergence
and blocked-face velocity. The configuration declares monotone refinement and
finest-level error tolerances, plus a pulse excess-diffusivity bound relative to
physical D. Tests retain failed gates and reject overwriting existing studies.
Numerical tolerances are working verification choices, not experimental errors.

The original uniform-grid source-matched study is
`results/solute_verification_final_5a015f5/`. All fifteen cases pass the original
protocol; its six historical sources are preserved with hashes and archived copies. All
raw fields are finite, and the three exported figures were visually inspected.
Maximum relative mass drift is 6.661e-16, concentration bound violations are zero,
and resolved uniform-solute error is exactly zero in this study.

| Finest numerical reference | Relative fluctuation RMS error |
|---|---:|
| Periodic 3D wave, 20x20x20 | 1.59195% |
| Neumann 3D diffusion, 16x16x16 | 0.05437% |
| High-Peclet pulse, 256x3x3 | 0.88601% |
| Semidiscrete time decay, dt=0.2 s | 0.00006836% |

Pulse error decreases from 9.2203% to 3.0294% to 0.8860%. The signed excess
diffusivity decreases from 1.8602e-9 to 6.9829e-11 to 6.4911e-12 m2/s,
or 3.2636 / 0.1225 / 0.01139 times the physical D. The coarsest pulse is
numerically broader than physical diffusion alone would imply. These measured
differences are why conservation and positivity alone are insufficient.

The companion eleven-case momentum rerun in
`results/flow_verification_solute_5a015f5/` also passes, with source hashes
checked. The full suite passes 138 tests, including failed-study retention,
atomic rollback, strong-forcing subcycling and sharp-front concentration bounds.

## Next integration

Focused/nonuniform momentum and scalar operators now have separate references
and a tracked summary in `RECTILINEAR_CHECKPOINT.json`. Next implement oblique
momentum cut faces and verify their pressure/viscous/advective operators, then couple interfacial uptake
and moving geometry. That work needs an explicit solution/solid-density and
partial-volume closure before claiming physical rising-level conservation.
Thermal properties, density derivatives, kinetics and independent quality data
remain separate dependencies. The current pulse test does not establish resolved
KDP boundary layers or experimental mass-transfer/growth accuracy.
