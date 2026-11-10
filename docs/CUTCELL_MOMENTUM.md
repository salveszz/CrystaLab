# Conservative cut-cell momentum and small cells

`simulation.momentum` adds fixed-domain nonlinear momentum transport to the
existing pressure/viscosity operator. It retains the same geometry, cell volume
means, face velocity means and pressure gauges. No geometry is voxelized, no
positive liquid capacity is discarded, and no velocity is clipped.

The method is **first order in space and time**. One implicit donor-cell
transport step uses the previous constrained face flow, then the existing
monolithic Stokes step applies viscosity, acceleration and pressure. The
advecting flow changes with the solution between steps, making this nonlinear
self-advection, with a lagged advecting velocity. It is not a fully implicit
nonlinear iteration or the canonical MAC solver's limited reconstruction.

## One conservative transport flux

`ImplicitMomentumTransport` uses the pressure operator's independent native
faces and periodic aliases. For a face directed from liquid cell i to j, its
actual volume flux is Q=A*u_normal. Both cells use the **same** momentum flux:

```
F_ij = Q * (u_i_new if Q >= 0 else u_j_new)
V_i*(u_i_new-u_i_old)/dt + sum_oriented(F) = 0
```

The linear system is `diag(V)+dt*C(Q)`. Each column of C sums to zero, giving
conservative transfers. With zero net volume flux in each cell, constant fields
are preserved and the implicit upwind system has a discrete componentwise
maximum principle. This bound applies to **transported cell means**; pressure
and physical forcing do not obey a componentwise velocity maximum principle.

Rows are scaled by their actual positive diagonal before sparse LU. Scaling
conditions the equations without changing their capacities or transfers. The
unscaled residual must pass the existing 3e-12 relative criterion. Input face
fluxes must be divergence-free, including a check of the volume error over the
requested timestep. Blocked-wall flow and mismatched periodic aliases are
rejected. Only the current flux/timestep factorization is cached.

The outgoing Courant number is reported as
`dt*sum_outgoing(Q)/V_liquid`. Implicit transport avoids the explicit small-cell
stability restriction. It does not establish accuracy at a large Courant number.
The distinction between implicit stabilization and accuracy on cut cells is
also discussed in [May and Laakmann](https://arxiv.org/pdf/2310.16101). This
implementation uses implicit first-order donor transport everywhere; it does
not reproduce their mixed second-order MUSCL-Trap method.

## Momentum and energy budgets

For a closed or periodic domain, transport conserves `sum(V*u)`. Density is
factored out; multiply by rho for physical linear momentum. Donor transport has
nonnegative numerical dissipation

```
D_upwind = 0.5 * sum_faces |Q| * |u_j_new-u_i_new|^2
E = 0.5 * sum_cells V*|u|^2
(E_new-E_old + 0.5*sum(V*|u_new-u_old|^2))/dt
    + D_upwind + 0.5*sum(net_Q*|u_new|^2) = 0
```

The net-Q term measures residual volume imbalance; it is retained, not silently
removed. For an incompressible flux it vanishes. Numerical dissipation is
separate from physical viscosity and must decrease under spatial refinement.

`CutCellMomentum.advance(state, dt, acceleration_m_per_s2=...)` accepts an
existing `StokesState`, returns a `MomentumStep`, and changes no input state.
The supplied state must belong to the same fixed geometry.
The return holds the final constrained state, intermediate advected means and
budgets. The combined work identity includes upwind dissipation, both implicit
Euler damping terms, viscous dissipation, pressure work and external work.
Pressure work remains negligible through the shared divergence/adjoint pair.
Momentum change is checked against body-force impulse and stationary-wall
traction impulse, including the pressure contribution. Periodic momentum has
no wall impulse. Solid-cell values and all boundary traces remain zero.

## Optional pressure Schur backend

Fine 3D reference meshes made the full coupled sparse LU expensive. The new
`CutCellStokes(..., linear_solver="schur")` solves the **same** condensed saddle
system, using one scalar viscous factorization shared by all three components
and pressure conjugate gradients. `linear_solver="direct"` remains the default.

Each Schur matvec applies the actual scalar viscous inverse and actual coupled
divergence. A pressure preconditioner combines inverse volume mass with an
inverse pressure Laplacian, following the constant-property
[Cahouet–Chabard form, equation 26 in Bootland et al.](https://strathprints.strath.ac.uk/68686/7/Bootland_etal_SIAM_2019_Preconditioners_for_two_phase_incompressible_Navier_Stokes.pdf).
In the project's scaled unknowns its action is
`L^2*(M_gauge_inverse + K_pressure_inverse/(dt*nu))`; omit the Laplacian term
for steady Stokes. For each pinned region, inverse gauge mass is
`diag(1/V_free)+ones*ones.T/V_pin`. That rank-one term respects the pressure
quotient instead of overlooking the pinned constant mode.

The preconditioner changes convergence, not equations. Pressure CG uses
rtol 2e-13; the final **original unscaled saddle residual** still must meet
3e-12. No artificial compressibility, pressure penalty or relaxed tolerance is
introduced. The previous pressure iterate may initialize the next solve, and
only the current system is cached. Tests compare direct/Schur velocity and
pressure, including a retained 1e-10-capacity geometry. This is measured solver
improvement, not proof of arbitrary-cut or production HPC scaling.

## Independent numerical references

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crystalab-momentum \
  .venv/bin/python scripts/check_momentum.py --output results/momentum_verification
```

The declared protocol is `configs/momentum_verification.json`. Sources, input
protocol, environment, raw states/geometry, budgets, figures and per-case times
are archived in a fresh directory. Existing output is rejected and failures are
kept. Acceptance tolerances are not changed after a failed measurement.

The study covers:

- Periodic vector-wave translation using exact volume averages. This isolates
  donor transport and quantifies numerical damping under refinement. A unit test
  also uses the analytic complex Fourier amplification of the implicit upwind
  scheme, independently of the implementation's matrix.
- A fully 3D periodic ABC/Beltrami velocity and its nonlinear pressure, including
  exact rectangular velocity and pressure averages. The pressure reference
  integrates cross products; it is not the square of averaged velocity.
- A fully 3D closed sloped-wall manufactured flow. The velocity is the curl of
  `psi*[1,1,1]`, with
  `psi=x^2(1-x)^2*y^2(1-y)^2*z^2(1-z)^2*(z-a-b*y)^2` in normalized coordinates.
  The curl vanishes on all stationary walls. Polynomial differentiation and
  multiplication give analytic viscous and nonlinear accelerations; add an
  independently prescribed affine pressure gradient. Split Gaussian integrals
  give actual liquid-volume/face means. Shared analytic edge integrals construct
  the normal curl flux through Stokes' theorem, preserving reference continuity
  without applying the numerical projection or repairing a numerical solution.
- A time-dependent version of that flow. Analytic forcing includes the time
  derivative, exponential viscous/pressure terms and the quadratic exponential
  convective term. Separate 128-step results provide a **time-only** reference
  for velocity and pressure at 4/8/16 steps. Spatial error remains separate.
- Rotated/subcell solids and a periodic sloped wall, checking repeated unforced
  momentum, pressure work, volume conservation, no slip and energy decrease.
- An exact four-face circulation through a retained triangular liquid sliver,
  with volume fraction near 1e-10 and outgoing Courant 1000. Check bounded vector
  transport, conservation and work, then the complete nonlinear/Stokes step.
  This is a numerical stress test, not a physically resolved velocity field.

The first 6/10/14 wall study in `results/momentum_initial_bb4264a/` **fails**
its wall accuracy gates. At 14, uniform/focused velocity errors are
13.7146%/43.3199% and pressure errors over the prescribed scale are
45.0318%/168.277%. Conservation, energy, bounds, wave, ABC, temporal and small-cell
gates pass. A separate linear-only diagnostic also produces large wall errors,
confirming that the high-curvature reference needs finer resolution in the
underlying viscous/pressure discretization. Preserve the original study and do
not claim its coarse focused mesh validated. The next aggressive-focus study retains uniform 24/36/48 results but still
fails focused 24/36 accuracy (25.560%/17.166% velocity error). The remaining
expensive focused 48 case was interrupted, with its source snapshot and partial
results kept in results/momentum_refined_bb4264a.

The accepted study retains the reference and every original tolerance, and
explicitly subdivides focused outer intervals wider than the background spacing
L/n. Nominal counts 12/18/24 produce actual focused grids 17^3/27^3/34^3. The
fine band, domain and solid are unchanged. This is a better resolved mesh with
extra cells, not acceptance of the original fixed-count meshes or an equal-cost
comparison. The generator change is limited to the verification helper; existing
production meshes/defaults are unchanged. Both nominal count and actual counts,
physical widths and edges are archived. A unit test checks the width cap and
unchanged domain/liquid volume/wall area.

## Scope, files and next work

The 2026-11-10 planning review puts the bounded synthetic data contract and
pilot recorder ahead of further CFD expansion. See [AI_READINESS.md](AI_READINESS.md).
The integration work below describes the next higher-fidelity physics stage,
when a required output or measured discrepancy justifies it. A first growth
regressor can use the existing illustrative engine with explicit assumptions.

The canonical `IncompressibleFlow` runner still rejects partial momentum cells.
This operator supplies the next verified building block; integrating it into the
existing runtime/API requires explicit state/flux adapters and combined scalar
checks. The growing-crystal engine is still illustrative and separate. No AI,
new package hierarchy, empirical forcing reinterpretation or ML dependency is
introduced. Existing growth/runner defaults and placeholders are preserved.
The initial adapter scope is stationary no-slip or periodic boundaries; the
canonical rigid-lid shear-free top needs its own stencil and verification.

| File | Change and purpose |
|---|---|
| `src/simulation/momentum.py` | New conservative implicit transport, split nonlinear step, bounded transport and combined momentum/energy diagnostics. |
| `src/simulation/stokes.py` | Optional Schur backend for the unchanged pressure/viscosity equations; the default direct backend and residual tolerance remain. |
| `scripts/check_momentum.py` | Independent analytic averages/forcing, tiny-cell circulation, refinement study, raw evidence, timings and plots. |
| `configs/momentum_verification.json` | Mathematical test inputs, original tolerances, refined mesh levels and explicit failure history. |
| `tests/test_momentum.py` | Sixteen operator, reference, mesh-cap, topology, conservation, solver-equivalence, input and evidence-retention regressions. |
| `docs/CUTCELL_MOMENTUM.md`, `docs/CUTCELL_MOMENTUM_CHECKPOINT.json` | Method, evidence, limits and source-matched numerical checkpoint. |
| `roadmap.md`, `docs/AI_READINESS.md`, related physical/flow/cut-cell method docs and `SOURCES.md` | Completed/remaining scope, source provenance and review links. |

Next integrate the accepted fixed cut-cell velocity/flux state
into the existing flow/scalar interfaces and verify passive-solute conservation,
resolution and time synchronization. Keep the runtime guard until those checks
pass. Physical interfacial uptake, moving geometry, density/partial-volume
closure, kinetics, measured quality and calibrated apparatus forcing follow.
Higher-order transport may be useful where donor diffusion is too large; it
requires a separate boundedness/energy/refinement study.

## Measured accepted checkpoint

All **20 cases** in `results/momentum_background_resolved_bb4264a/` pass their
original tolerances. [CUTCELL_MOMENTUM_CHECKPOINT.json](CUTCELL_MOMENTUM_CHECKPOINT.json)
records final sources/protocol/results, original failed/interrupted studies and
extra solver/linear-only diagnostics. Full suite: **214 tests PASS** in 87.997 s;
focused lint/format/whitespace pass. All current/archived final hashes match, all
20 field archives are finite and the refinement figure was inspected.

| Wall mesh | Nominal background count | Actual grid | Velocity relative RMS | Pressure RMS / prescribed scale |
|---|---:|---|---:|---:|
| Uniform | 12 | 12^3 | 18.162% | 57.722% |
| Uniform | 18 | 18^3 | 8.5603% | 29.286% |
| Uniform | 24 | 24^3 | 4.9190% | 17.387% |
| Resolved focused | 12 | 17^3 | 13.526% | 47.592% |
| Resolved focused | 18 | 27^3 | 4.8490% | 17.712% |
| Resolved focused | 24 | 34^3 | 3.3665% | 12.813% |

The finest gates are 5% velocity and 25% pressure over the prescribed scale;
coarse table entries need not satisfy the finest thresholds. Pressure's scale
is `rho*nu*U/L`, not measured pressure RMS. The focused comparison uses more
cells; its maximum interval is at most L/n. Uniform 48 from the preserved
refinement has 1.2449% velocity/4.6476% scaled pressure error, but is not required
for the accepted protocol. Resolution requirements depend on reference/forcing.

Periodic wave errors at 8/16/24 are 13.137%, 7.0413%, 4.9363%; timestep decreases
with grid spacing at the declared outgoing Courant bound. The diagnostic
separates donor and time damping. ABC velocity errors at 6/10/14 are 2.3020%,
0.79910%, 0.41758%; pressure RMS errors are 39.080%, 19.658%, 14.766%. These
regimes do not establish high-Reynolds-number or turbulent accuracy.

Time-refinement velocity differences against the 128-step reference are
2.1100%, 1.0667%, 0.50935%; scaled pressure differences are 6.8813%, 3.4870%,
1.6671%, consistent with first-order time convergence. That reference isolates
time differences on the fixed grid, not spatial/physical validation.

Maximum normalized work residual is 1.549e-15, momentum error 5.089e-17,
integrated volume flux 8.195e-15 and pressure work 1.795e-16. Actual maximum
divergence is 3.770e-14 1/s. Transport bound violation is zero. Unforced energy
decreases. Pure and complete coupled tiny-cell cases both reach Courant 1000
and retain minimum liquid fraction 9.99984e-11.

Additional direct/Schur comparisons pass on periodic, sloped, focused, rotated,
disconnected and tiny geometries. Maximum velocity difference is 1.606e-16 m/s;
volume-weighted relative pressure RMS difference is below 2.14e-13. The tiny
case's maximum pointwise pressure difference is 8.619e-12 Pa, retained alongside
weighted differences. The measured ABC-14 case takes 73.27 s with the previous
full direct solve and 5.73 s with Schur in the final study; this is a bounded
observation, not an HPC scaling guarantee. Full case times remain in the report.

These momentum/solver-option changes are included in the simulation/data
checkpoint authorized for commit and push on 2026-11-10. The next higher-fidelity
physics task, when needed, is the canonical closed/periodic flow/scalar adapter
and its conservation, resolution and clock checks. It is separate from the first
synthetic growth-model experiment. The existing runtime guard stays active.
