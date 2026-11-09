# Coupled cut-cell pressure and viscosity

`simulation.stokes.CutCellStokes` advances constant-property **linear Stokes**
flow on the existing fixed `CutCellGeometry`. It solves viscous momentum and
incompressibility together. The velocity representation is the already verified
viscosity operator's liquid-volume means and shared face means. Pressure acts
on those face means directly. This completes their numerical coupling without
interpolating a pressure correction between different energy metrics.

The module is an operator in the existing simulation package. It does not
replace the canonical `IncompressibleFlow` runtime or connect Navier–Stokes to
the growing-crystal demo. Conservative nonlinear momentum transport and
advective small-cell handling remain to verify before enabling partial cells
in that runtime. AI implementation and physical growth calibration remain
separate roadmap items.

## Shared velocity, divergence and pressure

For a liquid cell T, let u_T denote its velocity volume mean, lambda_F its shared
velocity face means, and pi_T=p_T/rho its pressure. Use the existing viscous B_T
from [CUTCELL_VISCOSITY.md](CUTCELL_VISCOSITY.md). Define an **integrated**
divergence D by the actual outward area vectors:

```
(D lambda)_T = sum_F A_F * n_TF dot lambda_F
```

Internal apertures contribute opposite signs to adjacent cells. Periodic end
faces share one trace. Actual embedded facets are included, with their existing
liquid-outward normals and zero stationary-wall trace. All outer walls default
to no slip; optional prescribed outer face means support manufactured tests.
The supplied outer flux must be compatible in every connected liquid region.
Moving embedded walls and free-surface stress conditions are unsupported.

The cell/face divergence is the k=0 specialization of the hybrid divergence in
[Di Pietro and Krell, equation 17](https://arxiv.org/pdf/1607.08159).
The implementation combines this coupling with this project's existing mimetic
viscosity, rather than claiming to reproduce the paper's entire HHO method or
its convergence theorem on arbitrary cuts.

Pressure enters momentum through **-D.T*pi**. Thus the velocity/pressure pair
uses an exact discrete integration-by-parts relation. No independent A*d
projection metric, artificial compressibility or pressure penalty enters the
coupled solve. The original `PressureProjection` remains unchanged: its
connectivity/gauge groups and normal-face layout are reused. Its scalar
projection is not applied after the Stokes solve.

## Combined equations and solution

The implicit Euler step solves

```
V_T*(u_new-u_old)/dt + nu*sum_F q_TF = V_T*acceleration_T
q_T = B_T @ (u_T - lambda_faces)
nu*sum_adjacent(q)_F + (D.T*pi)_F = 0     # free face components
D*lambda = 0                            # every liquid cell
```

The steady solve omits the time derivative. Let b=B*1, s=1.T*B*1,
`d=s+V/(dt*nu)` and `load=V*(u_old/(dt*nu)+acceleration/nu)`.
Local elimination gives `u=(load+b.T*lambda)/d` and
`S_T=B-b*b.T/d`. In steady mode use `d=s` and `load=V*acceleration/nu`.
The remaining system has the symmetric block form

```
[ S tensor I    -C.T ] [lambda_free] = [r             ]
[ -C              0 ] [w          ]   [D_boundary*g/L]
C = D_free/L;  w = pi*L/nu
```

L is the geometry's existing scale. Eliminate one pressure row/unknown per
connected liquid region, then restore a **liquid-volume mean zero** gauge in
that region. All remaining continuity equations, including the omitted dependent
row, are checked afterward. No liquid volume is removed, merged or floored.

This matrix is symmetric **indefinite**; the existing SPD-only solver is not
appropriate. A sparse LU factorization solves a congruently scaled system.
Velocity diagonal scaling and a diagonal pressure Schur estimate balance its
blocks. They change conditioning, not the equations. The actual unscaled
residual must satisfy 3e-12 relative tolerance. One current factorization is
cached per dt*nu; production/HPC scaling is not established by this solver.

## Energy and physical units

The pressure contribution to work is `-pi.T @ D @ lambda`, which vanishes for
incompressible flow. The combined backward-Euler identity is

```
(E_new-E_old + 0.5*sum_T V_T*|u_new-u_old|^2)/dt
    + nu*sum_T (u-lambda).T @ B_T @ (u-lambda)
    - pi.T @ D @ lambda
    = sum_T V_T*u_new dot acceleration + boundary_power
E = 0.5*sum_T V_T*|u_T|^2
```

For stationary outer walls boundary power is zero. Prescribed outer velocities
use the viscous **plus pressure** boundary traction in their power. Diagnostics
retain these terms separately, along with cell/face momentum residuals,
integrated volume flux and actual per-cell divergence. Density is factored out
of the energy/power diagnostics; multiply by rho for joules/watts. Returned
pressure is in **Pa**. Cell velocities and face means are in m/s.

Energy is a volume-weighted quadrature of cell means, not an exact integral of
unresolved subcell velocity. Continuity constrains face fluxes; it does not make
a piecewise-constant volume-mean field pointwise divergence-free. The linear
step has no pressure/viscosity splitting error. Time truncation and spatial
approximation errors remain and are measured separately.

## Accuracy limits and pressure-gradient forcing

A cell-based body force is not pressure-robust. Even a pure analytic pressure
gradient, whose continuum velocity is zero, can produce a finite grid-dependent
cell velocity. The verification reports that leakage rather than treating a
passing flux/energy check as sufficient accuracy evidence. Leakage and pressure
error must decrease under refinement, under unchanged thresholds. Strong
pressure gradients and smaller nu need their own resolution studies; a
pressure-preserving force reconstruction is a possible later improvement.

The 1e-7 liquid layer is a numerical stress test, with subnanometer physical
thickness; it does not assert a continuum-fluid experiment at that scale.
Geometry's existing moment-subtraction precision limits are unchanged. Rotated
and subcell obstacles test conservation/energy, not converged obstacle-flow
predictions. No calibrated KDP growth, thermal, quality or actuator response is
established here.

## Independent verification and failure history

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crystalab-stokes \
  .venv/bin/python scripts/check_stokes.py --output results/stokes_verification
```

The 17 cases include:

- Uniform/focused sloped-wall velocity **and pressure** refinement. Velocity is
  the independently integrated divergence-free reference from the viscosity
  study; add affine `pi=(nu*U/L)*(x/L+2*y/L-3*z/L)` and its analytic gradient
  to the forcing. Pressure volume means use true liquid centroids. A separate
  gradient-only solve measures spurious velocity and pressure accuracy.
- Random unforced steps around rotated/subcell solids, the tiny layer, an
  x-periodic sloped wall and two disconnected liquid regions. Check no slip,
  energy decrease, negligible pressure work, cellwise volume conservation and
  independent pressure gauges. Export native Cartesian normal fluxes and compare
  their direct face sum with D*lambda.
- Periodic sine decay at 4/8/16 steps against the analytic **semidiscrete** decay
  rate. This isolates Euler time error from spatial error.
- A time-dependent sloped-wall reference `(u,p)=exp(-t/tau)*(u0,p0)` with nonzero
  pressure and prescribed outer velocities. Analytic body forcing and Gaussian
  liquid means are independent of the numerical operator. A separate 128-step
  numerical trajectory supplies a **temporal** reference; compare velocity and
  pressure at 4/8/16 steps. It is not an analytic spatial-accuracy reference.

The first declared 6/12/18 study, `results/stokes_initial_d532f35/`, **fails**
the focused velocity/refinement gates: finest velocity error 4.4208% exceeds
3%, and the 6-to-12 focused pressure rate is about 0.507, below 0.8. Its
conservation, energy, time, uniform and gradient-leakage gates pass. Preserve
that evidence. The refined protocol measures 12/18/24 with **all tolerances
unchanged**, and adds explicit gradient-state balance and pressure time checks.
Acceptance of a refined regime does not validate those failed coarse meshes.

Each output directory records the complete protocol, git parent, environment,
source hashes/copies, geometry, states, pressure and figures. Existing output is
rejected. Failed cases/gates remain in reports. `tests/test_stokes.py` deliberately
checks both a failed accuracy gate and a failed geometry refinement, plus
rest, algebraic adjoint/nullspace, density units, periodic modes, coupled time
refinement, disconnected motion, tiny capacities and invalid boundary flux.
Every new code file begins with a purpose summary.

## File review and next work

| File | Change and purpose |
|---|---|
| `src/simulation/stokes.py` | New shared-trace pressure/viscosity operator, implicit/steady solves, pressure gauges, native flux export and combined work/conservation diagnostics. Reuses existing viscosity and pressure geometry/topology; those source files are unchanged. |
| `scripts/check_stokes.py` | Independent velocity/pressure forcing, gradient leakage, spatial/time studies, evidence archival and refinement plot. |
| `configs/stokes_verification.json` | Mathematical test inputs, refined grid protocol, original tolerances and explicit failure history. |
| `tests/test_stokes.py` | Twelve coupling/accuracy/invariant/input/evidence-retention regressions. |
| `docs/CUTCELL_STOKES_CHECKPOINT.json` | Final source identity, protocol, measurements, regression result and original failure record. |
| `docs/CUTCELL_STOKES.md` | This derivation, limits, evidence and file-by-file explanation. |
| `roadmap.md`, `docs/AI_READINESS.md` | Current completed/remaining milestones and correct commit status. |
| `docs/CUTCELL_VISCOSITY.md`, `docs/CUTCELL_PRESSURE.md`, `docs/FLOW_MODEL.md`, `docs/PHYSICAL_MODEL.md`, `docs/PHYSICAL_DATA.md`, `SOURCES.md` | Coupling status, references and links to measured evidence. |

Next implement conservative nonlinear momentum transport using a defined
reconstruction of these volume/face unknowns and the same conservative volume
flux. Verify momentum/energy and bounded small-cell advection before enabling
full cut-cell Navier–Stokes. Then connect passive solute and physical interfacial
uptake/motion with density/partial-volume closure and repeat coupled refinement.
The existing runtime guard and illustrative growing-crystal demo stay in place.

## Measured refined checkpoint

All 17 cases in `results/stokes_refined_d532f35/` pass the unchanged gates.
[CUTCELL_STOKES_CHECKPOINT.json](CUTCELL_STOKES_CHECKPOINT.json) preserves the
final and original failed protocols/results, source identities and regression
evidence. The full suite passes **198 tests** (76.905 s); all three new Python
files pass focused lint/format checks and have explanatory headers.

| Mesh, cells per axis | Velocity RMS / reference RMS | Pressure RMS / prescribed scale | Gradient-only velocity RMS / U |
|---|---:|---:|---:|
| Uniform 12 | 2.4374% | 5.4908% | 0.6069% |
| Uniform 18 | 1.1376% | 3.0874% | 0.2869% |
| Uniform 24 | 0.65395% | 1.9792% | 0.16581% |
| Focused 12 | 7.4122% | 8.4061% | 1.7933% |
| Focused 18 | 4.4208% | 5.7956% | 1.0738% |
| Focused 24 | 1.8883% | 3.0236% | 0.45290% |

The finest gate applies at 24; the coarse focused errors still exceed 3%.
Pressure error is normalized by `rho*nu*U/L`, not by measured pressure RMS.
Maximum normalized integrated flux is 5.622e-16, pressure work 1.555e-16 and
combined momentum/work residual 3.976e-15. Maximum actual divergence is
1.291e-14 1/s; the native normal-face volume sum agrees with D*lambda to
3.232e-18 relative to U*L^2. All stationary embedded wall traces are zero.
Unforced energy decreases on all five cut/topology stress cases.

Periodic time velocity errors at 4/8/16 steps are 0.25346%, 0.12815%, 0.064435%.
Coupled time velocity errors against the separate 128-step reference are
1.3826%, 0.69123%, 0.32811%; pressure errors over the prescribed scale are
7.2764%, 3.6376%, 1.7266%. Both velocity and pressure pass the unchanged minimum
0.8 time-order gate. These are temporal errors on the fixed reference grid.

All current/archived source hashes match and all 17 field archives are finite.
The original figure is retained. `refinement_readable.png` is a plot-only re-export
of the same archived measurements, using the archived `plot_refinement.py` helper
for nonoverlapping tick labels; it changes no source, solver or metrics.

The coupling work remains local and uncommitted. Current main/origin checkpoint
is d532f35; no new commit or push was made for this implementation.
