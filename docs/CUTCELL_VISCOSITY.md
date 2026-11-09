# Fixed embedded-wall viscosity

`simulation.viscosity.CutCellViscosity` is a standalone, constant-viscosity
vector diffusion operator on the existing `CutCellGeometry`. It uses actual
liquid volumes and the actual embedded facets. Each stationary wall facet has
zero velocity **face mean in all three components**, including tangential ones.
It does not replace `IncompressibleFlow` or advance the growing-crystal demo.

## Unknowns and local operator

The unknowns are three velocity **volume means** in each liquid cell and three
auxiliary velocity **face means** on each open Cartesian face. An interior face
has one shared trace. Periodic end faces alias the same trace. Embedded facets
and nonperiodic outer faces have prescribed traces; ordinary steps prescribe
zero. Manufactured steady tests may prescribe nonzero outer traces. Moving
embedded walls and the runtime's shear-free top condition are unsupported here.

For each cell, collect outward liquid normals in N and rows
`R_f = A_f * (x_face - x_liquid_centroid)`. Exact geometry gives
`R.T @ N = V*I`. For unit diffusion, construct

```
M0 = R @ R.T / V
M  = M0 + trace(M0)/3 * P_ker(N.T)
B  = diag(A) @ inverse(M) @ diag(A)
q  = B @ (u_cell - lambda_faces)
```

A full QR decomposition constructs the orthogonal projector; a Cholesky check
rejects a nonpositive metric. This follows the consistent positive metric
family in [Brezzi, Lipnikov and Simoncini (2005), equations 3.9–3.10](https://mate.unipv.it/brezzi/papers/strike1-9.pdf).
The stabilization factor is this project's declared choice. A simple
two-point difference between skew liquid centroids would lose affine
consistency on cut cells. For affine velocity, `M@N=R` gives the exact integrated
diffusive face flux. On a full orthogonal Cartesian cell the construction reduces
to the usual half-cell diffusion conductances.

The assembled equations are

```
V*(u_new-u_old)/dt + nu*sum(q) = V*acceleration
sum(q_from_adjacent_cells) = 0   # each interior trace
```

The steady equation omits the time derivative. Local elimination of each cell
mean leaves a symmetric face system, solved by the existing residual-checked
`solve_spd`. Its tolerances are unchanged. The operator caches one factorization
for the current `dt*nu`; it does not accumulate timestep caches. Steady solves
require a Dirichlet anchor in every connected region. Fully periodic implicit
solves retain their constant velocity mode.

All positive capacities are retained. The geometry's existing bounded
preprocessing still resolves disconnected pockets; viscosity introduces no
cell merge, solid voxelization, artificial capacity floor or velocity clipping.

## Energy, flux and representation limits

For zero wall traces, implicit Euler satisfies the discrete work identity

```
(E_new-E_old + 0.5*sum(V*|u_new-u_old|^2))/dt
    + nu*sum((u-lambda).T @ B @ (u-lambda))
    = sum(V*u_new*acceleration)
E = 0.5*sum(V*|u|^2)   # density factored out
```

Steady prescribed outer velocities also contribute boundary work, which is
reported with its sign. Diagnostics retain integrated cell momentum and shared
face flux residuals, dissipation and the work residual. E is a cell-mean
quadrature of kinetic energy; it does not resolve within-cell fluctuations.

`normal_face_velocity()` exports the traces' normal components in the existing
x/y/z MAC array shapes. This checks layout, blocked faces and periodic aliases.
It is **not an energy-compatible pressure transfer**. The existing pressure
operator acts on different unknowns and an A*d metric. Pressure/viscosity
coupling is now implemented directly on these shared traces in
[CUTCELL_STOKES.md](CUTCELL_STOKES.md), with a monolithic linear solve and
combined conservation/work checks. This does not turn normal-face export into
an energy-compatible interpolation or enable full nonlinear cut-cell flow. Nonlinear momentum fluxes and advective
small-cell handling remain subsequent work. The canonical flow's partial-cell
rejection remains active.

The thin-layer numerical stress test retains a liquid fraction of 1e-7; its
subnanometer thickness is not a physical continuum experiment. Geometric moment
subtraction there produces about 2.03e-8 relative defect when scaled by the
actual tiny liquid volume; this is recorded without repairing the geometry.
Energy stability does not establish affine accuracy on arbitrarily thin cells,
nor published shape-regular convergence bounds for all possible crystal cuts.
The rotated/subcell cases test invariants, not obstacle-flow accuracy.

## Independent numerical verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crystalab-viscosity \
  .venv/bin/python scripts/check_viscosity.py \
  --output results/viscosity_verification
```

The declared gates in `configs/viscosity_verification.json` were written before
the first measurement. The 13-case study checks:

- Affine tangential shear beside `z=a+b*y`, on uniform and focused meshes.
- Spatial refinement of a three-component, divergence-free manufactured field:
  `U*[f(d), sin(pi*x/L)*f(d), b*sin(pi*x/L)*f(d)]`, where
  `d=z/L-a/L-b*y/L`, `f(d)=d*(1-d)`. Its velocity averages and analytic
  `-nu*Laplacian(u)` forcing are integrated independently over the liquid by
  split Gaussian quadrature. Forcing is never manufactured by applying B.
- Viscous energy/flux balances on a rotated cube, a subcell cube, a 1e-7 liquid
  layer and an x-periodic sloped wall.
- Implicit Euler refinement at 4/8/16 steps. A periodic sine mode's analytic
  uniform-grid **discrete** decay rate isolates time error from spatial error.

Every run requires a fresh directory and archives protocol, source hashes and
copies, environment, fields, true geometry, refinement history and plots.
Failed numerical gates and failed geometry cases remain in the report. Unit
tests deliberately exercise both failure paths and output overwrite rejection.
`tests/test_viscosity.py` also checks closed-form reference integrals, forced
work balance, disconnected regions, periodic constant preservation, invalid
inputs and pressure-format export. Every new Python file has a purpose summary.

## Measured checkpoint and file review

The source-matched run is `results/viscosity_final_3b99100/`; its compact
record is [CUTCELL_VISCOSITY_CHECKPOINT.json](CUTCELL_VISCOSITY_CHECKPOINT.json).
All 13 cases pass the unchanged gates. Uniform relative volume RMS errors are
1.8943%, 0.48634%, 0.21695%; focused errors are 8.7717%, 2.0301%, 0.76926%.
Observed spatial rates exceed 1.9 in both families. Maximum affine error is
2.43e-13 of the declared speed. Time errors are 1.6681%, 0.85558%, 0.43342%,
with observed orders 0.96/0.98. Decay cases decrease energy; their largest
normalized flux/work balance error is below 2.5e-15. Wall traces are exactly zero.

| File | Change and purpose |
|---|---|
| `src/simulation/viscosity.py` | New operator, volume/trace state, implicit/steady solves, normal-face export and work/flux diagnostics. |
| `scripts/check_viscosity.py` | Independent references, fresh-directory study runner, source/geometry/field archives and refinement figure. |
| `configs/viscosity_verification.json` | Dimensions, numerical test viscosity/speed, geometry, refinement levels and predeclared gates; these are mathematical test inputs, not calibrated KDP data. |
| `tests/test_viscosity.py` | Eleven numerical, conservation, boundary, topology and evidence-retention regression tests. |
| `docs/CUTCELL_VISCOSITY_CHECKPOINT.json` | Source identity, measured results and gates from the final study. |
| `docs/CUTCELL_VISCOSITY.md` | Method, assumptions, evidence, limitations and this file-by-file review. |
| `SOURCES.md` | Primary method citation and numerical evidence entry. |
| `docs/AI_READINESS.md`, `docs/PHYSICAL_MODEL.md`, `docs/PHYSICAL_DATA.md`, `docs/FLOW_MODEL.md`, `docs/CUTCELL_PRESSURE.md` | Current status and links; full cut-cell dynamics and growth/AI readiness remain unfinished. |
| `roadmap.md`, `PROGRESS.md` | Local progress/status, including the approved checkpoint push and the next coupling step. |

Existing solver, runner and package layouts are preserved. No AI implementation
or ML dependency is added. The viscosity files and roadmap were subsequently
committed as `d532f35`, after the tracked documentation checkpoint `aa4669b`.
The roadmap is now tracked. The next local coupling work is described in
[CUTCELL_STOKES.md](CUTCELL_STOKES.md).
