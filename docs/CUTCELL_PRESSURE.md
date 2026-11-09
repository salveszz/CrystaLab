# Fixed sloped-wall geometry and pressure foundation

This completed foundation represents actual sloped crystal surfaces and verifies
their pressure projection. A subsequent standalone embedded viscosity study now
passes; see [CUTCELL_VISCOSITY.md](CUTCELL_VISCOSITY.md). Together these do **not**
yet enable full sloped-crystal momentum time stepping: their unknowns need an
energy-compatible pressure/viscosity transfer, followed by nonlinear momentum
transport and advective small-cell verification. `IncompressibleFlow` continues
to reject partial momentum cells.

The existing uniform/rectilinear runtime now delegates to the same
`PressureProjection` implementation used by the cut-cell references. This is a
shared operator, not a separate replacement simulation or a second planar path.
The earlier focused-grid and geometry/pressure checkpoints are committed.

## Geometry and conservation measures

`CutCellGeometry(grid, solid)` intersects a canonical convex `Crystal` with
actual rectilinear cells. It normalizes geometry by the largest domain length
before clipping and converts lengths/areas/volumes/moments back to SI. It stores:

- Liquid volume and volume centroid in every Cartesian cell.
- Open Cartesian face areas and their actual centroids.
- Embedded wall pieces, with the original crystal face ID, area, area centroid
  and normal pointing out of liquid into solid.
- Per-cell wall area vectors and first normal moments.

Wall pieces come from clipping the original facet polygons by the cell box.
The new `Crystal.clipped_face_moments()` preserves original face identities,
including redundant/inactive faces. It uses the same polygon clipping kernel as
the existing solid/Cartesian-face intersections. A wall exactly coincident with
a Cartesian face belongs to its liquid-side cell, so it is counted once.

Two divergence-theorem identities independently audit each liquid cell:
`sum(A*n)=0` and `sum(A*centroid outer n)=V_liquid*I`, including both Cartesian
apertures and embedded wall pieces. Rotated and subcell crystals are represented
by actual intersections rather than occupancy masks. Geometry arrays are read-only.

### Disconnected pockets and bounded resolution

One Cartesian box can contain more than one disconnected piece of liquid beside
a sharp crystal edge. Combining them into one pressure unknown creates a false
connection. The geometry checks the connectivity of outside-half-space pieces,
using shared outside corners and exact pair-intersection volumes when needed.
The direct constructor raises `SplitLiquidCellError` with physical box bounds.

`CutCellGeometry.resolved(..., max_axis_refinements=48)` can instead insert
Cartesian planes by bisecting the widest axis interval of a failed cell. It
rechecks the geometry after every split and records the axis, coordinate and box
bounds. The physical crystal is unchanged. This is bounded, static preprocessing;
it is not moving adaptive refinement or a hidden pocket merge. Exhausting the
declared limit remains an error.

The rotated-cube reference starts from 8/12/16 cells per axis and resolves to
14x12x12, 20x19x20 and 31x27x33, using 14/23/43 inserted planes. Actual final
edges and the full refinement history are exported. These are the grids used
for the measured projection invariants; their labels retain the starting count.

## Compatible pressure projection

`PressureProjection(grid, liquid_volumes_m3, open_areas_m2)` builds one graph
connection for every independent open face. Scalar data index z/y/x; aperture
arrays follow z/y/x normal axes, and velocity fields follow x/y/z components.
Periodic end faces are aliases, including unequal widths at opposite ends.

With open area A and nominal pressure-center distance d, a face has volume flux
`Q=A*u`, gradient `(phi_j-phi_i)/d` and conductance `A/d`. The symmetric graph
stiffness solves the **volume-integrated** pressure-correction equation. It is
not divided row by row by tiny liquid capacities. Actual V still defines
divergence and the volume-weighted pressure gauge. Each disconnected liquid
region has its own gauge and source/flux compatibility check.

`project()` returns corrected face velocities and phi in m2/s. The canonical
flow converts that potential to its pressure increment `rho*phi/dt` in Pa.
The projector is stateless apart from cached factorizations: it validates finite
fields, blocked normal velocities and periodic aliases, and returns independent
arrays. Its metric/connection arrays are read-only.

Pressure values are point approximations at nominal Cartesian centers, including
centers outside the liquid portion. They are **not liquid-cell averages**. This
first embedded operator uses two-point differences; it does not implement the
primary reference's centroid-interpolated second-order boundary stencil.

Projection decreases the positive face norm `sum(A*d*u²)`. For full regular
momentum cells this is the existing velocity-volume metric. For cut cells it is
a compatible projection norm, not a validated physical kinetic-energy integral.
Impermeability belongs to this operator. Tangential no slip requires the next
embedded viscosity/momentum stage.

`linear.py` contains the prior residual-checked SPD solve, now shared by
momentum and pressure. Its direct/CG tolerances are preserved. `flow.py` removes
its duplicate pressure assembly, gauge, gradient and projection implementations
and delegates those operations to `PressureProjection`.

## Independent numerical references

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_cutcell.py \
  --output results/cutcell_pressure
```

The fourteen-case protocol is `configs/cutcell_verification.json`. It covers
uniform/focused oblique-wall Poisson refinement; rotated, subcell and tiny-layer
geometry; periodic projection; separate disconnected-region gauges; and an
expected rejection of a split cell without preprocessing. Seven numerical source
files, exact settings and package versions are archived before measurement.
Case-level numerical/input errors stay in the report and make it fail. Existing
output directories are rejected. Raw fields include capacities, centroids,
apertures, wall patches, refined edges, refinement history and pressure potentials.

For the manufactured Poisson problem use dimensionless X/Y/Z, `d=Z-a-b*Y`, and
`phi=s*cos(pi*X)*Y²*(1-Y)²*Z²*(1-Z)²*d²`. Its normal derivative vanishes on the
outer box and sloped wall. The analytic Laplacian is integrated independently
over each actual liquid cell with split Gaussian quadrature; the x cosine
integral is analytic. The source is not computed by applying the discrete
matrix to the reference potential. Tests also compare analytic derivatives with
independent finite differences and compare quadrature orders.

The analytic gradient is sampled at open-face centroids and projected. Its
remaining norm is an additional gradient/projection refinement check. Random
fields check integral flux balance and projection-norm decrease independently
of the manufactured source. Absolute divergence and integral flux residuals are
both reported; tiny capacities can amplify a small integral residual.

The tiny-layer case retains a numerical liquid fraction of 1e-7 in its top cells.
Its relative volume criterion is explicitly 1e-6, accounting for subtractive
cancellation at that capacity ratio. The normal geometry cases retain a 1e-10
relative-volume criterion. This layer is a numerical conditioning stress test,
not a physical continuum experiment.

### Preserved failed stage

`results/cutcell_pressure_first_8011b79/` retains the original six measured
Poisson cases, sources, fields and console log. The initial runner aborted at a
rotated cube's split cell before it could write a final report; `failure.json`
explicitly records that reconstruction from the preserved log. It also records
the focused n=16 potential error of 6.4348%, above its declared 5% limit.

The follow-up keeps the 5% potential/gradient targets, increases the Poisson
levels from 8/12/16 to 8/16/24, resolves the rotated geometry with bounded axis
refinement, and records subsequent case errors directly. No solid is changed to
avoid the failed topology. The separate tiny-volume tolerance is stated above.

### Accepted local evidence

`results/cutcell_pressure_final_v2_8011b79/` passes all fourteen declared cases,
including the expected rejection. The seven measured sources match their hashes
and archived copies. Raw geometry/fields are finite and the final figures were
inspected. Wall face IDs and axis-refinement history are included in the raw
archives so future interfacial coupling can retain facet identity.

| Finest Poisson case | Potential RMS error | Analytic-gradient projection residual |
|---|---:|---:|
| Uniform, 24x24x24 | 0.41065% | 0.36367% |
| Focused, 24x24x24 | 2.17234% | 1.08533% |

Maximum integral flux residual relative to the initial face-flux scale is
4.616e-13. All blocked-face speeds are zero and all random projections decrease
the declared norm. Maximum normalized area/moment closure errors are
9.189e-14 / 8.782e-12. The finest resolved rotated grid contains capacities as
small as 1.546e-8 of a Cartesian cell; the middle grid reaches 2.681e-10.
The separately declared thin-layer volume error is 1.259e-8 relative.

Absolute divergence reaches 6.717e-8 1/s in the smallest rotated cells despite
the small integral residual. This conditioning diagnostic is retained. It is
not evidence of a stable or accurate cut-cell momentum/advection time step.
The future momentum/scalar integration must address small-cell behavior directly.

Current-source uniform momentum (eleven cases), scalar/high-Peclet transport
(fifteen cases) and rectilinear flow/solute (twenty cases) regressions also pass.
The full suite passes 165 tests. `CUTCELL_PRESSURE_CHECKPOINT.json` preserves the
protocol, sources, results, failed-stage history and current regression identities.

## Code-level change review

| File | Change and reason |
|---|---|
| `src/simulation/crystal.py` | Extract one shared polygon clipping helper and add original-facet clipping/moments for embedded wall pieces. Existing shape/growth laws retain their roles. |
| `src/simulation/cutcell.py` | Build exact SI liquid/wall geometry, closure diagnostics, connectivity checks and bounded static axis refinement. Export read-only geometry and refinement history. |
| `src/simulation/pressure.py` | Shared area/distance pressure graph, volume-integrated Neumann solves, per-region gauges, compatible gradients/divergence and stateless projection. |
| `src/simulation/linear.py` | Rehome the existing SPD solve and residual checks so both numerical components use one solver. |
| `src/simulation/flow.py` | Delegate pressure and linear solves to those shared components; continue guarding unsupported partial momentum cells. |
| `scripts/check_flow.py`, `scripts/check_solute.py`, `scripts/check_rectilinear_flow.py` | Include the new pressure/linear dependencies in archived source fingerprints. |
| `scripts/check_cutcell.py` | Run independent geometry/Poisson/projection references; export native fields and figures; preserve failures and reject overwrite. |
| `configs/cutcell_verification.json` | Declare geometries, actual refinement controls, independent reference settings and numerical acceptance criteria. |
| `tests/test_cutcell.py` | Fourteen tests for analytic geometry, closure, topology, bounded refinement, small capacities, pressure compatibility/gauges, independent MMS derivatives/refinement and study integrity. |
| `docs/CUTCELL_PRESSURE_CHECKPOINT.json` | Preserve final protocol, source identity, measured outcomes, failure history and current-source regression identities. |
| `docs/CUTCELL_PRESSURE.md`, model/readiness docs, `SOURCES.md`, local roadmap | Explain the scope, evidence, remaining dependencies and commit recommendation. |

The preceding local changes for shared mesh metrics, unequal-cell flow/solute,
their tests and runner are reviewed separately in
[RECTILINEAR_FLOW.md](RECTILINEAR_FLOW.md#file-by-file-review).
All code files retain purpose summaries at the top. The user approved committing
and pushing both milestones on 2026-11-08: rectilinear flow/solute in `ab0a6a1`
and cut-cell geometry/shared pressure in `cb83f64`.

## Commit boundary and next work

The committed scope is **verified rectilinear flow/solute plus cut-cell
geometry and pressure foundations**, checkpointed in two focused commits after
review. Their titles/descriptions must not claim
complete oblique no-slip Navier-Stokes or AI-ready growth physics.

Next implement and independently verify embedded tangential no-slip viscosity,
consistent nonlinear momentum fluxes and small-cell time-stepping behavior.
Only then connect the partial-cell geometry to the full resolved momentum/scalar
runtime. Interfacial uptake, moving geometry, the physical density/partial-volume
closure, KDP calibration and meaningful quality/actuator/observation contracts
remain prerequisites for simulation-driven growth-control training. Real-data
ingestion and an offline XGBoost baseline can proceed in parallel; see
[AI_READINESS.md](AI_READINESS.md).
