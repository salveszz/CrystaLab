# Open-tank crystal and liquid model

This is an uncalibrated faceted-growth research model. All lengths, times,
concentrations and coefficients use illustrative model units. Its displacement
field is quasistatic, not a Navier-Stokes solution or validated KDP hydrodynamics.

## Geometry and fluid accounting

The crystal is a convex polyhedron represented by outward face normals and offsets.
Vertices, actual face areas and volume follow from those planes. Geometric
intersections calculate solid volume inside each grid cell and blocked area on
each shared cell face. A seed smaller than one voxel therefore still displaces
its true volume. Planar sections in the renderer show the actual solid boundary.

Liquid is present only below a horizontal free surface and outside the crystal.
Each cell stores actual liquid capacity `V_i` and concentration per liquid volume,
not per whole voxel. Fully dry/solid cells have zero stored field values, which
are masked from plots and averages; zero is a storage convention, not a prediction
of temperature in the crystal or air.

`initial_liquid_height` specifies the level with the initial seed already present.
The default leaves headspace at 75% of the tank height. Growth raises the level H
so that `tank_base_area * H - submerged_solid_volume` remains constant. If the
crystal emerges, only the submerged part contributes to displacement. The model
pauses growth at a full rim rather than silently losing solution through overflow.

The carrier liquid is incompressible. Dissolved solute has negligible partial
volume in this dilute approximation. Crystallization removes dissolved solute
mass but leaves carrier volume constant. This assumption needs replacement with
measured solution density/partial-volume data for quantitative laboratory work.

Combined tracked solute mass is `sum(concentration_i * V_i) + density * crystal_volume`.
Face growth is scaled to locally available solute and headspace before consumption.
Only wetted face area drives deposition; one averaged normal speed still advances
an entire planar facet. Surface roughness and spatial growth variation within a
facet are not modeled.

Each wetted facet is extruded along its outward normal by the configured sampling
depth. Exact intersections of that prism with cells, tank walls and the free
surface define volumes `W_fi` and first moments. First moments also locate each
cell's actual liquid centroid. An aperture-weighted, inverse-distance least-squares
fit estimates concentration and temperature gradients from open-face liquid
neighbors. A rank-deficient stencil recovers only observable components; an
isolated cell uses zero gradient.

The sampler reconstructs both fields at the centroid of each prism/cell overlap,
computes supersaturation there and averages it with weights `W_fi`, scaled by the
wetted fraction of the facet. Concentration gradients are scaled if needed so
the reconstructed field stays nonnegative throughout the cell's wet bounding
box; stored cell means are unchanged. This integrates affine concentration
exactly at uniform temperature, but the nonlinear concentration/solubility ratio
uses one quadrature point per overlap. It does not resolve an interfacial
boundary layer. Keep the depth fixed in physical units when comparing grids.
Normal prisms over distinct facets of a convex crystal have disjoint interiors;
edge/corner neighborhoods outside those prisms are excluded by this convention.

The same weights define the local solute budget. With `W_i = sum_f W_fi`, the
available mass is `sum(C_i * W_i)`. Deposition removes the fraction
`(deposited_mass / available_mass) * W_i / V_i` from each cell concentration.
This keeps uptake conservative and nonnegative without treating every partly
intersected cell as a full donor. Facets currently share this pooled local budget;
there is no separate face-by-face diffusion-limited deposition solve.
Withdrawal still uses the uniform cell-mean approximation; the gradient correction
in this step improves the kinetic sampling only. It does not make the whole
coupled transport/uptake model second order. The initial affine temperature
profile is now averaged at actual liquid centroids, including cut/free-surface
cells, while preserving the same continuous profile and physical coefficients.

## Transport and moving boundaries

Liquid diffusion uses a finite-volume graph with conductance equal to open liquid
face area divided by the actual distance between neighboring Cartesian centers.
Fully blocked faces and dry cells have no connection. A local least-squares
gradient corrects the offset between liquid centroids and those nominal centers.
The corrected operator is assembled into the backward-Euler system. GMRES with
a diagonal preconditioner solves this nonsymmetric system, with residual checks
and sparse direct fallback. The two-point SPD system uses conjugate gradients.

If a corrected candidate leaves the input field range, limit opposite correction
transfers by cell mass budgets and solve the SPD base with that bounded RHS.
This preserves conservation and the range without clipping solved field values.
Case reports retain counts of bounded solves and direct fallbacks. Accuracy can
reduce near limiting; this is not a claim of global second-order convergence.

When the crystal grows, old and new liquid capacities determine a displacement
source: `L * potential = V_old - V_new`. Integrated edge volume fluxes follow
potential differences and satisfy discrete volume continuity. Edge conductance
uses the larger old/new liquid opening to approximate the transition geometry.
Aperture and cell first moments infer interval-averaged material displacement.
Local linear gradients and a midpoint predictor estimate the transported scalar
at each aperture. Flux limits preserve the reconstructed field bounds, scalar
mass and heat proxy; closing cells donate all content and newly wet cells receive
actual donor content. The donor transport is the conservative low-order base.
The exported displacement velocity now uses the same first-moment balance at
currently wet cells. The potential is not physical
pressure; inertia, viscosity, buoyancy, a resolved boundary layer, free-surface
waves and mechanical stirring velocities are absent.

Disconnected liquid that would need compression raises an explicit error rather
than routing liquid through solid. Temperature relaxation represents an external
bath. Stirring is a saturating effective diffusivity through existing liquid
connections, replacing global mixing that could cross a solid barrier. Latent heat,
solid thermal conduction, evaporation and gas-side transport are not modeled.

## Inspection and checks

Run `python scripts/run_simulation.py --headless`. The output directory contains
single-pass `growth.gif`, `final_tank.png`, `measurements.csv`, `config.json` and
`fields.npz`. The archive stores temperature/concentration, liquid capacities/centroids,
liquid/solid fractions, cell edges/volumes, aperture centroids and displacement
vectors. Scalar fields index `[z,y,x]`;
vector fields index `[z,y,x,component]` with components x/y/z. Display arrows are
scaled for visibility. The geometric free-surface height is not exaggerated.

Tests cover analytic solid intersections, volume sums for rotated cut cells,
blocked transport, positive/conservative small-cell diffusion, grid/time refinement,
carrier/solute/heat conservation during remapping, discrete continuity and rim
limits. Numerical tests do not validate the model against real KDP experiments.

Method references: [NIST finite-volume discretization](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)
and [SciPy sparse linear solves](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).

## Code responsibilities

This map explains the numerical checkpoint's code responsibilities.

| File | Purpose of the current change |
|---|---|
| `src/simulation/crystal.py` | Integrate solid/prism volumes and first moments, and polygon section areas/moments. This locates liquid averages and transport apertures geometrically. |
| `src/simulation/tank.py` | Build uniform/focused cell edges; calculate true capacities, centroids and openings; initialize averaged fields; connect reconstructed transport and interval displacement velocity. |
| `src/simulation/physics.py` | Estimate local gradients; correct implicit diffusive fluxes with bounded fallback; reconstruct conservative midpoint transfers during boundary motion. |
| `src/simulation/engine.py` | Sample reconstructed fields at exact prism-overlap centroids and use actual cell volumes for occupancy/uptake accounting. The uniform cell-mean withdrawal approximation remains explicit. |
| `scripts/check_spatial_accuracy.py` | Compare sampling, internal diffusion balances and prescribed liquid translation with independent analytic/geometric references. |
| `scripts/check_convergence.py` | Run independent cases in parallel, preserve protocol/source identity and export actual grid geometry and limiter/solver diagnostics. |
| `scripts/run_simulation.py` | Default to the focused configuration, draw unequal cells at their actual edges and export the complete mesh/moment fields. |
| `tests/test_spatial_accuracy.py` | Verify independent moments, local gradients, cut-cell diffusion bounds/fallbacks and analytic moving-layer transport. |
| `tests/test_focused_grid.py` | Verify unequal-cell geometry, positivity limiting at real bounds, sampling, diffusion refinement and moving-boundary conservation. |
| `tests/test_convergence.py` | Check that independent process execution agrees with serial histories and face motion. |

`configs/focused.json` selects the refined demonstration and
`configs/convergence_focused.json` declares its separate unchanged-tolerance
study. Published-reference metadata is in `configs/benchmarks/`; its limitations
and the future real-data comparison procedure are in `REAL_DATA_VALIDATION.md`.
