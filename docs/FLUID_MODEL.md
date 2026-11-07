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

## Transport and moving boundaries

Liquid diffusion uses a finite-volume graph with conductance equal to open liquid
face area divided by Cartesian center spacing. Fully blocked faces and dry cells
have no connection. Backward Euler solves `(diag(V) + D*dt*L) c_new = V*c_old`.
Residual-checked preconditioned conjugate gradients solve the SPD systems, with
sparse direct fallback for small systems or unsuccessful residual checks. This
conserves scalar amount and handles very small partial cells without a tiny
explicit timestep or clipping unstable negative values. It is a low-order two-point
flux approximation, not a corrected high-order cut-cell stencil.

When the crystal grows, old and new liquid capacities determine a displacement
source: `L * potential = V_old - V_new`. Integrated edge volume fluxes follow
potential differences and satisfy discrete volume continuity. Edge conductance
uses the larger old/new liquid opening to approximate the transition geometry.
Upwind donor transport moves solute and the sensible-heat proxy along that flux.
Newly wet cells receive actual donor content. The potential is not physical
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
`fields.npz`. The archive stores temperature/concentration, liquid capacities,
liquid/solid fractions and displacement vectors. Scalar fields index `[z,y,x]`;
vector fields index `[z,y,x,component]` with components x/y/z. Display arrows are
scaled for visibility. The geometric free-surface height is not exaggerated.

Tests cover analytic solid intersections, volume sums for rotated cut cells,
blocked transport, positive/conservative small-cell diffusion, grid/time refinement,
carrier/solute/heat conservation during remapping, discrete continuity and rim
limits. Numerical tests do not validate the model against real KDP experiments.

Method references: [NIST finite-volume discretization](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)
and [SciPy sparse linear solves](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
