# Fixed-geometry sampling and transport diagnostics

Historical output directories mentioned below were removed during the
2026-11-07 results cleanup. Re-run the diagnostic for raw measurements/figures.
The coupled checkpoint's compact record is [NUMERICAL_CHECKPOINT.json](NUMERICAL_CHECKPOINT.json).

The initial diagnostic step isolated spatial approximations at checkpoint
`561c551`. That checkpoint's coupled study has a 3.70% finest-pair face-advance
difference against its unchanged 2% working tolerance. Subsequent
sampling and transport reconstruction is described below. The diagnostics have no new pass/fail gate
for simulated growth and do not measure real KDP accuracy or crystal quality.

## Run and inspect

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_spatial_accuracy.py
```

The output directory defaults to `results/spatial_accuracy/` and must not already
exist. Use `--output results/another_name` for another experiment. The script
records the design, Git commit, source fingerprints and numerical package
versions in `study.json` before measuring. It saves `report.json`, `report.md`
and `summary.png`, with per-face measurements and resolved sampling settings.
The initial checkpoint used `results/spatial_accuracy_561c551/`, a historical
output directory removed during cleanup.

## Independent geometric reference

`BoxIntersectionMoments` enumerates feasible intersections of triples of planes
to recover a clipped solid's corners. Its triangular hull and an interior anchor
form tetrahedra. A tetrahedron's centroid is the mean of its four corners; adding
its volume-weighted centroid gives the first moment of the solid.

Subtracting the clipped solid's volume and first moment from the wet cell box
gives the actual liquid centroid. This reference uses a different clipping
algorithm from the runtime's edge intersections, although both use SciPy's hull
calculation. Tests independently check a rectangular box, a half-box, a
tetrahedron, a trapezoid and a cell with an occupied corner. For that last case,
removing the lower 1/8 corner of a unit cell shifts each liquid-centroid coordinate
from 1/2 to 15/28.

## Facet sampling check

Hold the default sloped seed fixed and prescribe dimensionless supersaturation
`sigma(x) = 0.35 + [0.08, -0.05, 0.06] dot (x - tank_center)`. Temperature is
constant and equilibrium concentration is one, so concentration is `1 + sigma`.
Use the same 0.75-unit normal-prism depth on the three original spatial grids.
Repeat with the same seed rotated about two axes and translated slightly.

Compare the production sampler using values at nominal Cartesian cell centers
with the same sampler using exact liquid-cell averages. An affine field's average
over a liquid cell equals its value at the liquid centroid. The analytic facet
reference is its value at the prism centroid: area centroid of the facet plus
half the outward extrusion. Independently integrating each prism/cell overlap
also gives that value to roundoff.

At checkpoint `561c551`, the engine weighted each entire liquid-cell average by its overlap volume. For a
nonuniform field, the whole-cell average need not equal the average inside that
smaller overlap. Exact volume weights therefore do not by themselves provide an
exact concentration integral.

Baseline maximum absolute supersaturation errors using true liquid averages:

| Pose | 20x16x16 | 30x24x24 | 40x32x32 |
|---|---:|---:|---:|
| Aligned | 0.000723461 | 0.00121980 | 0.000547395 |
| Rotated and shifted | 0.00299735 | 0.00147432 | 0.000597536 |

These are absolute supersaturation differences, not growth-error percentages.
The aligned errors are not monotonic as the face moves relative to cell edges.
Exact overlap-integral references agree with analytic prism means within 5.0e-16.
Independent cell-volume references agree with runtime volumes within 2.8e-16.

## Internal transport check

Use a 4x4x4 tank with fixed solid below `z = 1.27 + slope*x`, a full liquid level,
and cell counts 8, 12 and 16 per axis. Test slopes zero and 0.4. Prescribe the
affine concentration gradient `g = 0.1 * [1, 0, slope]`. This gradient is tangent
to the solid plane: its normal component is zero, and its Laplacian is zero.
Internal cut cells must therefore have zero net diffusive flux for this field.
Exclude the outer cell layer because the affine field does not satisfy the
tank's sealed outer-wall boundary conditions. No transient solution is claimed.

The checkpoint's diffusion operator uses actual liquid volumes/open areas, but its
two-point flux treats liquid-cell averages as values at nominal Cartesian
centers. The true liquid centroids shift when the solid cuts a cell. This is a
demonstrated consistency error for the oblique affine-field test.

For the aligned boundary, net-flux defects are at roundoff. For slope 0.4, maximum
interior cut-cell defects are 0.00279536, 0.00124952 and 0.000575700 in illustrative
concentration-volume/time units with unit diffusivity. Dividing by
`|g| * cell_width^2` gives 0.103817, 0.104413 and 0.0855238. This normalization is a
local flux scale, not a percentage error in growth or in a complete simulation.

An oracle using the prescribed exact gradient corrects the centroid offsets and
reduces these balance defects below 2.0e-16. It demonstrates the missing linear
information; it is not a usable runtime correction with estimated gradients.

## Sampling reconstruction

The runtime now computes solid/prism intersection first moments using its
edge-intersection geometry, independently checked against the plane-triple
reference above. Subtracting solid moments from wet-box moments locates actual
liquid centroids, including partially filled free-surface cells.

Local weighted least squares estimates field gradients from cell averages at
those centroids through open liquid faces. It uses no analytic gradient input.
Concentration slopes are scaled to keep reconstructed concentration nonnegative
on the wet bounding box. The sampler evaluates reconstructed concentration and
temperature at prism/cell centroids, then averages their supersaturation. This
is exact for affine concentration at uniform temperature; variable-temperature
ratios still require approximate quadrature. Stored field means and the existing
uniform fractional solute withdrawal remain unchanged by reconstruction.

The six aligned/rotated affine sampling cases in
`results/spatial_accuracy_reconstruction_561c551/` now have maximum errors below
5.0e-16. At that intermediate step the transport patch-test numbers remained
unchanged: diffusion and displacement equations were not modified. The initial temperature field now
stores true liquid averages of the same affine profile rather than nominal-center
samples. Final field archives include liquid centroids for inspection.

Tests compare runtime moments to the independent rotated reference, check actual
liquid centroids and initial temperature, recover affine gradients, handle a
one-dimensional stencil, reject reconstruction across a solid wall and verify
nonnegative slopes without modifying means. That intermediate checkpoint passed
all 80 tests; the complete focused-grid checkpoint passes 96.

File responsibilities: `crystal.py` computes geometric moments; `tank.py` stores
liquid centroids, prism moments and concentration slope limits; `physics.py` adds
the local gradient fit; `engine.py` uses reconstructed facet samples. The existing
demo and convergence runner export the new centroids. The crystal remains a
convex face-plane solid. No AI, flow-solver replacement or quality calibration
is part of this change.

Method reference: [OpenFOAM's least-squares and limited-gradient overview](https://doc.openfoam.com/2312/tools/processing/numerics/schemes/gradient/).

The full coupled rerun in `results/convergence_reconstruction_561c551/` completes
all five cases. Finest spatial differences are 0.87% net growth and 3.79% face
advance; the finest timestep pair passes (0.16% growth, 0.12% face advance).
Conservation, positivity and final time pass throughout. Analytic sampling is now
consistent, but coupled spatial agreement has not improved from the checkpoint's
0.30% growth and 3.70% face differences. The original 2% combined gate remains
failed, with transport correction as the next separate task.

## Corrected liquid transport

The production diffusive flux now corrects the offset between liquid centroids
and nominal Cartesian centers using the estimated local gradients. Merely using
the distance between two liquid centroids would not correct an oblique connection
to its shared face. The implicit corrected system retains paired conservative
edge fluxes. A bounded correction falls back to the positive two-point system
when needed; solved field values are not clipped.

For the sloped-plane patch test, the production correction reduces the three
interior balance defects from 0.00279536 / 0.00124952 / 0.000575700 to below
1e-15. Unlike the oracle reference, it estimates gradients from actual cell means.
The source-matched full study in `results/convergence_transport_561c551/` still
fails the original spatial face gate: 0.66% growth, 0.74% volume trajectory and
3.69% face advance. Its temporal and conservation gates pass. Analytic consistency
alone did not resolve the coupled spatial error.

## Moving-boundary reconstruction

An independent prescribed-motion case isolates remapping. A solid horizontal
layer rises from z=1.2 to 1.25 in a 4x4x4 tank while the free surface rises from
3.2 to 3.25. The initial field is `C=1+0.1*(z-1.2)`. Uniform upward translation
requires the new field `C=1+0.1*(z-1.25)` at the new liquid centroids.

The donor-cell reference gives maximum errors 0.00549775, 0.00383322 and
0.00299999 on 8, 12 and 16 cells per axis. Production remapping now combines
cell/aperture first moments, local linear reconstruction and a midpoint predictor.
Limits retain scalar mass, reconstructed bounds and complete evacuation of closing
cells. The same cases have concentration errors below 2e-13, temperature errors
below 4e-12, velocity errors below 2e-12 and relative mass errors below 3e-16.
These are model-unit analytic checks, not experimental accuracy claims.

The full original-grid study in `results/convergence_remap_561c551/` still has
0.70% growth, 0.77% volume trajectory and 3.74% face-advance differences. Temporal
and conservation gates pass. That failure is retained rather than interpreting
the analytic remap result as a coupled pass.

## Focused rectilinear mesh and limits

`--study-config configs/convergence_focused.json` repeats the sampling diagnostics
on the separately declared focused grids. The six aligned/rotated cases in
`results/spatial_accuracy_focused_v2_561c551/` have sampling errors below 6e-16.
Tests also exercise nonuniform-grid diffusion, prescribed translation and coupled
conservation. See `FOCUSED_GRID.md` for actual cell volumes, distances and rendering.
The completed final-source coupled study also passes its spatial, temporal and
conservation gates; the measured differences and scope are in `CONVERGENCE.md`.

These tests isolate known sampling/transport approximations. They do not apportion
every source of coupled growth error or establish accuracy of nonlinear depletion,
uniform cell-mean uptake, KDP kinetics, hydrodynamics or physical quality. Full
coupled refinement is still a separate acceptance check for each intended regime.

Method background: [NIST's cell averages and two-point diffusion discussion](https://pages.nist.gov/fipy/en/stable/numerical/discret.html)
and [NASA's verification assessment guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html).
