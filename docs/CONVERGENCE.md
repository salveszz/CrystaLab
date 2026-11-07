# Coupled growth/displacement convergence study

The study measures resolution sensitivity in the existing simulator. It keeps
the growth equations, geometry handling and liquid flux discretization fixed. It does not validate
physical KDP parameters or optical quality.

## Run and inspect

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_convergence.py
.venv/bin/python scripts/plot_convergence.py --report results/convergence/report.json
```

The default output is `results/convergence/`. A fresh study refuses to overwrite
an existing study. Use `--resume` with identical protocol, runner, solver sources
and numerical environment, or choose another `--output` directory. Completed
cases can be reused; an interrupted case restarts from time zero. For a short
pipeline check, use `--duration 2 --output results/convergence_smoke`.

`configs/convergence.json` declares the protocol and tolerances. Its
`simulation_config` path is relative to that configuration file. The protocol is
saved before any run begins. The manifest records fully resolved simulation
settings, solver/runner fingerprints, the starting Git commit, package versions
and numerical thread settings.

## Three controlled steps

1. Spatial refinement: run grids 20x16x16, 30x24x24 and 40x32x32 at dt=1.
2. Temporal refinement: run dt=1, 0.5 and 0.25 on the fixed 40x32x32 grid.
3. Compare growth and conservation with identical world-space inputs. The finest
   grid at dt=1 is shared, so five unique experiments cover these two sweeps.

Every run uses the same 120-unit duration, domain, seed geometry/orientation,
initial free level, analytic temperature profile, initial concentration and
physical coefficients. Concentration noise and stochastic defect events are
explicitly disabled. The sampling layer stays 0.75 length units thick. The runner
adjusts `boundary_layer_cells` for each spacing; retaining a fixed cell count
would change the physical layer while refining the grid.

This is an isolated-axis design, not the full 3x3 factorial study. It cannot by
itself rule out all grid/time interactions or establish an asymptotic convergence
order. The finest run is a reference, not a known exact solution.

## Metrics and predeclared gates

The default numerical refinement gate requires the finest adjacent spatial pair
and the finest adjacent temporal pair to differ by at most 2% in:

- Final net volume gain and maximum volume-trajectory difference.
- Final surface area and maximum face-position/advance trajectory difference.
- Liquid-level rise and final volume-weighted liquid concentration.

The active kinetic-face sets must match. Growth and level comparisons normalize
by the change from the initial state, rather than by large initial volume/height
baselines. Face-position differences normalize by the largest reference face
advance. Histories are linearly interpolated onto the union of their output times
for comparison; interpolation does not modify a simulated state.

Every case must also satisfy maximum relative tracked-solute mass error <=1e-10,
carrier-volume error <=1e-10, continuity residual divided by whole-cell volume
<=1e-8, concentration >=-1e-12 and exact final time. These are declared numerical
checks, not experimentally established physical tolerances. Tolerances must not
be changed after seeing results to turn a failure into a pass.

## Artifacts and interpretation

The runner saves `report.md`, `report.json` and `study.json`. The plot command reads
that report and creates `summary.png` without rerunning the solver. Each case also has
`case.json`, `measurements.csv` and `final_fields.npz`. Reports contain complete
histories, face offsets, final face rates, dimensions, conservation diagnostics
and runtime. Final 3D fields remain available for later spatial analysis.

A run completing successfully is separate from its refinement gate passing.
The report prints PASS/FAIL for spatial refinement, timestep refinement and
conservation. Coarser disagreements remain visible even if the finest pair passes.
The gate measures geometry and growth outcomes; it does not compare registered
full-field error norms on the moving liquid domain.

If the coupled gate fails, investigate numerical resolution, face-neighborhood
sampling, transport stencils and splitting errors before claiming resolution-
independent growth. A good mass balance alone cannot establish that growth rates
are accurate. Any subsequent solver change needs its own explained review and a
new source-matched study rather than reusing the old measurements as new results.

## Linear-algebra runtime improvement

The initial direct-LU attempt completed two baseline cases before its finest-grid
cost became excessive. Those measurements are retained separately as an incomplete
reference study. A matrix benchmark found an alternative permutation slower, while
Jacobi-preconditioned conjugate gradients agreed with direct solutions to about
1e-12 and reduced refined-grid solve cost substantially.

The core now uses a strict residual-checked iterative SPD solve, with the original
sparse direct solve for small systems or a failed iterative residual check. This
changes the linear-algebra implementation, not equations, fluxes, time steps or
physical coefficients. SciPy >=1.12 supplies the explicit rtol API. Tests compare
iterative/direct results and exercise fallback. One far-field test now allows
roundoff (12 decimal places) instead of requiring bitwise equality after remapping.
No study tolerance was relaxed.

All five study cases are rerun under the new solver fingerprint. The two completed
direct baselines are used separately for solution-agreement checks, never mixed
into the optimized study's refinement gate. Results directories keep their exact
source/environment identities. Any performance claim applies to measured runs and
includes setup/geometry costs separately from the matrix microbenchmark.

## First measured outcome

The full-duration working-tree study in `results/convergence_spd` completed all
five cases with the stated protocol and unchanged 2% threshold. Results:

| Finest adjacent comparison | Net growth change | Face advance change | Assessment |
|---|---:|---:|---|
| 30x24x24 to 40x32x32, dt=1 | 3.07% | 5.68% | FAIL |
| dt=0.5 to 0.25, grid=40x32x32 | 0.20% | 0.16% | PASS |

The spatial volume-trajectory change is 3.54%. All cases pass conservation,
positivity and exact final time; the maximum relative combined-mass error is
2.080e-12. Good mass balance therefore does not establish spatially
converged growth. The next investigation is to isolate face-neighborhood sampling,
cut-cell flux errors and seed resolution before changing the solver or using AI.
These were candidate explanations at the end of the first study.

Artifacts are ignored generated outputs; rerun commands above to reproduce the
study. Its manifest contains the exact working-tree source/environment hashes.

## Spatial sampling diagnosis and correction

A constant-field geometry check isolated a grid-dependent sampling domain in the
original center-selected face neighborhoods. For the same default seed and
0.75-unit sampling depth, its selected liquid volumes were:

| Grid | Original selected volume | Exact normal-prism volume |
|---|---:|---:|
| 20x16x16 | 4.564000 | 2.509447 |
| 30x24x24 | 3.099111 | 2.509447 |
| 40x32x32 | 1.844667 | 2.509447 |

The original donor region changed by +82%, +23% and -26% relative to the
continuous facet prisms. That is a demonstrated sampling-domain inconsistency;
it does not prove that all coupled growth error came from sampling.

The corrected implementation intersects each wetted facet's outward normal prism
with the liquid cells, tank and free surface. It uses those fractional volumes
for both supersaturation averaging and the solute withdrawal budget. The prism
depth and all protocol tolerances remain unchanged. The fully submerged initial
seed now integrates to `surface_area * depth = 2.509446968344222` on all three grids,
within floating-point roundoff. Tests also translate and rotate the seed, check
constant-field averages and verify deposited-mass conservation. Nearly coincident
facet corners are projected onto their defining plane and their convex perimeter
is recovered before extrusion, preventing invalid prism side planes.

This correction still samples piecewise-constant cell fields and uses the same
low-order transport stencil. Its coupled refinement result must be measured in
a fresh source-matched study; the exact sampling-volume check alone is not a
claim of converged growth. Diagnostic JSON files are in
`results/spatial_sampling/`.

## Corrected sampling outcome

The fresh study in `results/convergence_prism_v2/` completed all five full-duration
cases under the unchanged protocol. Its recorded solver and runner fingerprints
match the current source. The finest spatial pair gives:

| Metric | Original sampling | Geometric sampling | 2% gate |
|---|---:|---:|---|
| Net volume gain | 3.07% | 0.30% | PASS |
| Maximum volume-trajectory difference | 3.54% | 0.61% | PASS |
| Maximum face-advance difference | 5.68% | 3.70% | FAIL |

The other spatial gate metrics pass. The finest timestep pair passes, with
0.16% growth/trajectory change and 0.12% face-advance change. All five cases pass
conservation, nonnegative concentration and exact final time. Maximum relative
tracked-mass error is 3.520e-12, maximum carrier-volume error is 1.422e-15 and
maximum continuity residual divided by cell volume is 4.733e-11.

The combined numerical gate therefore still **fails**. The coarser spatial pair
also remains outside tolerance (2.63% growth and 8.31% face advance). These results
show improved volume agreement within this protocol, not complete spatial
convergence or validated physical growth.

The remaining finest-pair discrepancy is largest on the four lateral facets,
about 0.01723 length units of advance at time 120. Cap advances change in the
opposite direction, so close total volumes can hide shape differences. Per-face
measurements are saved in `spatial_face_diagnostic.json` alongside the report.

The next small investigation is a fixed-geometry, nonuniform-field check of
liquid-cell averaging and transport near planar and oblique facets. The current
piecewise-constant sampling and two-point cut-cell flux remain candidate error
sources; the remaining error has not yet been isolated between them. Keep the
same tolerances and physical coefficients when testing the next correction.

For code review: `crystal.py` constructs the wetted facet prisms; `tank.py`
integrates their cell overlaps; `engine.py` uses the weights for sampling and
fractional mass uptake. `tests/test_face_sampling.py` adds eight independent
geometry/conservation regressions. The complete suite has 66 passing tests.
The default visual demo also completes at time 120 and exports finite 3D fields
and a 61-frame single-pass GIF in `results/sampling_demo/`.
