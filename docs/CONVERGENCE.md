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
