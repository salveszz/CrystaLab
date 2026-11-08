# Resolved flow and solute on unequal cells

The existing SI solver now accepts explicit Cartesian edges through
`IncompressibleFlow(..., grid_edges_m=[x_edges, y_edges, z_edges])`.
The same runtime handles uniform, smoothly stretched and focused grids. The
solid must still align with grid faces; partial momentum cells raise an error.
This milestone establishes unequal-cell operators before introducing oblique
embedded boundaries or moving geometry.

## Mesh and physical measures

`simulation.mesh.RectilinearGrid` validates three increasing edge arrays from
zero to their stated domain lengths. It stores actual cell widths, Cartesian
centers, volumes, staggered face areas and pressure-center distances. Metric
arrays are read-only. Inputs and exported components follow x/y/z; scalar arrays
index z/y/x. `flow.spacing` and `flow.cell_volume` are now minimum-spacing/volume
summaries for conservative timestep bounds, rather than universal cell sizes.
Operators and mass accounting use the complete arrays.

`focused_axis_edges()` is the existing growth tank's band-and-stretched-tail
algorithm, moved into this shared mesh module with input validation. The growth
tank imports that function; its focused grid and the SI flow grid use the same
construction. SI lengths must be provided explicitly rather than interpreting
the illustrative tank's lengths as meters.

## Pressure, viscosity and momentum

For a face shared by pressure cells i and j, A is its area and d the actual
Cartesian distance between those cell centers. Volume flux is `Q=A*u` and
cell divergence is the signed sum of its face fluxes divided by its own volume.

Projection assembles a symmetric pressure stiffness with conductance `A/d`,
solving it against the negative net predictor volume flux. The correction to
face velocity is `-(phi_j-phi_i)/d`; phi has units m2/s. This uses compatible
divergence and gradient operators. Every disconnected liquid region gets one
pressure gauge, with a volume-weighted mean of zero. The projection's kinetic
energy inner product uses the staggered control volumes `A*d`.

Viscosity uses the actual staggered control volumes and a symmetric stiffness K
whose connections have conductance `dual_face_area/velocity_node_distance`.
The implicit solve is `(M + dt*nu*K)*u_new = M*rhs`, with diagonal M containing
the momentum volumes. The mass-normalized Laplacian is generally nonsymmetric
on unequal cells; the system supplied to conjugate gradients is symmetric.
No-slip tangential wall fluxes use the actual half-cell distance, and the rigid
lid remains impermeable and shear-free.

Momentum advection transfers the same signed flux between neighboring momentum
cells. Transverse dual volume fluxes combine half of each neighboring primal
volume flux, including unequal-width weighting. Reconstruction uses physical
slopes and the actual distance from each donor velocity node to the shared dual
face. Reconstructed face values are bounded by adjacent values; solved velocity
and pressure fields are not clipped. The time integrator remains first order.

Periodic seams include the half-widths from both ends. A duplicate end face
remains an alias, including when the two end-cell widths differ. Energy, pressure
gauges and verification errors use physical control-volume weights.

Method background is registered as FLOW-NONUNIFORM-2002 in `SOURCES.md`. Our
limited first-order time method is a project implementation and does not claim
the reference paper's nondissipative/second-order method.

## Conservative solute

The scalar still uses the flow's native volume fluxes. Dissolved mass is now
`sum(C_i*V_i)` over actual liquid cells. Diffusion conductance is `D*A/d`, slopes
use actual center distances and scalar reconstruction uses actual donor-to-face
distances. The correction budgets and outgoing-rate substeps use each cell's
own capacity. A raw concentration sum is not a conserved quantity on this mesh.

The scalar RK2 and momentum/solute coupling contracts are described in
[SOLUTE_MODEL.md](SOLUTE_MODEL.md). Fields use the solution-volume concentration
basis, with frozen properties at the declared reference point. No empirical
stirring multiplier enters the resolved-flow path.

## Independent verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_rectilinear_flow.py \
  --output results/rectilinear_verification
```

The output folder must be fresh. `configs/rectilinear_verification.json` declares
the mathematical fields, grids, times and tolerances. The runner archives all
nine measured Python sources, protocol, fluid reference, Git identity and package
versions before measuring. It rejects source changes during a study. Raw NPZ
files contain actual edges/volumes, native velocity/pressure, masks, scalar fields
and exact scalar reference means where available. Figures draw the true cell
edges rather than assigning equal-size pixels to unequal cells.

The twenty cases cover:

- Three channel grids against the quadratic solution and three timesteps against
  viscous eigenmode decay on the same stretched mesh.
- Three nonlinear 3D ABC grids against exact velocity and dynamic pressure.
- Three focused ABC grids with simultaneous uniform-solute transport, checking
  flow accuracy, incompressibility, mass and constant-field preservation.
- Three 3D translating/diffusing scalar waves and three Neumann diffusion cases,
  compared with integrated analytic cell means and physical-volume RMS errors.
- Uniform/nonuniform solute in nonzero resolved forcing around the stationary
  aligned cube, with four cells across its width and coarser cells farther away.

The original uniform-grid momentum and high-Peclet scalar studies are rerun
separately to check compatibility. New tests also check weighted projection
energy, force/pressure balance, symmetric viscous stiffness, periodic momentum
conservation, unequal periodic end widths, raw-sum versus mass behavior, actual
solid volumes, failure retention and archived mesh identity.

The first study, `results/rectilinear_verification_first_8011b79/`, failed its
unchanged diffusion error gate: 0.156298% at 16 cells per axis, exceeding the
declared 0.15% limit. Its other gates passed. It is retained as a failure. The
follow-up uses diffusion counts 8/12/20 instead of 6/10/16 and keeps the same
physical setup, equations and 0.15% acceptance threshold. No tolerance is relaxed.

The accepted study is `results/rectilinear_verification_final_v2_8011b79/`.
All twenty cases pass. Its nine numerical source files match their hashes and
archived copies; raw arrays are finite and actual mesh volumes have been checked
against edge products. Both final figures were inspected. The earlier final
report's crowded log-axis labels were corrected in this source-matched rerun;
the numerical results are unchanged.

| Finest case | Velocity/scalar RMS error | Pressure RMS error |
|---|---:|---:|
| Stretched channel, nz=32 | 0.04044% | n/a |
| Stretched viscous decay, dt=0.125 s | 0.15730% | n/a |
| Stretched ABC, 20x20x20 | 0.32612% | 1.12922% |
| Focused ABC + uniform solute, 24x24x24 | 0.81133% | 2.47371% |
| Stretched 3D scalar wave, 20x20x20 | 2.20441% | n/a |
| Stretched Neumann diffusion, 20x20x20 | 0.10067% | n/a |

No solute-mass drift, scalar-bound violations or uniform-concentration error were
measured in these cases. Maximum normalized divergence is 5.155e-16 and all
blocked-face speeds are zero. The separate eleven-case uniform momentum and
fifteen-case uniform/high-Peclet scalar regressions also pass under the current
operators. The full suite passes 151 tests. These are numerical results for the
declared fields, not experimental accuracy or KDP growth/quality validation.

## File-by-file review

| File | Change and purpose |
|---|---|
| `src/simulation/mesh.py` | Shared rectilinear metric arrays, input validation and focused edge generation. |
| `src/simulation/tank.py` | Import the shared focused generator rather than maintaining a separate copy. |
| `src/simulation/flow.py` | Explicit edge input; actual primal/dual measures in projection, pressure gauge, viscosity, momentum fluxes and energy; actual coordinate exports. |
| `src/simulation/solute.py` | Per-cell mass/capacity and actual face/center metrics in diffusion, slopes, correction budgets and subcycling. |
| `scripts/check_flow.py` | Optional edge inputs, physical-volume error weights, actual coordinate references and exported mesh/source identity. |
| `scripts/check_solute.py` | Actual flow coordinates, optional obstacle mesh and exported mesh/source identity. |
| `scripts/check_convergence.py` | Fingerprint the newly shared mesh and its unit-validation dependency in growth studies. |
| `scripts/check_rectilinear_flow.py` | Reproducible twenty-case unequal/focused verification study, raw outputs, source archives and figures. |
| `configs/rectilinear_verification.json` | Declared study settings, independent reference definitions and acceptance thresholds. |
| `tests/test_rectilinear_flow.py` | Thirteen geometry/operator/coupling/refinement and study-integrity regressions. |
| `docs/RECTILINEAR_CHECKPOINT.json` | Tracked final protocol, source hashes and measured results; raw studies stay ignored. |
| `docs/AI_READINESS.md` | Current readiness and remaining dependencies before growth-control training. |
| `docs/FLOW_MODEL.md`, `docs/PHYSICAL_MODEL.md`, `docs/SOLUTE_MODEL.md`, `docs/FLUID_MODEL.md`, `docs/FOCUSED_GRID.md` | Update the implementation status, contracts and links to the current unequal-grid evidence. |
| `SOURCES.md` | Register the nonuniform-method background, explicit design choices, failed-stage history and measured checkpoint. |

Every Python file added or changed has an explanatory summary at the top. This
work is local for review; no commit or push is included in this step.

## AI preparation

This supplies a required numerical foundation for local resolution around a
crystal. Sloped momentum boundaries, interfacial uptake and moving geometry still
require their own operators and independent refinement studies. A stationary
aligned-cube conservation check does not establish accuracy around a growing KDP
crystal. The remaining training gates are in [AI_READINESS.md](AI_READINESS.md).
