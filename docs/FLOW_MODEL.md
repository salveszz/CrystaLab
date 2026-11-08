# Fixed-crystal 3D incompressible flow

`src/simulation/flow.py` implements a constant-density, isothermal Navier-Stokes
component in SI units. It advances all three velocity components and dynamic
pressure. This is the first uniform-grid, fixed-geometry momentum milestone,
independent of the illustrative crystal-growth engine.

`configs/physical_reference.json` supplies the sourced KDP-water density and
viscosity at 30 C, the reference fixed cube and the rest-case definition.
`configs/flow_verification.json` declares independent analytical/manufactured
tests and their tolerances. Sources are recorded in root `SOURCES.md`, including
FLOW-PROJECTION-1968, FLOW-BOUNDARIES-2001, FLOW-BELTRAMI-1996 and DESIGN-FLOW-002.
No experimental velocity, pressure or quality data has calibrated this component.

## Fields and equations

The fluid obeys `du/dt + div(u*u) = -grad(p)/rho + nu*Laplacian(u) + a`,
with `div(u)=0`, `nu=mu/rho` and physical acceleration a in m/s2. Density and
viscosity are constant at the declared property reference point.

| Field | Location and array shape | Unit |
|---|---|---|
| Dynamic pressure p | Cell centers `[nz,ny,nx]` | Pa |
| x velocity | x-normal faces `[nz,ny,nx+1]` | m/s |
| y velocity | y-normal faces `[nz,ny+1,nx]` | m/s |
| z velocity | z-normal faces `[nz+1,ny,nx]` | m/s |
| Optional forcing | Evaluated at the corresponding velocity faces | m/s2 |

Pressure is not divided by density in storage. Periodic end faces are aliases,
not separate unknowns. A cell-centered velocity export averages the surrounding
face values for inspection; pressure projection acts on the face velocities.

The first-order time step is:

1. Form a momentum predictor from old velocity, conservative advective fluxes,
   prescribed acceleration and the old dynamic-pressure gradient. Solve viscosity
   implicitly using the symmetric staggered Laplacian.
2. Solve `(-D*G)*phi=-D*u_star` with the same divergence/gradient operators.
   Phi has units m2/s. Each disconnected fluid region receives one gauge.
3. Set `u_new=u_star-G*phi` and `p_new=p_old+rho*phi/dt`, then subtract each
   connected region's mean pressure. This is incremental pressure correction.

Shared momentum fluxes use upwind donor selection and monotonized central linear
slopes. Limited reconstruction reduces donor-cell numerical diffusion without
new face-reconstruction extrema. Boundary slopes fall back to lower order.
There is no post-step clipping of solved velocity or pressure. First-order time
integration and limiter/boundary errors remain; no global second-order claim is made.

Small sparse systems use cached factorization; larger ones use diagonal-
preconditioned conjugate gradients. Every solve checks its residual. Momentum
advection requires `dt*sum(max(abs(u_component))/component_spacing)<=0.45`.
`run_until` selects a step no larger than the requested maximum or a 0.4 CFL
step, and stops at the exact requested endpoint. Implicit viscosity does not
remove the explicit momentum-advection restriction.

## Solid and outer boundaries

The canonical `Crystal` geometry calculates solid/cell intersections after
normalizing coordinates to a safe geometric scale. This first momentum operator
accepts only fully fluid or fully solid cells. A partly occupied cell raises an
explicit error. The configured fixed cube aligns with the verification grid;
an arbitrary sloped KDP crystal is not silently approximated by that cube.

Blocked normal faces enforce zero penetration. Tangential reflected ghost values
impose no slip at stationary walls and the solid. The top uses zero normal flow
and zero tangential derivative: a fixed, shear-free **rigid-lid approximation**.
Its pressure correction follows impermeability; the fluid mean fixes the gauge.
It is inconsistent to additionally clamp pressure independently at every top cell
in a driven rigid-lid case. The original rest-only pressure wording was updated
when implementing this condition. A genuine moving free surface needs separate
normal-stress and kinematic conditions.

Uniform gravity is balanced in the hydrostatic reference pressure
`rho*g_abs*(H-z)`. The component stores the remaining dynamic pressure. It does
not yet calculate thermal/solutal buoyancy or infer a physical stirrer flow from
the old dimensionless agitation settings.

## Run the verification study

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_flow.py \
  --output results/flow_verification
```

The output directory must be fresh. Before measurements, `study.json` records the
resolved protocol/reference, actual source hashes, starting Git commit, package
versions and thread settings. The runner archives those source files and rejects
source changes during the study. Results include complete final staggered fields,
pressure, fluid masks, geometry, a Markdown/JSON report and figures. These are
new verification artifacts; the old growth results were removed earlier.

The test families are:

- Rest around the reference fixed cube through 120 s: zero velocity/dynamic
  pressure and exact excluded volume.
- A 3D periodic-horizontal channel domain with no-slip bottom and shear-free top:
  `u_x(z)=a_x*(2*H*z-z2)/(2*nu)`. Three vertical resolutions verify spatial error.
- The same channel's viscous eigenmode: `u_x=U*sin(pi*z/(2H))*exp(-nu*k2*t)`.
  Three timesteps isolate time error. All three coordinates remain in the solver.
- Periodic 3D ABC/Beltrami flow with all components nonzero. Here
  `curl(u)=k*u`, so nonlinear transport is `grad(|u|2/2)`, pressure is
  `-rho*|u|2/2` plus a gauge, and velocity decays as `exp(-nu*k2*t)`.
  This tests nonlinear momentum and pressure against an exact 3D solution.
- Nonzero vortical forcing around the fixed cube. This manufactured acceleration
  verifies exclusion, incompressibility and finite nonzero flow. It is not a
  measured pump/stirrer or a converged experimental KDP flow field.

Each case checks normalized divergence and blocked-face velocity. Channel and
decay errors must decrease and meet their finest-level tolerances; ABC velocity
and pressure errors must decrease and meet separate tolerances. These numerical
criteria are stated in the configuration and are not experimental error bounds.

## Limits and next integration

The first donor-only momentum study in `results/flow_verification_5505e9d/`
failed the declared monotonic ABC velocity-refinement gate. Although its velocity
errors were small, diffusion/time/spatial errors partly canceled. The report and
all five measured source files are retained with verified hashes. Tolerances
were not relaxed. The limited, conservative momentum reconstruction receives its
own source-matched study. Its first report in
`results/flow_verification_limited_5505e9d/` passed numerically but a plotting
call changed a dimensions array in memory before drawing the pressure slice.
The already saved numerical fields were unaffected. A copy at the plotting
boundary and a reporting-purity regression corrected the visual scale.

The final accepted study is `results/flow_verification_final_5505e9d/`.
All eleven cases pass the original criteria, and all five source files match
their recorded hashes and archived copies. Final error measurements:

| Test / finest setting | Relative RMS velocity error | Relative RMS pressure error |
|---|---:|---:|
| Channel, nz=32 | 0.03343% | n/a |
| Viscous decay, dt=0.125 s | 0.15657% | n/a |
| Nonlinear ABC, 24x24x24 | 0.20155% | 1.40745% |

The rest case remains exactly still through 120 s. All tested cases have zero
blocked-face speed; maximum normalized divergence is 3.075e-16. The manufactured
fixed-crystal case reaches maximum centered speed 3.879e-5 m/s at 2 s, with
nonzero motion in all three components. Its flow is an example, not an
experimentally validated or spatially converged obstacle prediction. The final
images were inspected and all field archives checked finite with correct SI
domain dimensions. The full working suite passes 122 tests.

Uniform grids, fixed aligned solids and constant properties are the current
scope. Focused/nonuniform momentum cells, oblique cut faces, moving/growing
solids, deformable free surfaces, turbulence closures, thermal/solute transport
and buoyancy remain unsupported by this component. The original growth demo
still uses its conservative displacement approximation; it has not quietly
become a calibrated CFD growth simulator.

Next connect conservative, adequately resolved solute transport to these face
volume fluxes. KDP's large solutal Peclet/Schmidt scales make numerical diffusion
an important separate check. Then extend the momentum discretization to actual
cut-cell geometry and couple interfacial uptake/motion. Keep empirical stirring
effects out of a resolved-flow path unless their remaining contribution is
explicitly justified and calibrated. Validate kinetics and quality independently.
