# Coupled heat, solute and buoyancy verification

The fixed-domain solver now transports **temperature and dissolved mass**, with
both fields affecting momentum through declared Boussinesq buoyancy. External
thermal walls support prescribed temperature or a finite heat-transfer film.
Nine numerical cases pass the frozen checks and a separate native-field audit.
This advances the pitch's heat/mass-transfer component. It supplies no real KDP
calibration, crystal growth, physical risk labels or control benefit.

The main camera milestone remains reviewed real images and independent model
evaluation, as described in [Next steps](NEXT_STEPS.md). Supporting numerical
experiments do not supply those labels.

## Equations and units

The existing fixed-grid [momentum solver](FLOW_MODEL.md) retains constant reference
density, incompressibility, projection and implicit viscosity. The new
`src/simulation/thermal.py` adds

```text
dT/dt + div(u T) = alpha laplacian(T), alpha = k / (rho0 cp)
dC/dt + div(u C) = D laplacian(C)
rho'/rho0 = -beta_T (T - T0) + beta_C (C - C0)
buoyancy acceleration = (rho'/rho0) gravity
```

Temperature is Kelvin; C is kg solute/m³ solution. cp is J/(kg K), k is W/(m K),
beta_T is 1/K and beta_C is m³/kg. Warmer liquid rises under downward gravity;
larger concentration increases density for the declared positive beta_C. An
explicit budget limits `abs(beta_T ΔT) + abs(beta_C ΔC)` rather than hiding
cancellation between large terms. The implementation refuses an exceeded budget.
This is a constant-property approximation, not a measured equation of state.

The shared `ConservativeScalarTransport` contains the existing finite-volume
advection/diffusion operator. `SoluteTransport` retains its public concentration
API and kg budget; `HeatTransport` uses its own Kelvin field and sensible energy
`rho0 cp sum(V (T - Tref))`. Internal face transfers use opposite signs in the
two adjacent cells. There is no conversion of concentration values into
temperature labels.

For an external thermal wall, conductance is
`G = A / (half_cell_width/k + 1/h)` in W/K. Prescribed temperature is the `h → ∞`
case; `h = 0` is insulation. Wall exchange is integrated in both SSP RK2 stages
and included in the stability rate, local bounds and heat budget. Time-dependent
ambient temperature uses simulation seconds. Fixed aligned solids remain
insulating; periodic directions cannot carry external thermal walls.

Buoyancy is evaluated from the old scalar fields and interpolated to momentum
faces using actual center-to-face distances. Mean old/new projected volume flux
advects both scalars. This coupling and the momentum method are **first order in
time overall**. RK2 scalar substeps do not make the full system second order.
Flow, heat, solute, clocks and budgets roll back together after a failed step.

## Frozen experiment and actual results

[`configs/thermal_verification.json`](../configs/thermal_verification.json) is
DESIGN-THERMAL-001. All fluid, heat and expansion constants are numerical design
inputs. They are **not measured KDP properties**. The 10 mm box uses rho0=1000
kg/m³, mu=0.001 Pa s, cp=4000 J/(kg K), k=0.4 W/(m K), D=5e-8 m²/s,
beta_T=2e-4/K and beta_C=8e-4 m³/kg. No material identity or physical operating
recipe is inferred from these numbers.

Accepted output is `results/thermal_verification_v2`, with nine native NPZ field
archives, protocol bytes, source snapshots, environment and artifact hashes.
The original `v1` output is retained; `v2` records the formatted source without
changing the frozen protocol. The seven frozen checks pass.

| Case | Actual result | What it establishes |
|---|---|---|
| Periodic buoyant shear, 16×4×4 | Velocity relative RMS error 0.012972, 0.006565, 0.003303, 0.001657 at dt=0.1, 0.05, 0.025, 0.0125 s | Approximately first-order time convergence of signed thermal/solute forcing plus viscous response. |
| Wall conduction, 8/16/32 x cells | Temperature RMS error 8.0373e-4, 2.0259e-4, 5.0753e-5 K at 10 s | Spatial refinement toward the continuous cell-average heat solution. |
| General 3D flow, buoyancy off/on, 8³ | Maximum component difference 3.3944e-4 m/s at 0.25 s | Scalar-dependent feedback changes the flow in this numerical example. No experimental reference is supplied. |

The shear reference is continuous in time on the **same spatial grid**. It isolates
time error; it does not claim spatial convergence of a physical buoyant flow.
The z velocity, temperature and concentration are z-invariant cosine modes in x,
so the nonlinear advection term vanishes for this case. Both signed buoyancy
sources remain nonzero. The wall case has prescribed 300/310 K boundaries and a
decaying sine perturbation around the linear steady solution; its reference uses
continuous space and exact cell averages.

`scripts/audit_thermal.py` imports no producer or simulation module. It verifies
every manifest hash, reconstructs the shear using a 3×3 matrix exponential and
an independent finite-step modal recurrence, reconstructs wall cell integrals,
and checks raw-field heat/mass budgets, bounds, periodic seams and divergence.
The general 3D comparison establishes a numerical response, not accuracy against
a known physical flow. The wall heat-budget audit checks the recorded integrated
exchange against the saved energy change; it is not an independent experimental
wall-flux measurement.

The audit is saved as `results/thermal_independent_audit_v2.json`. A test also
alters and rehashes a native velocity archive to verify that numerical auditing
rejects it after integrity checks have been satisfied. A PNG/SVG figure and its
source identities are in `results/thermal_figures_v2`; the rendered PNG was
inspected. The first plot is retained with its crowded tick labels.

The initial targeted run exposed missing duplicated periodic-end values in the
public buoyancy field. The momentum solver already ignored that duplicate, but
the returned field was inconsistent at the seam. The field now copies its first
periodic face to the last. The failed log is retained. No acceptance threshold
was relaxed. The [checkpoint](THERMAL_CHECKPOINT.json) records final verification
and exact source/artifact identities.

Final verification passed **457 tests and 323 subtests** in 116.64 s; the targeted
heat/solute suite passed 29 tests and 14 subtests. The legacy solute compatibility
check also produced identical numerical arrays and equal measurements for
six closed/periodic updates against the previous source. All nine NPZ archives
have identical arrays between retained runs v1/v2. These comparisons are saved in
`results/thermal_solute_compatibility_v1.json`.

## Reproduce

Use fresh output paths; scripts refuse to overwrite existing runs or audits.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_thermal.py --output results/thermal_verification_new
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_thermal.py results/thermal_verification_new --output results/thermal_audit_new.json
.venv/bin/python scripts/plot_thermal.py results/thermal_verification_new --output results/thermal_figures_new
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_thermal.py tests/test_solute.py
```

`FlowHeatSoluteSimulation` is a separate component. The legacy illustrative growth
engine, camera models, PID and operator replay are not connected to it. The
earlier CPU/hybrid Vulkan timing study remains an isothermal NS/solute result;
no new GPU performance or speedup claim is made for this coupled implementation.

## Public physical data and remaining work

The [Eisenschmidt thesis repository](https://opendata.uni-halle.de/handle/1981185920/12338)
and its [MPG PDF endpoint](https://pure.mpg.de/pubman/item/item_2636003_6/component/file_3007907/Eisenschmidt-2018_A_Cyclic_Growth-Dissolution_Process.pdf)
were checked as potential KDP process/property sources. The MPG retrieval returned
403. The Halle landing page and bitstream returned HTML bot challenges rather
than a usable PDF. Acquisition records and the response hashes are retained at
`data/external/KDP-EISENSCHMIDT-2018/acquisition_v1.json` and `acquisition_v2.json`.
No challenge was bypassed. Indexed text is a lead, not an accepted transcription
of uninspected tables; no thermal property or growth table from it enters this
experiment or training data.

Next physical gates are compatible, inspected solution properties and expansion
coefficients at declared temperature/composition; measured thermal boundary
conditions and flow forcing; independent coupled-flow evidence; and a separately
specified surface-uptake/growth/latent-heat model. Moving/oblique runtime solids,
solid heat conduction, free-surface heat transfer and interfacial kinetics remain
outside this component. Physical defect/risk interpretation, outcome validation
and a comparable CPU/GPU **coupled** benchmark remain open.

For the AI model, continue the public KDP review pack and seek independent public
evaluation groups with suitable labels, clocks and process/action records. These
manufactured fields must not become accepted real-image labels, real action
response, defect ground truth or evidence for improved yield, quality or energy.
