# Physical units and first flow definition

The physical-model work uses SI internally. The unit choices follow
[BIPM-SI](../SOURCES.md#bipm-si); numeric inputs have source IDs in the root
[source and data register](../SOURCES.md). Display conversions happen at input
and output boundaries. A changed label does not convert a model.

`configs/physical_reference.json` is an input definition for a stationary,
isothermal 3D flow case. `simulation.units` and `describe_physical_setup.py`
already validate that definition and calculate concentration/transport scales.
The descriptor does not run a flow or growth prediction; `check_flow.py` exercises
the new fixed-domain Navier-Stokes component described in `FLOW_MODEL.md`.
`check_solute.py` verifies passive scalar transport and fixed-domain flow coupling
as described in `SOLUTE_MODEL.md`.
The existing `Simulation`
and `configs/default.json` / `focused.json` remain the explicitly illustrative
numerical benchmark; none of their coefficients has been silently reinterpreted.

## Unit contract

| Quantity | Internal unit / meaning |
|---|---|
| Position, plane offset, sampling depth, grid spacing | m |
| Area / volume | m2 / m3 |
| Time, actuator delay, ramp duration | s |
| Temperature / temperature difference | K / K |
| Velocity / angular velocity | m/s / rad/s, with explicit axis and sign |
| Volumetric flow rate | m3/s, with inlet/outlet geometry and sign convention |
| Pressure | Pa; distinguish physical gauge pressure from pressure divided by density |
| Solution / crystal density | kg/m3; these are different properties |
| Dynamic / kinematic viscosity | Pa s / m2/s; nu=mu/rho |
| Solute / thermal diffusivity | m2/s; alpha=k/(rho*cp) when thermal properties are available |
| Thermal conductivity / specific heat capacity | W/(m K) / J/(kg K) |
| Solute mass fraction w | kg KDP/kg solution |
| Literature water-mass ratio q | kg KDP/kg water; q=w/(1-w) |
| Solute concentration C | kg KDP/m3 solution; C=rho_solution*w |
| Molality b | mol KDP/kg water; q=b*M_KDP |
| Normal growth speed / interfacial solute mass flux | m/s / kg/(m2 s) |
| Relative supersaturation sigma | fraction on a stated basis; percent=100*sigma |

`T_K=T_C+273.15`; the published solubility polynomial must receive Celsius.
`omega=2*pi*rpm/60`; `Q[m3/s]=Q[L/min]*1e-3/60`. Negative angular velocity
represents the opposite rotation. Axis orientation is separate from speed.

For a binary solution of volume V, `m_solute=C*V` and
`m_water=(rho_solution-C)*V`. This gives `q=C/(rho_solution-C)`, not
`C/rho_solution`. The old engine's concentration is per carrier volume, so a
direct assignment of these new solution-volume concentrations would be wrong.

For comparison with KDP-2025 use `sigma=(q-q_eq(T))/q_eq(T)`. Its mass-ratio
definition must not be replaced by `C/C_eq-1` after a nonlinear basis change.
A constant-density transport model can evolve C or w, then convert back to q
for the source's supersaturation definition.

## Sourced physical reference point

[VTT-2006](../SOURCES.md#vtt-2006), printed page 17 / PDF page 19, supplies a
binary KDP-water CFD property point at 30 C and molality 2.2 mol/kg water:
rho=1170 kg/m3, mu=0.0015 Pa s, D=5.7e-10 m2/s. Density/viscosity were calculated
and diffusivity estimated; numerical uncertainties are not provided. These are
reference inputs, not a validated property curve or independent measurements.
They are restricted here to the reported temperature and composition.

The converted q is 0.2993881 kg/kg, w is about 0.230407 and C about 269.576 kg/m3.
The solubility relation recorded in KDP-2025 gives an exploratory cross-source
supersaturation estimate of about 0.09662 at 30 C. That is a derived estimate,
not a VTT-reported measurement or a calibration result. Displayed extra digits
do not imply matching experimental precision.

No pure-water viscosity, heat capacity or density is silently substituted for
this concentrated solution. Heat capacity/conductivity, thermal/solutal density
derivatives, solid density, lattice metric and facet kinetics remain separate
inputs to obtain and audit before enabling the corresponding physics.

## First stationary open-tank case

DESIGN-REST-001 deliberately chooses a 10 mm cube tank, an 8 mm liquid level and
a submerged 1 mm cube obstacle centered at (5,5,4) mm. This is a verification
geometry, not a reconstruction of VTT's tube or the 2025 experimental cell.
The cube tests flow exclusion; it is not a physical KDP crystallographic habit.
Gravity (0,0,-9.81) m/s2 is a chosen terrestrial approximation, not a local
measurement. Duration 120 s is a benchmark choice.

The first case is at rest: fixed crystal, zero rotation/pump flow, uniform
temperature and composition, no growth or dissolution. Define:

- Stationary tank walls and crystal: no-slip, impermeable boundaries.
- Solute: zero normal flux on every boundary, including the fixed crystal.
- Top: flat, impermeable surface with zero tangential stress. This is a
  stationary free-surface approximation; evaporation and waves are disabled.
- Pressure: use a dynamic pressure in Pa relative to the hydrostatic rest
  profile, `p_h=rho*g_abs*(H-z)`. In the fixed impermeable-lid approximation,
  choose a fluid-volume mean-zero gauge. This is not atmospheric pressure fixed
  independently at every top cell. The pressure condition at impermeable walls
  follows the normal momentum/projection equation. A deformable free surface
  needs a separate normal-stress/kinematic boundary treatment.
- Temperature: a prescribed uniform 303.15 K for this isothermal benchmark;
  there is no claim of a solved heat equation or a calibrated bath model.

The expected rest solution has zero velocity, constant composition and balanced
hydrostatic pressure. This target is checked by the fixed-domain flow benchmark.
The rising-level growth condition remains the intended later open-tank model.

Inspect the definition without producing results files:

```bash
.venv/bin/python scripts/describe_physical_setup.py
.venv/bin/python scripts/describe_physical_setup.py --speed-scale 0.001
```

The optional speed is a diagnostic scale, not an applied actuator. For a 10 mm
length and 1 mm/s scale, Re=7.8, solutal Pe about 17544 and Sc about 2249. A
velocity scale does not establish the actual flow regime of a future apparatus.
High Pe also means near-surface solute gradients can require substantially finer
resolution than velocity alone. The code reports diffusion time as L2/D.

## Flow and growth integration sequence

1. Implement a separate tested flow component on the 3D grid. A staggered layout
   is the initial design: cell pressure `[nz,ny,nx]`, x velocity `[nz,ny,nx+1]`,
   y velocity `[nz,ny+1,nx]`, z velocity `[nz+1,ny,nx]`. Aperture-weighted face
   fluxes must satisfy incompressibility on actual liquid volumes.
2. Verify the rest state and a controlled 3D channel/shear-flow reference.
   Check pressure/velocity errors, divergence, blocked-wall leakage and grid/time
   sensitivity, including unequal cells. Audit geometric tolerances and time
   control in physical scales before connecting the current geometry engine.

   The uniform-grid fixed-box component now passes rest/channel, viscous-decay
   and nonlinear 3D ABC references; details are in `FLOW_MODEL.md`. Unequal-cell
   momentum and general cut faces remain follow-up work.
3. Advect solute conservatively with the same volume fluxes; check constant-field
   preservation, positivity and total solute mass. Add the heat equation only
   with sourced thermal properties and actual thermal boundary conditions.

   Fixed-grid passive solute coupling is now implemented, with 3D cell-average
   wave/diffusion references, a high-Peclet pulse broadening check and coupled
   fixed-cube tests. Heat and concentration-dependent properties remain pending.
4. Add a defined forcing case: rotation requires axis, angular velocity and ramp
   history; pumps require physical ports, flow rates and matching outlet/balance
   conditions. The illustrative `rotation`/`circulation` numbers are not rpm or
   m3/s. Remove or recalibrate empirical agitation boosts in resolved-flow mode
   to avoid counting the same transport twice.
5. Assess thermal and solutal buoyancy using sourced density derivatives. A
   Boussinesq model requires small relative density changes; define the solutal
   derivative against w (or explicitly convert it from another basis). Do not
   infer that the isothermal reference proves buoyancy can always be omitted.
6. Couple facet kinetics to interfacial solute flux and boundary motion. Preserve
   solute, solvent and volume consistently with solution/solid densities. At
   w about 0.23 the old negligible-solute-volume assumption is not established:
   conserving its carrier volume is not evidence for physical solution-volume
   conservation. Revisit the rising level with a density/partial-volume closure.
7. Repeat coupled refinement and compare with independent growth/flow experiments.
   Profile cost before synthetic batches. Geometry/transport fidelity does not
   supply optical quality labels; the quality target needs independent data.

The physical definition, conversions, initial uniform-grid momentum component and
conservative passive solute coupling are implemented. Arbitrary momentum cut
cells, thermal transport/buoyancy, growing-interface integration, kinetic
calibration and quality validation remain subsequent work, with
source/uncertainty tracking required throughout.
