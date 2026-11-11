# CrystaLab sources and data register

Last full source audit: 2026-11-08. Subsequent vision acquisition and control-method
updates are dated 2026-11-11; their artifacts are in the linked guides/checkpoints.
This file records the websites and numeric evidence used
or cited in the project. It separates imported values, method references,
project choices and sources awaiting audit. A search result alone is not an
experimental dataset. No imported record establishes calibrated optical quality.

Machine-readable rate records are in `configs/benchmarks/`; the physical input
definition is `configs/physical_reference.json`. Source IDs below must accompany
new physical parameters and future datasets. Update this register and the
corresponding JSON together when values, units or interpretations change.

## Source catalogue

| ID | Website / primary source | Project use and access status |
|---|---|---|
| BIPM-SI | [SI base units](https://www.bipm.org/en/measurement-units/si-base-units), [SI Brochure](https://www.bipm.org/en/publications/si-brochure) | SI convention; official base-unit page inspected. |
| NIST-KDP | [NIST Chemistry WebBook: KDP](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7778770&Mask=80) | Molecular weight for molality-to-mass-ratio conversion; official entry inspected. |
| COD-9010000 | [COD entry](https://www.crystallography.net/cod/9010000.html), [original paper DOI](https://doi.org/10.1088/0022-3719/15/1/005) | Hash-pinned local CIF, revision 291792, 295 K cell/density reference. Measurement pressure missing; original paper access blocked. |
| CIF-CORE | [IUCr density definition](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Iexptl_crystal.density_diffrn.html), [pressure definition](https://www.iucr.org/__data/iucr/cifdic_html/3/CORE_DIC/Icell_measurement.pressure.html) | Official field meanings/units: calculated density in Mg/m3 and cell measurement pressure in kPa, distinct from synthesis pressure. |
| KDP-2026 | [Li et al., primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919635/), [publisher](https://pubs.rsc.org/en/content/articlelanding/2026/ra/d5ra09916a) | Later acquisition retrieved the supplement and three phase-specific microscopy movies through official PMC cloud storage; decoded and hash-pinned. No reviewed event/mask or actuator-response labels. See [acquired vision data](docs/REAL_VISION_DATA.md). |
| LGA-LEEDS-2026 | [University of Leeds dataset record](https://archive.researchdata.leeds.ac.uk/1537/) | Later direct acquisition/import verified 2,123 microscope images from nine alpha-LGA runs and 79 source shape-fit sidecars. Reviewed masks remain unavailable. See [acquired vision data](docs/REAL_VISION_DATA.md). |
| ADP-NATURE-2024 | [Cho et al., primary article](https://www.nature.com/articles/s41467-024-47503-1) | One acquired/decoded supplementary movie explicitly depicts ADP in a levitated droplet, not KDP tank growth. Caption/license/hash evidence is retained in the [vision registry](configs/vision_sources.json). |
| CTRL-ANTIWINDUP | [MathWorks PID anti-windup documentation](https://www.mathworks.com/help/simulink/slref/anti-windup-control-using-a-pid-controller.html) | Primary method reference inspected 2026-11-11 for saturation/back-calculation/tracking concepts. Project Python implementation and test gains are independent; no physical parameters adopted. |
| CTRL-DERIVATIVE | [NI PID algorithm documentation](https://www.ni.com/docs/en-US/bundle/labview/page/implementing-the-pid-algorithm-with-the-pid-vis.html) | Primary method reference inspected 2026-11-11 for derivative on measurement. Our filter/discretization/units are declared separately; no vendor code, gains or equipment limits adopted. |
| MFA-STRATH-2022 | [University of Strathclyde dataset](https://pureportal.strath.ac.uk/en/datasets/data-for-direct-image-feature-extraction-and-multivariate-analysi/) | Other-material process/image CSV candidate; public file requests returned 403. No numeric observations imported. |
| KDP-IEEE-2024 | [IEEE DataPort listing](https://ieee-dataport.org/documents/images-micro-craters-and-micro-cracks-surface-kdp-crystals) | Surface-defect images, subscription required and no files listed; not a growth-control dataset. |
| KDP-2016 | [Wu et al., accepted manuscript](https://eprints.whiterose.ac.uk/105693/3/20160707%20Crystal%20Face%20Specific%20Growth%20Kinetics%20FINAL%20x-x.pdf), [repository record](https://eprints.whiterose.ac.uk/id/eprint/105693/), [DOI](https://doi.org/10.1021/acs.cgd.6b00189) | Two published face-distance fit summaries; manuscript inspected. |
| KDP-2025 | [Maksimovic et al., PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800150/), [ACS full text](https://pubs.acs.org/doi/10.1021/acsomega.4c09287), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39926492/) | Tables 1/2 and solubility relation transcribed from indexed primary full text; direct page access was intermittently blocked. |
| VTT-2006 | [Manninen (ed.), VTT Research Notes 2340](https://publications.vtt.fi/pdf/tiedotteet/2006/T2340.pdf) | Binary-solution CFD property point; original institutional PDF downloaded, text checked and Table 2.1 visually inspected. |
| CAL-NIST | [Calibration and uncertainty analysis](https://www.nist.gov/publications/calibration-and-uncertainty-analysis-predictions-computational-models) | Calibration/model-inadequacy distinction; methodological reference, not KDP data. |
| NUM-FV | [NIST FiPy discretization](https://pages.nist.gov/fipy/en/stable/numerical/discret.html) | Finite-volume background, not a source of physical coefficients. |
| NUM-GRAD | [OpenFOAM gradient schemes](https://doc.openfoam.com/2312/tools/processing/numerics/schemes/gradient/) | Least-squares/limited-gradient background. Our implementation and analytic verification are project code. |
| NUM-LIMIT | [OpenFOAM limited normal-gradient correction](https://doc.openfoam.com/2312/tools/processing/numerics/schemes/sngrad/rtm/limited/) | Background considered for bounded corrections; indexed content only, direct access blocked. Not an assertion that our limiter is OpenFOAM's implementation. |
| NUM-SOLVE | [SciPy sparse direct solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html), [CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html), [GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) | Sparse solver APIs; no experimental data. Record package versions with each study. |
| NUM-VERIFY | [NASA verification assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html) | Verification/convergence background cited in the diagnostic documentation. |
| SOLUTE-SSP-1998 | [Gottlieb and Shu, primary paper](https://www.cfm.brown.edu/people/sg/SSP1.pdf) | Two-stage SSP RK2, equation 2.4/3.2; institutional PDF inspected 2026-11-08. No physical parameters imported. |
| SOLUTE-FCT-2015 | [Chaplin and Colella, primary preprint](https://arxiv.org/html/1506.02999) | Section 4/4.1 correction budgets, equations 43-53; full text inspected 2026-11-08. Our lower-order, stagewise scheme is distinct. |
| FLOW-NONUNIFORM-2002 | [Ham, Lien and Strong, primary publisher record](https://doi.org/10.1006/jcph.2002.7006) | Staggered-grid nonuniform conservation background; indexed primary abstract inspected 2026-11-08, full publisher access blocked. No material constants or numerical results imported. |
| CUTCELL-FV-2004 | [Schwartz, Barad, Colella and Ligocki, primary institutional report](https://escholarship.org/content/qt0fp606kk/qt0fp606kk.pdf) | Geometry moments, volume-integrated Poisson form and nominal-center unknowns; full primary PDF inspected 2026-11-08, printed pages 2-4. Our two-point projection does not claim its second-order boundary method. |
| FLOW-2000 | [Natural/forced solutal convection above a growing crystal](https://www.sciencedirect.com/science/article/abs/pii/S0022024899007459) | KDP Navier-Stokes/Boussinesq background; indexed abstract inspected, full page blocked. No constants imported. |
| FLOW-2001 | [Transient three-dimensional KDP flow study](https://www.sciencedirect.com/science/article/abs/pii/S0022024800008460) | 3D hydrodynamics background; indexed abstract only. No numerical dataset imported. |
| FLOW-2005 | [Schlieren study of ramp rate and rotation](https://www.sciencedirect.com/science/article/abs/pii/S0022024804013612) | Candidate concentration-pattern validation; abstract inspected, observations not imported. |
| FLOW-2006 | [CFD modelling of single KDP crystal growth at 30 C](https://www.sciencedirect.com/science/article/pii/S002202480501095X) | Diffusion/surface-reaction coupling background; indexed abstract/sections only, DOI 10.1016/j.jcrysgro.2005.09.044. |
| KINETICS-2015 | [Face-specific batch growth/dissolution kinetics](https://doi.org/10.1021/cg501251e) | Calibration candidate; abstract inspected. Full thesis access was blocked; no kinetic numbers imported. |
| QUALITY-2019 | [KDP surface-height variation](https://doi.org/10.1021/acs.cgd.8b01543) | Growth/impurity/quality modelling background; abstract only. No quality calibration or numeric labels imported. |
| PH-2002 | [pH-dependent prismatic/pyramidal kinetics](https://www.sciencedirect.com/science/article/pii/S0022024802016585) | Earlier roadmap candidate, DOI 10.1016/S0022-0248(02)01658-5. Numeric data not audited/imported. |

Relevant background also consulted, with **no values adopted**: [2011 3D KDP
flow/mass-transfer study](https://www.sciencedirect.com/science/article/pii/S0022024810004379),
[2024 cuboid KDP flow study](https://www.sciencedirect.com/science/article/abs/pii/S0022024824000630),
[2026 confined-growth study](https://www.sciencedirect.com/science/article/pii/S0022024826000084),
[2016 extreme-supersaturation droplet study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5137746/),
and [1995 concentration-dependent solution properties](https://ntrs.nasa.gov/api/citations/19970021347/downloads/19970021347.pdf).
The droplet diffusion estimates are outside our selected reference regime and
were not substituted into the 30 C profile. ResearchGate/search mirrors were
discovery routes; retained physical evidence comes from primary/institutional
sources. Failed Europe PMC/PMC requests supplied no extra data.

## BIPM-SI

We choose m, s, kg and K internally, with derived Pa, Pa s, m2/s, m/s and m3/s.
This is a project convention based on [BIPM's SI definitions](https://www.bipm.org/en/measurement-units/si-base-units).
See `docs/PHYSICAL_MODEL.md` for the complete quantity/basis contract. Exact
unit conversions include K=C+273.15, 1 nm=1e-9 m, 1 um=1e-6 m,
1 L=1e-3 m3, 1 hour=3600 s and 1 day=86400 s.

## NIST-KDP

The [NIST entry](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7778770&Mask=80)
identifies KH2PO4, CAS 7778-77-0, with molecular weight 136.0855. The working
molar mass is 0.1360855 kg/mol. No numerical uncertainty is given in that entry;
the stored precision must not be interpreted as a calibrated uncertainty.

## KDP-2016

Source: [Wu et al. (2016)](https://doi.org/10.1021/acs.cgd.6b00189), accepted
manuscript pages 21-22, equations 10A/B and 11A/B. Values are published linear
fit summaries for center-to-face normal distance, not raw frame measurements.

| Face | Intercept (um) | Normal rate (um/h) | Fit R2 |
|---|---:|---:|---:|
| {100} prismatic | 296.71 | 13.56 | 0.9314 |
| {101} pyramidal | 418.54 | 118.03 | 0.9968 |

Reported context: about 290 K, 120 mL, 82 min, 21 stereo pairs and occasional
manual agitation. Methods concentration is about 0.247 g/mL; the figure caption
says 24.7 g/mL. The mass basis, supersaturation, rate uncertainties and action
history are unresolved. Temperature/concentration were not strictly controlled.
Quality labels are unavailable. R2 is not measurement uncertainty; points
generated from these lines would be synthetic. The full audited record is
`configs/benchmarks/kdp_wu2016.json`.

## KDP-2025

Source: [Maksimovic et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800150/),
DOI 10.1021/acsomega.4c09287. Tables below retain the reported statistics and
plus/minus terms, whose confidence/standard-uncertainty definitions are not
verified here. N counts face rates, not independent experiments; repeated faces
share growth history. Machine record: `configs/benchmarks/kdp_maksimovic2025.json`.

<!-- Published tables are transcribed below from the machine-readable record. -->

Table 1: most probable {100} rates from fitted distributions.

| T (C) | sigma (%) | N decreasing | Rate decreasing (nm/s) | N increasing | Rate increasing (nm/s) |
|---|---:|---:|---:|---:|---:|
| 24.0 | 14.7 | 64 | 39 +/- 1 | 73 | 30 +/- 0.7 |
| 25.0 | 12.6 | 64 | 26.6 +/- 0.8 | 77 | 23.1 +/- 0.8 |
| 26.0 | 10.4 | 64 | 19.1 +/- 0.7 | 80 | 15.5 +/- 0.5 |
| 27.0 | 8.3 | 60 | 9.6 +/- 0.5 | 80 | 7.8 +/- 0.4 |
| 28.0 | 6.2 | 55 | 2.8 +/- 0.3 | 80 | 2.2 +/- 0.3 |

Decreasing/increasing refer to supersaturation; temperatures run 24 to 28 C /
28 to 24 C respectively. Table 2 contains **means** from the paper's test
experiments at 28 C, not additional copies of the Table 1 modes.

| Saturation T (C) | Growth T (C) | sigma (%) | N | Mean rate (nm/s) |
|---|---:|---:|---:|---:|
| 34.8 | 28.0 | 14.7 | 139 | 38.8 +/- 1.3 |
| 33.9 | 28.0 | 12.6 | 58 | 24.7 +/- 1.2 |
| 32.9 | 28.0 | 10.4 | 66 | 18.3 +/- 0.9 |
| 32.0 | 28.0 | 8.3 | 54 | 9.2 +/- 0.9 |

Experimental context recorded: 31.0 +/- 0.1 C saturation temperature,
growth-temperature stability +/- 0.02 C, nominal cell capacity about 15 mL,
1 C transitions within 5 min, about 15 min stabilization and about +/-5 um
face-displacement accuracy, following dissolution/refaceting. These are not
complete individual experiment trajectories or optical-quality labels.

The experimental section uses `q_eq=0.17554+0.00102*T_C+0.0000743*T_C^2` in
kg KDP/kg water, with `sigma=(q-q_eq)/q_eq`. The source attributes solubility
data to [Mullin and Amatavivadhana (1967)](https://doi.org/10.1002/jctb.5010170508);
we have not independently audited that original dataset. The helper is restricted
by project choice to 24--34.8 C, covering reported growth/saturation temperatures,
without claiming a universal original validity range or coefficient uncertainty.

## VTT-2006

Source: [Modelling of multiphase chemical reactors (ModCheR), final report](https://publications.vtt.fi/pdf/tiedotteet/2006/T2340.pdf),
M. Manninen (ed.), VTT Research Notes 2340 (2006), section 2.2.3, printed page
17 / PDF page 19, Table 2.1 and its preceding viscosity paragraph.

| Binary KDP-water input | Published value | SI value used |
|---|---:|---:|
| Temperature | 30 C | 303.15 K |
| Average molality | 2.2 mol/kg water | 2.2 mol/kg water |
| Density | 1170 kg/m3 | 1170 kg/m3 |
| Dynamic viscosity | 1.5e-3 Pa s | 0.0015 Pa s |
| Binary diffusivity | 5.7e-6 cm2/s | 5.7e-10 m2/s |

These are the report's **CFD inputs**: density/viscosity calculated from cited
relations and diffusivity estimated from cited data. Uncertainty is unspecified.
No alcohol-mixture values are adopted. Underlying Wang et al. (2000) and
Bohenek et al. (1997) measurements have not been independently audited.
This is a single reference point, not an assumed property law over temperature
or concentration. Used in `configs/physical_reference.json`.

Institutional PDF SHA256:
`cd077bf21b86bbad50dcc5da15bc31ad313066bb75568e6490a7b7fc051aecfb`.
The relevant original page was rendered and visually checked. The PDF is not
bundled in the repository; use the institutional link to retrieve it.

## DESIGN-REST-001

Project choices for the first stationary isothermal flow verification case,
not experimental observations: tank (0.01,0.01,0.01) m, liquid height 0.008 m,
fixed cube (0.001,0.001,0.001) m centered at (0.005,0.005,0.004) m, duration
120 s, gravity (0,0,-9.81) m/s2 as an approximate terrestrial benchmark setting.
Walls/crystal are stationary no-slip, the flat top is impermeable/stress-free,
and solute flux/rotation/pump flow are zero. Pressure is relative to the
hydrostatic rest profile. Actual laboratory geometry and actuators remain unknown.

From VTT-2006 + NIST-KDP, arithmetic gives q=0.2993881 kg/kg,
w about 0.230407, C about 269.576 kg/m3 and nu about 1.28205e-6 m2/s.
Combining that q with KDP-2025 gives sigma about 0.09662: a **cross-source
estimate**, not a reported VTT supersaturation. Extra digits are computational.

For an optional **diagnostic** speed of 0.001 m/s and length 0.01 m, Re=7.8,
Pe about 17544, Sc about 2249 and L2/D about 175439 s. These are derived scales,
not measured velocities or a solved flow. The configured rest case has zero
forcing. The diagnostic speed does not actuate it.

## NUMERICAL-CHECKPOINT

Internal numerical evidence is separate from physical measurements.
`docs/NUMERICAL_CHECKPOINT.json` preserves the final focused-study protocol,
fully resolved inputs, environment, source hashes, case summaries and gates.
All old generated files in `results/` were removed at the user's request on
2026-11-07, including plots, fields, raw histories and archived source copies.
At cleanup the directory retained only `.gitkeep`; new flow verification runs
produce fresh artifacts. The compact record is not a resumable
raw dataset. New runs must use fresh output directories.

| Final focused benchmark | Net growth / trajectory difference | Face-advance difference |
|---|---:|---:|
| Spatial 32^3 to 40^3, dt=1 | 0.085985% | 0.097916% |
| Temporal dt=0.5 to 0.25, 40^3 | 0.156915% | 0.121856% |

All five 120-model-unit cases passed the predeclared 2% gate. Maximum relative
mass error 9.149e-13, carrier error 1.185e-15, continuity/minimum-cell-volume
error 1.857e-10 (gate 1e-8). These values do not establish physical KDP accuracy.
Core corrections and documentation were committed as 5eea924 / 5449af3, and
published benchmark records as 64210bd. The study manifest identifies its earlier
base commit plus the actual measured source hashes.

Earlier failed refinement stages and diagnostic values remain described in
`docs/CONVERGENCE.md` and `docs/SPATIAL_ACCURACY.md`. Their old results paths are
historical labels, not files retained after cleanup. Do not claim those raw
observations are still available or substitute them for the final-source study.

## DESIGN-LEGACY-001 and local legacy data

The local files described below are historical obsolete outputs and are no
longer present. Their hashes identify those old files, not active inputs.

The original `SimulationConfig` and `configs/default.json` / `focused.json`
contain project-chosen illustrative numbers, not measured KDP constants. Examples
are tank sizes 10x8x8 model-length units, density 8 model-mass/model-volume units,
concentration 1.35, growth coefficients 0.025/0.015, solute diffusivity 0.02,
thermal diffusivity 0.04 and dimensionless agitation settings 0.5/0.5. The exact
resolved settings are in the retained numerical checkpoint. These values must
not acquire SI labels by relabeling fields.

The local ignored `data/transitions.jsonl` and `data/transitions.manifest.json`
are an obsolete synthetic dataset from `polygon-local-quality-v1`, seed 19:
684 rows, 36 episodes (22 train / 7 validation / 7 test). They are not real
measurements, a current 3D dataset or calibrated quality labels. They were outside
the requested `results/` cleanup and are not bundled in Git. They are excluded
from physical-model calibration and future 3D training data.

- Manifest SHA256: `057bfd059a998a01302089e309c20762c4425c7fcf84cd8707a4e33fbcc174ff`.
- JSONL SHA256: `5d4e1f9bc726fe2e110f7add1db075795174347f570b146c6ea10e1e50fa4cf3`.

## FLOW-PROJECTION-1968

[Chorin, Numerical Solution of the Navier-Stokes Equations (1968)](https://math.berkeley.edu/~chorin/chorin68.pdf),
[DOI](https://doi.org/10.1090/S0025-5718-1968-0242392-2), supplies projection-method
background. The publisher abstract and scanned institutional paper were consulted
on 2026-11-07. Our staggered-grid discretization, boundary closures and code are
project implementations, not a transcription of the paper. No physical constants
are taken from it.

## FLOW-BELTRAMI-1996

[Switzer, NASA CR-4756, Validation Tests of TASS for Application to 3-D Vortex
Simulations (1996)](https://ntrs.nasa.gov/api/citations/19970001365/downloads/19970001365.pdf)
documents analytic viscously decaying Beltrami flow as a test of nonlinear 3D
Navier-Stokes solvers. Primary report text was inspected on 2026-11-07. Our
periodic ABC benchmark is derived from the Beltrami identity; it does not reproduce
the TASS discretization or claim the NASA report's numerical results.

## DESIGN-FLOW-002

New flow verification choices: uniform orthogonal 3D grids, constant density and
viscosity from VTT-2006, fixed geometry, isothermal fluid. Periodic channel and ABC
fields are manufactured/reference flows, not laboratory KDP experiments. Grid
sizes, speeds, acceleration, viscosity-decay times and tolerances must be recorded
with each generated verification report. Pressure correction uses impermeable
normal-velocity boundaries and a gauge; a rigid, shear-free top must not also
clamp pressure independently at every top cell in a driven flow. Arbitrary cut
cells and moving/growing interfaces are outside this first momentum solver.

## FLOW-BOUNDARIES-2001

[Brown, Cortez and Minion, Accurate Projection Methods (2001)](https://www.math.tulane.edu/~cortez/Prints/accproj.html),
[DOI](https://doi.org/10.1006/jcph.2001.6715), is consulted for the relationship
between pressure updates and numerical boundary conditions. Our initial solver
uses first-order time stepping, incremental pressure correction and limited
linear upwind momentum fluxes; it does
not claim the paper's second-order scheme or pressure accuracy. Boundary and
time errors are measured separately in project benchmarks.


## FLOW-NUMERICAL-20261107

New numerical evidence, not laboratory data. The declared inputs/tolerances are
in `configs/flow_verification.json`; physical coefficients come from VTT-2006.
The final source-matched study is `results/flow_verification_final_5505e9d/`,
with resolved settings/environment, all source copies, final native fields,
report and figures. All eleven cases pass; the complete suite passes 122 tests.

| Case | Relative velocity/RMS error (%) | Relative pressure error (%) | Scaled divergence |
|---|---:|---:|---:|
| rest | n/a | n/a | 0.000e+00 |
| channel_8 | 0.534881 | n/a | 1.847e-16 |
| channel_16 | 0.133721 | n/a | 1.720e-16 |
| channel_32 | 0.033430 | n/a | 1.945e-16 |
| decay_0.5 | 0.607019 | n/a | 1.723e-16 |
| decay_0.25 | 0.307717 | n/a | 1.709e-16 |
| decay_0.125 | 0.156570 | n/a | 1.771e-16 |
| abc_12 | 0.591286 | 2.839061 | 2.245e-16 |
| abc_18 | 0.306224 | 1.607187 | 2.630e-16 |
| abc_24 | 0.201550 | 1.407454 | 3.075e-16 |
| obstacle | n/a | n/a | 1.768e-16 |

All blocked-face velocities are zero. Rest runs to 120 s. The manufactured
obstacle flow runs to 2 s with acceleration scale 1e-4 m/s2, reaching centered
speed 3.879061e-5 m/s. These are computed outcomes under DESIGN-FLOW-002, not
measured circulation or calibrated quality labels. Neither focused/oblique
momentum cut cells nor growing-crystal coupling is established by these tests.

The donor-only study `results/flow_verification_5505e9d/` failed its original
ABC velocity-refinement monotonicity gate. Its exact five source files were
preserved and verified against the original recorded hashes. The limited-flux
follow-up kept the same protocol/tolerances. An intermediate figure-scale bug
in `flow_verification_limited_5505e9d` was corrected in the final runner; raw
numerical fields were unaffected. Both histories remain marked, not relabeled.

Final measured source SHA256 values:

- `src/simulation/flow.py`: `5b2bf56a05f9236eafe309768450f3185abd6ac4003bb808bda9ad47d315d64d`.
- `src/simulation/units.py`: `fc6147019552b30f482292c7b7bed80cad3ea04c0c50d8c878f849533e8a5bee`.
- `src/simulation/crystal.py`: `7a58827d41d1d4c6c629294ec955f049a426c781f1c7f487dada797db387aba3`.
- `scripts/check_flow.py`: `6ff1e9b5fd46f6c01d5c981391695d13fc0e0b91d706165ba35dd7fce2f4b68d`.
- `scripts/describe_physical_setup.py`: `c0bd7e2e22b24ecbda70b0f9a1d5bb7a5ace0c6f74868ac12f4e40e02ff0dece`.

## SOLUTE-SSP-1998

[Gottlieb and Shu, Total Variation Diminishing Runge-Kutta Schemes (1998)](https://www.cfm.brown.edu/people/sg/SSP1.pdf),
Mathematics of Computation 67, 73-85,
[DOI](https://doi.org/10.1090/S0025-5718-98-00913-2).
The institutional primary PDF was inspected on 2026-11-08: equation 2.4 on printed
page 76 and equation 3.2 on page 78 supply the two-stage convex-combination time
integrator. The scalar implementation combines two bounded Euler stages.
Limiter/boundary behavior and first-order momentum coupling prevent a global
second-order accuracy claim. No material coefficients come from this paper.

## SOLUTE-FCT-2015

[Chaplin and Colella, A Single Stage Flux-Corrected Transport Algorithm for
High-Order Finite-Volume Methods (2015)](https://arxiv.org/html/1506.02999),
arXiv:1506.02999. Primary full text inspected on 2026-11-08, especially section 4
and 4.1, equations 43-53. This supplies the framework of a conservative low-order
base plus paired, budget-limited flux corrections. Our donor/diffusion base,
MC reconstruction, immediate-neighbor bounds and RK2 stagewise limiting are
project choices; the cited paper instead develops higher-order fluxes, CTU,
RK4 and smooth-extremum treatments that we do not claim to implement. No physical
constants or experimental observations are imported.

## DESIGN-SOLUTE-003

`configs/solute_verification.json` declares fifteen fixed-grid numerical cases
and their acceptance limits. Use the VTT-2006 reference D with frozen density
and viscosity; concentration perturbations are mathematical verification fields,
not evidence that these properties are constant over that composition range.
Lengths, durations, velocity amplitudes, concentration waves/pulses, grids,
timesteps and tolerances are project verification choices.

Three-dimensional periodic wave and no-flux diffusion references use integrated
cell means. Temporal diffusion uses a semidiscrete eigenmode to isolate time
error. The periodic Gaussian pulse has Pe=17543.86 at 1 mm/s over 10 mm, and
explicitly measures numerical broadening using the difference of numerical and
exact center-weighted variances divided by twice elapsed time. This diagnostic
is not a universal numerical diffusivity or a KDP boundary-layer resolution
certificate. Fixed-cube forcing is manufactured, without calibrated pumps,
surface uptake, heat, buoyancy, growth or quality observations.

## SOLUTE-NUMERICAL-20261108

The accepted study `results/solute_verification_final_5a015f5/` passes all fifteen
cases and every original gate in DESIGN-SOLUTE-003. The retained
`docs/SOLUTE_CHECKPOINT.json` includes the exact protocol/reference, environment,
six source SHA256 values, complete per-case metrics, gates and the momentum
recheck identity. Generated raw fields/figures and source archives remain local,
ignored artifacts; their hashes were checked against the current sources and
all arrays checked finite. All three exported figures were inspected.

Finest relative fluctuation RMS errors are 1.59195% (periodic 3D wave), 0.05437%
(Neumann 3D diffusion), 0.88601% (pulse), and 0.00006836% (semidiscrete temporal
reference). Pulse excess-diffusivity diagnostics across nx=64/128/256 are
1.860245e-9 / 6.982865e-11 / 6.491146e-12 m2/s, or 3.26359 / 0.12251 / 0.01139
times physical D. Maximum relative solute-mass drift is 6.66134e-16; all measured
concentration-bound violations and coupled uniform-concentration errors are zero.
These are numerical results for the stated fields, not new physical constants.

The eleven-case momentum rerun `results/flow_verification_solute_5a015f5/` also
passes, with source identity checked. The full suite passes 138 tests. Fixed
uniform geometry and constant properties remain explicit; focused/sloped momentum
geometry, growing-interface coupling, kinetic calibration and experimental
validation are still pending.

## FLOW-NONUNIFORM-2002

[Ham, Lien and Strong, A Fully Conservative Second-Order Finite Difference Scheme
for Incompressible Flow on Nonuniform Grids (2002)](https://doi.org/10.1006/jcph.2002.7006),
Journal of Computational Physics 177, 117-133. Primary publisher metadata/abstract
was inspected through the index on 2026-11-08; direct full text was blocked.
It motivates keeping nonuniform staggered operators mutually consistent. Our
actual primal/dual finite-volume measures, symmetric stiffness and bounded
upwind flux implementation are project derivations checked independently. We
do not reproduce or claim its nondissipative second-order method. No physical
constants, experimental observations or reported numerical results are imported.

## DESIGN-FLOW-RECT-004

`configs/rectilinear_verification.json` declares twenty unequal/focused-grid
numerical cases. Smooth stretching is
`edge/L = s + amplitude*sin(2*pi*s)/(2*pi)`, with fixed x/y/z amplitudes
0.4 / -0.25 / 0.3. Focused nonlinear flow uses the existing 3.6 mm central-band
generator, now shared by the SI flow and illustrative growth tank. The aligned
fixed cube has four cells across its 1 mm width and coarser far-field cells.
These are design/verification choices, not an experimental KDP apparatus.

Channel/decay and nonlinear ABC fields use the existing analytical references;
3D scalar wave/diffusion references integrate actual cell means. Error norms,
pressure gauges, energy and solute mass use physical volumes. Density, viscosity
and solute diffusivity retain VTT-2006 provenance and the same property limitations.
No thermal, buoyancy, interfacial growth or quality mechanism is added.

The first study `results/rectilinear_verification_first_8011b79/` failed the
diffusion gate: relative fluctuation RMS 0.156298% at 16 cells per axis exceeds
0.15%. Its other gates passed. Its protocol, sources and failed result are
retained. The next protocol increases diffusion levels from 6/10/16 to 8/12/20,
retaining the same fields, properties, duration and 0.15% threshold. Resolution
is increased; the acceptance tolerance is not relaxed.

## FLOW-RECT-NUMERICAL-20261108

`results/rectilinear_verification_final_v2_8011b79/` passes all twenty cases of
DESIGN-FLOW-RECT-004 at the original acceptance thresholds. The diffusion grid
increase is described above; the failed first study remains retained. A later
plot-only fix replaces crowded logarithmic tick labels with declared grid/time
levels; numerical results are unchanged in the final source-matched rerun.

Finest volume-weighted relative RMS errors: channel 0.04044%, viscous time decay
0.15730%, stretched ABC velocity/pressure 0.32612%/1.12922%, focused ABC
velocity/pressure 0.81133%/2.47371%, 3D scalar wave 2.20441%, Neumann diffusion
0.10067%. All refinement gates pass. Maximum normalized divergence is 5.155e-16;
blocked-face velocity, measured solute-mass drift, scalar-bound violations and
coupled uniform-concentration errors are zero in this study.

`docs/RECTILINEAR_CHECKPOINT.json` preserves protocol/reference inputs, environment,
nine source hashes, measured metrics, failure history and regression identities.
Source archives match their hashes/current measured sources, raw fields are
finite and mesh volumes match their edge products. The final figures were
inspected. Separate current-source regressions in
`results/flow_rectilinear_regression_8011b79/` (eleven cases) and
`results/solute_rectilinear_regression_8011b79/` (fifteen cases) pass. The full suite
passes 151 tests. This establishes the fixed rectilinear/aligned-solid numerical
milestone; oblique momentum boundaries, growing-interface integration, physical
calibration and AI growth-control readiness remain pending.

## CUTCELL-FV-2004

[Schwartz, Barad, Colella and Ligocki, A Cartesian Grid Embedded Boundary Method
for the Heat Equation and Poisson's Equation in Three Dimensions](https://escholarship.org/content/qt0fp606kk/qt0fp606kk.pdf),
LBNL preprint dated 2004-11-02. Full institutional PDF inspected on 2026-11-08,
especially printed pages 2-4: geometric capacities/apertures/centroids,
nominal-center unknowns and a volume-integrated elliptic equation.
These motivate the representation and conditioning checks. Our exact polyhedral
clipping, connectivity guard, bounded Cartesian bisection and compatible A/d
pressure graph are project implementations. We do not implement the report's
centroid-interpolated second-order flux or complete heat/moving-boundary method.
No material constants or experimental observations are imported.

## DESIGN-CUTCELL-PRESSURE-005

`configs/cutcell_verification.json` declares fourteen geometry/pressure cases
with SI lengths and mathematical reference potentials. It introduces no new
fluid properties or calibrated growth coefficients. The manufactured potential
is `s*cos(pi*X)*Y^2*(1-Y)^2*Z^2*(1-Z)^2*(Z-a-b*Y)^2`; all outer/oblique normal
derivatives vanish. Analytic Laplacians are independently integrated over actual
liquid regions by split Gaussian quadrature with an analytic x integral.
The discrete operator is not used to generate its own validation source.

The original first study aborted at a genuinely split liquid cell beside the
same three-axis rotated cube. Its first six results, source archives and console
trace remain in `results/cutcell_pressure_first_8011b79/`; `failure.json` explicitly
records its reconstruction from that preserved log. Focused n=16 potential error
was 6.4348%, above the declared 5% limit. The follow-up keeps both 5% error gates,
increases Poisson levels to 8/16/24 and resolves the same rotated solid by bounded
static axis refinement. Per-case failures now remain in reports with a failed gate.

The tiny-layer conditioning case explicitly uses relative volume tolerance 1e-6
at liquid fraction 1e-7; ordinary geometry keeps 1e-10. This accounts for loss of
relative precision when subtracting nearly equal capacities. It is a numerical
stress test, not a physical subnanometer continuum experiment.

## CUTCELL-PRESSURE-NUMERICAL-20261108

`results/cutcell_pressure_final_v2_8011b79/` passes all fourteen cases. Uniform and
focused finest potential RMS errors are 0.41065% and 2.17234%; analytic-gradient
projection residuals are 0.36367% and 1.08533%. Maximum scaled integral residual
is 4.616e-13; all blocked-face speeds are zero and every random projection
decreases its A*d norm. Maximum normalized area/moment closure is
9.189e-14 / 8.782e-12. Actual refined rotated grids are 14x12x12, 20x19x20 and
31x27x33, obtained with 14/23/43 inserted axis planes. The solid is unchanged.

Minimum liquid fraction reaches 2.681e-10 in the middle rotated grid. Maximum
absolute divergence is 6.717e-8 1/s in tiny capacities; it is retained alongside
the small integral residual. The projection norm is not a physical cut-cell
kinetic-energy measurement, and this does not establish cut-cell advection or
momentum time-step stability. Embedded no-slip viscosity and nonlinear momentum
remain to implement before full growth/runtime integration.

`docs/CUTCELL_PRESSURE_CHECKPOINT.json` retains exact source/protocol/environment
identity, metrics, failure history and regression studies. Current-source
uniform momentum (eleven cases), scalar/high-Peclet transport (fifteen cases)
and rectilinear flow/solute (twenty cases) pass separately; the full suite passes
165 tests. Raw arrays and archived hashes are checked and figures inspected.
This is local numerical evidence, not calibrated KDP growth/quality data.

## COD-9010000 and bounded structural reference

Nelmes, Meyer and Tibballs (1982), *The crystal structure of tetragonal KH2PO4
and KD2PO4 as a function of temperature*, DOI 10.1088/0022-3719/15/1/005.
The tracked `data/COD.zip` contains the COD record for its KH2PO4 sample at
295 K. `configs/kdp_material_reference.json` pins archive/entry SHA256 and
transcribes the selected scalar fields; it does not import the paper's KD2PO4
structure or infer a temperature law. The original-paper request returned an
HTML access challenge, so missing pressure is retained as unknown.

The CIF gives a=b=7.4521 Angstrom, c=6.974 Angstrom, angles 90 degrees, Z=4,
V=387.293 Angstrom cubed, space group 122 and calculated density 2.334 Mg/m3.
Using NIST-KDP's molar mass and exact SI Avogadro constant gives derived density
2333.8948984452236 kg/m3; displayed digits are computational, not independent
measurement precision. The cell-volume check uses relative tolerance 1e-5 and
rounded-density agreement 1e-3, explicit numerical consistency choices.

`simulation.material` derives crystal-frame normals from the gradient of the
tetragonal plane equation. It does not supply facet kinetics, optical quality,
thermal expansion or a physical solution/solid partial-volume closure. It never
relabels the old model-unit growing-crystal runtime. The broader COD collection
includes different phases/conditions and known metadata conflicts; there is no
general all-phase CIF importer in this milestone.

## Measurement-source access audit

`configs/data_sources.json` registers the sources above, their materials,
stated licenses, local records and eligibility limits. `docs/DATA_AUDIT.json`
preserves the 2026-11-08 access results. The KDP-2026 paper lists one supplementary
PDF and three movies, with solubility/micrograph information, under CC BY-NC 3.0.
This audit received 404 for its PDF and 403 for its movie requests. The
Strathclyde README and paired CSV requests returned 403. These access failures
are specific observations, not evidence of permanent unavailability. Leeds
describes alpha-LGA vision data under CC BY 4.0; no direct file URLs were exposed
by the inspected archive record. IEEE lists a subscription requirement and no
uploaded files. No unavailable payload is represented as a downloaded dataset.

**2026-11-11 vision follow-up:** the Leeds direct file URLs are now verified;
2,123 real alpha-LGA images and 79 manual shape-fit sidecars were downloaded and
indexed. Three phase-specific KDP movies and supplementary methods were obtained
from the official PMC cloud mirror after primary links failed. Publisher MD5 and
local SHA256 checks pass. A separate Nature movie is explicitly ADP according to
its caption. Current URLs, license/phase limits, actual labels and import evidence
are in [the vision registry](configs/vision_sources.json) and
[real vision data](docs/REAL_VISION_DATA.md). The old failed audit remains historical.
These acquisitions do not supply reviewed segmentation masks, defect-onset labels
or calibrated controller responses.

No registered source currently supplies reviewed raw KDP facet trajectories.
KDP-2016 and KDP-2025 remain their existing
fit/aggregate records. Counts are not expanded into sample rows, and fitted
lines are not reconstructed as measured frames. Other-material candidates are
kept separate from KDP calibration and evaluation.

The standalone `scripts/audit_data_sources.py` records physical-data metadata
and access outcomes. It creates no models, forecast datasets or training labels.
Existing packages, starter scripts, solver code and runner defaults remain
unchanged. See `docs/PHYSICAL_DATA.md` for the bounded scope and its checks.

## Updating this register

For each new source or datum record: stable ID, title/author/year, direct URL/DOI,
page/table/equation, access date/method, original value/unit/basis, SI conversion,
experimental conditions, uncertainty meaning, validity limits and destination
config/code/dataset. Label values as measured, fitted, estimated, derived or
design choices. Record conflicts and missing fields explicitly.

Keep frames from an experiment together; an aggregate count is not a number of
independent experiments. Never expand fitted lines into supposedly measured
frames or replace unavailable quality with zero defects. Sources consulted but
not audited remain candidates; they do not provide active model coefficients.

## MFD-VISCOSITY-2005

Brezzi, F.; Lipnikov, K.; Simoncini, V. (2005), *A family of mimetic finite
difference methods on polygonal and polyhedral meshes*, author manuscript,
[primary PDF](https://mate.unipv.it/brezzi/papers/strike1-9.pdf), accessed
2026-11-09. Section 3, equations 3.6–3.10: consistent positive local flux metrics
using face moments and normals. `simulation/viscosity.py` uses unit diffusion
and an orthogonal-kernel stabilization with its explicitly declared scalar
factor. Shared trace continuity and local cell elimination provide the project's
implicit vector diffusion implementation. The paper's convergence assumptions
are not asserted for all nonconvex or arbitrarily thin crystal cuts. See
`docs/CUTCELL_VISCOSITY.md` for the derivation, representation and limits.

## CUTCELL-VISCOSITY-NUMERICAL-20261109

`configs/viscosity_verification.json` declares mathematical inputs and acceptance
gates for stationary, constant-nu viscosity, not measured KDP coefficients.
`results/viscosity_final_3b99100/` passes 13 cases: affine/sloped-wall manufactured
shear on uniform/focused meshes, four cut-geometry energy-decay checks and three
implicit Euler timestep checks. Finest spatial errors: 0.21695% / 0.76926%.
`docs/CUTCELL_VISCOSITY_CHECKPOINT.json` preserves source hashes, environment,
protocol, metrics and gates. This standalone cell-mean/face-mean operator still
needs energy-compatible pressure transfer, nonlinear momentum fluxes and
advective small-cell verification before cut-cell flow can run. Fixed geometry,
constant viscosity, stationary embedded walls and no free-surface stress are
current limits; no growth calibration or AI readiness is established.

## HYBRID-STOKES-2018

Di Pietro, D. A.; Krell, S., *A Hybrid High-Order method for the steady
incompressible Navier–Stokes problem*, author manuscript dated 2018-11-09,
[primary PDF](https://arxiv.org/pdf/1607.08159), accessed 2026-11-09.
Equation 17 defines the hybrid discrete divergence; its k=0 case is the
face-area-normal sum used in `simulation/stokes.py`. Section 3.5, equation 38,
uses the divergence/pressure pairing. The project's negative transpose pressure
force enforces that adjoint relation on its existing mimetic viscous operator.
This is not a reproduction of the paper's entire viscous/nonlinear scheme or
proof of its inf-sup/convergence bounds on all project cut geometries.
See `docs/CUTCELL_STOKES.md` for the independent project derivation, units,
energy identity and measured limits, including gradient-force leakage.

## CUTCELL-STOKES-NUMERICAL-20261109

`configs/stokes_verification.json` declares fixed-domain linear pressure/viscosity
coupling checks. Inputs are mathematical tests, not calibrated KDP conditions.
The first 6/12/18-grid study in `results/stokes_initial_d532f35/` failed focused
velocity accuracy (4.4208% at 18, gate 3%) and the coarse focused pressure rate
(0.507, gate 0.8). Conservation/energy/time gates pass. Retain this failed
protocol and results. The refined 12/18/24 study uses every original tolerance
unchanged, with added gradient-state balance and pressure-time checks. New
acceptance applies only to that measured refined regime. No nonlinear momentum,
growing interface, physical growth prediction or AI readiness is established.

The completed `results/stokes_refined_d532f35/` study passes all 17 cases.
Finest uniform/focused velocity errors are 0.65395%/1.8883%; pressure RMS over
the declared scale is 1.9792%/3.0236%. All 198 regression tests pass. Exact source
hashes, full metrics, original failed evidence and environment are retained in
`docs/CUTCELL_STOKES_CHECKPOINT.json`. The result verifies a fixed linear
operator at its declared resolution, not full nonlinear cut-cell growth flow.

## IMPLICIT-CUTCELL-TRANSPORT-2023

May, S.; Laakmann, F., *Accuracy analysis for explicit-implicit finite volume
schemes on cut cell meshes*, [primary manuscript](https://arxiv.org/pdf/2310.16101),
accessed 2026-11-09. Introduction and section 2 discuss the small-cell explicit
stability restriction and implicit treatments; stability does not establish
accuracy. The project uses fully implicit first-order upwind transport everywhere,
not the paper's mixed MUSCL-Trap method, and makes no second-order claim. Its
own finite-volume momentum/energy identities and independent tests are documented
in `docs/CUTCELL_MOMENTUM.md`.

## PRESSURE-SCHUR-2019

Bootland, N.; Bentley, A.; Kees, C.; Wathen, A. (2019), *Preconditioners for
Two-Phase Incompressible Navier–Stokes Flow*, SIAM Journal on Scientific
Computing 41(4), B843–B869,
[primary PDF](https://strathprints.strath.ac.uk/68686/7/Bootland_etal_SIAM_2019_Preconditioners_for_two_phase_incompressible_Navier_Stokes.pdf),
section 2.2, equation 26, accessed 2026-11-09. The constant-property
Cahouet–Chabard inverse combines pressure mass and Laplacian inverse terms.
The optional project Schur backend adapts that form to its scaled unknowns and
volume-mean pressure quotient. The actual Schur matvec and original saddle
residual use the unchanged project equations. No variable-density/two-phase
method, penalty or preconditioner-as-physics approximation is adopted.

## CUTCELL-MOMENTUM-NUMERICAL-20261109

`configs/momentum_verification.json` defines mathematical transport/nonlinear
verification inputs, not measured KDP data. The first 6/10/14 wall study in
`results/momentum_initial_bb4264a/` fails the high-curvature wall accuracy gate;
its conservation, energy, bounds, wave, ABC, time and tiny-cell gates pass.
Retain that failure and all original tolerances. The refined study uses the
same 3D curl reference on 24/36/48 wall grids, with a tested optional Schur solve
and an added complete coupled tiny-cell stress case at outgoing Courant 1000.
The donor scheme is first order and introduces separately measured numerical
dissipation. Canonical runtime integration, passive-solute/growth coupling,
physical calibration and AI readiness remain separate work.

The accepted `results/momentum_background_resolved_bb4264a/` study passes all
20 cases with the original tolerances/reference. Focused outer intervals are
explicitly subdivided to the background width L/n; nominal 12/18/24 gives actual
17^3/27^3/34^3 grids, so this is not equal-cell-budget evidence. Finest uniform/
focused velocity errors are 4.9190%/3.3665%, scaled pressure 17.387%/12.813%.
Both Courant-1000 cases retain ~1e-10 capacities and pass momentum/energy/bounds.
All 214 tests pass. Source-matched records and failed/interrupted history are
in `docs/CUTCELL_MOMENTUM_CHECKPOINT.json`; current regime limits and source
changes are explained in `docs/CUTCELL_MOMENTUM.md`.

## DESIGN-CONTROL-THERMAL-001

`configs/temperature_control_baseline.json` defines a first-order thermal software
fixture and four declared development scenarios. The time constants, loads,
sensor/actuator delays, noise, PID gains, fallback value and command range/slew
are project test assumptions. They are not measured KDP parameters, literature
operating bounds or an approved thermostat recipe. The SI-bearing interfaces do
not convert the older illustrative growth engine into a calibrated plant.

The method references above motivate anti-windup and measurement derivatives;
the actual equations, analytic tests, comparison results and limits are in
[Temperature control](docs/TEMPERATURE_CONTROL.md) and its
[checkpoint](docs/TEMPERATURE_CONTROL_CHECKPOINT.json). Physical equipment timing,
operating limits and action-response evidence remain separate requirements.

## DESIGN-EVENT-SHADOW-002

`configs/event_rule_policy.json` declares source-clock age/gap/expiry settings for
development review proposals. `configs/supervised_control_fixture.json` binds an
exact fixture video to a chosen software mapping of 100 thermal-model seconds per
display second, a scripted decision delay and bounded recipe pauses. These are
integration-test assumptions, not a physical acquisition clock, KDP operating
recipe, human review or a measured quality response. Real sources are restricted
to proposal-only shadow execution by this path.

[Event supervision](docs/EVENT_SUPERVISION.md) retains the actual shadow outcomes,
common-recipe tracking cost, proposal/evidence hashes and independent control
audit. It extends DESIGN-CONTROL-THERMAL-001 without adding physical parameters
or an adaptive forecast claim. The [checkpoint](docs/EVENT_SUPERVISION_CHECKPOINT.json)
records the verified source revision and explicit open pitch requirements.

## DESIGN-AREA-FORECAST-003

`configs/area_forecast.json` defines a 0.25-source-video-second projected-area
forecast horizon, causal two-observation histories, development gap/age settings
and a bounded one-coefficient ridge fit. One horizon equals one illustrative
optical geometry step at the exported 4 FPS, not a physical growth interval.
Only the existing optical pilot's training families fit the coefficient/prior;
the development comparison preserves startup coverage and separate current/future
emission-proxy masks. No 3D truth or action-response parameters enter inputs.

The actual [forecast report](docs/AREA_FORECAST.md) records that rate calibration
does not beat future-area persistence on the four shared validation windows.
Those families were already used during segmentation development. Real shadow
outputs remain unreviewed, unscored and unable to authorize actuation. The
[checkpoint](docs/AREA_FORECAST_CHECKPOINT.json) binds all model/data identities,
the independent audit and explicit remaining measurement/adaptation gates.
