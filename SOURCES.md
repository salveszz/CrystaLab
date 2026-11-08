# CrystaLab sources and data register

Last audit: 2026-11-07. This file records the websites and numeric evidence used
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
| KDP-2016 | [Wu et al., accepted manuscript](https://eprints.whiterose.ac.uk/105693/3/20160707%20Crystal%20Face%20Specific%20Growth%20Kinetics%20FINAL%20x-x.pdf), [repository record](https://eprints.whiterose.ac.uk/id/eprint/105693/), [DOI](https://doi.org/10.1021/acs.cgd.6b00189) | Two published face-distance fit summaries; manuscript inspected. |
| KDP-2025 | [Maksimovic et al., PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800150/), [ACS full text](https://pubs.acs.org/doi/10.1021/acsomega.4c09287), [PubMed](https://pubmed.ncbi.nlm.nih.gov/39926492/) | Tables 1/2 and solubility relation transcribed from indexed primary full text; direct page access was intermittently blocked. |
| VTT-2006 | [Manninen (ed.), VTT Research Notes 2340](https://publications.vtt.fi/pdf/tiedotteet/2006/T2340.pdf) | Binary-solution CFD property point; original institutional PDF downloaded, text checked and Table 2.1 visually inspected. |
| CAL-NIST | [Calibration and uncertainty analysis](https://www.nist.gov/publications/calibration-and-uncertainty-analysis-predictions-computational-models) | Calibration/model-inadequacy distinction; methodological reference, not KDP data. |
| NUM-FV | [NIST FiPy discretization](https://pages.nist.gov/fipy/en/stable/numerical/discret.html) | Finite-volume background, not a source of physical coefficients. |
| NUM-GRAD | [OpenFOAM gradient schemes](https://doc.openfoam.com/2312/tools/processing/numerics/schemes/gradient/) | Least-squares/limited-gradient background. Our implementation and analytic verification are project code. |
| NUM-LIMIT | [OpenFOAM limited normal-gradient correction](https://doc.openfoam.com/2312/tools/processing/numerics/schemes/sngrad/rtm/limited/) | Background considered for bounded corrections; indexed content only, direct access blocked. Not an assertion that our limiter is OpenFOAM's implementation. |
| NUM-SOLVE | [SciPy sparse direct solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html), [CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html), [GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) | Sparse solver APIs; no experimental data. Record package versions with each study. |
| NUM-VERIFY | [NASA verification assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html) | Verification/convergence background cited in the diagnostic documentation. |
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
