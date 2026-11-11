# Bounded physical-data preparation

The physical-data tools provide a checked KDP structural reference and a
standalone source audit. They do not calibrate the illustrative growing engine.
The separate [synthetic pipeline](SYNTHETIC_DATA.md) now generates tabular growth
data; model training, real forecasting and physical control remain future work.

## KDP structural reference

`configs/kdp_material_reference.json` pins `data/COD.zip` and its selected
`9010000.cif` entry by SHA256. `src/simulation/material.py` checks the scalar
transcription and derives solid density and crystal-frame facet normals.

The reference is tetragonal KH2PO4 at 295 K, with a=b=7.4521 Angstrom,
c=6.974 Angstrom, Z=4 and V=387.293 Angstrom cubed. The existing sourced molar
mass gives `rho=Z*M/(N_A*V)`, approximately 2333.8949 kg/m3. This agrees with
the CIF's rounded calculated density of 2334 kg/m3; it is not an independent
bulk-density measurement. A physical 1 mm3 crystal has reference mass about
2.333895 mg.

For the plane `h*x/a+k*y/a+l*z/c=constant`, its normal is proportional to
`(h/a,k/a,l/c)`. The `{101}` unit normal is approximately
`(0.683297,0,0.730140)` in crystal axes. World orientation needs a separate
rotation. Unit-cell dimensions do not prescribe the whole crystal's habit.

Measurement pressure is not reported in this CIF and remains unknown. This is
one temperature point, not a thermal-expansion law. It neither changes the
303.15 K solution-property reference nor redefines the illustrative growth
engine's model-unit coefficients.

```bash
.venv/bin/python scripts/check_material_reference.py
```

The archive is read directly. The duplicate `data/COD/` extraction was verified
byte-for-byte against all 22 ZIP entries and removed during cleanup on 2026-11-11.
The uploaded kinetic/export proposal remains unchanged and unused. The
[data guide](../data/README.md) identifies retained inputs and the active dataset.

## Source/access audit

`configs/data_sources.json` distinguishes structural records, published rate
summaries, candidate supplementary files and measurements of other materials.
The audit functions live directly in `scripts/audit_data_sources.py`; no new
package or folder layout is introduced.

```bash
# Inspect source metadata and existing caches without network access.
.venv/bin/python scripts/audit_data_sources.py
# Explicitly retry listed public files, preserving a new access record.
.venv/bin/python scripts/audit_data_sources.py --fetch \
  --output data/audits/new_access_check.json
```

Downloads have size limits and format/hash checks. HTML challenge pages are
rejected; failures remain visible. The last actual file-access attempts,
2026-11-08, returned 403/404 for the listed KDP supplementary and Strathclyde
files. Those are dated access observations, not proof of permanent unavailability.
The existing KDP tables/fitted rates remain aggregate calibration references;
they are not expanded into artificial measured trajectories. No verified raw
KDP growth time series has been imported. Evidence is in
[DATA_AUDIT.json](DATA_AUDIT.json).

## Measurement preparation and next work

Future measurements should retain source/experiment/crystal/facet IDs,
dependency relationships, clock origin, SI units and concentration basis,
camera scale, uncertainty definitions, missing values and logged command timing.
Measured sensors and equipment commands must remain distinct. A single camera
view does not independently measure 3D crystal mass or internal optical quality.
The separate synthetic recorder now provides explicitly idealized sensor proxies;
it does not implement a real-measurement adapter. That adapter must follow the
actual available measurements; see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md).

A standalone embedded tangential no-slip viscosity operator now passes its
independent numerical checks; see [CUTCELL_VISCOSITY.md](CUTCELL_VISCOSITY.md).
Pressure now couples to the same velocity representation in the
[linear Stokes step](CUTCELL_STOKES.md), with a compatible energy budget. The
accepted [split momentum operators](CUTCELL_MOMENTUM.md) remain separate from
the growth engine. Their flow/scalar adapter and subsequent uptake/motion/volume
closure are conditional higher-fidelity work. The immediate priority is the
bounded synthetic data pipeline and its pilot/resolution/action audit in
[AI_READINESS.md](AI_READINESS.md); full CFD is not required for the first
synthetic growth prototype. Real-data collection and physical calibration
remain important for claims about measured KDP behavior.

## Files changed

- `configs/kdp_material_reference.json`, `src/simulation/material.py` and
  `scripts/check_material_reference.py`: bounded structural checks only.
- `configs/data_sources.json` and `scripts/audit_data_sources.py`: standalone
  physical/measurement-source audit.
- `tests/test_material_reference.py` and `tests/test_data_sources.py`: ten
  checks for numerical identities, file integrity and honest access failures.
- Source/model documentation and `DATA_AUDIT.json`: provenance and limitations.

Existing package directories, starter scripts, numerical solvers and runner
defaults are preserved. No optional ML dependency or fitted-model artifact is
added. All new code starts with its purpose summary.
