# Documentation index

Start with the [project README](../README.md) for the local dataset location and
commands, then use the [roadmap](../roadmap.md) for current priorities. Historical
milestone notes are preserved separately in [development history](DEVELOPMENT_HISTORY.md).

## Product plan and datasets

| Document | Purpose |
|---|---|
| [Roadmap](../roadmap.md) | Pitch-led milestones: camera, image data, perception, alerts, PID/adaptation and demonstration, plus HPC/Qiskit extensions. |
| [Vision MVP](VISION_MVP.md) | Capture/annotation contract, observable targets, real/synthetic splits, evaluation and control interfaces. |
| [Video baseline](VISION_BASELINE.md) | Implemented file/camera observation, explicit clocks, projected seed tracking, validity events and reproducible fixture replay. |
| [Real vision data](REAL_VISION_DATA.md) | Acquired microscope images/videos, actual labels, frozen image splits, licenses, audits and development replay. |
| [Vision checkpoint](VISION_CHECKPOINT.json) | Source hashes, 244-test result, fixture/development replay checks and remaining gates. |
| [Dataset v1 card](SYNTHETIC_DATASET_V1.md) | The active 360-episode / 1,080-sample dataset, splits, missingness and use limits. |
| [Dataset v1 report](SYNTHETIC_DATASET_V1.json) | Actual hashes, groups, audit results and measured runtime. |
| [Data pipeline contract](SYNTHETIC_DATA.md) | Generation/audit commands, exact labels, observations, units and provenance. |
| [AI readiness](AI_READINESS.md) | Separate readiness gates for vision, synthetic growth prediction and physical control. |
| [Data directory guide](../data/README.md) | Local generated files, original inputs and the Git ignore policy. |

The file/camera adapter and OpenCV seed baseline are implemented; no model fitting,
evaluated defect detector or physical controller is implemented. Real image
annotation and optical synthetic data are next. YOLOv8-seg and event alerts
follow the image-data gate. Growth baselines and XGBoost remain a supporting
experiment using the frozen tabular splits, illustrative units and fixed law.

## Physics and verification

| Topic | Method and evidence |
|---|---|
| Growing-crystal liquid/displacement model | [Fluid model](FLUID_MODEL.md), [coupled convergence](CONVERGENCE.md), [numerical checkpoint](NUMERICAL_CHECKPOINT.json). |
| Facet sampling and spatial accuracy | [Spatial accuracy](SPATIAL_ACCURACY.md), [focused-grid convergence](FOCUSED_GRID.md). |
| Physical definitions and source data | [Physical model](PHYSICAL_MODEL.md), [physical data](PHYSICAL_DATA.md), [source audit](DATA_AUDIT.json). |
| Experimental claims and evaluation | [Real-data validation](REAL_DATA_VALIDATION.md). |
| Canonical fixed-grid flow/solute work | [Flow model](FLOW_MODEL.md), [solute model](SOLUTE_MODEL.md), [solute checkpoint](SOLUTE_CHECKPOINT.json). |
| Rectilinear/focused flow operators | [Rectilinear flow](RECTILINEAR_FLOW.md), [checkpoint](RECTILINEAR_CHECKPOINT.json). |
| Cut-cell geometry and pressure | [Pressure method](CUTCELL_PRESSURE.md), [checkpoint](CUTCELL_PRESSURE_CHECKPOINT.json). |
| Cut-cell viscosity | [Viscosity method](CUTCELL_VISCOSITY.md), [checkpoint](CUTCELL_VISCOSITY_CHECKPOINT.json). |
| Coupled pressure/viscosity | [Stokes method](CUTCELL_STOKES.md), [checkpoint](CUTCELL_STOKES_CHECKPOINT.json). |
| Conservative momentum and small cells | [Momentum method](CUTCELL_MOMENTUM.md), [checkpoint](CUTCELL_MOMENTUM_CHECKPOINT.json). |
| Synthetic-label numerical acceptance | [Growth-data checkpoint](SYNTHETIC_DATA_CHECKPOINT.json). |

The fixed-domain flow operators are separate from the growing engine used for
current data. Numerical success does not establish physical growth or optical
quality. Individual method documents retain their declared approximations.

## Sources and historical artifacts

[SOURCES.md](../SOURCES.md) retains source attribution and distinctions between
verified references, design assumptions and unvalidated proposals. The original
[project presentation](../Pitch_KDP.pptx) remains the high-level proposal.

Generated study outputs under `results/` and the superseded pilot datasets have
been removed during authorized cleanup. Their paths in historical reports are
provenance labels, not files promised to exist in this checkout. Checkpoint JSON
files are preserved unchanged; rerun the documented verification commands when
raw study arrays are needed. The complete active v1 dataset remains local and
fully auditable. It is excluded from Git along with future generated outputs.
