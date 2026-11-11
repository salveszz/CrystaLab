# Data directory

The active synthetic dataset is stored locally at:

```text
data/synthetic_growth_v1/samples.jsonl
```

It contains 1,080 JSONL rows from 360 episodes and 60 distinct starting-condition
groups. The frozen split is 648 training / 216 validation / 216 test samples.
There are 720 fully observed rows and 360 startup rows with explicitly missing
sensor/history inputs. All future growth labels are present.

## Why the dataset may be missing from your view

Generated data is Git-ignored. This README, the generation configuration and
compact reports are versioned, but the raw dataset is local to the machine where
it was generated. A fresh clone must generate it. Editors that hide ignored
files may also hide `synthetic_growth_v1/`.

The [dataset card](../docs/SYNTHETIC_DATASET_V1.md) explains the fields, split and
limits. The [pipeline guide](../docs/SYNTHETIC_DATA.md) gives generation/audit
commands. The frozen protocol is
[`configs/synthetic_growth_dataset_v1.json`](../configs/synthetic_growth_dataset_v1.json).
The dataset directory includes its own README, source snapshot, full manifest,
raw episodes, final fields, configuration, logs and independent audit.

## Retained source inputs

| Path | Role |
|---|---|
| `COD.zip` | Original structural-input archive; the checked material tool reads it directly. |
| `audits/` | Local dated source-access evidence. Reviewed summary: [DATA_AUDIT.json](../docs/DATA_AUDIT.json). |
| `config_kinetic_integration_code.py` | Uploaded integration proposal retained for reference; unused by the runtime. |
| `kdp_crystal_growth_training_dataset_documentation.md` | Uploaded proposal, not the authoritative specification for the current dataset or accepted kinetics. |

The ZIP is a structural reference, not measured growth history. The uploaded
proposal files have not been adopted as a calibrated KDP model; use
[PHYSICAL_DATA.md](../docs/PHYSICAL_DATA.md) for the accepted reference and limits.

## Cleanup policy

Only `synthetic_growth_v1/` is the current generated growth dataset. The obsolete
`synthetic_growth_pilot*` directories were removed after their relevant metrics
and provenance were retained in the numerical checkpoints. The duplicate `COD/`
extraction was byte-checked against all 22 original ZIP entries and removed;
`COD.zip` remains intact. Keep the active dataset's raw episodes and final fields:
they are required for the independent audit.

New generated data and downloads remain ignored unless explicitly selected for
publication. Do not combine overlapping old pilots with the frozen v1 splits.
