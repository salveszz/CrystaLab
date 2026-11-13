# Data directory

## Real vision observations

The acquired microscope data is indexed at `data/real_vision_leeds_v1/images.jsonl`:
2,123 real alpha-LGA images from nine runs, split 1,323/400/400 by whole source
run/family. Original images and 79 manual shape-fit sidecars remain in the pinned
ZIP under `data/external_vision/LGA-LEEDS-2026/`. Independently reviewed pixel masks
and event labels remain unavailable. This is another material, not KDP calibration.

`data/leeds_shape_proxies_v3/` separately holds 54 training and 12 validation
silhouette targets projected from the authors' manual 3D fits. These are weak
geometric targets rather than reviewed pixel outlines. The three
`data/source_silhouette_*_v1/` cohorts contain 84 / 54 / 138 training images and
the same 56 validation images (12 real proxies plus 44 synthetic images).
The [source-shape guide](../docs/SOURCE_SHAPE_TRAINING.md) records the frozen
three-way training comparison, source identities and limits. Final test groups
and the six existing unreviewed annotation proposals remain separate.

`data/external_vision/KDP-2026/` retains three real phase-specific KDP microscopy
movies and supplementary methods; `ADP-NATURE-2024/` retains a separately identified
ADP movie. Source licenses, clocks, integrity checks and import/replay commands
are in [Real vision data](../docs/REAL_VISION_DATA.md) and the
[source registry](../configs/vision_sources.json). These raw acquisitions and
generated indexes remain Git-ignored. A fresh clone must fetch/index them.
`data/real_vision_kdp_v2/frames.jsonl` now indexes all 881 native KDP movie frames,
with source pixel hashes, presentation clocks and one conservative study family.
`data/annotations_kdp_v1/` contains six native annotation sheets (two uncertain
outlines and four manual sheets), with zero accepted reviews. See the
[KDP review-pack guide](../docs/KDP_VIDEO_DATA.md) for import, clock limits and review.

`data/published_masks_chimp_v1/images.jsonl` separately indexes 587 real
protein-crystal images and 4,986 published crystal/drop instance masks. Original
JPEGs and NPZ masks stay in `data/external_vision/CHIMP-ZENODO-2024/`. These are
selected protein-crystal annotations, not KDP seed/secondary/inclusion labels.
See [published masks](../docs/PUBLISHED_CRYSTAL_MASKS.md) for the exact source
license snapshot, resource-safe import, author membership and training gate.

## Synthetic growth data

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

The separate [CrystalCV source](../docs/CRYSTALCV_DATA.md) is cached under
`external_vision/CRYSTALCV-ZENODO-2026/` and indexed in
`real_vision_crystalcv_v1/`. It contains public perovskite observations and source
measurements. Keep uncontoured images, automatic displays and published manual
references separate, and preserve exact-pixel dependencies before any training
split. The large archives use `scripts/acquire_crystalcv.py` with their pinned
protocol; they are not loaded by the general small-file download path.

`results/crystalcv_process_audit_v2/` contains 482 automatic contour observations
and 463 past-only development candidates from one source run. The [process guide](../docs/CRYSTALCV_PROCESS.md)
records exact original workbook agreement, retrospective-field exclusion and
next-row command availability. Targets stay in pixels and unreviewed. Original
FRC workbook/controller files under `external_vision/FRC-NATURE-2024/` are read-only
lineage references, with unverified repository reuse terms; they add no training cohort.
