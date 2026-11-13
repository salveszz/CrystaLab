# Published real crystal masks

The project now retains the [CHiMP Detector dataset, version v1](https://zenodo.org/records/11110373),
by King and colleagues at Diamond Light Source, under **CC BY 4.0**. It adds real
microscope images with published expert annotations. The masks select interesting
or mountable **protein crystals** and droplets; they do not define KDP seed,
secondary-crystal or inclusion classes. The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11448919/)
describes their preparation and intended detector task.

This is an auxiliary source for a separately declared segmentation experiment.
It is not automatically combined with the pitch's three-class training dataset.
The six Leeds outline proposals still need actual human review, and independent
KDP evaluation remains open. No new real-data training or accuracy result is
claimed by this acquisition. The later [published-mask pilot](PUBLISHED_MASK_PILOT.md)
now trains with the source classes preserved. It has weak selected-crystal
validation performance and is not promoted or mapped into the pitch classes.

## Actual contents and audit

The downloaded ZIP has **206,420,425 bytes**. Its published MD5 is
`5592a7e8fbe47a28bdc84c0248535361`; local SHA256 is
`0b7f117c85ec7cc24bbc29aa7ac4e430a5f0bacd26f9a08478daba72b3f65b4f`.
Both checks pass, as do ZIP CRC and source-member checks.

| Source set | Images | Author training CSV | Author validation CSV |
|---|---:|---:|---:|
| VMXi | 237 | 189 | 48 |
| XChem | 350 | 280 | 70 |
| Total | 587 | 469 | 118 |

Re-reading the original JPEGs and NPZ files reconstructs **4,986 source instance
masks: 4,463 `crystal` and 523 `drop`**. There are 280 images without a selected
crystal mask, including 73 with an empty source annotation stack. Neither category
is silently interpreted as an exhaustive negative for our generic crystal task.
Source masks retain their holes, separate instances and original coordinates.

The VMXi source CSVs contain 189/48 training/validation images, whereas the paper
describes 190/47. The import preserves the downloaded membership and records the
discrepancy. Native image dimensions are 1688 × 1352 for VMXi and 1224 × 1024 for
XChem. No exact decoded-image duplicates were found, including across source splits.
Physical run/plate grouping is unverified; there is no new independent-run test set.
Author validation membership is preserved without presenting it as a final KDP test.

One source mask stack expands to **992,758,996 bytes**, with up to 435 instances.
The reader opens the nested NPZ, checks its array headers and streams one native
boolean mask plane at a time. It never uses `np.load` on the full mask tensor;
pickle loading is disabled for the small class array. The complete reconstruction
audit took 10.82 seconds with peak resident memory 149,572 KiB on this machine.
These are import resource measurements, not inference performance.

The [checkpoint](PUBLISHED_MASKS_CHECKPOINT.json) retains source/config hashes,
actual counts, reconstruction audit, regression result and inspection evidence.

## Local files and reproduction

```text
data/external_vision/CHIMP-ZENODO-2024/chimp_detector_datasets.zip
data/published_masks_chimp_v1/manifest.json
data/published_masks_chimp_v1/images.jsonl
data/published_masks_chimp_v1/importer_snapshot.py
results/chimp_source_audit_v1.json
results/chimp_source_preview_v1/
```

Raw images and generated indexes remain Git-ignored. The compact dated
[Zenodo metadata snapshot](CHIMP_ZENODO_METADATA_20261111.json) is versioned to
preserve the exact license and file checksum evidence. The API includes volatile
usage statistics, so refetching it later need not reproduce the snapshot's bytes.
The import uses this preserved snapshot; it does not overwrite it with a new reply.

Use the [source registry](../configs/vision_sources.json) and existing acquisition
tool to fetch/check the registered public assets. No login or purchase was used.
Then import into an unused output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_published_masks.py \
  --archive data/external_vision/CHIMP-ZENODO-2024/chimp_detector_datasets.zip \
  --metadata docs/CHIMP_ZENODO_METADATA_20261111.json \
  --protocol configs/chimp_vision_import.json --output data/published_masks_chimp_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_published_masks.py \
  --output data/published_masks_chimp_v1 --audit
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_published_masks.py \
  --preview data/published_masks_chimp_v1 --output results/chimp_source_preview_v1
```

The audit re-decodes all original images, re-streams masks and reconstructs every
index record, including area, bounding box and exact boolean-plane hash. Rehashed
index edits still fail comparison with the source. Duplicate pixels across author
splits would be retained and flagged rather than hidden by reshuffling.

## Inspection and training gate

Six author-training images were inspected: one image without selected crystals,
one with a few and one with many, from each source set. Original/outline panels
show small targets, dark droplet boundaries, edge-adjacent annotations and clutter.
The appearance differs substantially from the current large seeded-crystal
optical pilot. Source validation images were decoded for integrity and were not
visually inspected for tuning. No labels were edited or human reviews fabricated.

Before any auxiliary training, declare the selected-object task and its class
coverage, treatment of unannotated objects, source grouping and evaluation rules.
Do not rename `crystal` to `seed` or `drop` to `inclusion_candidate`. Published
source annotations are distinct from a new local review receipt; no such receipt
is required merely to preserve a published label, and none is invented here.
The current pitch-class adapter has no automatic export route for this index.

The [OpenCrystalData primary paper](https://doi.org/10.1016/j.dche.2024.100150)
and its public Kaggle listing were also checked. They describe several distinct
process-image tasks, including impurity classification, aggregate particle-size
characterization and annotated polystyrene spheres. Those payloads were not
downloaded in this batch; their labels cannot be assumed to be KDP growth masks.
The main pitch sequence still needs reviewed relevant boundaries/events, longer
trajectories, mixed-training comparisons and separately reported real evaluation.
