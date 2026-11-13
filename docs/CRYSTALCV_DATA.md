# Public perovskite growth and control-data acquisition

We acquired the versioned [CrystalCV bulk dataset](https://zenodo.org/records/19140132)
and [software/example archive](https://zenodo.org/records/20655010), both declared
**CC BY 4.0** by their Zenodo records. The material is MAPbBr3, CsPbBr3 or FAPbI3
perovskite, separately identified by source folder. This adds relevant public
growth trajectories, published rectangle measurements and a control-data lead.
It supplies no KDP mask/event accuracy or validated KDP control policy.

The [primary study](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00140h)
describes growth analysis, a thermal-cycle trial and phase-change tracking. The
archives have their own CC BY 4.0 metadata; the article's CC BY-NC 3.0 terms and
the latest GitHub repository's absent license metadata are separate. We use the
versioned Zenodo assets and preserve attribution, license and change records.
No author software was executed, installed or incorporated into the runtime.

## Actual files and counts

| Archive | Bytes | Published MD5, verified locally |
|---|---:|---|
| `Crystallization_CaseStudydata.zip` | 6,717,870,074 | `7833cfb2a3e2eb6450eeb1718ad329aa` |
| `CrystalCV.zip` | 253,384,719 | `b43d54363bfb756b833ff2c5c62104c7` |

SHA256, exact version URLs and preserved license snapshots are pinned in
[`configs/crystalcv_vision_import.json`](../configs/crystalcv_vision_import.json).
The data concept DOI `10.5281/zenodo.19140131` resolved to version record
`19140132`; future concept updates must not silently replace that version.
Metadata contains volatile statistics, so reproduction uses the dated snapshots
rather than assuming a fresh API response has identical bytes.

The completed index is `data/real_vision_crystalcv_v1`. It retains 30,357 original
archive members, 29,974 decoded image entries, 226 source CSV tables and 54
published manual measurements. Images are classified before downstream use:

| Image role | Entries | Treatment |
|---|---:|---|
| Uncontoured source growth images | 28,279 | Preserve original JPEGs, timestamp overlays and source folders. There are **27,652 distinct native pixel arrays** after exact overlap is accounted for. |
| Automatic annotated output displays | 1,641 | Preserve as derivatives; not raw training inputs or human ground truth. |
| Published manual annotated displays | 54 | Preserve as references for the source measurement task; not clean training inputs or silhouette masks. |

The bulk archive itself has 29,569 image entries. The software archive adds 351
growth-image copies and 54 manual displays. All 54 manual displays use TIFF
content despite their `.jpg` names. Native decoding follows the content rather
than rewriting the extension or losing the original bytes.

There are 132 raw source-folder identities across the two archives. Exact pixel
overlap connects them into 129 components: the two example sequences appear in
both archives, and source folders `A_230726_0.2mmh` and `A_230802_0.2mmh` share
pixels. These are **not 129 independently verified physical experiments**.
`results/crystalcv_structure_v1.json` preserves the folder relationships. Every
entry currently stays in one study-level development family; no independent
evaluation groups or train/test image split have been assigned.

## What the labels and clocks mean

`SI_ACC_TEST.csv` supplies 54 manual FIJI rectangle areas for the author's selected
primary crystal, with automatic rectangle measurements for comparison. Each row
is bound to an original thermal-cycle image and its corresponding annotated
display. These are published **scalar rectangle measurements**, not segmentation
polygons, exhaustive negatives or inclusion labels. No local human review receipt
is fabricated. The source's declared 79.26 pixel²/mm² scale is retained as source
metadata; we do not independently certify physical calibration.

Filename elapsed seconds and `frame_index × 600 s` remain separate. The 54 manual
rows differ by up to five seconds between those clocks. Actual capture/run-start
time, dropped-frame history and first-hour membership remain unknown. The source
images retain their printed clocks; image cadence is not inferred from video
playback or a universal fixed frame interval.

The paper reports heater removal at 25 h 30 min and reapplication 80 minutes later.
The archived figure script marks 25 h 40 min–27 h. Both are preserved with separate
clock identities in `reported_actions.json`; their alignment is unresolved. Neither
is a sensor-temperature trace or acknowledged command log. The paper's thermal
case uses inverse-temperature solubility; its action direction must not become
a KDP recipe. The single trial cannot establish a counterfactual control benefit.

## Infusion/growth table: a useful separate lead

One bulk folder, `A_230326_0.1mmh`, includes **28,811 reported process/vision rows**
over 0–40.214444 h, plus experiment parameters and 482 source camera images.
The table has strictly increasing times, approximately five-second sampling and
fields including infusion rate, total infusion, camera-derived area/length,
growth estimates and controller error terms. The exact source tables/config and
their inventory are retained in `results/crystalcv_process_source_v2`.

The source's per-run config has a different basename than its parent folder;
the initial failed lookup is retained in `results/crystalcv_process_source_v1`.
Its config declares `FrameInterval=10`, whereas these image filenames advance
roughly five minutes, and its scale is `0.0375` under legacy semantics. Do not
apply the current thermal example's time/scale conversion to this run. Other
folders also contain tracking files whose names refer to different runs. No
automatic table-to-image association is made merely from a parent directory.

The process/image endpoint times are compatible, but per-image synchronization,
unit/calibration meaning, online versus later smoothing, and the meaning of
reported infusion rate still need checking. No measured pump-flow sensor,
command acknowledgment or time-resolved temperature sensor has been verified.
Treat this as a candidate action-response development source, not an accepted
physical response model or reviewed crystal-quality labels.

## Tools, verification and reproduction

`scripts/acquire_crystalcv.py` hashes files in chunks, checks publisher MD5 and
pinned SHA256/size, preserves partial attempts and refuses to overwrite accepted
files. Use it for these large archives; the general small-file source auditor
does not download them. `src/vision/crystalcv_data.py` reads one bounded archive
member at a time and stores source/pixel identities without extracting the whole
dataset or executing source code. The native loader checks bytes and decoded
pixels before returning a requested image.

`scripts/audit_crystalcv.py` imports no producer. It re-reads the original ZIP/CSV
files, decodes images with Pillow instead of OpenCV, reconstructs native BGR
hashes, clocks, coverage, duplicate groups and manual measurement bindings, and
checks the external frozen protocol. The [checkpoint](CRYSTALCV_CHECKPOINT.json)
records the actual audit, regression, source identities and preserved attempts.
The first single-process audit ended with signal 143 without a final report;
that incomplete attempt is retained. The accepted execution uses eight disjoint
pixel-audit shards and an explicit coverage merge. A shard alone cannot establish
full pixel verification; the merge checks every shard index, selected source-ID
digest, total coverage and identical source/protocol/auditor identities.
The PNG/SVG inspection figure is in `results/crystalcv_figures_v1`; its PNG was
visually checked. Source values and model evaluation remain different evidence.
All 29,974 native image entries pass the merged independent reconstruction.
The published automatic rectangles have 7.866% mean absolute percentage error
against the 54 source manual rectangle areas; this is a source-baseline
comparison, not a newly trained model or silhouette score. Final verification
passed 477 tests and 323 subtests in 133.10 s; 20 source-specific tests passed.

```bash
# Offline check; add --fetch when acquiring into a fresh cache.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/acquire_crystalcv.py --cache data/external_vision/CRYSTALCV-ZENODO-2026 --output results/crystalcv_acquisition_new.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_crystalcv.py --cache data/external_vision/CRYSTALCV-ZENODO-2026 --protocol configs/crystalcv_vision_import.json --output data/real_vision_crystalcv_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_crystalcv.py --index data/real_vision_crystalcv_new --protocol configs/crystalcv_vision_import.json --output results/crystalcv_audit_new.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/plot_crystalcv.py --index data/real_vision_crystalcv_new --output results/crystalcv_figures_new
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_crystalcv_data.py
```

Use fresh output paths. Archives, generated indexes and analysis outputs stay
Git-ignored. Source code, protocol, metadata snapshots and checkpoint are tracked.

For bounded audit jobs, replace the single audit command above with:

```bash
mkdir results/crystalcv_shards_new
for crystalcv_shard in 0 1 2 3 4 5 6 7; do
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_crystalcv.py --index data/real_vision_crystalcv_new --protocol configs/crystalcv_vision_import.json --shard-index "$crystalcv_shard" --shard-count 8 --output "results/crystalcv_shards_new/shard_${crystalcv_shard}.json"
done
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_crystalcv.py --index data/real_vision_crystalcv_new --protocol configs/crystalcv_vision_import.json --output results/crystalcv_audit_new.json --merge-inputs results/crystalcv_shards_new/shard_*.json
```

Next, audit the candidate process/image synchronization and causal fields, define
the compatible observable target, and review source/run dependencies before an
action-response development experiment. Independently reviewed KDP masks/events,
representative KDP evaluation, faces, first-hour coverage and physical outcomes
remain separate requirements under the [pitch contract](VISION_MVP.md). The
current pitch detector, forecasts, PID and operator controls are unchanged.
