# Acquired real vision data

The first real-data acquisition and image import are complete. **Independently
reviewed Leeds/KDP pixel masks and visible-event annotations are still missing.** A separate
[published-mask acquisition](PUBLISHED_CRYSTAL_MASKS.md) adds 587 real protein-crystal
images and 4,986 source crystal/drop masks, with their original selected-object
semantics. They are not mapped to the pitch classes. No pitch model has been fitted
on independently reviewed real pixel/event labels. A separate
[source-shape experiment](SOURCE_SHAPE_TRAINING.md) now derives 54 training and
12 validation silhouette proxies from the authors' manual 3D fits and completes
synthetic-only, source-projection-only and mixed fits. Its positive mixed result
measures proxy agreement on alpha-LGA; it does not establish real defect accuracy.
[Synthetic-only pitch-class fits](VISION_PILOT.md) and
[learned development replay](LEARNED_VIDEO.md) exist; their real accuracy remains
unevaluated.
The [additional public KDP review](PUBLIC_KDP_SOURCE_REVIEW.md) records surface
crack/crater leads, access attempts and missing reuse/label evidence. Those leads
have supplied no accepted training payload and are not growth-event validation.
A separate [CrystalCV acquisition](CRYSTALCV_DATA.md) now adds 27,652 distinct
uncontoured perovskite image arrays, 54 published manual rectangle measurements,
automatic tracking tables and an infusion/growth-log lead. Its source roles,
duplicate folders, clocks and scale semantics remain separate; these are not
accepted KDP masks or a validated action-response model.
The [source registry](../configs/vision_sources.json) pins URLs, licenses, citations
and actual SHA256 values. The [compact report](REAL_VISION_DATA.json) records the
download/decode/import evidence; raw archives remain local and Git-ignored.
The [KDP frame index/review pack](KDP_VIDEO_DATA.md) now makes all 881 movie frames
traceable and prepares six native annotation sheets with phase/timing context.
Two contain uncertain outline drafts; four need manual outlines. None is reviewed
or accepted as real training truth. All three clips stay in one development family.
A second [KDP gel-growth source](KDP_GEL_DATA.md) now adds nine small publication
panels and uncertain sheets in a separate family. Its sparse figure-relative
clocks, source marks and imaging-domain limits remain explicit. It supplies no
accepted pixel/event labels or raw camera sequence.

## What is actually present

| Source | Local observations | Permitted project interpretation |
|---|---|---|
| [CrystalCV, Zenodo data V1](https://zenodo.org/records/19140132), [software/examples V1](https://zenodo.org/records/20655010) | 29,974 image entries across both archives; 28,279 uncontoured entries represent 27,652 distinct arrays. Also 54 manual rectangle areas, 1,641 automatic output displays and a 28,811-row process/vision table. Both archives CC BY 4.0. | Perovskite development data. No silhouette/inclusion masks, independent KDP evaluation or accepted causal control model. See [source guide](CRYSTALCV_DATA.md). |
| [Cai et al., KDP gel-growth, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210260/) | Nine integer crops from three published figures, including a sparse 0/10/70-minute gel-growth sequence. CC BY 4.0; original marks retained. | Separate publication-panel development candidates. No raw camera pixels, reviewed masks, first-hour origin, event delay or control validation. See [source/import guide](KDP_GEL_DATA.md). |
| [University of Leeds, DOI 10.5518/1684](https://archive.researchdata.leeds.ac.uk/1537/) | 2,123 original microscope JPEGs from nine alpha-LGA runs; 79 source manual crystal-shape fit sidecars. CC BY 4.0. | Real other-material vision development and a source of shape-fit supervision after its geometry/units are checked. The fits are not reviewed pixel masks or defect labels. |
| [Li, Zhang and Ren, KDP study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919635/) | Supplementary PDF and three experimental KDP movies, all decoded; 881 total video frames. CC BY-NC 3.0. | Phase-specific microscopy development examples. Conditions include transition, dissolution and growth; these are not all defect-free tetragonal optical-crystal growth. |
| [Cho et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-47503-1) | One decoded-format ADP crystallization movie and its caption. CC BY 4.0. | Other-material experimental video. The caption explicitly identifies ADP, despite the paper also studying KDP. |
| [CHiMP Detector, Zenodo v1](https://zenodo.org/records/11110373) | 587 real protein-crystal microscope images, 4,463 selected crystal masks and 523 drop masks; source CSV membership preserved. CC BY 4.0. | Auxiliary source with published expert annotations. No exhaustive generic negatives, KDP role/event labels or physical run-group evaluation is established. See [source-mask guide](PUBLISHED_CRYSTAL_MASKS.md). |

Leeds `videos.zip` contains five result/overlay videos; these are preserved as
reference, not imported as raw microscope training sequences. Its `validation.zip`
contains synthetic reconstruction checks, not independent real validation runs.
The large synthetic-image archives have not been downloaded.

The KDP publisher links returned 404/403 during this acquisition. Its media were
then retrieved through the [official PMC cloud service](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/),
with repository MD5 values checked in addition to our SHA256 hashes. The earlier
failed requests remain in dated audit files. No login or access purchase was used.
Preserve CC BY-NC restrictions for the KDP files separately from CC BY sources.

## Local layout and reproducible acquisition

```text
data/external_vision/LGA-LEEDS-2026/experiments.zip
data/external_vision/LGA-LEEDS-2026/README_ILETT_2026.txt
data/external_vision/LGA-LEEDS-2026/videos.zip
data/external_vision/LGA-LEEDS-2026/validation.zip
data/external_vision/KDP-2026/
data/external_vision/ADP-NATURE-2024/
data/real_vision_leeds_v1/manifest.json
data/real_vision_leeds_v1/images.jsonl
data/audits/vision_sources*_20261111.json
```

Download/recheck the pinned source files into a cache, preserving a new report:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_data_sources.py \
  --registry configs/vision_sources.json --cache data/external_vision \
  --fetch --output data/audits/vision_sources_new_check.json
```

Omit `--fetch` for an offline audit. Existing cached bytes are checked; changed
checksums fail rather than silently replacing an accepted source. ZIP members and
CRC are checked before acceptance. No archive is blindly extracted by this tool.
The source registry retains attribution and derivative records; keep them with
any later training export.

## Real-image index and frozen splits

The [import protocol](../configs/leeds_vision_import.json) groups whole runs and
keeps all `20-R1`, `20-R2` and `20-R3` replicates in one conservative source-condition
family. The source identifiers are not independently transcribed sensor readings.
Runs 12 and 14 were used for pilot inspection and remain development/training data.

| Split | Source runs | Images |
|---|---|---:|
| Train | 12, 14, 20-R1, 20-R2, 20-R3 | 1,323 |
| Validation | 16, 22 | 400 |
| Test | 18, 26-R0 | 400 |

This evaluates within-source transfer. It does not supply independent laboratories,
campaigns or KDP test experiments. Exact duplicate images cannot cross splits;
related clips and future augmentations inherit their source run/family split.
Holdout images were decoded for integrity, not inspected for visual tuning.

The archive contains 523 images in run 14; the article describes a 522-image long
sequence. Preserve all source images and record this discrepancy instead of
dropping one to match the prose. Its filenames use an `LGA_alpha_` prefix, unlike
the other runs. The first import's unsupported-prefix failure remains locally in
`data/real_vision_leeds_v1_attempt_1_failed/`; the corrected import is complete.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_vision_data.py \
  --archive data/external_vision/LGA-LEEDS-2026/experiments.zip \
  --protocol configs/leeds_vision_import.json --output data/real_vision_leeds_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/import_vision_data.py \
  --output data/real_vision_leeds_v1 --audit
```

Use an unused output for a new import. The audit re-decodes every original image,
rechecks sidecars, source identity, run membership and sample hashes, and compares
the reconstructed index. Original files stay in the ZIP; the loader reads them
lazily with per-image verification. `images.jsonl` exposes provenance and missing
labels, not a fabricated segmentation training set.

## Timing and labels

Leeds nominal elapsed time uses the paper's five-minute cadence. Per-frame clock
transcription/drop checks are unfinished, so actual acquisition time remains null.
The manual sidecars describe source 3D shape fits, including rendering assumptions;
they are preserved verbatim. Do not reinterpret their distances, volumes or fitted
parameters as independently measured masks, SI quantities or optical quality.

KDP movie metadata from supplementary page S8:

| File | Frames / display time to last frame | Published context and speed-up |
|---|---|---|
| `d5ra09916a2.mov` / S1 | 306 / 10.167 s | Defective-seed transition under tetragonal-phase supersaturation; 14x. |
| `d5ra09916a3.mov` / S2 | 286 / 9.500 s | Monoclinic/tetragonal seeds under slight monoclinic undersaturation; 25x. |
| `d5ra09916a4.mov` / S3 | 289 / 9.600 s | Both seed types at 4.18% monoclinic relative supersaturation; 15x. |

The clips contain text overlays and truncated fibers. Retain phase identities,
visible/occluded regions and source-family grouping during annotation. Keep all
three as development-only until their target and dependency review is complete.
Published speed-up factors support a declared relative-clip time mapping; they do
not establish the growth-run start or first-hour defect coverage. None of these
Leeds/KDP/ADP sources provides controller action-response or optical-quality ground
truth. The separate CrystalCV source adds reported thermal/infusion evidence
whose timing, causality and material limits are described in its own guide.

## Inspectable development replay

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/export_vision_sequence.py \
  --index data/real_vision_leeds_v1 --run-id 12 \
  --output results/leeds_pilot12_v1 --max-frames 30
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/observe_video.py \
  --video results/leeds_pilot12_v1/sequence.avi \
  --config configs/leeds_pilot12_baseline.json \
  --source-info results/leeds_pilot12_v1/source_info.json \
  --run-id leeds_12_development --output results/leeds_pilot12_baseline_v2
```

The export records every resize and original source image in `frame_map.jsonl`.
Display timing stays separate from acquisition timing. The pilot command refuses
test/validation runs for visual tuning. The initial threshold 150 yielded 6/30
valid observations; threshold 180 on the same development clip yielded 30/30.
Both result directories remain. The final outline was visually inspected, but
there are no reviewed mask-error metrics yet. Valid-frame counts are not accuracy.

Next: establish reviewed pixel/event annotations and their uncertainty, freeze
performance budgets and complete relevant real pitch-class evaluation. Synthetic
YOLO fits and the separate source-fit silhouette comparison are already complete
at their documented scopes. Current access is public datasets only; relevant KDP
capture/action/outcome evidence must come from documented public sources or remain
an explicit later gate. The [roadmap](../roadmap.md) keeps predictive control and
the HPC/Qiskit extensions open.
