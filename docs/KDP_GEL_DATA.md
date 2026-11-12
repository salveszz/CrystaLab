# Public KDP gel-growth publication panels

A second KDP study now supplies **nine source-linked publication crops** and nine
uncertain annotation sheets. No accepted labels or human reviews were created.
These are small published images, with their original marks retained; they are
development candidates rather than raw camera frames or a scored evaluation set.
The existing KDP movies, Leeds groups and trained models are unchanged.

The source is [Cai, Ren, Yao and Li (2026), DOI 10.3390/molecules31101744](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210260/),
published under **CC BY 4.0**. The paper identifies tetragonal KDP and describes
secondary formation along needles in silica gel. Its public payload provides
figures and article content; no raw video, segmentation masks or actuator logs
were found in the versioned PMC metadata. The official XML preserves the license.

| Published panels | Count | Context | Clock |
|---|---:|---|---|
| Figure 1b/c photographs | 2 | Solution-/gel-grown crystal microscopy | Not supplied |
| Figure 2a sequence | 3 | In-situ growth in silica gel | Published 0, 10, 70 min relative to its first panel |
| Figure 2c | 1 | Needle-like gel-grown crystal | Not supplied |
| Figure 3g/h/i | 3 | Solution-grown microscopy under different nominal conditions | Not supplied |

The nominal medium/chemistry descriptions are source context, not measured
process traces. A solution-grown specimen photograph does not establish that the
camera viewed live growth through solution. The gel domain differs from the
pitch's stirred seeded tank. The paper's secondary-growth interpretation does
not supply per-instance polygons or an independently reviewed visible-onset time.

## Import and review status

`configs/kdp_gel_panels.json` freezes original JPEG hashes/dimensions, integer crop
coordinates, context, permission, family and clock semantics. Selected outputs
are **200–230 px wide / 151–171 px high**. No resizing, colour change, annotation
removal or inpainting occurs. Text, scale bars, ROI boxes and crossing arrows
remain visible. Schematics, diffraction patterns, plots, SEM and dissolution
panels are excluded from these camera-image candidates.

All nine belong to `KDP-GEL-2026-study`, assigned to development/training candidates
only. They are not nine independent experiments; specimen/run relationships are
unknown. The source adds a distinct published study, not a frozen independent
KDP evaluation cohort. Check image overlap and applicable label definitions before
any later fit. The default model and training cohorts are not modified by import.

The local paths are:

- `data/external_vision/KDP-GEL-2026`: original figures, versioned PMC JSON/XML,
  source page and acquisition records.
- `data/real_vision_kdp_gel_v1`: nine lossless PNG crops, source records, frozen
  protocol and producer snapshot.
- `data/annotations_kdp_gel_v1`: nine empty, explicitly uncertain review sheets.
- [Contact sheet](../results/kdp_gel_preview_v2/contact_sheet.png): native-sized
  images with source IDs and unreviewed status, visually checked locally.

The isolated specimen views offer candidate outline examples. The growth sequence
has low contrast and attached/overlapping structures; the needle view is clipped.
Main-seed identity, secondary-object boundaries and internal optical appearances
remain unresolved. A blank sheet is not a reviewed negative image. Actual review
must retain uncertainty where source marks, clipping or resolution hide boundaries.

```bash
.venv/bin/python scripts/review_annotations.py \
  --bundle data/annotations_kdp_gel_v1 --read-only
```

Omit `--read-only` only for actual annotation work. Drafts and previews create no
review receipts. The existing [review/export workflow](ANNOTATION_REVIEW_UI.md)
still requires explicit per-image review; export of this unreviewed bundle was
actually refused before an output folder was created.

## Clock and validation limits

Source Figure 2a marks **0/600/4,200 figure-relative seconds**. Its first panel is
not a documented growth-run start. Acquisition timestamps, first-hour membership,
continuous negative observation, event-onset intervals, pixel calibration,
temperature/rotation/flow logs, commands and quality labels remain null. Do not
infer alert recall/delay or a physical response from these three sparse stills.

The [checkpoint](KDP_GEL_CHECKPOINT.json) preserves actual source/artifact identities
and verification. An independent Pillow decoder reconstructs all nine crops from
the original JPEGs with **zero pixel difference** from the OpenCV import. It also
checks the official XML/figure MD5 values, source DOI/license, external protocol,
grouping/clocks, unknown fields and unreviewed bundle records without importing
the producer. Source auditing decodes all three original JPEGs and verifies five
pinned assets. Read-only store access to all nine sheets leaves bundle bytes
unchanged; this is software verification, not human annotation or a browser test.
The full regression suite passes **434 tests / 315 subtests in 113.79 s**. The
focused panel/source checks pass 12 tests / 6 subtests, covering crop integrity,
provenance reconstruction, grouping, unknown-label export and image formats.

The first targeted test expected the review store's status name to be `uncertain`;
the existing interface correctly calls untouched sheets `proposed` and separately
sets their uncertainty flag. The test was corrected to check both fields. An
initial contact-sheet attempt failed on an unavailable font; v2 uses Pillow's
bundled font. Both attempts are retained in verification history.

## Reproduction

Raw images and generated outputs are Git-ignored. The new source is pinned in
[the vision registry](../configs/vision_sources.json). The existing source-audit
fetch command can populate missing public assets; it checks the registry's other
sources too. The importer itself performs no network requests.

With the pinned assets cached, select unused output paths:

```bash
.venv/bin/python scripts/import_publication_panels.py import \
  --output data/real_vision_kdp_gel_new
.venv/bin/python scripts/import_publication_panels.py audit \
  --index data/real_vision_kdp_gel_new
.venv/bin/python scripts/import_publication_panels.py bundle \
  --index data/real_vision_kdp_gel_new --output data/annotations_kdp_gel_new
.venv/bin/python scripts/audit_publication_panels.py \
  --index data/real_vision_kdp_gel_new --bundle data/annotations_kdp_gel_new \
  --output results/kdp_gel_audit_new.json
```

Accepted training still needs compatible reviewed outlines. The core next action
remains the original small KDP review pack, followed by the bounded real/mixed
comparison. These additional panels widen the available source material; reliable
inclusions, face/asymmetry labels, first-hour detection and physical control
remain open pitch requirements.
