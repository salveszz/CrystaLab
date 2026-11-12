# Local image review

The local editor shows an original microscope image and the computer's proposed
outline. A person checks whether the outline follows the crystal's visible edge,
moves points where necessary, and records a review for that image. No code review
is involved. These checked outlines can then become real segmentation labels for
the pitch's camera model.

This is a still-image annotation tool. The separate
[operator replay console](OPERATOR_REPLAY.md) now adds video/process display,
proposal review and simulated pause/manual resume. A separate
[adaptive software diagnostic](ADAPTIVE_SUPERVISION.md) now compares changing
forecast state and bounded pauses. Its [operator forecast view](OPERATOR_FORECAST.md)
now preserves that context with explicit reviews. Live equipment/operator
response validation remains open.
The six existing Leeds run-12 proposals still have **zero actual human reviews**.
They are alpha-LGA development images; they do not establish KDP accuracy.

## Open the existing proposals

From the repository root, with the existing environment and local data:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/review_annotations.py \
  --bundle data/annotations_leeds12_v1
```

Open `http://127.0.0.1:8765` in a local browser. Stop the server with Ctrl+C.
Use `--port` if that port is occupied. Add `--read-only` to inspect without
allowing any label, receipt, revision or export writes. No images are uploaded.
The interface uses Python's standard-library HTTP server and buildless local
HTML/CSS/JavaScript; the annotation pipeline requires the `vision` dependencies.
PySide6 is only an optional browser-QA tool and is not needed to use the editor.

The repository contains preparation and import instructions rather than the raw
images. On a fresh clone, follow [real data](REAL_VISION_DATA.md) and
[annotations](ANNOTATIONS.md) first. Only prepared training bundles can be opened;
final evaluation annotation requires a separately frozen protocol.

## Check one image

1. Start with **Focus outline**, then **Fit image** to inspect the whole field.
   Scroll to zoom, use **Pan** or Space + drag to move, and **Hide outlines** to
   see an edge without the colored line. The pixels retain their native size;
   screen zoom does not change saved coordinates.
2. Drag a point to move it. Click an edge to insert a point; Delete removes the
   selected point. **New outline** adds an object: click its corners, choose its
   label, then press Enter or **Finish outline**. **Undo** restores prior edits.
   **Remove selected outline** removes an object rather than a single point.
3. Label supported visible objects as **Main crystal**, **Other crystal** or
   **Possible inclusion**. These map to `seed`, `secondary_crystal` and
   `inclusion_candidate`. A reflection, bubble or unclear speck must not silently
   become a crystal or inclusion label. Follow the [annotation rules](ANNOTATIONS.md).
4. If an edge, identity or whole image is unclear, mark the relevant uncertainty
   and **Save draft**. Drafts, including uncertain drafts, never count as reviews
   or accepted training labels. Moving between images automatically saves valid
   edits; finish or cancel an unfinished outline first. Invalid geometry stays
   unsaved with an error, so it can be corrected or discarded.
5. Once the image and all visible target outlines have actually been checked,
   enter the reviewer's name, check the acknowledgment and press **Record review**.
   Empty images need a separate acknowledgment that no target objects are visible.
   Editing clears the acknowledgment. Uncertain images or objects cannot be reviewed.

A receipt binds the reviewer, UTC time, source sample and exact image/annotation
hashes. A name is a traceable attestation, not identity authentication. Viewing,
draft saving, navigation and export do not manufacture reviews.

## Corrections and export

Reviewed images are locked. **Revise selected image** creates a sibling bundle
such as `annotations_leeds12_v1_revision_001` and opens that version. The selected
image needs a fresh review; other unchanged valid receipts are copied. Original
files and earlier receipts stay intact. Stale receipts are not carried forward
as valid reviews. The footer shows the active bundle path; use that path with
`--bundle` when restarting work on a revision. Another open page must reload
before saving against the changed version.

**Export reviewed labels** creates an unused sibling directory ending in
`_reviewed_001`, `_reviewed_002`, etc. It uses the existing guarded exporter:
only reviewed images are included, unreviewed images are counted and excluded,
and stale receipts block export. Original images, edited annotations, YOLO polygon
labels, receipts, license and source-family/split metadata are retained. Export
does not train a model or establish an event/quality label.

Writes require the current bundle and annotation hashes. A conflicting edit in
another window is refused rather than overwriting it. The server binds only to
`127.0.0.1`, checks local Host/Origin and a per-server session token, and serves
only the selected bundle and editor assets. Keep one server per working bundle.

## Verification and next gate

The [checkpoint](ANNOTATION_REVIEW_CHECKPOINT.json) records API/regression checks
and 20 offscreen QtWebEngine browser checks. These cover native canvas gestures,
uncertain drafts, explicit fixture reviews, negative acknowledgment, export,
revision, navigation and a 390-pixel mobile layout. Actual real images were opened
read-only at native 2880 × 2160 dimensions; before/after hashes prove the entire
bundle remained unchanged. Test receipts refer only to generated fixture pixels.
Browser-QA servers were stopped after verification.

Reproduce in an environment with optional PySide6 installed, using a new output:

```bash
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tests/check_annotation_review_ui.py \
  --output results/annotation_review_ui_next \
  --real-bundle data/annotations_leeds12_v1
```

Next, a person must check the six proposals and settle the boundary/class rules.
Then expand the training pilot across the frozen development families, collect
contiguous event and negative periods, freeze evaluation budgets and run the
real/synthetic experiment. No reviewed real training cohort, independent real
accuracy, first-hour detection or physical control benefit is claimed here.
