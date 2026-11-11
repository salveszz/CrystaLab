# Real-image polygon annotation workflow

An annotation is a label attached to an image. Here, an outline tells the model
which pixels belong to the crystal. Review means checking that the outline
follows the visible crystal edge and correcting it where needed, for example
where a reflection or bubble confused the automatic proposal. It does not require
reviewing code. A separate event label records what happens over time, such as
when a second crystal first becomes visible.

The first bundle contains six alpha-LGA training images from source run 12,
frames 0, 40, 80, 120, 160 and 199. The original decoded image pixels are preserved
at full resolution. OpenCV proposed a seed outline in each image. **None of these
proposals has a human review receipt or is exported as reviewed training truth.**

Source manual 3D shape fits remain separate. This workflow annotates visible
projected outlines; it does not establish 3D volume, optical quality or event onset.
The [real-data guide](REAL_VISION_DATA.md) retains source attribution and split rules.

## Prepare and inspect

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/annotate_vision.py prepare \
  --index data/real_vision_leeds_v1 --run-id 12 \
  --frames 0 40 80 120 160 199 \
  --baseline configs/leeds_pilot12_baseline.json \
  --output data/annotations_leeds12_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/annotate_vision.py preview \
  --bundle data/annotations_leeds12_v1 \
  --output results/annotations_leeds12_preview_v1
```

Existing outputs are refused. Preparation currently accepts training runs only;
annotation of final evaluation runs needs a separately frozen protocol. Proposal
resizing is used only by the baseline; polygons map back to the original pixels.
Every preview is marked proposed/not reviewed and creates no receipt.

The bundle layout is:

```text
data/annotations_leeds12_v1/
  manifest.json
  images/12_0000.png
  annotations/12_0000.json
  reviews/
```

The JSON annotations use the polygon structure supported by
[LabelMe](https://github.com/wkentaro/labelme). Open the annotation JSON in a local
polygon editor that preserves its relative image reference, or edit the points
directly. No online upload is needed by this workflow. The source image, dimensions,
source ID, run and family must remain unchanged.

## Annotation rules for the pilot

- Outline the seed's outer visible silhouette, including its visible border.
  Preserve the actual projection rather than fitting a rectangle or reconstructing
  an unseen face. Shadows and reflected duplicates are not additional seed objects.
- Inspect all visible target instances. The permitted labels are `seed`,
  `secondary_crystal` and `inclusion_candidate`. Do not identify a bubble or speck
  as a secondary crystal without supporting evidence; uncertain identity remains
  flagged and excluded from reviewed training export.
- Keep an uncertain shape's `flags.uncertain` true. The exporter rejects uncertain
  polygons. Review the observation/target definition rather than silently converting
  uncertainty into a healthy/negative label.
- An empty proposal means the baseline did not produce an outline. It is not a
  negative example. Only an explicitly inspected image with no visible target
  instances may receive a reviewed-negative acknowledgment.
- Keep polygon coordinates finite, distinct and within the original image.
  Self-crossing and degenerate polygons are rejected. Save any relevant visibility
  or truncation notes in the annotation/receipt; do not infer full dimensions for
  an object truncated by the field of view.

These rules and the class coverage need reviewer assessment before acceptance
budgets and real evaluation are frozen. The six seed proposals do not supply
reviewed examples of every event or class from the pitch.

## Record a review and export

After a reviewer has inspected the image and current polygons, that operator can
record the individual attestation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/annotate_vision.py review \
  --bundle data/annotations_leeds12_v1 --annotation 12_0000.json \
  --reviewer 'reviewer name' --ack-image-reviewed --note 'review notes'
```

For a verified empty image, add `--empty-reviewed`. There is no automatic
approve-all operation. A receipt records reviewer, time and exact image/annotation
hashes. Changing a polygon or source image after review invalidates that receipt.
Previous receipts are never overwritten; use a new bundle for a revised review.
This is a traceable operator attestation, not authentication of reviewer identity.

Export the accepted subset into an unused directory:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/annotate_vision.py export \
  --bundle data/annotations_leeds12_v1 --output data/reviewed_leeds12_v1
```

The command fails when no reviews exist. It exports accepted images, exact edited
annotation JSON and YOLO polygon labels, with receipts and original source-family
metadata in the manifest. Unreviewed images are counted and excluded explicitly.
Source material, license and split assignments remain attached. A reviewed seed
polygon does not become an optical-quality or defect-prevention label.
