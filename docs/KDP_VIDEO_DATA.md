# Public KDP video index and annotation pack

The three previously acquired public KDP movies now have an audited **881-frame
native-pixel index** and a **six-image annotation pack**. Two sheets contain
uncertain computer outlines; four need manual outlines. All six remain unreviewed
and are excluded from accepted training exports. These are derived observations
from the existing movies, not newly acquired independent experiments.

The [import protocol](../configs/kdp_video_import.json) preserves the original
MOV/SHA256, source attribution, CC BY-NC 3.0 license, phase captions and whole-study
grouping. Captions were checked on supplementary page S8 of
[Li, Zhang and Ren's primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919635/).
The [checkpoint](KDP_VIDEO_CHECKPOINT.json) binds actual import, audit, UI and
regression evidence to exact source/artifact hashes.

| Clip | Context from the source caption | Frames | Last display time | Published speed-up | Selected native frames |
|---|---|---:|---:|---:|---|
| S1 | Defective seed undergoing phase transition under tetragonal-phase supersaturation | 306 | 10.1667 s | 14× | 0, 305 |
| S2 | Co-seeded monoclinic/tetragonal fibers under slight monoclinic undersaturation | 286 | 9.5000 s | 25× | 0, 285 |
| S3 | Co-seeded fibers at published 4.18% supersaturation relative to monoclinic KDP | 289 | 9.6000 s | 15× | 0, 288 |

All clips belong to one conservative `KDP-2026-study` family, assigned to training/
development only. Their physical run/specimen independence is not established.
They supply no independent KDP validation/test cohort. Existing Leeds holdouts
and synthetic final groups are unchanged.

## Check one image

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/review_annotations.py \
  --bundle data/annotations_kdp_v1
```

Open the printed loopback URL. Add `--read-only` for inspection without writes.
The editor shows the original image, source phase, video time, published-speed-up
estimate and license. It explicitly states that acquisition timestamps and the
growth-run start are unknown. No image is uploaded.

Start with the first S1 frame: compare the dashed outline with the visible
crystal edge. Move or add points where necessary. The crystal is clipped at the
native top edge; this does not reveal its missing extent. The final S1 frame has
attached growth and uncertain instance boundaries, so leave unresolved identities
uncertain rather than accepting the entire connected shape as one crystal.

S2/S3 start as blank **unknown annotation sheets**, with image uncertainty set.
Draw supported visible instances using the existing **New outline** tool. Resolve
which object is the main seed and distinguish co-seeded crystals without calling
them new parasitic nucleation. The water interface, source text, reflections and
internal phase-transition texture must not automatically become crystal or
inclusion labels. These stills provide no onset/negative-period annotation.

Only clear uncertainty after actually resolving it. A real reviewer checks the
original image and every supported target, enters their name and explicitly
records the review. Draft navigation, previews and import do not create reviews.
Blank sheets must not be accepted as negatives just because no polygon exists.
If interpretation is unclear, retain the draft. The
[editor guide](ANNOTATION_REVIEW_UI.md) describes point editing, revision and export.

## Provenance and clocks

The index decodes every frame sequentially and records its native BGR SHA256,
dimensions, original clip hash and rational presentation timestamp. Bundle PNGs
preserve the selected decoded pixels exactly, with no resizing, crop or overlay
removal. The source protocol and producer code are snapshotted. An audit also
requires an external frozen protocol, preventing a self-consistent edit of index
metadata from silently changing phase, license or family provenance.

Container presentation time is display time. Multiplication by the published
speed-up yields only an inferred relative elapsed time since clip start:
142.3333/237.5/144.0 s at the respective final frames. It does not establish actual
acquisition timestamps, growth-start time, first-hour membership or per-frame
actuator response. Pixel scale, temperature/rotation/flow measurements and quality
labels remain null. Nominal publication conditions are not invented sensor logs.

The independent audit reconstructs all 881 source-frame pixels and presentation
clocks without importing the producer. All six annotation PNGs additionally
match an independent FFmpeg CLI decode exactly. Counterfeit record hashes, changed
source bytes, duplicate payloads and altered license protocols are tested.
The UI browser checks all six native images read-only, preserves their phase
context on navigation and verifies that real bundle files stay byte-identical.
Actual human reviews remain zero.

## Reproduction

The import requires the `vision` Python dependencies and `ffprobe`; the independent
decoder comparison also requires the `ffmpeg` command. Versions are recorded in
the index and decoder reports. No tool downloads are performed by these commands.

With the pinned files in `data/external_vision/KDP-2026`, choose unused outputs:

```bash
.venv/bin/python scripts/import_real_video.py --output data/real_vision_kdp_new
.venv/bin/python scripts/import_real_video.py \
  --index data/real_vision_kdp_new --audit
.venv/bin/python scripts/import_real_video.py \
  --index data/real_vision_kdp_new --bundle --output data/annotations_kdp_new
.venv/bin/python scripts/annotate_vision.py preview \
  --bundle data/annotations_kdp_new --output results/kdp_previews_new
.venv/bin/python scripts/audit_kdp_review_pack.py \
  --index data/real_vision_kdp_new --bundle data/annotations_kdp_new \
  --cross-decoder results/kdp_decoder_new.json --output results/kdp_audit_new.json
```

`data/real_vision_kdp_v2` is the accepted local index. The first import remains
preserved; its frame-index bytes match v2, which adds source snapshots and the
external-protocol guard. The raw movies and generated images remain Git-ignored.
Fresh acquisition instructions are in [real vision data](REAL_VISION_DATA.md).
No new source download, author contact or permission purchase occurred here.

After actual reviews, the existing guarded exporter can supply image-training
labels with their source family and permissions. Mixed training can then use
those reviewed examples; it still needs a separately frozen, independent real
evaluation cohort. Phase-specific fibers do not by themselves validate the
pitch's tetragonal tank-growth defects, visible inclusions, face asymmetry,
first-hour detection, physical control or optical-quality claims.
