# Anchored projected-edge diagnostic

The first projected-edge measurement diagnostic is complete on **96 artificial
frames across eight declared cases**. It measures directional silhouette advance
after image-reference registration, retains unavailable views, and emits a
review-only proposal for the supported unequal-advance fixture. Attachment,
comparable object pose and complete visibility are **external input assertions**;
the fixture construction supplies them here. They are not learned predictions
or human reviews. The six real KDP sheets lack this evidence and produce no
numerical edge measurements or proposals.

This implements a bounded measurement prerequisite for the pitch's asymmetry
work. Visible-face identification, reviewed real accuracy and rotation/flow
control remain open. The current YOLO observer, default model, PID and operator
console are unchanged. This diagnostic's proposal records are standalone; the
existing operator/action path does not consume them.

## What is measured

Three distinct coloured references locate a planar image frame. Their centroids
come from the RGB pixels. An orientation-preserving least-squares similarity
maps current pixel coordinates into the initial reference frame. Its residual
and scale must satisfy declared limits. An anchor is a fixed combination of the
three reference points, independent of the seed mask's centroid.

For foreground points `p` and anchor `a`, the four supports are:

```text
h(+x) = max(p.x - a.x)    h(-x) = max(a.x - p.x)
h(+y) = max(p.y - a.y)    h(-y) = max(a.y - p.y)
advance = current support - earlier comparable support
difference = abs(positive advance - negative advance)
fraction = difference / (positive advance + negative advance)
```

The axes are reference-image x/right and y/down. A support may touch a vertex or
an edge; it is not an identified crystallographic face or Miller plane. The
outputs are pixel advances over source-clock windows, without physical scale or
growth-time conversion. Mask errors, visibility and optics still affect them.

The [frozen protocol](../configs/projected_edge_diagnostic.json) declares a
two-source-second window, maximum 1.5 s observation gap, minimum total advance
4 px, at most 1.5 px negative advance for raster uncertainty, a 4 px difference
and 0.5 fraction gate, and two consecutive qualifying windows. These are fixture
test assumptions, not approved real performance budgets. Missing/uncertain views,
new pose/reference IDs and long gaps clear comparison history and persistence.
Binary seed masks must be connected, sufficiently large and unclipped, with the
independent anchor inside them.

The coloured-marker detector is deliberately limited to these controlled images.
The three-point similarity is not a 3D pose estimator or a liquid/vessel optical
calibration. OpenCV's [limited affine transform reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
describes this four-degree-of-freedom transformation class. Its
[PnP documentation](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html)
instead requires calibrated camera parameters and corresponding 3D/object and 2D
image points to estimate object pose. Those prerequisites are absent from the
real KDP pack. Stationary external markers alone do not prove that the crystal
itself retained its pose or attachment.

## Actual execution and baselines

Each case has 12 frames at 1 fixture FPS, covering 0–11 source seconds. RGB images
feed the existing OpenCV `SeedTracker`; construction masks do not supply its
measurements. The coloured references remain separate from the tracked seed.
All 96 seed masks are observed; the qualified edge method accepts 60 frames and
forms 42 causal comparison windows. The fixtures are opaque rendered polygons,
not a photorealistic or independently held-out camera dataset.

| Case | Qualified frames | Qualified review proposals | Result |
|---|---:|---:|---|
| Symmetric, fixed view | 12 | 0 | Numerical imbalance at round-off level |
| Unequal advance, attachment asserted | 12 | 1 at frame/time 3 | First two eligible windows trigger the declared review gate |
| Symmetric with camera translation/rotation/scale | 12 | 0 | Maximum registered imbalance 0.24010 px, below frozen 2.5 px check |
| Symmetric plus free translation | 0 | 0 | Attachment not established |
| Changed/unverified object pose | 4 | 0 | Eight later views unavailable |
| Missing reference marker | 8 | 0 | Four unavailable views; history must mature again |
| Occluded silhouette | 8 | 0 | Four unavailable views; history must mature again |
| Non-similarity view change | 4 | 0 | Eight later reference fits fail the residual gate |

Maximum accepted camera-motion reference RMS is 0.09953 px against the declared
0.75 px limit. The review proposal retains baseline/current frame IDs, clocks,
image/mask/context hashes and the affected image axis. It asks for reference,
pose and rotation/circulation review. It neither confirms a physical defect nor
contains an actuator value or command. No operator review was recorded.

Two simple baselines use the same masks, horizon, thresholds and persistence:

- An unregistered fixed image origin raises a review at frame 3 for the unequal
  case, camera-motion case and free-translation case. The latter two illustrate
  misleading apparent edge motion in these fixtures.
- Recentring every mask on its own centroid raises no reviews, including the
  unequal-advance case. Recentring removes the directional motion relative to the
  independent attachment reference that this experiment is intended to measure.

The [PNG](../results/projected_edge_figures_v2/projected_edges.png) and
[SVG](../results/projected_edge_figures_v2/projected_edges.svg) show source RGB,
observed masks and these comparisons. The first and third raw images match;
reference/anchor overlays differ because their input qualifications differ.
The green vertical line marks the one qualified review proposal. Generated
artifacts remain Git-ignored; the plotting script is tracked.

## Why the preconditions are necessary

The diagnostic preserves an exact ambiguity, rather than solving it with a mask
centroid assumption. One construction keeps an independent object origin fixed
and advances its right silhouette by 4 px per step while its left remains fixed.
Another expands symmetrically by 2 px per side and translates the object by
2 px per step. Both give **identical RGB images and observed masks in all 12
paired frames**, despite different underlying construction laws.

The correct distinction in this experiment comes from externally supplied
attachment assertions. It is not an automatic classifier of growth versus
translation. Incorrect assertions could still produce a misleading result.
Real use therefore needs source-supported mount/reference evidence, pose and
visibility observations, calibration through the intended optics and independently
reviewed boundaries. A coloured reference outside the crystal does not supply
these by itself.

All six existing native KDP sheets were checked with missing external context.
The method returns `missing_external_view_reference`, null supports and zero
proposals for every sheet. Their pixels, annotations and review files stay
byte-identical. No reference, physical clock, facet label, seed mask, quality
measurement or action response was invented for those images.

## Independent verification and retained failure

The [checkpoint](PROJECTED_EDGES_CHECKPOINT.json) binds sources, protocol, runs,
raw artifacts and logs. `scripts/audit_projected_edges.py` imports neither the
tracker nor fixture producer. Pillow decodes the saved pixels; the gray fixture
body reconstructs the observed seed masks independently. An SVD/Procrustes fit
checks the producer's linear least-squares registration. The audit reconstructs
supports, naive baselines, causal windows, qualification decisions, persistence,
proposal contents, the identical-image pairs and all real-image refusals.

Maximum support and transform-matrix disagreements are **1.23e-12 px** and
**1.25e-12**, respectively. All frozen checks pass. The complete regression suite
passes **444 tests / 323 subtests in 113.87 s**; the focused checks pass
10 tests / 8 subtests. No physical rig, trained face detector or live camera is
tested by these results.

The first v1 run is retained. Its independent audit found an equality-boundary
disagreement: one fit produced total advance `4.0`, another
`3.9999999999999716`. The corrected v2 implementation treats differences within
1e-9 px and fractions within 1e-12 as numerical equality. Pixel thresholds,
cases, nominal comparison windows and acceptance limits are unchanged. Raw
support values remain unrounded. All source images and observed masks are
unchanged between versions; proposal outcomes are unchanged. Both early failed
audit logs and the later successful independent audits are preserved.

The first attempt to run the full suite was not executed because automatic
approval review hit an account usage limit. A later retry through the normal
approval path succeeded; this did not require a test workaround.

## Reproduce and continue

Use an unused output directory and the original six-image KDP pack pinned by the
protocol. No files are downloaded and no hardware is contacted:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_projected_edges.py \
  --output results/projected_edge_new
.venv/bin/python scripts/audit_projected_edges.py results/projected_edge_new \
  --output results/projected_edge_audit_new.json
.venv/bin/python scripts/plot_projected_edges.py results/projected_edge_new \
  --output results/projected_edge_figures_new
```

Next obtain reviewed visible edges/faces and supported reference/pose evidence
for a bounded real comparison. Evaluate current learned-mask errors separately;
this diagnostic uses a controlled OpenCV baseline. Actual optical distortion,
marker correspondence, attachment/pose quality and timing need their own tests.
Only then can an evaluated measurement feed the operator/supervisory path.
Physical rotation/circulation responses, first-hour detection and effects on
crystal quality remain separate pitch requirements. The public-data annotation
work in [Next steps](NEXT_STEPS.md) remains the core gate.
