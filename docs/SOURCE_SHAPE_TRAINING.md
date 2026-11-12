# Real images with author-derived silhouette targets

The Leeds archive contains 79 manual 3D crystal fits linked to original microscope
images. The authors describe fitting projected wireframes to those images using
their interactive tool. These are published shape measurements, distinct from
our six unreviewed computer-generated outline proposals. See the
[primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13070466/) and
[source dataset](https://archive.researchdata.leeds.ac.uk/1537/).

The project now derives **66 silhouette proxies** from those fits: **54 training
and 12 validation images**. The remaining 13 fitted test images are excluded;
the original full-image test runs remain reserved. A projected shape is useful
supervision for a separate main-crystal silhouette task. It is not an independently
reviewed pixel boundary, secondary-crystal/inclusion label or KDP accuracy result.
The material is alpha-LGA. No local human review receipt is created.

## Projection and provenance

The [projection policy](../configs/leeds_shape_projection.json) pins the source
index/archive and admits complete orthorhombic plane sets with an explicit pose.
CrystaLab's existing half-space geometry reconstructs the polyhedron. A standard
orthographic projection uses the recorded camera distance/FOV and mean transformed
vertex depth. The native center-square crop is resized to 512 × 512 pixels.
The mask is the filled integer convex hull of nearest-pixel projected vertices.
This raster contract avoids numerical ambiguity from nearly collinear corners.

Each target preserves its original image/fit references, camera member and hashes,
source vertices, crop/resize, projection scale, split and condition family. Source
model lengths remain uncalibrated units here. Refracted back edges and visible
faces are not separate pixel targets in this experiment.

An independent reconstruction uses triple-plane intersections and Rodrigues
rotation rather than the exporter/geometry engine. It reconstructs all 66 targets,
with exact mask agreement and maximum vertex discrepancy below 8.3e-11 source
units. This proves geometric conversion consistency; it does not prove agreement
with newly reviewed image boundaries. Earlier projection attempts remain local:
numerically different corner lists motivated the explicit raster contract before
training or validation scoring.

Method conventions were checked against the paper-linked
[CrystalSizer3d source at the recorded commit](https://github.com/Shape4PPD/crystalsizer3d/tree/62cc090e09f1f794582a8099c9f230f7dcffced8).
The cached reference code carries CC BY-NC 4.0; it is not a projection/training
dependency. The source images/manual-fit data retain their CC BY 4.0 attribution.
The independent CrystaLab implementation uses its existing geometry engine.

## Separate training contract

The [frozen experiment](../configs/source_silhouette_experiment.json) compares:

| Training condition | Training images | Epochs | Planned updates |
|---|---:|---:|---:|
| Synthetic only | 84 | 10 | 210 |
| Real images with source-projected targets | 54 | 15 | 210 |
| Mixed | 138 | 6 | 210 |

Every fit starts from the same pinned YOLOv8n-seg COCO weights and uses CPU,
256-pixel input, batch four, fixed seed, full-resolution mask supervision,
non-overlapping mask targets, no warmup and a constant learning rate of 0.001.
The final batch can be smaller than four. Equal planned update counts do not
equalize per-domain sample exposure or establish a general learning curve.

All three fits have the same **56 validation images**: 12 real source-fit proxies
from two alpha-LGA condition families and 44 synthetic images. Whole source
families retain their existing splits, including all 20-degree replicates together.
These are within-source development comparisons, not an independent laboratory
or KDP test. Training validation is retained for logging; the comparison uses the
**last checkpoint**, selected before fitting, rather than choosing a best epoch
from these results. No validation retuning or automatic promotion is allowed.

The single class is `main_crystal_silhouette`. Synthetic seed proxies and the
source authors' fitted main crystal support that declared target. This does not
relabel protein masks, infer unannotated secondary/inclusion truth or use local AI
proposals as reviewed labels. The ordinary pitch exporter continues to require
real review receipts; the pitch observer explicitly rejects auxiliary weights.

Evaluation selects the highest-confidence instance at the frozen 0.25 threshold.
It never chooses a detection by overlap with its target. Missing predictions
remain zero masks and zero IoU. A mean mask computed from the 54 real training
targets supplies a static baseline. Reports retain image-weighted and family-macro
proxy IoU, Dice, missing masks, extra predictions and area error per evidence kind.
Extra predictions concern this one-target task, not confirmed defects/false alerts.

## Actual comparison and verification

The [checkpoint](SOURCE_SHAPE_CHECKPOINT.json) retains the three completed fits,
their distinct last-weight identities, source/data hashes and execution evidence.
On the common validation targets:

| Method | Real proxy IoU, 12 images | Synthetic proxy IoU, 44 images | Missing real / synthetic masks |
|---|---:|---:|---:|
| Static real-training template | 0.6256 | 0.6989 | 0 / 0 |
| Synthetic only | 0.6203 | 0.8783 | 1 / 0 |
| Real source-projection only | 0.9073 | 0.0000 | 0 / 44 |
| Mixed | 0.9255 | 0.9078 | 0 / 0 |

These are image-weighted means. Real families contain six images each, so their
macro means agree. Synthetic family-macro IoU is 0.7048 / 0.8688 / 0.0000 / 0.8970
in the same order. Extra predictions are respectively 0 / 1 / 2 / 0 on real images
and 0 / 11 / 0 / 5 on synthetic images. All failures remain in the reports.

Mixed training has the highest aggregate proxy agreement in this fixed experiment.
The comparison is not uniform across real families: real-only scores 0.9473 versus
mixed 0.9402 on condition 16; mixed scores 0.9107 versus real-only 0.8672 on condition
22. Synthetic-only slightly loses to the static baseline on the real aggregate.
The real-only model misses all 44 synthetic targets at the frozen threshold.
Neither that failure nor the inference threshold was retuned after scoring.

See the [comparison figure](../results/source_silhouette_figures_v1/proxy_comparison.png)
and [six training projections](../results/source_silhouette_figures_v1/training_projection_preview.png).
Each fit used two CPU threads; concurrent elapsed times were 666–675 seconds,
including validation. These timings are not isolated performance benchmarks.

Full regression passed **411 tests / 300 subtests** in 215.54 seconds. Independent
checks reconstruct all 66 source-fit targets, the training-only template, frozen
training configurations/epoch counts and all 224 selected-prediction score records.
The auditor does not rerun neural inference or reconstruct raw detection selection;
the frozen implementation and a targeted test enforce highest-score selection.
The later audit/plot CLIs were also run on the actual artifacts and the figures
were visually inspected. No human label reviews or physical commands were created.

Default pytest discovery is now limited to `tests`. The initial full run discovered
a downloaded upstream example and stopped on its missing `wx` import. That failure
is retained; no upstream geometry implementation is a runtime dependency.

## Reproduce

Raw data, targets, weights and results stay Git-ignored. Use unused outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/project_source_shapes.py \
  --output data/leeds_shape_proxies_next
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/source_silhouette.py prepare \
  --projection data/leeds_shape_proxies_v3 --synthetic data/yolo_optical_refinement_v1 \
  --mode mixed --output data/source_silhouette_mixed_next
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/source_silhouette.py train \
  --dataset data/source_silhouette_mixed_v1 \
  --config configs/source_silhouette_train_mixed.json \
  --output results/source_silhouette_fit_mixed_next
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/evaluate_source_silhouette.py \
  --output results/source_silhouette_evaluation_next
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_source_silhouette.py \
  --evaluation results/source_silhouette_evaluation_v1 \
  --output results/source_silhouette_execution_audit_next.json
```

The frozen study binds its original local paths/hashes. Reproduced datasets or
runs need a new declared study with their identities; the old protocol remains
unchanged. The source images, manual fits, existing annotation proposals and final
test groups are preserved.

The next accepted-real-label gate still requires pixel/event review and relevant
independent KDP evidence. Action response, face/asymmetry/inclusion interpretation,
first-hour clocks, equipment/outcomes, HPC and Qiskit remain in the
[pitch-led roadmap](../roadmap.md).
Current user access is public datasets only. Further data work must preserve
public-source licensing and task/clock provenance; a local growth rig is not assumed.
