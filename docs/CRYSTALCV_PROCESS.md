# Public process/image audit and development examples

The public-data route now has **463 past-only forecasting candidates** from one
MAPbBr3 growth run: 482 native images and 28,811 source log rows over about
40.21 hours. This is a development resource for camera measurements and process
history. It supplies no reviewed KDP masks, independent evaluation, measured
pump response or demonstrated control benefit. No model was fitted in this step.

The accepted export is `results/crystalcv_process_audit_v2`, under frozen
[`configs/crystalcv_process_audit_v2.json`](../configs/crystalcv_process_audit_v2.json).
Its [checkpoint](CRYSTALCV_PROCESS_CHECKPOINT.json) binds sources, results,
independent reconstruction, retained earlier attempts and regression evidence.

## Source lineage and reuse

Images and the CSV come from the already acquired CC BY 4.0
[CrystalCV Zenodo archive V1](https://zenodo.org/records/19140132), folder
`Source_datasets/1_BulkAnalysis/MA/A_230326_0.1mmh`.
The original experiment is described by [Haruta et al., Nature Synthesis (2024)](https://www.nature.com/articles/s44160-024-00576-8).
Its [public reference repository](https://github.com/Yuki-Haruta/2024NatureSynthesis-FRC)
contains the matching workbook and acquisition/control source. We read selected
versioned files as references; no author notebook, camera, pump or notification
code was executed. The repository's license metadata is absent. Its additional
workbooks/outcomes are discovery leads, not an adopted training cohort. The
article and each data archive have separate reuse terms.

The acquired reference workbook is
`Data/00_FRC-excels/230326_A_0.1gr48hrapprox.xlsx`, SHA256
`3b8bca53405d4639abe6d3ea77fc73944f9339c930ee6cf35e5158f70b2e8c6c`.
Read-only artifact-tool inspection and native XML inspection found three sheets
and no Excel formulas. All **374,543 numeric cells** in `Data!A2:M28812` match the
13 corresponding CSV columns, with zero numerical difference. The workbook was
neither edited nor recalculated. Stored values can still be retrospective Python
calculations; absence of worksheet formulas does not establish online availability.
Reference paths, immutable blob IDs and hashes are recorded in
[`FRC_REFERENCE_PROVENANCE.json`](FRC_REFERENCE_PROVENANCE.json).

## What the checks establish

| Question | Result and practical consequence |
|---|---|
| Do images and rows share a compatible recorded clock? | Published source uses elapsed integer seconds, exported as six-decimal hours. Inverting that representation needs at most 0.0016 s rounding correction. All 482 filename times match a log second exactly, assuming zero origin offset. This is source-record consistency, not measured hardware synchronization or a trusted KDP growth-start clock. |
| Is the saved smooth history available online? | A centered quadratic filter with window 505 reconstructs exported smoothed length to 2.31e-14 source-reported mm. Editing only values after row 5000 changes 252 earlier/current rows, by up to 0.248886 mm. These fields must not enter a forecast as recorded online inputs. |
| Is the same-row infusion value available at image time? | Reference code captures/saves the image, computes control and then appends the infusion command. V2 exposes row `j` only at row `j+1`'s timestamp. A held reported command is an observation-history feature; there is no verified flow sensor or command acknowledgment. |
| Can the sidecar scale establish physical size? | Original workbook `Parameters!F5` is blank; acquisition cadence, mask and smoothing parameter cells are also absent. The later JSON's `0.0375` fails both tested legacy interpretations. Keep the candidate target in pixels. |
| Do native thresholds recover the logged object's progression? | Original workbook RGB bounds are retained, with an explicitly declared circular analysis ROI. Largest-first/nearest-centroid tracking produces 482 integrity-valid contours. This is a computational continuity check, not a person-checked silhouette or exhaustive crystal/defect labels. |

The native frames are 1080 × 1080. Their filename spacing is roughly five
minutes, unlike the later sidecar's `FrameInterval=10`. We use actual filenames.
The circle center `(540,540)` and radius `427` come from that sidecar as analysis
choices; the original acquisition-mask field is blank. No geometry/scale is
silently inherited from CrystalCV's separate thermal-cycle example.

As a source-consistency diagnostic only, the first six hours (72 images) estimate
83.0719 contour-pixel² per source-log mm². Keeping that ratio fixed gives 2.206%
area disagreement in the prefix and 1.586% on the later 410 images. Both values
compare automatic measurements of the same source, not independent ground truth.
This ratio is not physical calibration and never enters the candidate features
or targets. The sidecar linear-scale hypothesis has about 88.5% later disagreement;
its area-divisor hypothesis is much worse. No scale hypothesis is accepted.

## Candidate contract

Each candidate uses one hour of previous images and reported-command history,
and a future image at least 30 minutes later, at most 30 seconds late. At least
four valid history images and fresh process availability are required. The five
inputs are fixed:

1. Current contour area, in pixels².
2. Current square-root area, in pixels.
3. A trailing linear slope of square-root area, in pixels/hour.
4. Latest reported infusion command already available under the next-row bound.
5. Time-weighted mean of that available reported state over the previous hour.

The target is future square-root contour area in pixels. It is an **unreviewed
automatic observation proxy**, not true face advance, mass, defect status or
quality. Source log smoothed length/growth, growth rate, estimated mass and
controller error/derivative are excluded. Their deployed lineage/bootstrap
availability is not certified. Final measured evaporation is also not an input.
Future image values enter the target only. Per-sample image IDs, source log row
ranges and availability bounds make the prefix inspectable.

There are 463 accepted candidates, 12 insufficient-image-history rejections and
seven missing-future-target rejections. All belong to one study/run development
family, with **zero independent evaluation groups and zero new human reviews**.
Nearby windows overlap heavily. Do not randomly split them or count 463 windows
as 463 independent experiments. No pitch class/default detector, recipe or
physical command changes.

## Independent verification and retained correction

The producer is `scripts/check_crystalcv_process.py`, with reusable input logic
in `src/vision/process_series.py`. It reads source workbook values without
recalculation and binds original bytes, native pixels and masks to their index.
`scripts/audit_crystalcv_process.py` imports no producer. It independently checks:

- Original ZIP/CSV/workbook identities and all 374,543 numerical workbook cells.
- All 482 native images with Pillow; RGB selection, filled connected components,
  polygon area/centroid and source clock association without OpenCV.
- All 463 candidate inputs, availability/row lineage and future targets, using
  separate least-squares and held-state integration calculations.
- Source smoothing with direct local quadratic regression, including a
  future-only perturbation, and the separate scale-consistency diagnostics.

Its smoothing reconstruction differs by at most 1.88e-11 source-reported mm,
within the protocol's original 1e-10 tolerance. The PNG/SVG figure in
`results/crystalcv_process_figures_v2` was inspected visually. Targeted tests
check same-cycle/future perturbation invariance, time precision, stale histories,
native threshold identity and rejection cases.
The final regression passed **492 tests and 323 subtests in 125.42 seconds**.
The source/process test selection passed 35 tests, including 15 new process tests.

V1 protocol/export/auditor results remain intact. V1 reconstruction passed but
treated same-timestamp commands as available, before source statement order was
checked. It is **superseded for forecasting** by V2's next-source-row bound.
Its numerical reconstruction pass is not a causal-availability approval. The
protocol changed to tighten availability, without relaxing thresholds or selecting
on forecast scores; no forecasting fit or model score was used. Earlier failed
source-basename lookup and original acquisition/audit attempts also remain intact.

## Reproduce and proceed

Use fresh output paths and the cached source archives/reference files:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/check_crystalcv_process.py --output results/crystalcv_process_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_crystalcv_process.py --result results/crystalcv_process_new --output results/crystalcv_process_check_new.json
MPLCONFIGDIR=/tmp/crystalab-matplotlib .venv/bin/python scripts/plot_crystalcv_process.py --result results/crystalcv_process_new --output results/crystalcv_process_figures_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_process_series.py tests/test_crystalcv_data.py
```

The [frozen forecast comparison](CRYSTALCV_FORECAST.md) is now complete. It trains
on 275 prefix windows and scores 69/66 windows in two later blocks, with disjoint
full supports. Recent rate beats both ridge models, including the model with
available process features. Both fail the predeclared descriptive improvement
rule and remain unpromoted. These are within-run proxy results; command/growth
associations do not establish the effect of choosing a different command.

In parallel, continue the small public KDP outline/event review and seek suitably
licensed independent KDP evidence. Real pitch-class segmentation, visible faces,
first-hour timing, rotation/flow response and physical quality/yield/energy
comparisons remain open. See the [next steps](NEXT_STEPS.md) and [pitch roadmap](../roadmap.md).
