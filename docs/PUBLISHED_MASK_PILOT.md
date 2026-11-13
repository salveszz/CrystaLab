# First fit on published real pixel labels

A separate YOLOv8n-seg pilot now trains on **author-supplied protein crystal/drop
masks** from the [CHiMP source](PUBLISHED_CRYSTAL_MASKS.md). Selected-crystal
validation recall is only **8/74 (10.8%)** at the frozen settings. The real-label
training path runs; the model stays experimental and is refused by the pitch
observer. It is not a KDP seed/secondary/inclusion model.

## Data and fixed budget

The [protocol](../configs/chimp_mask_pilot.json) selects 32 training and 16
validation images per source domain by a fixed sample-ID hash, preserving
original author membership. Selection precedes conversion checks. Three of the
96 selected images have disconnected masks that a single polygon cannot
represent. They are excluded completely without replacement, with originals
retained; unsupported objects never become negatives.

The resulting **61 training / 32 validation images** contain 687 source masks:
602 selected `crystal` and 85 `drop`. All 781 original masks in the 96-image
selection, including rejected images, are preserved. Original holes remain
evaluation truth; polygon loss is recorded separately. Independent Pillow
rasterization has minimum native polygon IoU 0.998817, above the frozen 0.95 gate.

The [training configuration](../configs/chimp_mask_pilot_train.json) fixes 15 CPU
epochs at 512 input, batch 4, seed 20261113, AdamW learning rate 0.001, no warmup,
constant learning rate, separate masks and no image augmentation. Existing pinned
COCO initialization is used locally. The run took 264.97 seconds. Evaluation uses
the **last epoch**, confidence 0.25, NMS IoU 0.7, maximum 512 detections,
original-size masks and native match IoU 0.5. No hyperparameter search or
comparative improvement claim is made.

## Original-mask results

| Split / class | Published selected targets | Matched | Missed | Unmatched predictions |
|---|---:|---:|---:|---:|
| Training crystal | 528 | 240 | 288 | 102 |
| Validation crystal | 74 | 8 | 66 | 67 |
| Training drop | 55 | 55 | 0 | 11 |
| Validation drop | 30 | 27 | 3 | 9 |

Validation selected-crystal precision is 8/75 (10.7%); matched mean native IoU is
0.7363. Drop recall is 90%, with matched mean IoU 0.8928. Crystal matches are
3/18 for VMXi and 5/56 for XChem. Four validation images have no source-selected
objects, which does not establish absence of all visible crystals. Unmatched
predictions are errors against this selected-object task, not confirmed KDP
false alerts. The result exposes poor crystal generalization within this pilot.

The source selects interesting or mountable protein crystals, rather than
exhaustively labeling every crystal. Physical run/plate grouping is unknown.
This adds no independent KDP evaluation, seed roles, visible-event labels or
local human-review receipts. No model, default detector or physical recipe is promoted.

## Evidence and reproduction

Local artifacts are `data/chimp_selected_mask_pilot_v1`,
`results/chimp_selected_mask_train_v1`, `results/chimp_selected_mask_evaluation_v1`
and `results/chimp_selected_mask_independent_audit_v1.json`.
Final weights have SHA256
`3a403fc68bac82d4a3ffc6dce31c50e8e74e258f4277fdf6fbef1f7dae11a529`.
The [checkpoint](PUBLISHED_MASK_PILOT_CHECKPOINT.json) binds source/config/code,
artifacts, runtime, source-access attempts and verification.

The independent auditor imports no producer/evaluator. It verifies source
identities with Pillow, polygon fidelity with another rasterizer, every native
score assignment with Python integer bitsets and four fixed neural replays.
The real pitch observer additionally refuses this task before inference. Four
fixed source/prediction views were inspected after scoring; that creates no human
annotation receipt. The full suite passed **511 tests and 323 subtests in 114.96 s**.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/chimp_mask_pilot.py prepare --dataset data/chimp_selected_mask_pilot_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/chimp_mask_pilot.py train --dataset data/chimp_selected_mask_pilot_new --output results/chimp_train_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/evaluate_chimp_pilot.py --dataset data/chimp_selected_mask_pilot_new --run results/chimp_train_new --output results/chimp_evaluation_new
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_chimp_pilot.py --dataset data/chimp_selected_mask_pilot_new --run results/chimp_train_new --evaluation results/chimp_evaluation_new --output results/chimp_audit_new.json
```

Use fresh paths. Large datasets, masks, weights and generated outputs remain
Git-ignored; protocols, code and provenance support reproduction. The main next
gate remains relevant reviewed KDP boundaries/events and diverse independently
grouped data. This supporting experiment supplies no physical control benefit.
