# Synthetic growth dataset v1

The generation checkpoint below predates model fitting. The new
[full pipeline](../artifacts/README.md) uses this same numerical protocol and
includes a fresh versioned dataset, complete XGBoost training and independent
prediction audits. Its model results are separate from this historical report.

Status: complete. All 360 episodes and 1,080 forecast samples passed generation
and independent auditing. Numerical sources match the accepted checkpoint. No ML
training was performed. The compact provenance, hashes, actual split groups,
missingness and validation results are in
[SYNTHETIC_DATASET_V1.json](SYNTHETIC_DATASET_V1.json).

Use `data/synthetic_growth_v1/samples.jsonl` as the input dataset for the first
synthetic growth-model experiment. Superseded `synthetic_growth_pilot*` directories
were removed during cleanup on 2026-11-11. Their accepted numerical summaries
remain in the checkpoint; do not merge overlapping pilot conditions into the
working train/validation/test sets. See the tracked [data guide](../data/README.md)
if your editor or a fresh clone does not show generated files.

## Fixed design

The user selected 360 episodes / 1,080 forecast samples. The complete design is
[`configs/synthetic_growth_dataset_v1.json`](../configs/synthetic_growth_dataset_v1.json).
It keeps the accepted 32^3 focused grid, dt=1, 30-unit duration, 0.75 sampling
thickness, seed geometry, kinetics, controls and observation contract unchanged.
Only the initial concentration groups and their fixed split assignments expand.
The source/environment fingerprint matches the accepted
[SYNTHETIC_DATA_CHECKPOINT.json](SYNTHETIC_DATA_CHECKPOINT.json).

| Split | Base groups | Episodes | Forecast samples |
|---|---:|---:|---:|
| Training | 36 | 216 | 648 |
| Validation | 12 | 72 | 216 |
| Test | 12 | 72 | 216 |
| Total | 60 | 360 | 1,080 |

The 60 concentrations are `round(1.25 + 0.20 * i / 59, 9)` for i=0..59. Twelve
consecutive five-concentration strata each receive three training groups, one
validation group and one test group. A deterministic shuffle with seed 20261110
fixes their positions; both overall range endpoints remain training anchors.
Group IDs are `base_000` through `base_059`; physics seeds are 20261110+i.
Seeds alone are not treated as independent scenarios: all concentrations differ.

Every base group runs all six accepted recipes: baseline, cooling, warming, low
agitation, high agitation and staged controls. Each recipe starts from the same
state within that group. All related recipes/windows stay in that group's split.
Observation streams are independently seeded by group and paired across recipes,
separate from simulation RNG. This is a concentration-interpolation benchmark
under one fixed model, not a test of unseen kinetics, geometry or real experiments.
Only the six recorded recipes are represented; their existence does not validate
arbitrary new combinations of temperature, agitation and action history.

## Inputs and labels

Each episode provides decisions at t=0, 10 and 20, with exact future volume-gain
labels ending at t=10, 20 and 30. Labels are reconstructed from recorded truth;
there is no future-state interpolation. Mean volume-growth rate and mass gain are
auxiliary labels. Physical quality remains null/unavailable.

Use only `manifest.json`'s 14 `feature_columns` and their missing flags as model
inputs. The fields are idealized noisy/delayed geometric and solution proxies,
past observed volume/rate, observation age, the held temperature/agitation
commands and horizon. Episode/group/split IDs and future label timestamps are
metadata. Complete simulator parameters, exact truth and final fields are for
provenance and auditing, not additional prediction-time inputs.

The two-unit sensor delay makes all 360 startup samples explicitly miss their
sensor/history features. The remaining 720 samples have all 14 features
available. The final report records actual per-feature missingness. Null
features are not null targets. Evaluate startup and observed-state samples
separately in the later model phase, and avoid future-value imputation.

All quantities retain illustrative model units. A single real camera cannot
supply all these global 3D proxies directly. Detailed units, causal construction,
raw artifact layout and limitations are in [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md).

## Generate and verify

The accepted run used eight process workers, with each numerical library limited
to one thread and bytecode writes disabled:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/generate_dataset.py \
  --config configs/synthetic_growth_dataset_v1.json \
  --output data/synthetic_growth_v1 --jobs 8
```

A fresh run refuses to overwrite this directory. After stopping an interrupted
run, use the same command with `--resume`; do not launch a concurrent resume while
the original run is still active. Complete episodes are verified/reused; failed
attempts remain recorded and restart from time zero. Resume requires the same
protocol, source contents and numerical environment. Worker count can change.

After successful generation:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_dataset.py data/synthetic_growth_v1
```

The run preserves source snapshots, full resolved cases, raw truth/observations/
controls, final fields, invariant checks and hashes. The independent audit
reconstructs every sample and its split from those raw records. Actual wall time
and the command are recorded in `generation_run.json`. The first run's console
log is kept alongside the completed output. `protocol.json` is an exact copy of
the frozen configuration, and `README.md` describes the data when the output is
used separately from this repository. Data files remain ignored by Git;
the configuration and compact final report are the reviewable repository record.

## Accepted results

- Independent audit: PASS; 360 complete episodes, 60 groups, 1,080 unique samples.
- Whole-group splits: 648 training / 216 validation / 216 test; zero overlapping
  base groups between splits.
- Every forecast has exact start/end records; zero incomplete horizons.
- All 360 episodes completed on their first attempt; no failed cases were omitted.
- Maximum relative tracked-mass error: 5.214e-13
  (tolerance 1e-10).
- Maximum relative carrier-volume error: 2.371e-16
  (tolerance 1e-10).
- Maximum continuity residual per minimum cell volume:
  1.605e-11 (tolerance 1e-8).
- Volume-gain labels span 0.09745566–0.29880348
  model volume units. Quality labels are unavailable throughout.
- Measured wall time: 63.80 minutes with eight workers.
  Mean measured episode-worker time: 83.58 seconds.
- Dataset identity: `829185e2a78a90c5256cfa7c17a4936b148c861ddccd0837bbee153f9f999088`.
- Complete raw/final arrays, source snapshots, configuration, logs and the fresh
  independent audit are retained in `data/synthetic_growth_v1/`.

The pilot and previous numerical checkpoint establish the accepted fixed regime;
this batch checks every new episode's invariants and every sample's construction.
It does not repeat the expensive refinement study for every intermediate
concentration, establish a formal continuous error bound, fit a model or establish
real KDP prediction accuracy. Learning curves and model comparisons are later work.

## Workspace cleanup

Generated `results/` contents and project bytecode/lint/font caches were removed
on 2026-11-10. On 2026-11-11 the two superseded pilot datasets and the duplicate
`COD/` extraction were removed. Every extracted CIF was byte-checked against the
retained original `data/COD.zip`. Original source inputs/access evidence, private
notes, `.venv/`, committed numerical checkpoints and the entire active v1 dataset
are preserved. Historical result/pilot paths no longer describe live folders.

The compact report records generation-time provenance and cleanup, not live Git
status. Its bytes and the dataset's manifest/samples are preserved unchanged.
The current [roadmap](../roadmap.md) replaces the archived development checklist.

This cleanup changes no simulation equations or model code and adds no ML
dependency. Growth baselines and XGBoost remain a separate supporting experiment.
The current [roadmap](../roadmap.md) prioritizes the camera/image-data path; real
KDP, quality and physical control require their own experimental assessment.
