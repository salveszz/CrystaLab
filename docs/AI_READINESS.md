# Readiness for the first growth model

**Ready for a first synthetic, growth-only prediction experiment.** The data gate
is complete; no additional CFD feature is required for that experiment. Model
training has not started, and the training entry point remains a placeholder.

## Available data and evidence

Use `data/synthetic_growth_v1/samples.jsonl` locally. The
[dataset card](SYNTHETIC_DATASET_V1.md) and [report](SYNTHETIC_DATASET_V1.json)
describe the accepted 360 episodes / 1,080 samples across 60 concentration groups.
The frozen split is 648/216/216 by whole base group. The 720 fully observed rows
and 360 startup rows with missing sensor/history inputs must be assessed
separately. Do not merge overlapping pilot data or reshuffle episode windows.

The generation/audit pipeline records exact future volume-gain labels, bounded
held controls, independently seeded delayed/noisy sensor proxies, complete
configurations, source snapshots and final fields. The independent audit
reconstructs every sample from raw records and checks the split and provenance.
The [pipeline contract](SYNTHETIC_DATA.md) defines the units and allowed features.

The [numerical checkpoint](SYNTHETIC_DATA_CHECKPOINT.json) retains the accepted
32^3 focused grid, timestep/refinement results, source hashes and original 2%
tolerances. Coarser failures remain recorded. Its source-verification checkpoint
passed 227 tests; the simulation source is unchanged by the documentation cleanup.
The old raw pilots and study outputs were removed after their summaries were
preserved. The complete active v1 dataset remains auditable.

## Next implementation phase

1. Add a loader that uses the 14 saved feature names and their missing flags.
   Keep hidden configuration, exact truth, identifiers and future labels out of
   prediction-time inputs. Do not fill startup values from later observations.
2. Evaluate zero-growth persistence and recent observed volume-rate-times-horizon
   baselines, with the documented fallback when history is unavailable.
3. Compare a small XGBoost regressor for exact volume gain over the next 10 model
   time units. Fit/select using training and validation; keep final test groups
   for the agreed final comparison.
4. Report MAE/RMSE, signed error, macro-averages by base group, recipe results and
   startup/observed-state strata. Avoid percentage errors near zero growth.
5. Use learning curves over training groups before deciding to generate more data
   or increase model complexity.

The deliverable is a reproducible comparison showing whether the model improves
on the recent-growth baseline within this simulator. No fitting, model artifact,
ML dependency or controller is part of the current cleanup commit.

## Boundaries of this result

The first dataset tests interpolation over initial concentration under one fixed
illustrative law, geometry, time range and six recipes. It uses model units and
idealized 3D proxies; a single real camera does not supply exact 3D volume/area.
Several small early action effects do not consistently exceed the recorded
resolution-change estimates. The saturating agitation law has no validated
quality penalty or equipment-cost tradeoff. These data do not establish an
optimal real recipe, optical quality or real KDP prediction accuracy.

| Later result | Additional preparation |
|---|---|
| Real KDP growth predictions | Measured trajectories/conditions, relevant calibration and whole held-out experiments. A reduced/empirical model may be sufficient. |
| Camera/quality model | Images and valid spatial/quality labels, uncertainty definitions and separate real-data evaluation. |
| Physical controller | Valid action/quality responses, process/equipment limits, actuator timing and independent closed-loop assessment. |
| Resolved local flow/transport outputs | Verified coupling, intended forcing/boundaries and relevant experimental comparison. Moving growth additionally needs uptake/motion/volume closure. |

The [original presentation](../Pitch_KDP.pptx) proposes real plus synthetic data
and treats detailed Navier–Stokes/HPC as an extension. Its synthetic image examples
are a separate task from the later tabular XGBoost growth dataset. More samples
cannot independently correct a biased model or an unsupported quality label.

Collect suitable measured growth data in parallel. COD is structural evidence;
published aggregates are not independent raw growth trajectories. Resume detailed
CFD when a required output or measured discrepancy justifies it. Specify the
validation experiment first. See [physical data](PHYSICAL_DATA.md),
[real-data validation](REAL_DATA_VALIDATION.md) and the current [roadmap](../roadmap.md).
