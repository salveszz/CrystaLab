# Temporal candidate inspection and evaluation

The learned replay now has a causal tracker for non-seed crystal and inclusion
candidates. It retains mask evidence, object/event IDs and startup, later-appearance
and unknown-interval distinctions. Every emitted event requests operator review.
The shared recorder works on completed replays and during single-pass observation,
flushing decisions before the next prediction. The integrated path was exercised
on recorded files; physical camera/equipment timing remains unverified. It issues
no actuator commands.

This is a software milestone toward the pitch's visible-event alerts. The model
still mistakes bubbles for crystals and misses inclusions. No real event labels
have been reviewed, no physical first-hour result is available, and no supported
face-growth asymmetry measurement exists. See the
[saved-replay checkpoint](TEMPORAL_CANDIDATES_CHECKPOINT.json) and subsequent
[streaming checkpoint](STREAMING_CANDIDATES_CHECKPOINT.json) for exact evidence.

## Policy and clocks

`configs/candidate_policy.json` declares a fixed view, at least two consecutive
usable observations, mask IoU at least 0.2, a maximum gap of one source-video
second and at most 128 retained tracks. These are development settings, not
validated physical budgets. Confirmation duration is configurable; its default
is zero, so the two-observation requirement determines the initial delay.

The known seed and its compatible alternative masks are excluded. Other raw
crystal classes share a candidate identity type; changing the visual class does
not create a new object. Nearly identical masks are aliases. Clipping, multiple
components, unresolved seed overlap, poorly localized inclusions and competing
associations make that candidate type unobservable. Seed/view uncertainty makes
both types unobservable. Missing observations reset confirmation, and do not
prove that an object disappeared or that quality is healthy.

An object present in the first usable view is a startup candidate. Later objects
are distinguished by whether the previous view was usable. A possible match to
an expired track retains unresolved identity rather than being called new.
Retired footprints remain within the declared capacity; reaching capacity fails
and requires explicit review/reinitialization instead of silently forgetting
identities. This bounded development policy is not a continuous multi-day system.

Onset intervals describe detector observations, not microscopic nucleation.
`source_video_time_s` is the original replay clock. Display FPS, nominal acquisition
cadence and physical growth time remain separate. Usable intervals cover adjacent
processed observations within the gap limit; they are neither independently
annotated negative time nor proof of uninterrupted physical visibility.

## Reproduce the saved-replay analysis

Use unused output paths for reruns. The named paths already exist locally.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/analyze_candidates.py \
  --replay results/optical_pilot13_learned_v3 \
  --config configs/candidate_policy.json \
  --output results/optical_pilot13_candidates_v2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_optical_events.py \
  --replay results/optical_pilot13_learned_v3 \
  --output results/optical_pilot13_event_truth_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/evaluate_candidate_events.py \
  --analysis results/optical_pilot13_candidates_v2 \
  --annotations results/optical_pilot13_event_truth_v1/annotations.json \
  --output results/optical_pilot13_event_eval_v3
```

The analysis retains parent/source hashes and source snapshots, per-frame
`candidate_observations.jsonl`, separate `candidate_events.jsonl`, component masks,
first-detection masks, evidence images and an overlay video. The parent's
`events.jsonl` still means seed-observation transitions. Source artifacts are
checked before and after analysis. Final test inputs are excluded from this
development path.

Optical event truth comes from original scene identities and emission-proxy mask
eligibility, independently of detector predictions. It describes constructed
proxies rather than exact empirical visibility. The source frames are three
correlated construction steps in one development scene, not three independent
physical experiments.

## Single-pass observation

The same candidate policy can run directly on native observation masks, without
rereading saved masks or rerunning the detector. Add both candidate options to
the learned observer command:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-review-mpl .venv/bin/python scripts/observe_video.py \
  --video results/optical_pilot13_sequence_v1/sequence.avi \
  --config configs/optical_learned_observation.json \
  --model-run results/vision_training_pilot_v1 \
  --source-info results/optical_pilot13_sequence_v1/source_info.json \
  --frame-map results/optical_pilot13_sequence_v1/frame_map.jsonl \
  --run-id optical_pilot13_stream_v1 --output results/optical_pilot13_stream_v1 \
  --candidate-config configs/candidate_policy.json \
  --candidate-output results/optical_pilot13_stream_candidates_v1
```

Observation and candidate output directories must be separate and unused. Each
candidate event and observation is flushed during processing, while manifests
remain `running`. After observation closes, its immutable manifest and files are
verified and bound into the completed candidate manifest. An error marks both
runs failed and retains partial evidence. This avoids circular manifest hashes.
Candidate options require the learned observer; the seed-only baseline cannot
supply candidate absence evidence. Final test inputs remain excluded.

The 48-frame fixture can also exercise this path with
`scripts/make_candidate_fixture.py --output results/candidate_stream_fixture_v1
--candidate-config configs/candidate_policy.json`; its adjacent
`results/candidate_stream_fixture_v1_candidates` directory holds decisions.
The fixture still supplies imperfect masks and makes no detector accuracy claim.

The actual three-frame optical, 30-frame real and 48-frame fixture runs produce
the same per-frame/event JSON values as subsequent saved-replay analysis of each
parent. Tests observe flushed alerts before the following prediction and verify
sampled clocks, interrupted runs and rejected input/output combinations.

The real development run's decoded-frame-to-flushed-record processing latency was
0.617 seconds initially, then 0.024 seconds median; overall p95 was 0.037 seconds.
The metric includes inference, mask/overlay writes and candidate decisions. It
excludes decoding, model setup, final hashing and camera exposure/transport. It
does not establish an equipment deadline or sensor-to-command latency. Hardware
and exact values are retained in the streaming checkpoint; no camera was opened.

## Actual results

| Evidence | Observed result | Interpretation |
|---|---|---|
| Trained model, optical scene 013, three frames / 0.5 display seconds | One later crystal match, two false startup crystal inspections; one inclusion unmatched. Matched mask IoU 0.711, alert 0.25 source seconds after its first eligible frame. | Both target matching windows are censored by the short clip. No full-deadline recall or physical delay claim. Startup false alarms remain in all-alert counts. |
| Trained model, real alpha-LGA training run 12, 30 frames | Zero candidate events, 11 seed/view-uncertain frames; 4.25 usable source-video seconds per type. | No reviewed events or negative periods. Zero alerts cannot establish absence, recall, false-alert rate or KDP performance. |
| Constructed 48-frame policy fixture / 4.7 seconds | Two crystal matches, one inclusion match, one false startup crystal alert; transient noise suppressed and appearance after occlusion distinguished. | Predictions are supplied imperfect masks. All three deadlines fit within the clip, but only one crystal window and the inclusion window have uninterrupted class-usable evidence. The other crystal window overlaps an unknown view. |

The fixture has 0.9 usable annotated-negative source seconds for crystals and 2.2
for inclusions after excluding the unknown interval. It has no physical-time
interpretation. Generate it with `scripts/make_candidate_fixture.py --output ...`,
then use `analyze_candidates.py` on its `observations/` directory and evaluate
against its `annotations.json`.

The current evaluator reconstructs usable intervals from adjacent class flags,
timestamps and the configured gap bound. Its `observable_deadline_followup`
requires uninterrupted usable evidence across the full matching window; the
older `complete_followup` field retains elapsed-clip coverage for comparability.
The stricter fixture rerun is preserved at
`results/temporal_fixture_coverage_v2/report.json`. Ordinary hits, misses and false
inspections remain reported independently of follow-up exclusions. These usability
flags do not establish independently measured visibility or healthy crystal quality.
The [longer optical diagnostic](OPTICAL_TEMPORAL.md) supplies timed construction
profiles and a frozen-model workflow for clips with more post-onset observations.

## Event annotation and matching

`crystalab_visible_event_annotations_v1` binds the exact source video hash and
declares `evidence_kind` (`fixture`, `synthetic` or `real`) and the source-video
clock. Each event has a unique ID, candidate type, origin, last-absent/first-visible
onset interval, and a binary first-visible mask with its hash. A null lower bound
means startup onset is unknown. Separate negative intervals specify the candidate
type and start/end times; uncertain events and negative intervals cannot be
scored as certain truth. See the generated fixture annotation for the schema.

Evaluation matches events one to one by type, onset window and first-mask IoU
(default at least 0.5), maximizing valid match count and then IoU. The default
matching allowance is one source-video second after first visibility. Reports
retain unmatched/duplicate alerts, misses, origin confusion, interval-valued
delay, available-clip counts and a separate complete-follow-up denominator.
Events first visible outside the processed window are reported as excluded.
Negative coverage is the intersection of reviewed negative intervals and usable
observation intervals, with overlaps merged.

Real labels require a named person's explicit review of source events, boundaries
and negative periods. After that inspection, `scripts/review_candidate_events.py`
can save an exact-byte receipt using `--annotations`, `--source-video`, `--reviewer`
and `--ack-event-source-reviewed`. A separate `--ack-physical-clock-reviewed`
attests clock inspection. Neither option performs the inspection or authenticates
the reviewer; the agent must not use it to invent a human review. Editing labels
or masks invalidates the receipt. **Actual real review receipts remain zero.**

Physical false alerts per negative hour additionally require real evidence, a
reviewed physical clock, supported original clock origin/scale and source
provenance. These conditions are unmet by the current runs, so physical rates
are null. First-hour evaluation remains explicitly unfinished.

## Next pitch work

Review real masks/events, expand relevant independent training
coverage, investigate bubble/inclusion errors and freeze budgets before final
evaluation. Add comparable-view growth/face measurements where labels support
them. Verify physical capture/cadence before claiming live equipment performance.
The [temperature PID baseline](TEMPERATURE_CONTROL.md) now runs on a declared
simulated thermal plant. [Event-rule integration](EVENT_SUPERVISION.md) now emits
review proposals and exercises a declared fixture-to-PID pause. Adaptive policy,
real review and equipment characterization remain open.
The [roadmap](../roadmap.md) keeps physical, operator, HPC and Qiskit gates explicit.
