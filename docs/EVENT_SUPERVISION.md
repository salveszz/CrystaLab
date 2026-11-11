# Event-rule supervision and shadow proposals

Visible candidate evidence now reaches a stateful rule supervisor. Recorded
development clips produce evidence-linked review proposals, and a separate
software fixture connects an explicitly scripted pause decision through temperature
PID to delayed simulated actuation. This implements the rule baseline in roadmap
Batch 4, with actual results in the [checkpoint](EVENT_SUPERVISION_CHECKPOINT.json).

The adaptive forecast/state policy, operator interface, supported face asymmetry
and physical action-response evidence remain unfinished. These rules react to
visible candidates; they do not predict events before visibility. Real labels
remain unreviewed and the source detector's bubble/inclusion errors remain.

## Rules and review state

`controllers.supervision.RuleEvidence` binds run, source clock, original frame,
observation/receipt times, evidence kind, candidate visibility, visible track IDs,
issues and candidate events. The reader verifies completed candidate artifacts
and copied observation context before use. Final test inputs are excluded from
this development path.

| Source evidence | Review proposal | Numerical action |
|---|---|---|
| Crystal candidate newly visible after a clear view | Review ramp pause, retaining event/track/onset evidence. | None in shadow mode. |
| Startup crystal, appearance after an unknown interval or unresolved old identity | Inspect candidate identity. | None. These origins do not prove new nucleation. |
| Inclusion candidate | Inspect and mark the retained location/mask. | None. |
| Unknown, missing, stale or interrupted view | One inspection proposal per unknown-view episode. | None. Returning visibility does not certify healthy quality. |

Repeated packets/events cannot create duplicate proposals or silently change
their evidence. Proposals have stable payload hashes and explicit expiry. Expired
proposals remain in history; an extended unknown episode does not generate a new
notification every tick. The current unknown state remains visible. Proposal and
event memory are bounded; reaching capacity fails and requires explicit review
or reinitialization rather than forgetting identities.

`configs/event_rule_policy.json` declares a one-source-second evidence age/gap,
two-source-second proposal lifetime and a capacity of 128. These are development
settings in the bound source clock, not equipment timeout recommendations.
Packets separated by a long gap remain quarantined until a new usable packet.
Pending review, expiry and visibility restoration are distinct from authorization.

`decide_for_simulation` accepts exact proposal hashes and unique scripted decision
IDs only for fixture/synthetic evidence. Authorizing a ramp proposal additionally
requires a fresh usable view in which the candidate track is still observed.
The decision records `operator_reviewed=false` and
`physical_hardware_authorized=false`. It is an explicit test-policy assumption,
not a fabricated human review. Real evidence cannot use this simulation-authorization
path. Human review/override and actual hardware authorization need the later
operator/equipment integration.

## Actual shadow runs

| Evidence | Frames | Proposals |
|---|---:|---|
| Supplied-mask fixture | 48 | Two candidate inspections, one ramp review, one view inspection and one localized inclusion alert. |
| Trained model, optical development scene 013 | 3 | Two startup candidate inspections and one later ramp review. The inclusion remains missed. |
| Trained model, real alpha-LGA training run 12 | 30 | One view-inspection episode spanning the uncertain portion; no candidate-event proposal. |

All three shadow runs create **zero numerical actuator commands** and no process
sensor readings or operator reviews. The two optical bubble errors remain
inspection workload; routing them by startup origin does not improve the model's
classification accuracy. Zero real candidate alerts does not establish absence,
recall or quality. The real view proposal expires in the display clock; this is
not a measured physical response deadline.

Each output preserves `decisions.jsonl`, immutable `proposals.jsonl`, source and
policy identities, copied mask/evidence images, an overlay and final proposal
states. A 48-pixel footer labels shadow mode below the original image; mask and
location coordinates retain the native origin and dimensions. The independent
artifact check reconstructs source/frame links, proposal hashes, copied native
masks and decoded video counts.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_supervisory_shadow.py \
  --analysis results/leeds_pilot12_stream_candidates_v1 \
  --policy configs/event_rule_policy.json --output results/leeds_pilot12_shadow_v1
```

The named output exists locally; choose an unused path for reruns. Substitute
`results/optical_pilot13_stream_candidates_v1` or
`results/candidate_stream_fixture_v1_candidates` for the other recorded examples.
The core supervisor accepts causal packets during a stream, but this CLI exercises
saved candidate evidence. Physical camera/equipment timing remains unverified.

## Rule-to-PID software comparison

`configs/supervised_control_fixture.json` declares a separate 600-second cooling
fixture and a source-bound software clock map. One display second is deliberately
mapped to 100 thermal-model seconds. The exact input-video hash, both clock origins,
target clock ID, scale and test provenance are recorded. This mapping is never
presented as a physical growth-time conversion, and real sources are refused.

The supplied-mask fixture provides the original frame/event sequence. Only due
packets are consumed. A declared scripted policy waits 0.2 source seconds after
a new-crystal ramp proposal, checks current visibility/track and fresh simulated
temperature feedback, and authorizes a 60-model-second recipe pause. The total
pause budget is 120 model seconds. No decision is authorized after the source
clip's observed window. Missing visibility requests inspection; it does not invent
a corrective setpoint. PID continues using its separate process-sensor contract.

The observed pause begins at model time 130 s and ends at 190 s. Recipe phase is
held at 130 s during that interval, then resumes; final phase is 540 s at model
time 600 s. The event, exact proposal, scripted decision, sensor sample and later
commands remain linked. Temperature requests still obey the existing range/slew,
PID fallback and delayed-acknowledgment contracts.

| Variant | MAE to its commanded target (K) | MAE to the common original recipe (K) | Final recipe lag (model s) |
|---|---:|---:|---:|
| Fixed recipe | 0.18735 | 0.18735 | 0 |
| PID-only | 0.01053 | 0.01053 | 0 |
| Rule-supervised PID | 0.01910 | 0.15122 | 60 |

PID-only tracks the original recipe more closely. The rule intentionally changes
the target and incurs a recipe-delay cost; tracking its altered target alone
would conceal that cost. This plant has no modeled defect/quality response, so
the comparison establishes software behavior and traceability, not a beneficial
action or improved yield. It uses one declared thermal case and one analytic
supplied-mask sequence, not independent physical experiments or model accuracy.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/crystalab-review-mpl .venv/bin/python scripts/run_supervised_control.py \
  --analysis results/candidate_stream_fixture_v1_candidates \
  --policy configs/event_rule_policy.json --config configs/supervised_control_fixture.json \
  --output results/supervised_control_fixture_v2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit_control_baseline.py \
  --run results/supervised_control_fixture_v2 \
  --output results/supervised_control_fixture_v2_audit.json
```

These paths already exist. Three variants retain 1,803 process/control records.
The auditor independently reconstructs candidate lineage, software clock mapping,
scripted decision eligibility, exact bounded recipe-phase pause, causal command
IDs, delayed application, thermal response and both target-error metrics. The
original two-method thermal benchmark remains supported. Tests also reject a
rehashed record that falsely advances recipe phase through a pause.

## Next pitch work

Specify an observable short-horizon forecast, its causal history, objective,
changing adaptive state/policy and permitted actions. Compare it with persistence
and the implemented rule baseline using appropriately grouped evidence. A fixed
candidate-to-pause rule is not the adaptive-agent deliverable. Continue reviewed
real mask/event coverage, measurement/clock calibration and capture planning.
Add an operator view with actual review/override input, supported projected
growth/face measurements and equipment acknowledgment/fallback characterization.
Real first-hour, physical action benefit, HPC and Qiskit remain open in the
[pitch-led roadmap](../roadmap.md).
