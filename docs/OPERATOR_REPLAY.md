# Operator video and process replay

The local console displays recorded crystal outlines, candidate proposals and
their evidence alongside linked process records. An operator can record an
inspection or decline a proposal. For a declared fixture/synthetic source, an
explicit review can approve a bounded simulated recipe pause and resume it early.
Every accepted review preserves a separate counterfactual result; the original
video, proposal and earlier results remain unchanged.

This implements the pitch's operator demonstration at software-replay scope.
Real-video sessions retain shadow review without numerical commands or invented
temperature readings. Optional [forecast context](OPERATOR_FORECAST.md) now adds
the changing adaptive state and suggested pause duration. Live equipment,
face measurements, rotation/flow actions and real defect accuracy remain open. A replay review does
not become a segmentation label or prove historical/physical action benefit.

## Open a prepared session

With the local data and optional `vision` dependencies available:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --session results/operator_synthetic_v2
```

Open `http://127.0.0.1:8766`; stop with Ctrl+C. This session uses the frozen
fullmask model's lossless optical development replay and a separately declared
thermal software fixture. The [configuration](../configs/operator_optical_demo.json)
binds the exact input video and chooses **one source second = 100 thermal model
seconds**. Optical geometry is not coupled to this thermal response. The mapping
does not establish physical growth time, equipment bounds or defect prevention.

To view real alpha-LGA evidence with no process data or review writes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --session results/operator_real_v2 --read-only
```

No Site is registered, synchronized or published. The server uses standard-library
HTTP on loopback and local HTML/CSS/JavaScript, with no browser runtime framework
or external assets. The optional Qt browser is a verification tool only.

## Prepare a different replay

Choose an unused output directory. Existing outputs are refused.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/operator_console.py \
  --analysis results/vision_codec_fullmask_ffv1_v1/candidates/optical_temporal_003 \
  --simulation-config configs/operator_optical_demo.json \
  --session results/operator_synthetic_next --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/operator_console.py \
  --analysis results/leeds_pilot12_stream_candidates_v1 \
  --session results/operator_real_next --prepare-only
```

Preparation verifies the candidate/source artifacts and policy, retains immutable
shadow proposals and decoded preview frames, and links native seed measurements.
For eligible simulated sessions it first records fixed recipe, PID-only and
operator-supervised variants with **zero approvals**. Unreviewed proposals do not
pause the recipe. Each process result passes the independent thermal/action audit.
Attaching a simulation configuration to real evidence is refused.

These data/model paths exist on the development machine and are Git-ignored.
A fresh clone needs the source acquisition, observation and training/replay steps
in [real vision data](REAL_VISION_DATA.md), [training](VISION_TRAINING.md) and
[encoding](VIDEO_ENCODING.md). A different video needs its own explicit clock
mapping and source hash; the console does not infer one.

## Review and compare

1. Play, step or scrub the recorded frames. **Next proposal** jumps to its first
   retained evidence. Source-video time, original frame mapping and experiment
   time stay separate; unavailable experiment time remains unavailable.
2. Inspect the selected proposal and its evidence. Startup/unknown-origin crystal
   candidates request identity review. Newly visible candidates may request ramp
   review. Possible inclusions retain localized evidence. None confirms a defect.
3. Enter a reviewer name and explicitly acknowledge the selected replay frame.
   **Record inspection** saves a review; **Decline** records a disposition. A name
   is a self-attestation, not authenticated identity. Image-training annotations
   remain separate and unchanged.
4. In a simulated session, **Approve simulated pause** requires an unexpired ramp
   proposal, a fresh observable tracked candidate, current process feedback and
   remaining pause budget. The declared fixture permits 60 model seconds per
   pause, with 120 model seconds reserved in total. Requests that overlap, exceed
   bounds or use unknown/stale evidence fail without creating a review receipt.
5. At a later frame during an approved pause, select that proposal, acknowledge
   the new frame and choose **Resume recipe early**. This ends the pause at that
   simulation time and preserves the earlier requested duration and result.
   Reserved budget is not refunded by early resume.

The selected frame and proposal image must finish decoding before review buttons
enable. Slow/cancelled frame loads cannot pair a new clock with old pixels. Moving
to another frame clears the acknowledgment. After a saved decision, earlier
frames remain viewable but cannot create retrospective reviews out of order.

The process view separates sensor observation, recipe target, accepted bath
request and applied input. The cursor follows the declared clock mapping; chart
and table retain PID-only and original-recipe references. Both error to the
changed target and error to the original recipe are shown, together with recipe
lag. Tracking a paused recipe does not by itself demonstrate an improvement.

## Receipt and result trail

The session retains `manifest.json`, `frames.json`, immutable shadow evidence,
`reviews/` and separate `runs/` results. `history.json` links each receipt to its
result manifest and retains the latest revision. Receipts bind source candidate
manifest, clock/frame, exact proposal, displayed frame and proposal-image hashes,
session identity, actor kind, acknowledgment, UTC recording time and previous
receipt hash. Conflicting revisions and changed inputs are refused. Earlier
receipts and results are never overwritten. An incomplete receipt requires
recovery rather than silent replacement.

Simulations consume only due reviews and candidate packets. A review cannot
arrive late, use a future proposal or skip its actual visibility/expiry checks.
Commands retain proposal/review/sensor links, separate accepted/applied receipts
and the bounded PID response. The auditor reconstructs source/clock lineage,
pause and manual-resume intervals, recipe phase, commands and thermal response
without running the supervisor/controller/plant. Rehashed wrong-clock, wrong-source
or false-pause claims fail semantic checks.

The console is a replay workbench: operators can inspect recorded future portions
of a run. It does not measure prospective human reaction time or establish an
independent adaptive-policy evaluation. No physical hardware driver is exposed.

## Verification

The [checkpoint](OPERATOR_REPLAY_CHECKPOINT.json) records the final full regression,
browser checks, source/result identities and independent execution audit.
Full regression passed **388 tests / 280 subtests** in 113.56 seconds; the browser
passed **19 checks** without JavaScript errors. Independent reconstruction covers
94 cached preview frames and 7,212 process records across the preserved comparison
states. These repeated software fixtures are not independent physical experiments.
Writable browser interactions use generated fixture pixels and
`automated_fixture_qa` receipts, with **zero human reviews claimed**. They approve
a 60-model-second pause at time 130, then resume at 150: actual recipe lag is
20 model seconds. Both intermediate and final results remain independently
auditable. Actual learned synthetic and real sessions are previewed read-only;
source/session hashes and zero human-review counts are retained.

Browser checks cover playback, named acknowledgment, delayed image readiness,
pause/resume, evidence localization, seek restrictions, absent real process data
and a 390-pixel layout. The connected browser surface was unavailable; the
installed offscreen QtWebEngine browser provided these checks instead. QA servers
are stopped afterward. To reproduce with optional PySide6 and unused outputs:

```bash
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python tests/check_operator_console_ui.py \
  --output results/operator_browser_next \
  --synthetic-session results/operator_synthetic_v2 \
  --real-session results/operator_real_v2
```

The separate [adaptive diagnostic](ADAPTIVE_SUPERVISION.md) now connects a changing
observed-area model and forecast-scaled pause policy to a four-way PID comparison.
The [forecast integration](OPERATOR_FORECAST.md) now displays its source-bound
forecast/state/recommendation here and preserves it with explicit reviews.
Retain the action-response limits. Continue actual
real image/event review, compatible mixed training and independent KDP evidence.
First-hour coverage, face/asymmetry/inclusions, real equipment/outcome evaluation,
HPC and Qiskit remain explicit requirements in the [roadmap](../roadmap.md).
