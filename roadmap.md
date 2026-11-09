# Current local scope — 2026-11-09

The original package layout, starter commands and runner defaults are preserved.
Physical-reference/source audit is committed as 3b99100 (`llalal`). The standalone
embedded no-slip viscosity and roadmap are committed in d532f35, after the
tracked-documentation checkpoint aa4669b. Those milestones contain no AI work.

Current local work: couple pressure and viscosity on the same liquid-volume and
shared-face velocity means, using a monolithic fixed-domain linear Stokes solve.
Pressure is the negative adjoint of integrated face divergence, so pressure work
vanishes with conserved volume. The implementation and 12 new tests are present;
all 198 tests pass. See docs/CUTCELL_STOKES.md for the derivation/file review.

Numerical acceptance: all 17 cases in results/stokes_refined_d532f35 pass the
unchanged gates on 12/18/24 grids. Finest uniform/focused velocity RMS errors
are 0.654%/1.888%; pressure RMS over the prescribed scale is 1.979%/3.024%.
Pressure work and integrated volume residuals are near roundoff; pressure and
velocity pass time refinement. The original 6/12/18 study fails focused
accuracy/refinement and remains in results/stokes_initial_d532f35. Acceptance
does not validate those coarse focused meshes. All source hashes/fields were
checked; docs/CUTCELL_STOKES_CHECKPOINT.json retains both protocols/results.
This coupling work remains local and uncommitted.

Next: conservative nonlinear momentum fluxes on the shared representation,
then momentum/energy and advective small-cell handling under refinement before
enabling partial cells in the full Navier–Stokes runtime. Follow with passive
solute, physical uptake/motion and explicit solution/solid density/partial-volume
closure. The illustrative growth engine and canonical partial-cell guard remain
in place. AI training, new experiment packages, placeholder deletion and ML
changes remain outside this work.

The longer sections below contain development history, including superseded
local experiments. Their historical claims do not override this current scope.
Explain every affected file; further commit/push requires separate authorization.

# KDP AI — Simulation-First Coding Goal

Build a complete Python-based KDP crystal-growth simulation with AI prediction and automated control.

## Objective and Control Architecture

Maximize growth while maintaining crystal quality over the full experiment period, subject to explicit quality, equipment and process constraints. Quality is a required constraint, not an optional secondary objective. The implemented 3D simulator measures growth from actual crystal volume and density. Use net crystal mass gain per elapsed time as the primary growth metric. The old planar runtime has been removed; numerical and physical validation remain separate from implementation completion. Report face growth rates and shape separately. Define quality limits before controller tuning; the current synthetic defect count cannot establish real optical quality.

The intended loop is: camera frames and timestamped sensor measurements -> vision and temporal state estimation -> XGBoost outcome predictions for candidate actions -> constrained action selection -> temperature, rotation and circulation setpoints -> next measurements. XGBoost remains the primary predictive model. A separate controller searches action sequences over the remaining growth period using its predictions, applies the first action, and replans after new observations. Train XGBoost to predict action-conditioned future growth and quality so this search selects the best tested schedule. PID tracks equipment setpoints where appropriate.

Vision estimates silhouette, face motion, growth trends and visible anomaly indicators, with confidence scores. Temperature and concentration come from sensors or explicitly labeled estimates. Keep simulator ground truth for training labels and evaluation, separate from controller observations. A single 2D view provides projected geometry, not independently measured 3D volume or internal optical quality.

**Success criteria:** Compare fixed-recipe, PID and XGBoost-based control with the same starting conditions, durations, solute budgets and action limits. Report growth, quality proxies, constraint violations and intervention frequency across held-out experiments. Set numeric acceptance thresholds before evaluating the final test set. Do not claim a guaranteed global optimum or validated KDP performance from a synthetic demonstration.

**Review workflow:** Work in small steps. Explain every affected file and its code before committing. Wait for the user to review and authorize the commit, and push only when the user authorizes the push. Each approved step gets a short, focused commit.

## Implementation Status (3D Simulation Committed)

The user requested migrating the existing modules completely, with no parallel planar simulation path. `crystal.py`, `tank.py`, `physics.py`, `engine.py`, the existing demo, configuration and tests now operate in 3D. The temporary `geometry3d.py`, `tank3d.py` and `diffusion3d.py` runtime prototypes have been removed. User reviewed and authorized this migration; commit `748ca80` (`3d simulation`) is pushed to private GitHub main. All 52 tests pass. Further changes still require explanation and explicit commit/push authorization.

**Last completed review:** Explained the canonical 3D migration and open-tank displacement, audited comments, ran all 52 tests, and committed/pushed the authorized changes as `748ca80`. Geometry uses face-plane matrices; fields are `[nz, ny, nx]`; the coupled engine samples wetted face neighborhoods, conserves dissolved plus actual crystal-volume mass and incompressible carrier volume, clips to six tank walls and stops exactly at duration. The existing window renders the actual 3D polyhedron and volumetric field slices. Both window playback and GIF export run once. The default seed has sloped caps, with illustrative coefficients rather than calibrated KDP facet data.

**Simulation before AI:** No AI, controller or vision implementation is added in this migration. First establish trustworthy units, transport, growth and quality assumptions. The user chose an open tank with rising liquid level. Actual solid intersections now exclude fluid, partial-cell apertures block transport, and a quasistatic conservative displacement field moves carrier/solute/heat as the boundary grows. This remains a dilute approximation with no resolved Navier-Stokes/stirring/buoyancy solver. See `docs/FLUID_MODEL.md`.

**Representation:** Use unit outward normals `normals[n_faces, 3]`, offsets `offsets[n_faces]`, vertices `[n_vertices, 3]` and ordered face connectivity. Inside points satisfy `normals @ position <= offsets`. Surface area and volume come from the actual convex solid. Kinetic coefficients follow crystal faces, not global axis directions. Six extra planes have zero growth coefficients and enforce tank walls. Rendering triangles do not represent separate kinetic faces. Convex geometry and fixed face directions are current assumptions.

**Coordinates:** x/y are horizontal; z is vertical. Dimensions are width/depth/height. All scalar fields use `[nz, ny, nx]`; cells have volume `dx * dy * dz`. The tank exposes `[n_cells, 3]` positions for matrix-based face sampling. A future velocity field would need an extra three-component axis. Rotation/circulation remain illustrative mixing/transport parameters. Their effective diffusivity now acts only through open liquid faces; it is separate from the computed quasistatic displacement field.

**Completed review for `748ca80` (future changes still require approval):**

- [x] Crystal face geometry and its analytic tests.
- [x] Tank configuration, 3D diffusion and conservation/convergence tests.
- [x] Coupled growth, local uptake, wall containment, volume mass accounting and final-time tests.
- [x] Open-tank partial-cell geometry, conservative displacement, liquid-only transport and their tests.
- [x] Default sloped seed, face-rate coloring, normals/displacement arrows, free surface, single-pass export and volumetric NPZ outputs.

The items above were reviewed and committed. Future work must remain small and individually reviewed: explain every affected file and get explicit authorization before a commit or push. Keep one canonical 3D runtime.

**Next work after review:** Validate convergence of the coupled growth/displacement system beyond the tested diffusion refinement; establish physical units, traceable face kinetics and independently meaningful quality measurements. Quantitative fluid dynamics would require a real flow solver, measured boundary conditions and solution-density/partial-volume data. A synthetic defect counter cannot establish optical quality.

## Phase 1 — Working Simulator (Highest Priority)

- [x] Create the project structure and Python environment.
- [x] Migrate the existing tank and engine to a configurable 3D open tank with solid exclusion, partial liquid volumes and a rising free surface.
- [x] Implement volumetric temperature and solute-concentration fields, with liquid-aperture finite-volume diffusion, blocked solid/air transport and sealed outside walls.
- [x] Calculate supersaturation using a configurable illustrative solubility model; physical calibration remains pending.
- [x] Replace the existing crystal representation with a convex 3D polyhedron, configurable face planes and seed orientation.
- [x] Implement outward normal growth with coefficients attached to crystal faces.
- [x] Sample volumetric solution neighborhoods across arbitrary 3D face footprints and reconstruct vertices, including redundant/disappearing faces.
- [x] Derive actual surface area/volume/bounds and conserve dissolved plus crystal mass using density times volume gain.
- [x] Limit growth by all six tank walls and available local dissolved mass.
- [x] Implement simplified rotation/circulation effects as saturating liquid-only mixing diffusivity and transport enhancement; resolved stirring flow remains pending.
- [x] Implement optional impurity growth suppression and a labeled synthetic defect-event counter; physical defect geometry/quality prediction remains pending.
- [x] Add configurable timestep/duration and exact final-time stopping, including a final partial step.
- [x] Ensure reproducible initialization and synthetic events using random seeds.
- [x] Add tests for numerical stability, valid concentrations and combined solute/crystal mass conservation (66 tests in the committed suite).

Use simplified, explicitly labeled physics approximations initially. Do not fabricate validated KDP physical constants.

**Implementation milestone complete:** The existing 3D path integrates geometry, cut-cell fields, liquid-only transport, local uptake, rising surface and conservative displacement, committed as `748ca80`. The default 120-unit run preserves carrier volume and has maximum combined-mass drift about 1.21e-11 model units. Checked items mean implemented/tested at the stated approximation level, not validated against real KDP.

**Completed measurement step — coupled convergence:** Ran all five unique 120-unit experiments: three grids at dt=1 and three timesteps on the finest grid, sharing one case. Noise/defect randomness was disabled and physical sampling thickness held at 0.75. The first reports and histories are in `results/convergence_spd/`; tracked instructions are in `docs/CONVERGENCE.md`. At that stage, all 58 working-tree tests passed. A residual-checked iterative SPD solve with direct fallback made the refined runs practical; two completed direct baselines agree in volume histories within about 3e-12. All cases were recomputed under the new solver fingerprint and the original 2% gate.

- [x] Run spatial refinement: 20x16x16, 30x24x24, 40x32x32 at dt=1.
- [x] Run temporal refinement: dt=1, 0.5, 0.25 at 40x32x32.
- [x] Compare identical-setup growth, shape, liquid rise and conservation; export reports/figure and raw state.
- [x] Pass the coupled numerical gate for the declared focused benchmark. Finest spatial growth/face differences are 0.086%/0.098%; finest timestep differences are 0.157%/0.122%. All five 120-unit cases pass conservation, positivity and exact time. Historical uniform-grid failures remain recorded separately.

**Spatial sampling correction measured and committed:** A controlled geometry check found that the original donor-volume selection changed from 4.564000 to 3.099111 to 1.844667 across the three grids, although the same normal-prism neighborhood has volume 2.509447. Exact cell/prism overlaps now define supersaturation weights and fractional uptake. All 66 tests pass, including rotated/translated seeds, partial wetting, wall/surface clipping and nearly coincident corners. The fresh five-case full-duration study in `results/convergence_prism_v2/` reduces net-growth disagreement from 3.07% to 0.30% and face-advance disagreement from 5.68% to 3.70% under the unchanged 2% gate. Maximum relative tracked-mass error is 3.520e-12. This corrects a demonstrated sampling-domain inconsistency, while coupled spatial convergence remains incomplete. Explain each affected file before any commit/push.

- [x] Isolate the grid-dependent facet sampling volume with a constant-field geometry diagnostic.
- [x] Replace whole-cell selection with exact facet-prism weights and consistent fractional uptake.
- [x] Recheck the coupled spatial/time gate after the sampling correction; the completed study still fails the spatial face-advance criterion.

**Next step before AI:** Fixed-aligned-crystal Navier-Stokes/passive solute are verified on uniform and unequal/focused grids. Actual sloped/rotated cut-cell geometry, bounded topology resolution and shared compatible pressure projection are now independently verified. Next implement embedded no-slip viscosity and nonlinear momentum transport before full partial-cell time stepping and growing-boundary integration. Current evidence is in docs/CUTCELL_PRESSURE.md and docs/CUTCELL_PRESSURE_CHECKPOINT.json; AI gates remain in docs/AI_READINESS.md. Explain changes and retain file-top summaries. Both local milestones remain uncommitted for review. The user requests a commit-readiness recommendation, not a push: do not commit or push without explicit subsequent authorization.

### Simulation validation gate before AI

- [x] Check analytic cube/tetrahedron/octahedron geometry, rotations, invalid solids and disappearing faces.
- [x] Check exact solid/cell intersections, blocked-face areas and sub-cell seed volume.
- [x] Check grid-independent facet-prism volumes, rotated/translated neighborhoods, partial wetting, wall/free-surface clipping and conservative fractional uptake.
- [x] Check explicit full-box diffusion and implicit liquid diffusion against analytic solutions under refinement.
- [x] Check moving-boundary carrier/solute/heat conservation, discrete continuity, emerging crystals, trapped-fluid rejection and the full-rim growth limit.
- [x] Verify deterministic runs and exact final simulation time.
- [x] Quantify coupled growth/displacement sensitivity across grid and timestep changes using the completed full-duration study.
- [x] Establish coupled predictions within the predeclared 2% refinement gate for the focused benchmark; broader randomized training conditions still need regime-specific checks.
- [x] Define SI units, explicit concentration bases, sourced solubility relation and an isothermal fluid-property reference point with provenance in root SOURCES.md.
- [ ] Complete temperature/composition-dependent properties, solid density/lattice metric and face-kinetic calibration before claiming physical growth predictions.
- [ ] Compare memoryless and history-dependent facet kinetics against controlled published measurements before optimizing schedules; model/calibrate dissolution if the operating range includes it, otherwise explicitly exclude that regime.
- [ ] Validate growth predictions on independent experimental measurements, including uncertainty and operating ranges.
- [ ] Define independently measurable quality criteria and quality limits; a synthetic event counter is not optical-quality validation.
- [ ] Calibrate any inclusion/nucleation/quality response with meaningful data before using it for optimization.
- [ ] Model actuator delay, command ramp limits and control timing before control experiments.
- [ ] Validate a resolved flow model if quantitative stirring/buoyancy predictions are required; current displacement is quasistatic and stirring is effective diffusion.

**Geometry acceptance checks:** Known cube and tetrahedron volume/surface area; rotated seeds with unchanged volume; valid sloped faces; bounded/nondegenerate geometry; unchanged solid at zero growth; expected normal displacement under prescribed face motion. The coupled engine must additionally show 3D containment, conservation, nonnegative concentration, adequate timestep/grid convergence, and exact final-time termination when dt does not divide duration. Visual checks must show actual 3D vertices rather than an extruded 2D picture.

**Control-readiness checks:** Establish documented units and parameter ranges. Add face-specific growth calibration against traceable experimental measurements, including temperature, supersaturation and solution conditions. Test for trivial optimization: finite solute, local depletion and response delays must prevent the demonstrator from teaching an unsupported "always maximize circulation/rotation" strategy. The current engine uses wetted-area face kinetics, saturating effective stirring through liquid apertures, local uptake and moving-boundary displacement. These approximations still require calibration and a proper resolved-flow model before quantitative conclusions. Add transport and quality responses only when supported by data, and label uncalibrated mechanisms as synthetic. Define equipment lag, setpoint ramp limits and the control interval separately from the numerical timestep.

**Deliverable:** A functioning simulation where crystal geometry changes over time in response to environmental conditions.

## Phase 2 — Visualization

- [x] Render the 3D tank and actual crystal polyhedron, including sloped faces, with physical axis proportions.
- [x] Display labeled temperature/concentration slices through volumetric fields.
- [x] Visualize crystal volume, face normal growth rates, surface normals, displacement vectors and rising liquid level.
- [ ] Implement and visualize spatial defect mechanisms with meaningful labels; the current event count has no physical defect locations.
- [x] Play the window/GIF once to the exact final simulation time and hold the final state.
- [ ] Add start, pause, explicit restart and simulation-speed controls.
- [ ] Allow manual changes to temperature, rotation and flow.
- [x] Display recorded volume/liquid-level histories and status values during replay.
- [ ] Display genuinely live measurements while the numerical simulation is running; the current window replays a completed run.
- [x] Export a single-pass GIF, final PNG, measurement CSV, resolved JSON settings and actual volumetric state in fields.npz.

**Deliverable:** A usable visual simulation with interactive controls.

## Phase 3 — AI Environment

- [ ] Wrap the simulation in a Gymnasium environment.
- [ ] Implement `reset()`, `step()`, `render()` and `close()`.
- [ ] Define actions: temperature, rotation and circulation adjustments.
- [ ] Define observations: measurable simulation variables and fixed-size geometry summaries such as projected area, projected aspect ratio and orientation; use volume/surface estimates only with explicit measurement assumptions; avoid observations whose length depends on face count.
- [ ] Define objectives for growth, quality and constraint compliance.
- [ ] Enforce safe action bounds.
- [ ] Implement complete episode recording and replay.
- [ ] Test determinism and API compatibility.
- [ ] Define a timestamped observation contract with units, history windows, measurement confidence and missing-data flags, shared by simulator and future hardware adapters.
- [ ] Model sensor noise, delay and actuator response; keep exact simulator state out of deployment observations.
- [ ] Implement a configurable objective, hard process limits, rate limits and logged operator override.

**Deliverable:** A complete environment suitable for automated experiments.

## Phase 4 — Predictive AI

- [ ] Implement fixed-recipe and PID baselines.
- [ ] Generate varied simulation trajectories after the 3D simulator acceptance checks pass; version the simulator and geometry configuration with each dataset.
- [ ] Build training datasets containing states, actions and subsequent outcomes.
- [ ] Train XGBoost models to predict growth rate, asymmetry and simulated defect risk.
- [ ] Evaluate models on unseen episodes, seed geometries, orientations and parameter configurations.
- [ ] Save models, metrics and configurations.
- [ ] Implement candidate-action scoring using model predictions.
- [ ] Generate bounded experiments that vary actions independently enough to learn their effects, covering seed shape, initial conditions and disturbances. Save action timing and prediction horizon with each example.
- [ ] Split complete runs into training, validation and test groups before creating temporal windows; keep frames and neighboring samples from the same run together to prevent leakage.
- [ ] Train XGBoost using recent measured growth, sensor histories, current geometry and candidate actions to predict future growth over an explicit horizon. Add anomaly-risk targets only when meaningful labels exist.
- [ ] Implement a receding-horizon controller using bounded candidate search or sampled action sequences: predict outcomes, reject constraint violations, choose the best feasible sequence, apply only its first action and repeat from new observations.
- [ ] Assess prediction uncertainty and unsupported operating conditions on held-out runs. Fall back to a validated recipe/PID or request operator input when measurements or predictions are unreliable.
- [ ] Evaluate multi-step prediction error and actual closed-loop performance alongside regression metrics. Compare growth and quality outcomes against baselines across repeated seeds and held-out conditions.

**Deliverable:** An XGBoost-driven predictive controller.

## Phase 5 — Computer Vision

- [ ] Generate simulated camera images from the environment.
- [ ] Generate camera-projected segmentation masks from the same 3D crystal polyhedron and camera pose used by the renderer.
- [ ] Implement basic OpenCV geometry tracking.
- [ ] Train a YOLO segmentation model if OpenCV proves insufficient.
- [ ] Extract measurements and anomaly indicators from images.
- [ ] Integrate extracted features into XGBoost.
- [ ] Test performance with image noise and visual variations.
- [ ] Calibrate image scale and camera pose, track the same crystal over time and distinguish apparent motion from actual face growth. Estimate growth rates from time windows rather than one frame pair.
- [ ] Define separate labels and evaluation metrics for segmentation, face displacement, growth-rate estimation, visible inclusions and parasitic nucleation. Leave anomaly classes unsupported until corresponding data exists.
- [ ] Add a synthetic image/defect generator with explicit spatial and temporal ground truth; a scalar synthetic defect count alone cannot train a detector of visible defects.
- [ ] Test reflections, refraction, lighting changes, bubbles, occlusion and camera dropout. Report anomaly false alarms and detection delay, including the pitch's first-hour detection goal as an experimental target.
- [ ] Integrate estimated visual measurements into the controller and compare closed-loop performance with the same controller using ideal geometry observations.
- [ ] Evaluate on real annotated KDP images separately from synthetic data. Record dataset provenance, permissions and experiment identifiers; keep real test experiments out of training and calibration.

**Deliverable:** A vision-assisted control pipeline.

**Dependency:** Establish a minimal vision measurement contract and synthetic camera adapter during Phase 3. Phase 4 may use explicitly labeled noisy measurement proxies until Phase 5 replaces them with image-derived estimates. The integrated demonstration must use the vision path and must not silently fall back to exact simulator geometry.

## Phase 6 — Integration and Optimization

- [ ] Connect all components into one end-to-end system.
- [ ] Compare fixed-recipe, PID and AI control.
- [ ] Evaluate growth rate, asymmetry, defects and constraint violations.
- [ ] Test unseen disturbances and simulation configurations.
- [ ] Optimize performance and simulation speed.
- [ ] Create a Streamlit dashboard.
- [ ] Add automated benchmarking and reporting.
- [ ] Verify all components with integration tests.
- [ ] Demonstrate the full feedback loop on recorded or simulated camera streams, including an induced disturbance, uncertainty-triggered fallback and operator intervention. Log observations, model versions, candidate scores, commands and outcomes for replay.
- [x] Deliver a standalone 3D simulation demonstrator before AI integration.
- [ ] Complete the separate real-experiment validation milestone with calibrated sensors, operating limits and shadow-mode recommendations before automatic hardware control.
- [ ] Report usable growth, target-size time, anomaly detection delay, false alarms and constraint violations; add energy/time savings only when measured or explicitly modeled.

**Pitch alignment:** Early visible-change detection and human oversight remain core requirements. XGBoost supplies the predictive layer behind adaptive control. HPC fluid simulation and Qiskit remain optional future extensions, outside this MVP. The pitch's four-week schedule is a proposed scope/timebox, not evidence that real-world detection or growth gains have been demonstrated.

**Calibration references:** [KDP face-specific growth kinetics and supersaturation](https://www.sciencedirect.com/science/article/pii/S0022024802016585), [measurement of face-specific growth using imaging and concentration sensing](https://eprints.whiterose.ac.uk/id/eprint/105693/), [growth-rate fluctuations and inclusions in rapid KDP growth](https://pubs.acs.org/cgdefu/article/19/1/460/800810/Toward-the-Understanding-of-KDP-Crystal-Surface). These identify measurements and model gaps to investigate; they do not supply validated parameters for the current simulator.

**Deliverable:** A complete working KDP AI simulation demonstrator.

## Required Code Structure

/

```text
kdp-ai/
├── configs/
├── src/
│   ├── simulation/
│   │   ├── physics.py
│   │   ├── crystal.py
│   │   ├── tank.py
│   │   └── engine.py
│   ├── environment/
│   │   └── kdp_env.py
│   ├── rendering/
│   ├── vision/
│   ├── models/
│   ├── controllers/
│   └── dashboard/
├── scripts/
│   ├── run_simulation.py
│   ├── generate_dataset.py
│   ├── train_xgboost.py
│   └── benchmark.py
├── tests/
├── data/
├── results/
└── docs/FLUID_MODEL.md
```

## Agent Execution Rules

1. Start with the smallest functional simulation, then improve it.
2. Never leave an entire phase as placeholder code.
3. Test each component before integrating it.
4. Keep the physics model independent from AI and visualization.
5. Use reproducible simulation seeds and experiment configurations.
6. Do not use hidden simulator state as model inputs unless it could be measured in a real system.
7. Track progress in `PROGRESS.md`.
8. Explain each affected file and its changes before committing; wait for user review and explicit commit authorization. Use a short, focused commit per approved step.
9. Do not begin optional HPC, Qiskit or reinforcement learning work.
10. Continue through the phases in small reviewed steps. Fix failures before moving on, and push only with explicit user authorization.
11. Record websites, physical values, unit/basis conversions, uncertainty and provenance in root SOURCES.md before using new evidence. Distinguish measured/fitted/estimated values from derived values and project design choices.

## First Coding Milestone — Completed

Implement `src/simulation/` and `scripts/run_simulation.py`.

The committed runnable version now does the following:

- [x] Initialize a tank containing a supersaturated solution.
- [x] Place a crystal seed in the tank.
- [x] Simulate simplified crystal growth over time.
- [x] Respond to configurable temperature and concentration.
- [x] Update the 3D crystal polyhedron and volumetric solution concentration, deriving volume and dimensions from the resulting geometry.
- [x] Render the crystal's changing 3D geometry and hold the final state at the configured final time.
- [x] Run without machine learning or external datasets.

**Completed first milestone:** Running `python scripts/run_simulation.py` visibly demonstrates crystal growth over simulated time.

After completing this milestone, explain the result and wait for the user to choose and approve the next small step.

## Current checkpoint and active step

- [x] Commit and push the reviewed convergence runner, faster transport solver and exact facet sampling as b0ea6d3, 486adca and 561c551 to private salveszz/CrystaLab main.
- [x] Add fixed-geometry nonuniform-field diagnostics to separate sampling and cut-cell transport errors; this new work is uncommitted for review.
- [x] Implement and analytically verify geometry-aware liquid-field reconstruction/fluxes, then rerun the original coupled gate; its spatial face criterion still fails.

**Fixed-geometry diagnostic outcome (uncommitted):** `scripts/check_spatial_accuracy.py` separates exact facet-prism integration from piecewise-constant liquid averages, and tests the internal diffusion operator beside flat/sloped solid planes. Eight analytic-reference tests bring the working suite to 74 passes (66 committed). True liquid averages still give facet-mean errors; a tangent affine field produces spurious local net flux beside the sloped boundary. Exact-gradient oracle corrections remove that patch-test defect to roundoff. Reports are in `results/spatial_accuracy_561c551/`, method/limits in `docs/SPATIAL_ACCURACY.md`. These checks identify two spatial approximation errors without attributing all coupled face disagreement to either. Growth/quality physics, the runtime solver and the 2% coupled gate are unchanged.

## Sampling reconstruction checkpoint (uncommitted)

- [x] Compute runtime solid/prism first moments and actual liquid centroids.
- [x] Estimate local concentration/temperature gradients through open liquid faces and limit concentration slopes to stay nonnegative.
- [x] Reconstruct facet-neighborhood samples at overlap centroids; retain conservative uniform fractional uptake and existing transport equations.
- [x] Verify affine sampling against independent geometric references and run 80 tests (66 in the pushed checkpoint).
- [x] Complete a fresh five-case full-duration coupled study under the unchanged 2% gate; spatial face advance still fails at 3.79%.

The affine-field sampling error is now below 5e-16 across all six pose/grid cases. The known sloped-boundary transport patch-test defect is unchanged and remains the next separate correction. The same physical initial temperature profile is now stored as true liquid-cell averages. No commits or pushes are authorized for this new work.

**Sampling correction measured:** All five full-duration cases completed in `results/convergence_reconstruction_561c551/`. The sampler is affine-exact in independent checks, and all 80 tests pass. Coupled spatial net-growth disagreement is 0.87% and face advance 3.79%; neither improves on the previous finest-pair 0.30%/3.70% result. Timestep, conservation, positivity and final time pass. The next task is a separate geometry-aware cut-cell transport correction. This work remains local and uncommitted; no further commit or push without explicit user approval.

## Current numerical fixes and published-data preparation (uncommitted)

- [x] Correct diffusive fluxes for actual liquid-centroid offsets using estimated gradients, conservative edge corrections and bounded implicit fallback.
- [x] Recover tangent affine diffusion beside a sloped solid to roundoff; rerun the original coupled study (spatial face change 3.69%, still outside 2%).
- [x] Replace donor-only displacement with bounded linear midpoint transport using cell/aperture first moments; recover analytic translating concentration, temperature and velocity.
- [x] Rerun the original coupled study after remapping (spatial face change 3.74%, still outside 2%); preserve failed results.
- [x] Add a focused rectilinear liquid grid with actual capacities, openings, neighbor distances and correctly positioned field plots; keep the crystal continuous.
- [x] Check focused-grid geometry, affine sampling, bounded diffusion, diffusion refinement, translating layers and coupled conservation. All 96 tests pass.
- [x] Reproduce and fix the missing actual-bounds forwarding in concentration slope limiting; archive/interrupt the earlier focused attempt rather than mix its source with final results.
- [x] Complete the final-source five-case focused study in `results/convergence_focused_v2_561c551/` and pass all declared numerical gates. Verified exact source hashes and finite archives; inspected the exported comparison figure.
- [x] Verify the full focused demo/export through time 120: finite 40^3 fields, nonnegative wet concentration, conserved carrier volume, correct cell-edge rendering and 61 single-pass GIF frames.
- [x] Record a first published KDP face-rate reference with DOI, units, conditions, uncertainty gaps and explicit supported/unsupported uses (`configs/benchmarks/kdp_wu2016.json`). This is a regression summary, not raw experimental trajectories.
- [x] Transcribe controlled prismatic growth-rate Tables 1/2 from Maksimovic2025, preserving histories, counts, distinct modes/means and published uncertainty terms; do not treat aggregates as independent per-face trajectories.
- [x] Document physical-unit matching, separate calibration/whole-experiment validation, observation uncertainty, synthetic-data splits, action-ranking checks and independent quality targets in `docs/REAL_DATA_VALIDATION.md`.
- [ ] Establish physical units and audit controlled published growth/solubility measurements before fitting KDP parameters or generating an optimization training set.
- [ ] Obtain independent quality observations; do not treat missing quality labels as zero defects.

The user has no experimental measurements yet and selected published sources.
Real KDP calibration/validation, quantitative stirring response and quality
prediction remain unfinished. No AI model, commit or push is part of this step.

## Reviewed numerical checkpoint pushed

- [x] User authorized committing and pushing the complete numerical changes.
- [x] Commit `5eea924`: improve 3D liquid transport and add analytic/focused-grid checks. The exact staged tree passed 95 tests.
- [x] Commit `5449af3`: focused configurations, demo/refinement tools and method documentation. The exact staged tree passed all 96 tests.
- [x] Commit `64210bd`: published KDP reference records and the real-data validation procedure; metadata parsed and table shapes checked.
- [x] Push to private `salveszz/CrystaLab` main; local and remote HEAD verified at `64210bd9a16bda936be30c35e1896d1abbb22ad0`. Tracked working tree clean; formatting passes all 27 code files.

## Physical-flow sequence (uniform fixed flow and passive solute verified)

- [x] Define the SI unit/concentration contract, sourced isothermal KDP-water density/viscosity/diffusivity point, solubility relation and rotation/flow conversions. Heat properties, density derivatives and calibrated growth coefficients remain to obtain; see docs/PHYSICAL_MODEL.md and SOURCES.md.
- [x] Specify the first stationary isothermal open-tank rest boundary conditions and diagnostic Reynolds/Peclet/Schmidt scales in configs/physical_reference.json. Actual apparatus geometry, forcing histories and buoyancy regime remain separate inputs to audit.
- [x] Add and verify the initial uniform-grid, stationary-aligned-crystal 3D incompressible Navier-Stokes solver, including velocity/pressure, viscosity, nonlinear conservative momentum transport, rest/channel/decay/ABC references and blocked-face/divergence checks.
- [x] Extend momentum and passive solute to nonuniform/focused Cartesian cells with actual primal/dual volumes, face areas and distances. Independent numerical references pass.
- [x] Implement actual oblique geometry and shared compatible pressure projection, with independent manufactured Poisson, geometry-closure and topology-resolution verification.
- [x] Verify stationary embedded no-slip viscosity on real cut facets, with independent affine/shear refinement and energy/flux checks; see docs/CUTCELL_VISCOSITY.md.
- [x] Verify monolithic fixed cut-cell pressure/viscosity coupling (linear Stokes), including cellwise volume, stationary no slip, pressure work, energy, velocity/pressure and gradient-force refinement. All 198 tests and 17 refined numerical cases pass. The failed coarse focused study is retained; see docs/CUTCELL_STOKES.md and its checkpoint.
- [ ] Add conservative nonlinear cut-cell momentum fluxes, verify momentum/energy and advective small-cell handling, then enable the combined partial-cell runtime only after spatial/time verification.
- [x] Couple conservative passive solute advection-diffusion to resolved face volume fluxes, with high-Peclet broadening/refinement, positivity, mass and fixed-obstacle coupling checks. Empirical mixing/agitation multipliers do not enter this resolved path.
- [ ] Add heat advection-diffusion only after obtaining sourced thermal properties and physical boundary conditions.
- [ ] Add thermal and solutal buoyancy where the physical scales justify them. Use a Boussinesq approximation only when relative density variations are small enough.
- [ ] Couple the verified flow to facet-specific growth/solute flux, crystal motion and rising liquid level, retaining mass balance and a new coupled convergence study. Keep the initial free-surface approximation explicit; add waves only if the experiment requires them.
- [ ] Compare flow/mass-transfer behavior with published experiments and assess runtime before generating large synthetic batches. Navier-Stokes alone does not validate kinetics, optical quality or control recommendations.

Primary research supports resolving natural/forced convection in KDP growth:
[Navier-Stokes/Boussinesq KDP study](https://www.sciencedirect.com/science/article/abs/pii/S0022024899007459).
A [KDP schlieren experiment](https://www.sciencedirect.com/science/article/abs/pii/S0022024804013612)
is a candidate for concentration-pattern validation; its numerical observations
and uncertainty have not been imported or used to fit this simulator.

## Results cleanup and physical definition (local, uncommitted)

- [x] Clear all 34 old generated top-level entries in results/, retaining only .gitkeep; preserve the final numerical protocol/metrics/source hashes in docs/NUMERICAL_CHECKPOINT.json before deletion.
- [x] Create root SOURCES.md with websites, exact imported growth tables, solubility coefficients, fluid-property inputs, uncertainty gaps, design choices and legacy dataset provenance. Add a source-register maintenance rule.
- [x] Implement SI temperature/rotation/flow/growth-rate conversions, binary-solution concentration conversions, bounded-range KDP solubility and transport-scale calculations in simulation.units.
- [x] Add configs/physical_reference.json and scripts/describe_physical_setup.py to validate/inspect the stationary isothermal physical definition without generating result files or pretending a flow solve has run.
- [x] Verify published supersaturations at their rounding precision, source-unit conversions, mass identities, physical scales and input rejection. All 106 tests pass; no commit/push authorized for this new step.
- [x] Document physical units, boundary/pressure assumptions, planned field layout, forcing, buoyancy, growth-interface accounting and the integration sequence in docs/PHYSICAL_MODEL.md. Mark old raw-result references as removed.
- [x] Implement and verify the first stationary uniform-grid 3D momentum component; see the next section. Conservative scalar/growth coupling remains pending and the illustrative growth engine is still separate.

## SI checkpoint pushed; first 3D momentum component verified locally

- [x] User authorized commit/push of the units/source checkpoint, then roadmap implementation. Committed 018fe15 (sources/checkpoint; staged96 tests) and 5505e9d (SI definition; staged106 tests), then pushed. Verified matching remote SHA5505e9d67d9530d11ef5c4d8a6fbb3f15265e948 and PRIVATE salveszz/CrystaLab.
- [x] Add simulation.flow: 3D staggered native velocity/pressure, implicit physical viscosity, limited conservative nonlinear momentum fluxes, incremental pressure correction, wall/solid no slip and an explicit rigid-lid/gauge treatment.
- [x] Use canonical normalized Crystal intersections for exact fixed-cube exclusion; reject unsupported partial momentum cells.
- [x] Reproduce and retain the failed donor-only ABC refinement study with all five source SHA256 values verified; improve momentum reconstruction without relaxing the original protocol/tolerances.
- [x] Complete all11 cases in results/flow_verification_final_5505e9d: PASS. Finest errors: channel0.03343%, time decay0.15657%, nonlinear ABC velocity0.20155%/pressure1.40745%; all refinement trends pass. Zero blocked-face velocities; max normalized divergence3.075e-16.
- [x] Export and inspect final flow/pressure and refinement figures; verify all raw fields finite/correctly dimensioned and exact source hashes/archives. Fix a Matplotlib input-array mutation and add a reporting-purity regression.
- [x] Pass all122 tests; record source/method choices, failed-stage history and full measured metrics in root SOURCES.md and docs/FLOW_MODEL.md.
- [x] User authorized flow review plus verified solute transport as the current commit/push scope on 2026-11-08. Reviewed and committed momentum as 5a015f5 after all 122 baseline tests passed.
- [x] Implement conservative high-Peclet passive solute transport and fixed-flow coupling. The next momentum/growth milestones remain separate.

## Fixed-grid solute checkpoint (2026-11-08)

- [x] Add simulation.solute: solution-volume SI concentration, shared face mass fluxes, MC reconstruction, multidimensional correction budgets, SSP RK2 and explicit stability subcycling.
- [x] Couple momentum and solute with old/new mean native face volume fluxes, synchronized clocks and atomic failure rollback. Add volume-flux export to simulation.flow.
- [x] Verify continuous 3D wave/diffusion cell averages, high-Peclet periodic Gaussian transport and semidiscrete diffusion time order. Check sharp-front bounds, zero wall leakage and actual nonzero fixed-cube coupling.
- [x] Pass all 15 cases in results/solute_verification_final_5a015f5 under the original protocol, with exact source/archive hashes, finite fields and visually inspected figures. Maximum relative mass drift 6.661e-16; all bound violations and coupled uniform concentration errors zero. Finest pulse error 0.88601%; excess broadening 0.01139 times physical D.
- [x] Repeat all 11 momentum cases with final sources: results/flow_verification_solute_5a015f5 passes. Full suite passes 138 tests; formatting and focused lint checks pass.
- [x] Record sources, method choices, limitations and a tracked numerical summary in SOURCES.md, docs/SOLUTE_MODEL.md and docs/SOLUTE_CHECKPOINT.json. Generated raw studies remain ignored.
- [x] Commit and push the authorized milestone: 5a015f5 (verified fixed-domain 3D flow), then 8011b79 (conservative solute transport/coupling). Local HEAD and origin/main independently verified at 8011b790f4469b78ac0cbdf9ef509d425050329d; tracked working tree clean.
- [ ] Next: nonuniform/focused and oblique cut-cell momentum, followed by interfacial uptake and moving-boundary integration. Source density/partial-volume closure before claiming physical growth/rising-level conservation.

## Rectilinear/focused momentum checkpoint (local, uncommitted; no push authorized)

- [x] User requested continued simulation preparation for AI, an explanation of all changes and a purpose summary at the top of every code file. Do not push until the user approves. Keep this step local for review.
- [x] Add shared simulation.mesh metrics and move the existing focused-axis generator there. Tank imports the same generator; all 39 Python source/test/script files have explanatory top summaries.
- [x] Extend canonical simulation.flow with explicit edge arrays, compatible actual-volume pressure projection, volume-weighted gauges/energy, symmetric viscous stiffness and conservative dual momentum fluxes. Periodic end widths can differ.
- [x] Extend simulation.solute with actual capacities, face conductances and reconstruction offsets. Keep solution-volume concentration, bounded flux corrections and stable subcycling.
- [x] Update flow/scalar verification builders to use actual mesh coordinates/error weights and export real edges/volumes; include new dependencies in source fingerprints.
- [x] Add configs/rectilinear_verification.json, scripts/check_rectilinear_flow.py and thirteen operator/refinement/study-integrity tests.
- [x] Retain the failed initial diffusion study (0.156298% at n=16 exceeds 0.15%); increase only diffusion grid levels from 6/10/16 to 8/12/20. Keep the same physical fields and original threshold.
- [x] Final source-matched results/rectilinear_verification_final_v2_8011b79 passes all twenty cases. Finest errors: channel0.04044%, time decay0.15730%, stretched ABC velocity/pressure0.32612%/1.12922%, focused ABC0.81133%/2.47371%, scalar wave2.20441%, diffusion0.10067%. Max normalized divergence5.155e-16; no measured scalar mass drift/bound violations or uniform scalar error.
- [x] Current-source uniform momentum (eleven cases) and uniform/high-Peclet scalar (fifteen cases) regressions pass. Full suite passes151 tests; retain exact sources/mesh outputs and inspect figures.
- [x] Add docs/RECTILINEAR_FLOW.md with a file-by-file explanation, docs/RECTILINEAR_CHECKPOINT.json with reproducible metrics and docs/AI_READINESS.md with outstanding training gates; update model/source documentation.
- [ ] User review of local changes; no commit or push in this step.
- [ ] Next: oblique momentum cut-cell operators, then conservative uptake/motion with a physical density/partial-volume closure, coupled refinement and calibrated growth/quality/actuator/observation contracts before growth-control training.

## Cut-cell geometry/pressure foundation (local, uncommitted; recommendation only)

- [x] User requested the next implementation step, a full code-wise explanation and a recommendation on committing. Push remains explicitly forbidden without approval. No commit or push performed.
- [x] Extract canonical polygon clipping and add Crystal.clipped_face_moments for stable original-facet wall pieces.
- [x] Add simulation.cutcell for normalized SI capacities/apertures/centroids/wall moments, surface/first-moment closure and explicit disconnected-pocket detection.
- [x] Add bounded static axis bisection with recorded refined edges/history. Preserve the same three-axis rotated crystal; do not merge disconnected liquid through one pressure unknown. Starting grids8/12/16 resolve to14x12x12/20x19x20/31x27x33 with14/23/43 splits; configured limit48 remains explicit.
- [x] Add simulation.pressure and simulation.linear, sharing compatible integrated pressure solves, gauges and residual checks with the canonical flow runtime. Keep projection A*d norm separate from physical cut-cell kinetic energy and impermeability separate from tangential no slip.
- [x] Add fourteen cut-cell tests and the fourteen-case independently sourced protocol/runner. Volume-average manufactured sources use analytic Laplacians and Gaussian integration, not the numerical matrix. Preserve case failures and reject overwritten study folders.
- [x] Retain the aborted first rotated-cell attempt and its focused n=16 potential error6.4348% above5%. Keep5% gates and increase Poisson levels to8/16/24; record the distinct tiny-layer precision criterion explicitly.
- [x] Final source-matched results/cutcell_pressure_final_v2_8011b79 passes14 cases. Finest uniform/focused potential errors0.41065%/2.17234%; gradient projection residuals0.36367%/1.08533%. Max relative integral residual4.616e-13; max area/moment closure9.189e-14/8.782e-12; blocked-face speed0. Absolute divergence6.717e-8 1/s in tiny cells remains reported; this is not a cut-cell time-step stability claim.
- [x] Re-run current-source rectilinear20, uniform-flow11 and scalar15 cases: all pass. Full exact-source suite165 tests passes. Verify archives/hashes/finite fields and inspect final figures; all44 Python files have top summaries.
- [x] Add docs/CUTCELL_PRESSURE.md with full method/file review, docs/CUTCELL_PRESSURE_CHECKPOINT.json with reproducible evidence, and update SOURCES.md/model/readiness docs.
- [ ] Recommend checkpointing the verified foundations in two focused commits after user review: rectilinear flow/solute, then cut-cell geometry/shared pressure. Do not describe them as complete sloped no-slip Navier-Stokes or AI-ready growth. Await explicit commit authorization and separate push approval.
- [ ] Next: embedded no-slip viscosity/nonlinear momentum and small-cell dynamics, followed by physical density/partial-volume closure, uptake/moving-interface coupling, calibrated kinetics/quality and actuator/observation contracts.

## Approved checkpoints and real-data development (2026-11-08)

- [x] User explicitly approved commit and push. Pushed rectilinear flow/solute as `ab0a6a1`, cut-cell geometry/shared pressure as `cb83f64`, and documentation status clarification as `4bcb129` to origin/main. Verified local/remote HEAD `4bcb129bef6faec9b1a04baac7fbc4e2615a501e` and a clean tracked working tree.
- [x] Tested the extracted first staged tree: 151 tests pass. Final numerical source: 165 tests pass, 44 Python files have explanatory headers, formatting/focused lint pass, and archived/current source hashes match the 14-case cut-cell study and 46 regression cases.
- [x] Reviewed all twelve slides of Pitch_KDP.pptx, including its embedded crystal image. Slides 7-9 put real/synthetic data before model development and describe HPC as an extension. The roadmap chooses XGBoost for the predictive layer; camera monitoring and human oversight remain part of the intended MVP.
- [x] Clarified docs/AI_READINESS.md: real-data preparation and an offline supervised growth baseline can proceed while simulation work continues. Training/environment scripts are still placeholders. Published aggregate records and obsolete ignored planar trajectories do not supply a validated current training dataset.
- [ ] Next data milestone: define experiment IDs, timestamps, unit/concentration basis, measured features, uncertainty, action histories and a prediction horizon; import/audit a real dataset; split whole experiments before temporal windows; compare a simple reference predictor with XGBoost on held-out real runs. Add quality targets only with meaningful labels.
- [ ] Parallel physics milestone: embedded no-slip viscosity, conservative nonlinear momentum and small-cell dynamics, followed by density/partial-volume closure, interfacial uptake/motion and independent growth calibration. These remain necessary for credible simulation-driven growth-control training, rather than prerequisites for importing real observations.

## Bounded structural reference and offline growth data (local, 2026-11-08)

- [x] User authorized proceeding with the bounded COD check, source audit and offline pipeline. Commit/push approval has not been requested or granted for this new milestone.
- [x] Pin COD-9010000 archive/entry hashes, verify scalar transcription and formula/cell consistency, calculate solid density 2333.8948984452236 kg/m3 and reciprocal-lattice normals at 295 K. Missing pressure remains unknown; illustrative runtime parameters are unchanged.
- [x] Add configs/data_sources.json and a bounded optional-download audit. Seven listed asset requests failed (403/404); preserve actual failures in docs/DATA_AUDIT.json. No candidate is misrepresented as an imported raw KDP trajectory dataset.
- [x] Define SI facet observations with stable identities, dependency groups, uncertainty/confidence, missing sensors, measurement origins and separate logged setpoints/timestamps. Reject aggregates, inconsistent identities and duplicate timestamps.
- [x] Implement whole-group split manifests and causal history/future-horizon datasets, then zero-growth/training-mean/recent-rate baselines and optional native CPU XGBoost. Validation-only early stopping and best-iteration reload preserve holdout isolation. No quality or controller implementation is claimed.
- [x] Exact final full suite: 189 tests PASS (24 new), 51.492 s; 54 Python files have purpose summaries and pass formatting; focused lint and whitespace checks pass. Complete synthetic CLI software check succeeds with XGBoost 3.3.0. Current/archive hashes match all five saved model source files.
- [x] Preserve synthetic diagnostics: held-out recent-rate MAE 0.4 nm/s is below XGBoost 0.601996 nm/s. This is software evidence only, with one held-out fixture group; do not claim KDP predictive performance.
- [ ] Next data task: obtain accessible measured KDP facet trajectories (timestamps, scale, experiment IDs, conditions), normalize into the documented contract and evaluate whole held-out experiments before interpreting ML improvements. Keep other materials, aggregate calibration and synthetic checks separate.
- [ ] Physical growth/flow integration, partial-volume closure, kinetic/quality calibration, vision inference and hardware/control adapters remain unfinished. Avoid a broad CIF framework or additional physics expansion as a prerequisite for inspecting real observations.
