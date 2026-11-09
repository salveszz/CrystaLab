# Validation for a synthetic XGBoost training environment

The simulator does not need to reproduce every physical detail. It needs enough
accuracy for the decisions the model will make, with known limits. Passing a
grid-refinement check verifies numerical behavior; it does not validate KDP
kinetics, mixing or quality. A synthetic-only XGBoost evaluation measures agreement
with the simulator and cannot by itself show real-world control performance.

NIST distinguishes measurement/input uncertainty from inadequacy of the model
itself, and describes using limited physical experiments to calibrate simulations:
[calibration and uncertainty analysis](https://www.nist.gov/publications/calibration-and-uncertainty-analysis-predictions-computational-models).
The steps below are our proposed validation workflow, not completed validation.

[SOURCES.md](../SOURCES.md) is the central website/data register.
[PHYSICAL_MODEL.md](PHYSICAL_MODEL.md) defines the new SI reference inputs and
concentration conversions. That reference does not yet calibrate the growth engine.

[PHYSICAL_DATA.md](PHYSICAL_DATA.md) documents the bounded COD reference and
standalone source/access audit. Model training remains deferred. Published fit
summaries and aggregate tables are not raw trajectories, and no inaccessible
candidate file is represented as an imported experimental dataset.

## First published reference

`configs/benchmarks/kdp_wu2016.json` records the two face-growth regression
summaries from the KDP case in [Wu et al. (2016)](https://doi.org/10.1021/acs.cgd.6b00189),
equations 11A/B in the [accepted manuscript](https://eprints.whiterose.ac.uk/105693/3/20160707%20Crystal%20Face%20Specific%20Growth%20Kinetics%20FINAL%20x-x.pdf).
Reported prismatic/pyramidal normal rates are 13.56/118.03 micrometres/hour,
respectively. Their fitted intercepts are 296.71/418.54 micrometres. The batch was
approximately 82 minutes at 290 K, with occasional manual stirring. Temperature
and concentration were not tightly controlled. A concentration caption conflicts
with the methods text. Raw frame trajectories, rate uncertainty, measured
supersaturation and quality labels are absent from this record.

This is a useful first rate/geometry reference, not sufficient evidence for an
action-conditioned growth law. The stored values are published fit summaries;
generating points from those lines would create derived synthetic samples, not
new experimental observations. Fit R-squared is not a measurement-error interval.

`configs/benchmarks/kdp_maksimovic2025.json` also transcribes Tables 1/2 from
[Maksimovic et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11800150/).
It preserves separate supersaturation histories, sample counts, Table 1 modes,
Table 2 means and the published plus/minus values. These aggregate measurements
are a stronger starting point for prismatic kinetics. They are not independent
raw trajectories: faces share crystals/history, and uncertainty definitions need
audit before weighting a fit. Pyramidal kinetics, agitation response and optical
quality remain unsupported. No runtime parameter has been fitted to either source.

Our inference for the model design is that a memoryless growth law should be
tested against history-dependent alternatives before optimizing temperature
schedules. The present law depends on local supersaturation and illustrative
coefficients; it does not represent facet surface recovery or dissolution.
Negative supersaturation currently pauses growth. If candidate schedules include
dissolution, that response needs its own measured law and conservative boundary
motion, or the intended operating range must exclude it explicitly.

## Match the experiment before comparing

1. Establish physical units for length, time, concentration, density and diffusion.
   The current illustrative defaults cannot be compared numerically with
   micrometres/hour by choosing an arbitrary scale to make one curve match.
2. Match seed geometry/orientation and face-family mapping, tank size, solution
   volume, initial temperature/concentration, sampling definition and recorded
   action history. Record unreported conditions as unknown rather than inventing
   precise setpoints. Configured vectors are Cartesian geometric normals; a
   physical crystal's Miller indices also require its lattice metric before
   conversion to those normals.
3. Compare the same observation: normal face displacement with normal displacement,
   measured mass with mass, projected camera dimensions with matching projections.
   A bounding-box dimension is not a facet-normal measurement, and one camera
   does not independently measure full 3D volume or internal quality. Match the
   distance origin too: a center-to-face distance differs from a world plane
   offset if the crystal center moves. The 2016 fitted intercepts are not world
   coordinates for the simulator's seed planes.
4. Include instrument uncertainty, repeated-batch variability and remaining
   numerical error in the comparison. Report raw signed/absolute errors alongside
   relative errors; near-zero growth makes percentages unstable.

Runtime concentration is dissolved mass per carrier-liquid volume. The 2025
reference uses mass of KDP per mass of water. Convert both concentration and
solubility using a consistent carrier-density basis before combining them with
solid density and crystal volume. A unit conversion alone does not validate the
current negligible-solute-partial-volume approximation.

## Calibrate, freeze, then validate

Collect multiple experiments spanning the intended temperature, supersaturation,
seed size/orientation and agitation range. Use traceable KDP solubility/density
data and fit face-specific growth parameters on a calibration subset. Do not
absorb all errors in arbitrary kinetic coefficients: transport and sensor bias
can otherwise be mistaken for surface kinetics.

Reserve whole experiments, preferably a separate campaign/source, for validation.
Keep every frame from a batch in the same split. Freeze model choices and
parameters before looking at the held-out results. Predeclare acceptable errors
for the intended decision, based on measurement precision and the growth/quality
difference between candidate actions. The numerical 2% working tolerance is not
a universal experimental acceptance criterion.

If the calibrated simulator retains a systematic bias, a real-data residual
model can predict the difference between its outcomes and observed outcomes.
That hybrid remains subject to a separate real test set and an explicitly
limited operating range; fitting residuals is not evidence that the bias will
stay the same under unseen actions.

For each held-out run report face displacement/rate errors, volume or mass gain,
temperature/concentration response and timing. Inspect residuals against
temperature, supersaturation, face family and action. Systematic residuals show
where a model correction or a narrower valid operating range is needed. A single
fit summary cannot support independent temporal residual analysis or calibration
and validation splits by itself.

## Protect the XGBoost training task

- Train on candidate action schedules and future growth/quality outcomes over
  defined horizons. Evaluate whether action rankings and constraint decisions
  transfer, as well as average prediction error. One manual-stirring batch cannot
  establish causal responses to automated rotation/circulation schedules.
- Generate an ensemble using calibration-supported uncertainty ranges for
  kinetics, transport and observations. Store the sampled parameters and source
  provenance for audit, separately from controller-observable inputs.
- Keep synthetic training/validation seeds and entire parameter combinations
  separate. Keep the final physical validation set wholly real. Evaluate a
  fixed-recipe baseline before claiming benefit from an XGBoost controller.
- Identify simulator shortcuts. Current stirring has a saturating positive
  transport effect and no calibrated stirring-related quality cost. A model can
  learn "always maximize stirring" from that assumption. Arbitrary reward
  penalties or random defect labels would not validate the opposite response.
- Quality requires an independently measured target: for example annotated
  visible inclusions, measured transmission or a defined usable-crystal criterion.
  These are different measurements. The existing synthetic event counter cannot
  supply ground-truth optical quality. Leave unsupported quality labels marked
  unavailable; do not substitute zero defects for missing measurements.
- Use prediction uncertainty/out-of-range detection and a conservative recipe
  outside the validated range. Evaluate on real recorded experiments, then in
  shadow mode before making physical action recommendations.

An imperfect simulator can support software development and synthetic pretraining.
Real-data comparisons determine where its learned recommendations are credible;
they do not require every residual to become zero. Until units, action response
and quality are supported, this project remains a synthetic demonstrator rather
than a validated growth optimizer.

The [KDP batch growth/dissolution kinetics study](https://doi.org/10.1021/cg501251e)
is another candidate to obtain and audit. Its numerical dataset has not been
imported. The two reference records above do not constitute completed physical
calibration or independent validation of this simulator.
