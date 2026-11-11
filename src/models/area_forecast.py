# FILE PURPOSE: Forecast projected pixel-area changes using strictly causal estimate histories.
# Missing/uncertain views reset history. Run/clock/view/shape changes cannot share
# state; fitted gain calibration is distinct from an action-response or quality model.
"""Bounded projected-area trend features and transparent forecast baselines."""

from collections import deque
from dataclasses import dataclass

from controllers.contracts import finite, identifier

METHODS = (
    "persistence",
    "recent_rate",
    "train_mean_fraction",
    "calibrated_recent_rate",
)
FEATURES = (
    "current_area_px2",
    "recent_rate_px2_per_source_s",
    "history_duration_source_s",
    "horizon_source_s",
    "frame_pixels",
)


@dataclass(frozen=True)
class AreaForecastConfig:
    schema: str = "crystalab_projected_area_forecast_v1"
    horizon_source_s: float = 0.25
    maximum_gap_source_s: float = 0.5
    maximum_age_source_s: float = 0.5
    ridge: float = 0.0001
    coefficient_min: float = 0.0
    coefficient_max: float = 3.0

    def __post_init__(self):
        if self.schema != "crystalab_projected_area_forecast_v1":
            raise ValueError("Unsupported projected-area forecast configuration")
        for key in (
            "horizon_source_s",
            "maximum_gap_source_s",
            "maximum_age_source_s",
            "ridge",
        ):
            if finite(getattr(self, key), key, minimum=0) == 0:
                raise ValueError(
                    "Forecast horizons, clock limits and ridge must be positive"
                )
        finite(self.coefficient_min, "coefficient minimum", minimum=0)
        finite(self.coefficient_max, "coefficient maximum", minimum=0)
        if self.coefficient_max <= self.coefficient_min:
            raise ValueError("Forecast coefficient interval is invalid")


@dataclass(frozen=True)
class AreaEstimate:
    run_id: str
    clock_id: str
    view_id: str
    frame_index: int
    observed_at_s: float
    received_at_s: float
    image_shape: tuple[int, int]
    area_px2: float | None
    valid: bool
    requires_inspection: bool
    evidence_kind: str
    mask_sha256: str | None = None

    def __post_init__(self):
        for key in ("run_id", "clock_id", "view_id"):
            identifier(getattr(self, key), key)
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("Forecast estimates need original frame indices")
        if len(self.image_shape) != 2 or any(
            type(v) is not int or v <= 0 for v in self.image_shape
        ):
            raise ValueError("Forecast estimates need native image dimensions")
        finite(self.observed_at_s, "observation time", minimum=0)
        finite(self.received_at_s, "receipt time", minimum=0)
        if self.observed_at_s > self.received_at_s:
            raise ValueError("Area estimate cannot precede its measurement")
        if type(self.valid) is not bool or type(self.requires_inspection) is not bool:
            raise ValueError("Validity and inspection state must be explicit")
        if self.evidence_kind not in {"synthetic", "fixture", "real"}:
            raise ValueError("Declare the forecast evidence kind")
        if self.area_px2 is not None:
            finite(self.area_px2, "projected area", minimum=0)
            if not 0 < self.area_px2 <= self.image_shape[0] * self.image_shape[1]:
                raise ValueError("Projected area lies outside the image")
        if self.valid and (
            self.area_px2 is None or not self.mask_sha256 or len(self.mask_sha256) != 64
        ):
            raise ValueError("Valid area estimates require an explicit mask identity")


class AreaTrend:
    """Two usable comparable estimates supply the recent-rate feature.

    Context changes reset history; a source clock never becomes physical growth
    time implicitly. A valid seed measurement can still require inspection and
    is then excluded from automatic forecasting.
    """

    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=2)
        self.context = self.previous = None
        self.last_received = None

    def update(self, estimate, *, now_s=None):
        if not isinstance(estimate, AreaEstimate):
            raise ValueError("Use AreaEstimate records")
        now = (
            estimate.received_at_s
            if now_s is None
            else finite(now_s, "forecast time", minimum=0)
        )
        if estimate.received_at_s > now:
            raise ValueError("Forecast estimate has not been received")
        context = (
            estimate.run_id,
            estimate.clock_id,
            estimate.view_id,
            estimate.image_shape,
        )
        reset = None
        if self.context != context:
            self.history.clear()
            reset = (
                "initial_context" if self.context is None else "stream_context_changed"
            )
            if self.previous is None or (
                self.previous.run_id,
                self.previous.clock_id,
            ) != (estimate.run_id, estimate.clock_id):
                self.previous = None
                self.last_received = None
        if self.last_received is not None and now <= self.last_received:
            raise ValueError("Forecast receipt clock must strictly increase")
        if self.previous is not None:
            if (
                estimate.frame_index <= self.previous.frame_index
                or estimate.observed_at_s <= self.previous.observed_at_s
            ):
                raise ValueError("Forecast source frame/time must strictly increase")
            if (
                estimate.observed_at_s - self.previous.observed_at_s
                > self.config.maximum_gap_source_s
            ):
                self.history.clear()
                reset = "observation_gap"
        self.context = context
        self.previous = estimate
        self.last_received = now
        reason = (
            "invalid_measurement"
            if not estimate.valid
            else "inspection_required"
            if estimate.requires_inspection
            else "stale_measurement"
            if now - estimate.observed_at_s > self.config.maximum_age_source_s
            else None
        )
        if reason:
            self.history.clear()
            return dict(
                features=None, history_ready=False, reason=reason, reset_reason=reset
            )
        self.history.append(estimate)
        features = dict(
            current_area_px2=float(estimate.area_px2),
            recent_rate_px2_per_source_s=None,
            history_duration_source_s=None,
            horizon_source_s=self.config.horizon_source_s,
            frame_pixels=estimate.image_shape[0] * estimate.image_shape[1],
        )
        ready = len(self.history) == 2
        if ready:
            first, last = self.history
            duration = last.observed_at_s - first.observed_at_s
            features.update(
                recent_rate_px2_per_source_s=(last.area_px2 - first.area_px2)
                / duration,
                history_duration_source_s=duration,
            )
        return dict(
            features=features,
            history_ready=ready,
            reason=None if ready else "insufficient_history",
            reset_reason=reset,
        )


def predictions(features, *, coefficient, mean_fraction):
    """Return comparable baselines; a missing history does not become a zero rate."""
    if features is None:
        return {method: None for method in METHODS}
    if set(features) != set(FEATURES):
        raise ValueError("Forecast inputs differ from the causal feature whitelist")
    area = finite(features["current_area_px2"], "current area", minimum=0)
    horizon = finite(features["horizon_source_s"], "horizon", minimum=0)
    pixels = finite(features["frame_pixels"], "image pixels", minimum=0)
    if area <= 0 or horizon <= 0 or area > pixels:
        raise ValueError("Forecast area/horizon/image bounds are invalid")
    finite(coefficient, "gain calibration")
    finite(mean_fraction, "mean relative gain")
    recent = features["recent_rate_px2_per_source_s"]
    if recent is not None:
        finite(recent, "recent rate")
    gains = dict(
        persistence=0.0,
        train_mean_fraction=area * mean_fraction,
        recent_rate=None if recent is None else recent * horizon,
        calibrated_recent_rate=None
        if recent is None
        else recent * horizon * coefficient,
    )
    output = {}
    for method in METHODS:
        gain = gains[method]
        if gain is None:
            output[method] = None
            continue
        requested = area + gain
        finite(requested, "forecast arithmetic result")
        future = min(pixels, max(0.0, requested))
        output[method] = dict(
            future_area_px2=future,
            gain_px2=future - area,
            relative_gain=(future - area) / area,
            clipped_to_image=future != requested,
        )
    return output


def fit_calibration(samples, config):
    eligible = [
        s
        for s in samples
        if s["split"] == "train"
        and s["trend"]["history_ready"]
        and s["labels"] is not None
    ]
    if len({s["family_id"] for s in eligible}) < 2:
        raise ValueError(
            "Forecast calibration needs at least two training families with causal histories"
        )
    numerator = denominator = 0.0
    fractions = []
    for sample in eligible:
        f = sample["trend"]["features"]
        if set(f) != set(FEATURES):
            raise ValueError("Training inputs violate the feature whitelist")
        x = (
            f["recent_rate_px2_per_source_s"]
            * f["horizon_source_s"]
            / f["current_area_px2"]
        )
        y = sample["labels"]["relative_proxy_gain"]
        finite(x, "training relative rate")
        finite(y, "training relative label")
        numerator += x * y
        denominator += x * x
        fractions.append(y)
    raw = numerator / (denominator + config.ridge)
    if denominator == 0:
        raise ValueError("No observed rate signal supports gain calibration")
    coefficient = min(config.coefficient_max, max(config.coefficient_min, raw))
    return dict(
        coefficient=coefficient,
        unbounded_coefficient=raw,
        coefficient_clipped=coefficient != raw,
        mean_fraction=sum(fractions) / len(fractions),
        fit_samples=len(eligible),
        rate_signal_sum_squares=denominator,
        training_sample_ids=[s["sample_id"] for s in eligible],
        training_family_ids=sorted({s["family_id"] for s in eligible}),
        feature_whitelist=list(FEATURES),
        target="next visible optical-proxy area and gain; not volume or action-conditioned response",
    )
