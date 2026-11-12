# FILE PURPOSE: Update a bounded observable-area predictor from matured causal measurements.
# Source truth, future observations, process-hidden state and action effects are not inputs.
"""Online gain calibration and a declared forecast-informed pause recommendation."""

from dataclasses import asdict, dataclass
import copy
import math

from controllers.contracts import finite
from controllers.supervision import value_sha256
from models.area_forecast import AreaEstimate, AreaForecastConfig, AreaTrend


@dataclass(frozen=True)
class AdaptiveAreaConfig:
    schema: str = "crystalab_adaptive_area_policy_v1"
    description: str = "Software-only forecast-informed recipe-pause diagnostic"
    horizon_source_s: float = 0.25
    maximum_gap_source_s: float = 0.5
    maximum_age_source_s: float = 0.5
    initial_coefficient: float = 1.0
    coefficient_min: float = 0.0
    coefficient_max: float = 3.0
    prior_precision: float = 3.0
    forgetting_factor: float = 0.95
    error_ema_alpha: float = 0.2
    minimum_matured_updates: int = 3
    maximum_error_ratio: float = 3.0
    reference_positive_gain_quantile: float = 0.9
    minimum_reference_relative_gain: float = 0.001
    minimum_pause_model_s: float = 15.0
    maximum_pause_model_s: float = 60.0

    def __post_init__(self):
        if self.schema != "crystalab_adaptive_area_policy_v1":
            raise ValueError("Unsupported adaptive projected-area policy")
        for key in (
            "horizon_source_s",
            "maximum_gap_source_s",
            "maximum_age_source_s",
            "prior_precision",
            "maximum_error_ratio",
            "minimum_reference_relative_gain",
            "minimum_pause_model_s",
            "maximum_pause_model_s",
        ):
            if finite(getattr(self, key), key, minimum=0) == 0:
                raise ValueError("Adaptive limits must be positive")
        for key in (
            "forgetting_factor",
            "error_ema_alpha",
            "reference_positive_gain_quantile",
        ):
            if not 0 < finite(getattr(self, key), key) <= 1:
                raise ValueError("Adaptive weights/quantile must be in (0,1]")
        for key in ("initial_coefficient", "coefficient_min", "coefficient_max"):
            finite(getattr(self, key), key, minimum=0)
        if (
            not self.coefficient_min <= self.initial_coefficient <= self.coefficient_max
            or self.coefficient_min >= self.coefficient_max
        ):
            raise ValueError("Adaptive coefficient bounds are invalid")
        if (
            type(self.minimum_matured_updates) is not int
            or self.minimum_matured_updates < 1
        ):
            raise ValueError("Declare a positive matured-update requirement")
        if self.minimum_pause_model_s > self.maximum_pause_model_s:
            raise ValueError("Adaptive pause bounds are reversed")

    def trend_config(self):
        return AreaForecastConfig(
            horizon_source_s=self.horizon_source_s,
            maximum_gap_source_s=self.maximum_gap_source_s,
            maximum_age_source_s=self.maximum_age_source_s,
        )


class AdaptiveAreaModel:
    def __init__(self, config, reference):
        if not isinstance(config, AdaptiveAreaConfig):
            raise ValueError("Use AdaptiveAreaConfig")
        if reference.get("schema") != "crystalab_adaptive_area_reference_v1":
            raise ValueError("Declare a training-only area reference")
        self.reference = copy.deepcopy(reference)
        self.config = config
        self.gain_reference = finite(
            reference["positive_relative_gain_reference"], "growth reference", minimum=0
        )
        self.initial_error = finite(
            reference["initial_error_ratio"], "initial error ratio", minimum=0
        )
        if self.gain_reference <= 0:
            raise ValueError("Growth reference must be positive")
        self.trend = AreaTrend(config.trend_config())
        self.context = None
        self.latest = None
        self.reset()

    def reset(self):
        self.xx = self.config.prior_precision
        self.xy = self.config.prior_precision * self.config.initial_coefficient
        self.coefficient = self.config.initial_coefficient
        self.error_ratio = self.initial_error
        self.matured_updates = 0
        self.pending = None

    def update(self, estimate, *, now_s=None):
        if not isinstance(estimate, AreaEstimate):
            raise ValueError("Use native AreaEstimate records")
        context = (
            estimate.run_id,
            estimate.clock_id,
            estimate.view_id,
            estimate.image_shape,
        )
        changed = self.context is not None and context != self.context
        # Validate order/receipt and reset history through the shared causal contract first.
        trend = self.trend.update(estimate, now_s=now_s)
        if changed:
            self.reset()
        self.context = context
        before = dict(
            coefficient=self.coefficient,
            xx=self.xx,
            xy=self.xy,
            error_ratio=self.error_ratio,
            matured_updates=self.matured_updates,
        )
        update = None
        cancelled = None
        if self.pending is not None:
            p = self.pending
            if (
                trend["features"] is not None
                and trend["reset_reason"] != "observation_gap"
                and math.isclose(
                    estimate.observed_at_s, p["target_time_s"], abs_tol=1e-9, rel_tol=0
                )
            ):
                scale = p["cut_area_px2"] * self.gain_reference
                x = p["recent_gain_px2"] / scale
                y = (estimate.area_px2 - p["cut_area_px2"]) / scale
                residual = (estimate.area_px2 - p["forecast_area_px2"]) / scale
                self.xx = self.config.forgetting_factor * self.xx + x * x
                self.xy = self.config.forgetting_factor * self.xy + x * y
                self.coefficient = min(
                    self.config.coefficient_max,
                    max(self.config.coefficient_min, self.xy / self.xx),
                )
                self.error_ratio = (
                    1 - self.config.error_ema_alpha
                ) * self.error_ratio + self.config.error_ema_alpha * abs(residual)
                self.matured_updates += 1
                update = dict(
                    cut_frame_index=p["cut_frame_index"],
                    target_frame_index=estimate.frame_index,
                    target_mask_sha256=estimate.mask_sha256,
                    x_normalized=x,
                    y_observed_normalized=y,
                    pre_update_residual_normalized=residual,
                    coefficient_used_for_previous_prediction=p["coefficient"],
                )
            else:
                cancelled = "unusable_or_unaligned_target"
            self.pending = None
        forecast = None
        received_now = estimate.received_at_s if now_s is None else now_s
        if (
            trend["features"] is not None
            and trend["history_ready"]
            and estimate.observed_at_s + self.config.horizon_source_s > received_now
        ):
            features = trend["features"]
            gain = (
                features["recent_rate_px2_per_source_s"] * self.config.horizon_source_s
            )
            requested = estimate.area_px2 + self.coefficient * gain
            future = min(features["frame_pixels"], max(0.0, requested))
            forecast = dict(
                future_area_px2=future,
                gain_px2=future - estimate.area_px2,
                relative_gain=(future - estimate.area_px2) / estimate.area_px2,
                target_time_s=estimate.observed_at_s + self.config.horizon_source_s,
                clipped_to_image=future != requested,
            )
            self.pending = dict(
                cut_frame_index=estimate.frame_index,
                cut_area_px2=estimate.area_px2,
                recent_gain_px2=gain,
                forecast_area_px2=future,
                target_time_s=forecast["target_time_s"],
                coefficient=self.coefficient,
            )
        state = dict(
            estimate=asdict(estimate),
            computed_at_source_s=received_now,
            trend=trend,
            forecast=forecast,
            parameter_before=before,
            parameter_after=dict(
                coefficient=self.coefficient,
                xx=self.xx,
                xy=self.xy,
                error_ratio=self.error_ratio,
                matured_updates=self.matured_updates,
            ),
            matured_update=update,
            cancelled_update=cancelled,
            context_parameters_reset=changed,
            uncertainty_semantics="EW absolute observed forecast residual divided by training growth reference; not a calibrated confidence interval",
            action_response_model=False,
        )
        state["state_sha256"] = value_sha256(state)
        self.latest = copy.deepcopy(state)
        return copy.deepcopy(state)

    def recommend(self, now_s, *, base_pause_model_s, model_step_s):
        now = finite(now_s, "recommendation clock", minimum=0)
        base = finite(base_pause_model_s, "base pause", minimum=0)
        step = finite(model_step_s, "model step", minimum=0)
        if base <= 0 or step <= 0:
            raise ValueError("Pause scale and model step must be positive")
        state = self.latest
        reason = None
        if state is None or state["forecast"] is None:
            reason = "forecast_unavailable"
        elif now + 1e-12 < state["estimate"]["received_at_s"]:
            reason = "forecast_not_received"
        elif (
            now - state["estimate"]["observed_at_s"] > self.config.maximum_age_source_s
        ):
            reason = "stale_area_evidence"
        elif now >= state["forecast"]["target_time_s"] - 1e-12:
            reason = "forecast_target_elapsed"
        elif self.matured_updates < self.config.minimum_matured_updates:
            reason = "insufficient_matured_updates"
        elif self.error_ratio > self.config.maximum_error_ratio:
            reason = "observed_forecast_error_too_large"
        elif state["forecast"]["relative_gain"] <= 0:
            reason = "no_positive_projected_growth"
        ratio = (
            None
            if state is None or state["forecast"] is None
            else state["forecast"]["relative_gain"] / self.gain_reference
        )
        duration = None
        if reason is None:
            bounded = min(
                self.config.maximum_pause_model_s,
                max(self.config.minimum_pause_model_s, base * ratio),
            )
            duration = math.floor(bounded / step + 1e-12) * step
            if duration < self.config.minimum_pause_model_s:
                reason = "pause_bounds_do_not_fit_model_step"
                duration = None
        return dict(
            eligible=reason is None,
            forecast_available=state is not None
            and state["forecast"] is not None
            and reason
            not in {
                "forecast_unavailable",
                "forecast_not_received",
                "stale_area_evidence",
                "forecast_target_elapsed",
            },
            reason=reason,
            pause_model_s=duration,
            forecast_reference_ratio=ratio,
            state_sha256=None if state is None else state["state_sha256"],
            objective="Scale a new-crystal recipe-pause review to positive forecasted projected growth while bounding recipe delay; no validated physical utility",
            can_authorize_hardware=False,
        )
