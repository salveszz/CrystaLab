# FILE PURPOSE: Compute bounded temperature PID commands from explicit causal feedback.
# Filtered derivative uses measurements; back-calculation accounts for output/rate
# limits. Missing/stale feedback and scripted manual mode freeze or track state.
"""Software temperature PID; fallback values and gains require application evidence."""

from dataclasses import dataclass
import math

from .contracts import ActuatorLimits, ScalarObservation, finite, identifier


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 1.5
    ki_per_s: float = 0.05
    kd_s: float = 2.0
    derivative_filter_s: float = 3.0
    anti_windup_time_s: float = 5.0
    integral_min_K: float = -20.0
    integral_max_K: float = 20.0
    maximum_feedback_age_s: float = 4.0
    maximum_update_gap_s: float = 5.0
    initial_output_K: float = 298.15
    fallback_output_K: float = 298.15

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            finite(getattr(self, key), key)
        if any(
            getattr(self, key) < 0
            for key in (
                "kp",
                "ki_per_s",
                "kd_s",
                "derivative_filter_s",
                "maximum_feedback_age_s",
            )
        ):
            raise ValueError("PID gains/filter/feedback age must be nonnegative")
        if self.anti_windup_time_s <= 0 or self.maximum_update_gap_s <= 0:
            raise ValueError("PID time constants must be positive")
        if (
            not self.integral_min_K <= 0 <= self.integral_max_K
            or self.integral_min_K == self.integral_max_K
        ):
            raise ValueError("Integral range must contain zero and have positive width")
        if min(self.initial_output_K, self.fallback_output_K) < 0:
            raise ValueError("PID outputs use Kelvin")


class TemperaturePID:
    """Update on the controller clock using feedback already received on that clock.

    Output is a bath/jacket temperature request in K, not heater power or measured
    solution temperature. The initial output is held at the first zero-dt tick.
    No hardware fallback policy is implied by the caller's configured value.
    """

    def __init__(self, config: PIDConfig, limits: ActuatorLimits, *, run_id, clock_id):
        if limits.channel != "bath_temperature" or limits.unit != "K":
            raise ValueError("Temperature PID needs bath-temperature limits in K")
        for value in (config.initial_output_K, config.fallback_output_K):
            if not limits.minimum <= value <= limits.maximum:
                raise ValueError("Initial/fallback output must lie within limits")
        identifier(run_id, "run_id")
        identifier(clock_id, "clock_id")
        self.config, self.limits = config, limits
        self.run_id, self.clock_id = run_id, clock_id
        self.previous_time = None
        self.last_output = config.initial_output_K
        self.integral = self.derivative = 0.0
        self.last_feedback_time = self.last_feedback_value = None
        self.derivative_measurement = None
        self.needs_recovery = False

    def feedback_reason(self, observation, now):
        if observation is None:
            return "missing_feedback"
        if not isinstance(observation, ScalarObservation):
            raise ValueError("Feedback must use ScalarObservation")
        if observation.run_id != self.run_id or observation.clock_id != self.clock_id:
            return "feedback_identity_mismatch"
        if observation.channel != "temperature" or observation.unit != "K":
            return "feedback_channel_mismatch"
        if observation.received_at_s > now or (
            observation.measured_at_s is not None and observation.measured_at_s > now
        ):
            return "feedback_not_yet_received"
        if not observation.valid:
            return "invalid_feedback"
        if now - observation.measured_at_s > self.config.maximum_feedback_age_s:
            return "stale_feedback"
        if self.last_feedback_time is not None:
            if observation.measured_at_s < self.last_feedback_time:
                return "out_of_order_feedback"
            if (
                observation.measured_at_s == self.last_feedback_time
                and observation.value != self.last_feedback_value
            ):
                return "conflicting_feedback_timestamp"
        return None

    def update(self, now_s, setpoint_K, observation, *, manual_output_K=None):
        now = finite(now_s, "controller time", minimum=0)
        target = finite(setpoint_K, "setpoint_K", minimum=0)
        if self.previous_time is not None and now <= self.previous_time:
            raise ValueError("Controller time must strictly increase")
        dt = 0.0 if self.previous_time is None else now - self.previous_time
        reason = self.feedback_reason(observation, now)
        if dt > self.config.maximum_update_gap_s:
            reason = "controller_update_gap"
        valid = reason is None
        error = target - observation.value if valid else None
        p = self.config.kp * error if valid else None
        if valid:
            finite(p, "PID proportional state")
        old_integral = self.integral
        d, integral = self.derivative, self.integral
        if manual_output_K is not None:
            requested = finite(manual_output_K, "manual output", minimum=0)
            mode = "manual"
        elif not valid:
            requested = self.config.fallback_output_K
            mode = "fallback"
        else:
            mode = "recovering" if self.needs_recovery else "tracking"
            if self.needs_recovery:
                d = 0.0
                integral = (
                    self.last_output - target - p if self.config.ki_per_s > 0 else 0.0
                )
            elif self.derivative_measurement is not None and dt > 0:
                stamp, value = self.derivative_measurement
                measurement_dt = observation.measured_at_s - stamp
                if measurement_dt > 0:
                    raw_d = (
                        -self.config.kd_s * (observation.value - value) / measurement_dt
                    )
                    alpha = (
                        math.exp(-measurement_dt / self.config.derivative_filter_s)
                        if self.config.derivative_filter_s > 0
                        else 0.0
                    )
                    d = alpha * d + (1 - alpha) * raw_d
            if not self.needs_recovery:
                integral += self.config.ki_per_s * error * dt
            integral = min(
                self.config.integral_max_K, max(self.config.integral_min_K, integral)
            )
            requested = target + p + integral + d
        finite(requested, "PID arithmetic result")
        output, saturated, rate_limited = self.limits.limit(
            requested, self.last_output, dt
        )
        if mode == "manual" and valid and self.config.ki_per_s > 0:
            integral = output - target - p
            d = 0.0
        elif mode in {"tracking", "recovering"} and self.config.ki_per_s > 0:
            integral += (output - requested) * (
                -math.expm1(-dt / self.config.anti_windup_time_s)
            )
        integral = min(
            self.config.integral_max_K, max(self.config.integral_min_K, integral)
        )
        if self.config.ki_per_s == 0:
            integral = 0.0
        finite(integral, "PID integral state")
        finite(d, "PID derivative state")
        if valid:
            self.last_feedback_time, self.last_feedback_value = (
                observation.measured_at_s,
                observation.value,
            )
        if mode in {"manual", "fallback"}:
            self.derivative_measurement = None
            self.needs_recovery = True
            d = 0.0
        else:
            self.derivative_measurement = (observation.measured_at_s, observation.value)
            self.needs_recovery = False
        self.integral, self.derivative = integral, d
        self.previous_time, self.last_output = now, output
        return dict(
            mode=mode,
            reason=reason,
            feedback_sample_id=None if observation is None else observation.sample_id,
            feedback_age_s=None
            if observation is None or observation.measured_at_s is None
            else now - observation.measured_at_s,
            setpoint_K=target,
            error_K=error,
            proportional_K=p,
            derivative_K=d,
            integral_before_K=old_integral,
            integral_after_K=integral,
            requested_output_K=requested,
            output_K=output,
            saturated=saturated,
            rate_limited=rate_limited,
            update_interval_s=dt,
            feedforward_K=target,
        )
