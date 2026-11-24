# FILE PURPOSE: Define unit-bearing process observations, proposals and actuator limits.
# Temperature, bath setpoint, rotation and flow remain separate channels with
# explicit run/clock identity. These records do not open hardware or authorize it.
"""Causal scalar process records shared by controller and adapter implementations."""

from dataclasses import dataclass
import math

UNITS = {
    "temperature": "K",
    "bath_temperature": "K",
    "rotation": "rad/s",
    "flow": "m3/s",
}


def finite(value, name, *, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A nonempty {name} is required")


def channel_unit(channel, unit):
    if channel not in UNITS or unit != UNITS[channel]:
        raise ValueError("Unsupported channel/unit pair")


@dataclass(frozen=True)
class ScalarObservation:
    run_id: str
    sample_id: str
    channel: str
    value: float | None
    unit: str
    measured_at_s: float | None
    received_at_s: float
    clock_id: str
    source: str
    valid: bool = True
    reason: str | None = None

    def __post_init__(self):
        for key in ("run_id", "sample_id", "clock_id", "source"):
            identifier(getattr(self, key), key)
        channel_unit(self.channel, self.unit)
        finite(self.received_at_s, "received_at_s", minimum=0)
        if type(self.valid) is not bool:
            raise ValueError("Observation validity must be explicit")
        if self.measured_at_s is not None:
            finite(self.measured_at_s, "measured_at_s", minimum=0)
            if self.measured_at_s > self.received_at_s:
                raise ValueError(
                    "Measurement cannot follow receipt on the declared clock"
                )
        if self.value is not None:
            finite(self.value, "observation value")
            if self.unit == "K" and self.value < 0:
                raise ValueError("Kelvin temperature cannot be negative")
        if self.valid and (self.value is None or self.measured_at_s is None):
            raise ValueError("Valid observations require a value and measurement time")
        if not self.valid and not self.reason:
            raise ValueError("Invalid observation requires a reason")


@dataclass(frozen=True)
class ActuatorLimits:
    channel: str
    unit: str
    minimum: float
    maximum: float
    maximum_rate_per_s: float
    provenance: str

    def __post_init__(self):
        channel_unit(self.channel, self.unit)
        if self.channel == "temperature":
            raise ValueError("Measured temperature is not a command channel")
        for key in ("minimum", "maximum", "maximum_rate_per_s"):
            finite(getattr(self, key), key)
        if self.minimum >= self.maximum or self.maximum_rate_per_s <= 0:
            raise ValueError("Invalid actuator range/rate")
        if self.unit == "K" and self.minimum < 0:
            raise ValueError("Kelvin command range cannot be negative")
        identifier(self.provenance, "limit provenance")

    def limit(self, requested, previous, dt):
        finite(requested, "requested output")
        finite(previous, "previous output")
        finite(dt, "output interval", minimum=0)
        if not self.minimum <= previous <= self.maximum:
            raise ValueError("Previous output is outside configured limits")
        bounded = min(self.maximum, max(self.minimum, requested))
        change = self.maximum_rate_per_s * dt
        actual = min(previous + change, max(previous - change, bounded))
        return actual, bounded != requested, actual != bounded


@dataclass(frozen=True)
class CommandProposal:
    command_id: str
    run_id: str
    channel: str
    value: float
    unit: str
    proposed_at_s: float
    expires_at_s: float
    clock_id: str
    evidence_ids: tuple[str, ...]
    source: str

    def __post_init__(self):
        for key in ("command_id", "run_id", "clock_id", "source"):
            identifier(getattr(self, key), key)
        channel_unit(self.channel, self.unit)
        if self.channel == "temperature":
            raise ValueError("Use a separate actuator command channel")
        finite(self.value, "command value")
        finite(self.proposed_at_s, "proposal time", minimum=0)
        finite(self.expires_at_s, "expiry time", minimum=0)
        if self.expires_at_s < self.proposed_at_s:
            raise ValueError("Command expiry precedes proposal")
        if self.unit == "K" and self.value < 0:
            raise ValueError("Kelvin command cannot be negative")
        if not isinstance(self.evidence_ids, tuple):
            raise ValueError("Evidence IDs must be an immutable tuple")
        for evidence in self.evidence_ids:
            identifier(evidence, "evidence ID")
