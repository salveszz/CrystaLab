# FILE PURPOSE: Exercise process commands on an explicitly uncalibrated thermal test plant.
# A delayed simulated actuator records acceptance/application separately and enforces
# declared range/rate/identity/expiry constraints. No hardware driver is provided.
"""Analytic first-order thermal fixture and auditable simulated actuator."""

import math

from .contracts import ActuatorLimits, CommandProposal, finite, identifier


class SimulatedActuator:
    """Constant-delay sample-and-hold actuator; rate bounds apply to accepted requests.

    This is a software fixture. Authorizations are explicit simulation policies,
    never an inferred operator approval or permission to contact equipment.
    """

    def __init__(
        self, limits: ActuatorLimits, *, run_id, clock_id, delay_s, initial_value
    ):
        identifier(run_id, "run_id")
        identifier(clock_id, "clock_id")
        finite(delay_s, "actuator delay", minimum=0)
        finite(initial_value, "initial actuator value")
        if not limits.minimum <= initial_value <= limits.maximum:
            raise ValueError("Initial actuator value outside limits")
        self.limits, self.run_id, self.clock_id = limits, run_id, clock_id
        self.delay_s = delay_s
        self.applied_value = self.last_accepted_value = initial_value
        self.last_dispatch_time = self.time_s = 0.0
        self.pending, self.identifiers = [], set()

    def submit(self, proposal: CommandProposal, now_s, *, authorization):
        now = finite(now_s, "dispatch time", minimum=0)
        reason = None
        if proposal.command_id in self.identifiers:
            reason = "duplicate_command_id"
        elif proposal.run_id != self.run_id or proposal.clock_id != self.clock_id:
            reason = "command_identity_mismatch"
        elif (
            proposal.channel != self.limits.channel or proposal.unit != self.limits.unit
        ):
            reason = "command_channel_mismatch"
        elif authorization not in {"simulation_policy", "scripted_simulation_override"}:
            reason = "simulation_authorization_required"
        elif now < self.time_s or now < self.last_dispatch_time:
            reason = "out_of_order_dispatch"
        elif proposal.proposed_at_s > now:
            reason = "proposal_not_yet_valid"
        elif proposal.expires_at_s < now + self.delay_s:
            reason = "expires_before_application"
        elif not self.limits.minimum <= proposal.value <= self.limits.maximum:
            reason = "command_outside_range"
        elif (
            abs(proposal.value - self.last_accepted_value)
            > self.limits.maximum_rate_per_s * (now - self.last_dispatch_time) + 1e-10
        ):
            reason = "command_exceeds_rate"
        self.identifiers.add(proposal.command_id)
        receipt = dict(
            command_id=proposal.command_id,
            run_id=self.run_id,
            clock_id=self.clock_id,
            channel=self.limits.channel,
            unit=self.limits.unit,
            event="rejected" if reason else "accepted",
            time_s=now,
            effective_at_s=None if reason else now + self.delay_s,
            value=None if reason else proposal.value,
            reason=reason,
            authorization=authorization,
            source="simulated_actuator_v1",
        )
        if reason is None:
            self.last_accepted_value, self.last_dispatch_time = proposal.value, now
            self.pending.append(receipt)
        return receipt

    def advance(self, now_s):
        now = finite(now_s, "actuator clock", minimum=0)
        if now < self.time_s:
            raise ValueError("Actuator clock cannot reverse")
        applied = []
        while self.pending and self.pending[0]["effective_at_s"] <= now:
            receipt = self.pending.pop(0)
            self.applied_value = receipt["value"]
            applied.append(
                dict(receipt, event="applied", time_s=receipt["effective_at_s"])
            )
        self.time_s = now
        return applied


class ThermalPlant:
    """Exact piecewise-constant integration of dT/dt=(u + disturbance - T)/tau.

    T and u use K and time uses s solely to exercise the unit-bearing interfaces.
    Tau/disturbance are declared test assumptions, not KDP parameter estimates.
    Disturbance changes and fractional actuator delays are integrated at their
    actual timestamps, rather than moved to the following controller tick.
    """

    def __init__(self, *, initial_K, initial_input_K, time_constant_s, disturbances):
        finite(initial_K, "initial temperature", minimum=0)
        finite(initial_input_K, "initial input", minimum=0)
        finite(time_constant_s, "thermal time constant", minimum=0)
        if time_constant_s == 0:
            raise ValueError("Thermal time constant must be positive")
        if not disturbances or disturbances[0]["time_s"] != 0:
            raise ValueError("Disturbance schedule must start at zero")
        previous = -1.0
        for point in disturbances:
            finite(point["time_s"], "disturbance time", minimum=0)
            finite(point["offset_K"], "equilibrium offset")
            if point["time_s"] <= previous:
                raise ValueError("Disturbance times must strictly increase")
            previous = point["time_s"]
        self.temperature_K, self.input_K = initial_K, initial_input_K
        self.tau = time_constant_s
        self.disturbances = [dict(point) for point in disturbances]
        self.time_s = 0.0

    def disturbance(self, when):
        return next(
            point["offset_K"]
            for point in reversed(self.disturbances)
            if point["time_s"] <= when
        )

    def advance(self, end_s, applications):
        end = finite(end_s, "plant time", minimum=0)
        if end < self.time_s:
            raise ValueError("Plant clock cannot reverse")
        previous = self.time_s
        for event in applications:
            if (
                event["event"] != "applied"
                or event["channel"] != "bath_temperature"
                or event["unit"] != "K"
                or not previous <= event["time_s"] <= end
            ):
                raise ValueError("Invalid thermal actuator application sequence")
            finite(event["value"], "applied input", minimum=0)
            previous = event["time_s"]
        boundaries = sorted(
            {
                end,
                *(e["time_s"] for e in applications),
                *(
                    p["time_s"]
                    for p in self.disturbances
                    if self.time_s < p["time_s"] <= end
                ),
            }
        )
        for when in boundaries:
            equilibrium = self.input_K + self.disturbance(self.time_s)
            self.temperature_K = equilibrium + (
                self.temperature_K - equilibrium
            ) * math.exp(-(when - self.time_s) / self.tau)
            self.time_s = when
            for event in applications:
                if event["time_s"] == when:
                    self.input_K = event["value"]
        finite(self.temperature_K, "thermal plant result", minimum=0)
        return self.temperature_K
