# FILE PURPOSE: Record controlled growth episodes for synthetic data, without ML.
# Exact simulator truth, independently seeded sensor proxies, bounded held actions
# and invariant diagnostics stay separate. Clocks use decimal-rational event times;
# labels can join exact endpoints without interpolation or future observations.
"""Minimal observation/action recorder for the existing illustrative 3D engine."""

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral, Real

import numpy as np

from simulation.engine import Simulation

OBSERVABLES = (
    "volume",
    "width",
    "depth",
    "height",
    "surface_area",
    "temperature",
    "concentration",
    "liquid_level",
)
UNITS = dict(
    time="model_time",
    length="model_length",
    volume="model_length^3",
    surface_area="model_length^2",
    mass="model_solute_mass",
    temperature="model_temperature",
    concentration="model_solute_mass/model_carrier_volume",
    agitation="illustrative_dimensionless_command",
)
MODEL_MODE = "illustrative_growth_3d_v1"


def number(value, name, *, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or positive
        and value <= 0
        or nonnegative
        and value < 0
    ):
        raise ValueError(f"Invalid {name}: {value!r}")


def tick(value):
    """Use decimal input semantics, avoiding accumulated sampling-clock drift."""
    return Fraction(str(value))


@dataclass(frozen=True)
class RecordingConfig:
    sample_interval: float = 2.0
    decision_interval: float = 10.0
    forecast_horizon: float = 10.0
    history_samples: int = 2
    sensor_delay: float = 0.0
    relative_observation_noise: float = 0.0
    temperature_observation_noise: float = 0.0
    observation_seed: int = 1
    temperature_min: float = 22.0
    temperature_max: float = 28.0
    agitation_min: float = 0.0
    agitation_max: float = 2.0
    max_temperature_change: float = 3.0
    max_agitation_change: float = 2.0
    mass_relative_tolerance: float = 1e-10
    carrier_relative_tolerance: float = 1e-10
    continuity_cell_tolerance: float = 1e-8
    negative_concentration_tolerance: float = 1e-12

    def __post_init__(self):
        for name in (
            "sample_interval",
            "decision_interval",
            "forecast_horizon",
            "mass_relative_tolerance",
            "carrier_relative_tolerance",
            "continuity_cell_tolerance",
        ):
            number(getattr(self, name), name, positive=True)
        for name in (
            "sensor_delay",
            "relative_observation_noise",
            "temperature_observation_noise",
            "max_temperature_change",
            "max_agitation_change",
            "negative_concentration_tolerance",
        ):
            number(getattr(self, name), name, nonnegative=True)
        for name in ("history_samples", "observation_seed"):
            v = getattr(self, name)
            if (
                isinstance(v, bool)
                or not isinstance(v, Integral)
                or v < (1 if name == "history_samples" else 0)
            ):
                raise ValueError(f"Invalid {name}")
        for stem in ("temperature", "agitation"):
            lo, hi = getattr(self, stem + "_min"), getattr(self, stem + "_max")
            number(lo, stem + "_min")
            number(hi, stem + "_max")
            if hi < lo or stem == "agitation" and lo < 0:
                raise ValueError(f"Invalid {stem} bounds")
        for name in ("decision_interval", "forecast_horizon", "sensor_delay"):
            if tick(getattr(self, name)) % tick(self.sample_interval):
                raise ValueError(f"{name} must be a multiple of sample_interval")
        if self.forecast_horizon > self.decision_interval:
            raise ValueError("Horizon cannot cross the next action decision")


def expand_schedule(config, recording, schedule):
    """Validate sparse commands, then explicitly log every held decision interval."""
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("A nonempty action schedule is required")
    previous = dict(
        temperature=config.temperature, agitation=config.rotation + config.circulation
    )
    commands, last = {}, None
    for row in schedule:
        if not isinstance(row, dict) or set(row) != {
            "time",
            "temperature",
            "agitation",
        }:
            raise ValueError("Each command needs exactly time, temperature, agitation")
        number(row["time"], "command time", nonnegative=True)
        t = tick(row["time"])
        if (
            t >= tick(config.duration)
            or t % tick(recording.decision_interval)
            or (last is None and t != 0)
            or (last is not None and t <= last)
        ):
            raise ValueError(
                "Commands must start at zero and increase on decision ticks"
            )
        for name in ("temperature", "agitation"):
            value = row[name]
            number(value, name)
            if (
                not getattr(recording, name + "_min")
                <= value
                <= getattr(recording, name + "_max")
            ):
                raise ValueError(f"Command outside {name} bounds")
            if abs(value - previous[name]) > getattr(
                recording, "max_" + name + "_change"
            ):
                raise ValueError(f"Command exceeds {name} change limit")
            previous[name] = value
        commands[t], last = dict(previous), t
    expanded, t, held = [], Fraction(0), None
    while t < tick(config.duration):
        held = commands.get(t, held)
        expanded.append(dict(time=float(t), **held))
        t += tick(recording.decision_interval)
    return expanded


def make_observations(truth, recording):
    """Rebuild causal sensor proxies from truth; independent RNG never affects growth."""
    rng = np.random.default_rng(recording.observation_seed)
    by_time = {tick(row["time"]): row for row in truth}
    cache, observations = {}, []

    def sensor(t):
        if t not in by_time:
            return dict.fromkeys(OBSERVABLES)
        if t not in cache:
            values = {}
            for key in OBSERVABLES:
                v = by_time[t][key]
                sigma = (
                    recording.temperature_observation_noise
                    if key == "temperature"
                    else abs(v) * recording.relative_observation_noise
                )
                value = float(v + rng.normal(0, sigma)) if sigma else float(v)
                values[key] = None if key != "temperature" and value < 0 else value
            cache[t] = values
        return cache[t]

    for row in truth:
        t = tick(row["time"])
        source = t - tick(recording.sensor_delay)
        values = sensor(source)
        past = sensor(
            source - tick(recording.sample_interval) * recording.history_samples
        )
        lag, current = past["volume"], values["volume"]
        rate = (
            None
            if lag is None or current is None
            else (current - lag)
            / (recording.sample_interval * recording.history_samples)
        )
        observation = dict(
            time=float(t),
            sensor_time=float(source) if source in by_time else None,
            age=recording.sensor_delay,
            origin="synthetic_sensor_proxy",
            quality_target_available=False,
            **values,
            volume_lag=lag,
            recent_volume_rate=rate,
        )
        observation["missing"] = {
            key: observation[key] is None
            for key in (*OBSERVABLES, "volume_lag", "recent_volume_rate")
        }
        observations.append(observation)
    return observations


@dataclass
class EpisodeResult:
    truth: list
    observations: list
    controls: list
    diagnostics: dict
    final_fields: dict


def record_episode(config, recording, schedule, *, on_event=None):
    """Run one bounded episode; callback preserves raw evidence as it is produced."""
    controls = expand_schedule(config, recording, schedule)
    sim = Simulation(config)
    initial_mass, initial_carrier = sim.total_mass, sim.tank.carrier_volume
    diagnostics = dict(
        max_mass_relative_error=0.0,
        max_carrier_relative_error=0.0,
        max_continuity_cell_error=0.0,
        minimum_concentration=float("inf"),
        steps=0,
    )
    truth = []
    action_at = {tick(row["time"]): row for row in controls}
    events = {Fraction(0), tick(config.duration), *action_at}
    t = tick(recording.sample_interval)
    while t < tick(config.duration):
        events.add(t)
        t += tick(recording.sample_interval)

    def emit(kind, row):
        if on_event:
            on_event(dict(kind=kind, **row))

    def check():
        tank = sim.tank
        arrays = (
            tank.concentration,
            tank.temperature,
            tank.liquid_volumes,
            sim.crystal.vertices,
            sim.crystal.offsets,
        )
        if not all(np.isfinite(a).all() for a in arrays):
            raise ArithmeticError("Nonfinite simulation state")
        mass = abs(sim.total_mass - initial_mass) / max(abs(initial_mass), 1e-30)
        carrier = abs(tank.liquid_volume - initial_carrier) / max(
            initial_carrier, 1e-30
        )
        continuity = abs(tank.continuity_error) / tank.cell_volume
        minimum = float(tank.concentration.min())
        diagnostics["max_mass_relative_error"] = max(
            diagnostics["max_mass_relative_error"], mass
        )
        diagnostics["max_carrier_relative_error"] = max(
            diagnostics["max_carrier_relative_error"], carrier
        )
        diagnostics["max_continuity_cell_error"] = max(
            diagnostics["max_continuity_cell_error"], continuity
        )
        diagnostics["minimum_concentration"] = min(
            diagnostics["minimum_concentration"], minimum
        )
        if (
            mass > recording.mass_relative_tolerance
            or carrier > recording.carrier_relative_tolerance
            or continuity > recording.continuity_cell_tolerance
            or minimum < -recording.negative_concentration_tolerance
        ):
            raise ArithmeticError(
                f"Episode invariant failed at {sim.time}: {diagnostics}"
            )
        if np.any(sim.crystal.vertices < -1e-10) or np.any(
            sim.crystal.vertices
            > np.array([config.width, config.depth, config.height]) + 1e-10
        ):
            raise ArithmeticError("Crystal escaped tank")

    for t in sorted(events):
        target = float(t)
        while sim.time < target:
            old = sim.time
            sim.step(min(config.dt, 1.0, target - old))
            diagnostics["steps"] += 1
            check()
            if not old < sim.time <= target:
                raise ArithmeticError("Simulation clock did not advance exactly")
        check()
        row = dict(
            sim.measurements(),
            face_offsets=sim.crystal.offsets[:-6].tolist(),
            face_normals=sim.crystal.normals[:-6].tolist(),
            face_areas=sim.crystal.face_areas[:-6].tolist(),
        )
        truth.append(row)
        emit("truth", row)
        if t in action_at:
            command = action_at[t]
            sim.set_controls(
                temperature=command["temperature"], agitation=command["agitation"]
            )
            command.update(
                applies_until=min(
                    target + recording.decision_interval, config.duration
                ),
                rotation=sim.config.rotation,
                circulation=sim.config.circulation,
                actuator_delay=0.0,
            )
            emit("control", command)
    observations = make_observations(truth, recording)
    for row in observations:
        emit("observation", row)
    diagnostics.update(
        final_time=sim.time,
        exact_final_time=sim.time == config.duration,
        initial_total_mass=initial_mass,
        initial_carrier_volume=initial_carrier,
        diffusion=sim.tank.diffusion_diagnostics,
        remap=sim.tank.remap_diagnostics,
    )
    fields = dict(
        temperature=sim.tank.temperature,
        concentration=sim.tank.concentration,
        liquid_volumes=sim.tank.liquid_volumes,
        cell_volumes=sim.tank.cell_volumes,
        normals=sim.crystal.normals,
        offsets=sim.crystal.offsets,
        vertices=sim.crystal.vertices,
        x_edges=sim.tank.x_edges,
        y_edges=sim.tank.y_edges,
        z_edges=sim.tank.z_edges,
    )
    return EpisodeResult(truth, observations, controls, diagnostics, fields)
