# FILE PURPOSE: Compare fixed-recipe and PID feedback on declared simulated thermal scenarios.
# Causal noisy/delayed observations, scripted faults/overrides, proposal dispatch,
# delayed acknowledgments and separate plant truth are retained for independent audit.
"""Reproducible software control experiment; no KDP calibration or yield claim."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time

import numpy as np

from .contracts import (
    ActuatorLimits,
    CommandProposal,
    ScalarObservation,
    finite,
    identifier,
)
from .pid import PIDConfig, TemperaturePID
from .thermal import SimulatedActuator, ThermalPlant

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_row(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def validate_config(config):
    if config["schema"] != "crystalab_temperature_benchmark_v1":
        raise ValueError("Unsupported temperature benchmark schema")
    identifier(config["clock_id"], "clock_id")
    step = finite(config["step_s"], "step", minimum=0)
    duration = finite(config["duration_s"], "duration", minimum=0)
    if (
        step == 0
        or duration < step
        or not math.isclose(duration / step, round(duration / step))
        or duration / step > 100_000
    ):
        raise ValueError(
            "Benchmark needs a bounded whole number of positive time steps"
        )
    limits = ActuatorLimits(**config["limits"])
    pid = PIDConfig(**config["pid"])
    TemperaturePID(pid, limits, run_id="validation", clock_id=config["clock_id"])
    if step > pid.maximum_update_gap_s:
        raise ValueError("Benchmark tick exceeds the PID update-gap limit")
    if (
        not config["recipe"]
        or config["recipe"][0]["time_s"] != 0
        or config["recipe"][-1]["time_s"] != duration
    ):
        raise ValueError("Recipe must span the declared experiment window")
    previous = -1
    for point in config["recipe"]:
        finite(point["time_s"], "recipe time", minimum=0)
        finite(point["temperature_K"], "recipe temperature", minimum=0)
        if (
            point["time_s"] <= previous
            or not limits.minimum <= point["temperature_K"] <= limits.maximum
        ):
            raise ValueError("Recipe times/range are invalid")
        previous = point["time_s"]
    ids = set()
    if not config["scenarios"]:
        raise ValueError("At least one declared scenario is required")
    for scenario in config["scenarios"]:
        name = scenario["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,48}", name) or name in ids:
            raise ValueError("Scenario IDs must be unique safe names")
        ids.add(name)
        for key in ("actuator_delay_s", "sensor_delay_s", "sensor_noise_std_K"):
            finite(scenario[key], key, minimum=0)
        if type(scenario["seed"]) is not int or scenario["seed"] < 0:
            raise ValueError("Scenario seed must be a nonnegative integer")
        ThermalPlant(
            initial_K=pid.initial_output_K,
            initial_input_K=pid.initial_output_K,
            time_constant_s=scenario["time_constant_s"],
            disturbances=scenario["disturbances"],
        )
        for kind in ("missing_intervals", "held_intervals", "manual_intervals"):
            previous = -1
            for interval in scenario[kind]:
                start = finite(interval["start_s"], "interval start", minimum=0)
                end = finite(interval["end_s"], "interval end", minimum=0)
                if not previous <= start < end <= duration:
                    raise ValueError(
                        "Fault/override intervals must be ordered and disjoint within each type"
                    )
                previous = end
                if kind == "manual_intervals":
                    finite(interval["output_K"], "scripted override", minimum=0)
    return limits, pid


def in_interval(intervals, now):
    return next((p for p in intervals if p["start_s"] <= now < p["end_s"]), None)


def summarize(rows):
    errors = np.array(
        [r["plant_truth"]["temperature_K"] - r["setpoint_K"] for r in rows]
    )
    commands = np.array([r["decision"]["output_K"] for r in rows])
    return dict(
        samples=len(rows),
        sample_mae_K=float(np.abs(errors).mean()),
        sample_rmse_K=float(np.sqrt(np.mean(errors**2))),
        trapezoidal_IAE_K_s=float(
            np.sum(
                np.diff([r["time_s"] for r in rows])
                * (np.abs(errors[1:]) + np.abs(errors[:-1]))
                / 2
            )
        ),
        maximum_above_target_K=max(0.0, float(errors.max())),
        maximum_below_target_K=max(0.0, float(-errors.min())),
        requested_input_total_variation_K=float(np.abs(np.diff(commands)).sum()),
        saturated_steps=sum(r["decision"]["saturated"] for r in rows),
        rate_limited_steps=sum(r["decision"]["rate_limited"] for r in rows),
        fallback_steps=sum(r["decision"]["mode"] == "fallback" for r in rows),
        scripted_manual_steps=sum(r["decision"]["mode"] == "manual" for r in rows),
        invalid_observation_steps=sum(not r["observation"]["valid"] for r in rows),
    )


def run_scenario(config, scenario, method, output, *, target_provider=None):
    if method not in {"fixed_recipe", "pid", "rule_supervised", "adaptive_supervised"}:
        raise ValueError("Unsupported control comparison method")
    if (method in {"rule_supervised", "adaptive_supervised"}) != (
        target_provider is not None
    ):
        raise ValueError("A supervised method needs its explicit target provider")
    output.mkdir()
    limits, pid_config = ActuatorLimits(**config["limits"]), PIDConfig(**config["pid"])
    run_id = scenario["id"] + ":" + method
    clock_id = config["clock_id"]
    pid = TemperaturePID(pid_config, limits, run_id=run_id, clock_id=clock_id)
    actuator = SimulatedActuator(
        limits,
        run_id=run_id,
        clock_id=clock_id,
        delay_s=scenario["actuator_delay_s"],
        initial_value=pid_config.initial_output_K,
    )
    plant = ThermalPlant(
        initial_K=pid_config.initial_output_K,
        initial_input_K=pid_config.initial_output_K,
        time_constant_s=scenario["time_constant_s"],
        disturbances=scenario["disturbances"],
    )
    times = (
        np.arange(round(config["duration_s"] / config["step_s"]) + 1) * config["step_s"]
    )
    noise = np.random.default_rng(scenario["seed"]).normal(
        0, scenario["sensor_noise_std_K"], len(times)
    )
    measured, rows, frozen = [], [], {}
    last_output = pid_config.initial_output_K
    with (
        (output / "observations.jsonl").open("x") as records,
        (output / "commands.jsonl").open("x") as commands,
    ):
        for index, stamp in enumerate(times):
            now = float(stamp)
            applied = actuator.advance(now)
            value = plant.advance(now, applied)
            for event in applied:
                write_row(commands, event)
            measured.append(value + float(noise[index]))
            available = (
                int(
                    np.searchsorted(
                        times, now - scenario["sensor_delay_s"] + 1e-10, side="right"
                    )
                )
                - 1
            )
            missing = in_interval(scenario["missing_intervals"], now)
            held = in_interval(scenario["held_intervals"], now)
            if held is not None:
                key = held["start_s"]
                if key not in frozen:
                    frozen[key] = available
                available = frozen[key]
            invalid = available < 0 or missing is not None
            observation = ScalarObservation(
                run_id=run_id,
                sample_id=f"{run_id}:sensor:{index if invalid else available:06d}",
                channel="temperature",
                value=None if invalid else measured[available],
                unit="K",
                measured_at_s=None if invalid else float(times[available]),
                received_at_s=now,
                clock_id=clock_id,
                source="simulated_delayed_temperature_sensor_v1",
                valid=not invalid,
                reason="scripted_missing_packet"
                if missing is not None
                else "sensor_not_started"
                if invalid
                else None,
            )
            target = float(
                np.interp(
                    now,
                    [p["time_s"] for p in config["recipe"]],
                    [p["temperature_K"] for p in config["recipe"]],
                )
            )
            nominal_target = target
            supervision = None
            if target_provider is not None:
                supervision = target_provider(now, nominal_target, observation)
                target = finite(supervision["target_K"], "supervised target", minimum=0)
                if not limits.minimum <= target <= limits.maximum:
                    raise ValueError(
                        "Supervised target is outside declared process limits"
                    )
            manual = in_interval(scenario["manual_intervals"], now)
            if method in {"pid", "rule_supervised", "adaptive_supervised"}:
                decision = pid.update(
                    now,
                    target,
                    observation,
                    manual_output_K=None if manual is None else manual["output_K"],
                )
            else:
                requested = target if manual is None else manual["output_K"]
                bounded, saturated, rate_limited = limits.limit(
                    requested, last_output, 0 if index == 0 else config["step_s"]
                )
                decision = dict(
                    mode="open_loop" if manual is None else "manual",
                    reason=None,
                    requested_output_K=requested,
                    output_K=bounded,
                    saturated=saturated,
                    rate_limited=rate_limited,
                    feedback_sample_id=None,
                )
            last_output = decision["output_K"]
            evidence = (f"{run_id}:recipe:{index:06d}",) + (
                (observation.sample_id,)
                if method in {"pid", "rule_supervised", "adaptive_supervised"}
                else ()
            )
            if supervision is not None:
                evidence += tuple(supervision["evidence_ids"])
            proposal = CommandProposal(
                command_id=f"{run_id}:command:{index:06d}",
                run_id=run_id,
                channel="bath_temperature",
                value=last_output,
                unit="K",
                proposed_at_s=now,
                expires_at_s=now + scenario["actuator_delay_s"] + 2 * config["step_s"],
                clock_id=clock_id,
                evidence_ids=evidence,
                source="fixed_recipe_v1"
                if method == "fixed_recipe"
                else "temperature_pid_v1",
            )
            authorization = (
                "simulation_policy"
                if manual is None
                else "scripted_simulation_override"
            )
            dispatch = actuator.submit(proposal, now, authorization=authorization)
            write_row(
                commands,
                dict(
                    event="proposed",
                    time_s=now,
                    proposal=asdict(proposal),
                    authorization=authorization,
                ),
            )
            write_row(commands, dispatch)
            if dispatch["event"] != "accepted":
                raise ValueError("Simulation command rejected: " + dispatch["reason"])
            row = dict(
                run_id=run_id,
                clock_id=clock_id,
                time_s=now,
                setpoint_K=target,
                observation=asdict(observation),
                decision=decision,
                proposal=asdict(proposal),
                authorization=authorization,
                dispatch=dispatch,
                applied_receipts=applied,
                plant_truth=dict(
                    temperature_K=value,
                    applied_bath_K=plant.input_K,
                    equilibrium_disturbance_K=plant.disturbance(now),
                ),
                evidence_kind="simulation",
                physical_hardware_contacted=False,
            )
            if supervision is not None:
                row.update(
                    nominal_recipe_setpoint_K=nominal_target, supervision=supervision
                )
            write_row(records, row)
            rows.append(row)
    return dict(
        run_id=run_id,
        scenario=scenario["id"],
        method=method,
        records_path=str((output / "observations.jsonl").relative_to(output.parent)),
        commands_path=str((output / "commands.jsonl").relative_to(output.parent)),
        metrics=summarize(rows),
        accepted_not_yet_applied_at_end=actuator.pending,
    ), rows


def plot_comparison(config, all_rows, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(
        len(config["scenarios"]),
        2,
        figsize=(11, 2.7 * len(config["scenarios"])),
        squeeze=False,
    )
    for i, scenario in enumerate(config["scenarios"]):
        for method, color in [("fixed_recipe", "#b47717"), ("pid", "#216bc4")]:
            rows = all_rows[(scenario["id"], method)]
            times = [r["time_s"] for r in rows]
            axes[i, 0].plot(
                times,
                [r["plant_truth"]["temperature_K"] for r in rows],
                label=method,
                color=color,
            )
            axes[i, 1].step(
                times,
                [r["plant_truth"]["applied_bath_K"] for r in rows],
                where="post",
                label=method,
                color=color,
            )
        axes[i, 0].plot(
            times,
            [r["setpoint_K"] for r in rows],
            "k--",
            label="recipe target",
            linewidth=1,
        )
        for axis in axes[i]:
            axis.grid(alpha=0.2)
            axis.set_xlabel("Simulation time (s)")
            axis.legend(fontsize=8)
        axes[i, 0].set_title(scenario["id"] + " / simulated solution")
        axes[i, 0].set_ylabel("Temperature (K)")
        axes[i, 1].set_title("Applied simulated bath setpoint")
        axes[i, 1].set_ylabel("Bath request (K)")
        for interval in scenario["missing_intervals"] + scenario["held_intervals"]:
            for axis in axes[i]:
                axis.axvspan(
                    interval["start_s"], interval["end_s"], color="grey", alpha=0.15
                )
    fig.suptitle(
        "Software thermal fixture / uncalibrated plant / no KDP quality or equipment claim",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def run_benchmark(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    config_hash = sha256(config_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "config.json")
    sources = [
        *sorted((ROOT / "src/controllers").glob("*.py")),
        ROOT / "scripts/run_control_baseline.py",
    ]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = sha256(target)
        if sha256(source) != source_hashes[str(relative)]:
            raise ValueError("Control source changed during snapshot")
    manifest = dict(
        schema="crystalab_temperature_benchmark_run_v1",
        status="running",
        config_sha256=config_hash,
        source_sha256=source_hashes,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            platform=platform.platform(),
        ),
        scope="Declared uncalibrated first-order thermal software fixture; no physical KDP/equipment/quality claim",
        evidence_kind="simulation",
        physical_hardware_contacted=False,
        controller_reads_hidden_plant_truth=False,
        scenarios_role="development software scenarios, not independently sampled physical runs",
    )
    save_json(output / "manifest.json", manifest)
    start = time.perf_counter()
    runs, all_rows = [], {}
    try:
        for scenario in config["scenarios"]:
            for method in ("fixed_recipe", "pid"):
                report, rows = run_scenario(
                    config, scenario, method, output / (scenario["id"] + "_" + method)
                )
                runs.append(report)
                all_rows[(scenario["id"], method)] = rows
        plot_comparison(config, all_rows, output)
        if (
            sha256(config_path) != config_hash
            or sha256(output / "config.json") != config_hash
        ):
            raise ValueError("Control configuration changed during experiment")
        manifest.update(
            status="complete",
            runs=runs,
            elapsed_seconds=time.perf_counter() - start,
            files={
                str(p.relative_to(output)): sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            completed_runs=runs,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise
