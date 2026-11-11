# FILE PURPOSE: Independently reconstruct thermal experiment records and causal provenance.
# Artifact hashes, sensor delays/noise, command acknowledgments, exact plant response
# and reported metrics are checked without calling the controller or plant classes.
"""Independent saved-artifact audit for the temperature software benchmark."""

import json
import math
from pathlib import Path

import numpy as np

from .benchmark import sha256


def require(condition, reason):
    if not condition:
        raise ValueError("Control audit failed: " + reason)


def near(a, b, reason):
    require(
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9),
        reason,
    )


def local(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "artifact path escapes the run")
    return path


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def reference_target(points, now):
    if now == points[-1]["time_s"]:
        return points[-1]["temperature_K"]
    a, b = next(
        (a, b) for a, b in zip(points, points[1:]) if a["time_s"] <= now < b["time_s"]
    )
    fraction = (now - a["time_s"]) / (b["time_s"] - a["time_s"])
    return a["temperature_K"] + fraction * (b["temperature_K"] - a["temperature_K"])


def audit_control_benchmark(path):
    root = Path(path).resolve()
    manifest_hash = sha256(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest.get("schema") == "crystalab_temperature_benchmark_run_v1"
        and manifest["status"] == "complete",
        "completed temperature experiment required",
    )
    inventory = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    require(inventory == set(manifest["files"]), "artifact inventory changed")
    for name, digest in manifest["files"].items():
        require(
            sha256(local(root, name)) == digest, "artifact identity changed: " + name
        )
    require(
        sha256(root / "config.json") == manifest["config_sha256"],
        "configuration identity",
    )
    for name, digest in manifest["source_sha256"].items():
        require(
            sha256(local(root, "source_snapshot/" + name)) == digest,
            "producer source snapshot",
        )
    config = json.loads((root / "config.json").read_text())
    dt, duration = config["step_s"], config["duration_s"]
    limits = config["limits"]
    count = round(duration / dt) + 1
    reports = []
    require(
        len(manifest["runs"]) == 2 * len(config["scenarios"]), "comparison coverage"
    )
    require(
        {(r["scenario"], r["method"]) for r in manifest["runs"]}
        == {(s["id"], m) for s in config["scenarios"] for m in ["fixed_recipe", "pid"]},
        "comparison identities",
    )
    for run in manifest["runs"]:
        scenario = next(s for s in config["scenarios"] if s["id"] == run["scenario"])
        rows = json_rows(local(root, run["records_path"]))
        commands = json_rows(local(root, run["commands_path"]))
        require(len(rows) == count, "row count")
        noise = np.random.default_rng(scenario["seed"]).normal(
            0, scenario["sensor_noise_std_K"], count
        )
        accepted, applied_ids, expected_commands = {}, set(), []
        temperature, bath, previous_output = [config["pid"]["initial_output_K"]] * 3
        errors, total_variation = [], 0.0
        previous_time = 0.0
        for index, row in enumerate(rows):
            now = row["time_s"]
            near(now, index * dt, "controller clock")
            require(
                row["run_id"] == run["run_id"]
                and row["clock_id"] == config["clock_id"],
                "row identity",
            )
            target = reference_target(config["recipe"], now)
            near(row["setpoint_K"], target, "recipe interpolation")
            applications = row["applied_receipts"]
            for event in applications:
                command_id = event["command_id"]
                require(
                    command_id in accepted and command_id not in applied_ids,
                    "application without unique acceptance",
                )
                original = accepted[command_id]
                require(
                    event
                    == dict(
                        original, event="applied", time_s=original["effective_at_s"]
                    ),
                    "application receipt changed",
                )
                require(
                    previous_time <= event["time_s"] <= now, "application timestamp"
                )
                applied_ids.add(command_id)
            boundaries = sorted(
                {
                    previous_time,
                    now,
                    *(e["time_s"] for e in applications),
                    *(
                        p["time_s"]
                        for p in scenario["disturbances"]
                        if previous_time < p["time_s"] < now
                    ),
                }
            )
            for a, b in zip(boundaries, boundaries[1:]):
                for event in applications:
                    if event["time_s"] == a:
                        bath = event["value"]
                midpoint = (a + b) / 2
                offset = next(
                    p["offset_K"]
                    for p in reversed(scenario["disturbances"])
                    if p["time_s"] <= midpoint
                )
                temperature = math.exp(
                    -(b - a) / scenario["time_constant_s"]
                ) * temperature + (
                    1 - math.exp(-(b - a) / scenario["time_constant_s"])
                ) * (bath + offset)
            for event in applications:
                if event["time_s"] == now:
                    bath = event["value"]
            near(
                row["plant_truth"]["temperature_K"],
                temperature,
                "analytic thermal response",
            )
            near(
                row["plant_truth"]["applied_bath_K"],
                bath,
                "applied/observed input distinction",
            )
            sensor = row["observation"]
            require(
                sensor["run_id"] == run["run_id"]
                and sensor["clock_id"] == config["clock_id"]
                and sensor["unit"] == "K"
                and sensor["channel"] == "temperature",
                "sensor identity/units",
            )
            near(sensor["received_at_s"], now, "sensor receipt clock")
            source_index = math.floor((now - scenario["sensor_delay_s"]) / dt + 1e-9)
            held = next(
                (
                    p
                    for p in scenario["held_intervals"]
                    if p["start_s"] <= now < p["end_s"]
                ),
                None,
            )
            if held:
                first_hold_tick = math.ceil(held["start_s"] / dt - 1e-9) * dt
                source_index = math.floor(
                    (first_hold_tick - scenario["sensor_delay_s"]) / dt + 1e-9
                )
            missing = any(
                p["start_s"] <= now < p["end_s"] for p in scenario["missing_intervals"]
            )
            require(
                sensor["valid"] == (source_index >= 0 and not missing),
                "sensor availability",
            )
            if sensor["valid"]:
                require(0 <= source_index <= index, "future sensor truth used")
                near(
                    sensor["measured_at_s"],
                    source_index * dt,
                    "sensor delay/held clock",
                )
                near(
                    sensor["value"],
                    rows[source_index]["plant_truth"]["temperature_K"]
                    + float(noise[source_index]),
                    "sensor value/noise reconstruction",
                )
            else:
                require(
                    sensor["value"] is None and sensor["measured_at_s"] is None,
                    "missing sensor filled with truth",
                )
            proposal, dispatch, decision = (
                row["proposal"],
                row["dispatch"],
                row["decision"],
            )
            require(
                proposal["channel"] == "bath_temperature" and proposal["unit"] == "K",
                "command channel/units",
            )
            require(
                proposal["run_id"] == run["run_id"]
                and proposal["clock_id"] == config["clock_id"],
                "command identity",
            )
            output = decision["output_K"]
            near(output, proposal["value"], "proposal/output distinction")
            near(output, dispatch["value"], "accepted value")
            require(limits["minimum"] <= output <= limits["maximum"], "output range")
            require(
                abs(output - previous_output)
                <= limits["maximum_rate_per_s"] * (0 if index == 0 else dt) + 1e-9,
                "output slew",
            )
            require(
                dispatch["event"] == "accepted"
                and dispatch["command_id"] == proposal["command_id"]
                and dispatch["command_id"] not in accepted,
                "dispatch acceptance",
            )
            near(proposal["proposed_at_s"], now, "proposal clock")
            near(dispatch["time_s"], now, "dispatch clock")
            near(
                dispatch["effective_at_s"],
                now + scenario["actuator_delay_s"],
                "actuator delay",
            )
            require(
                dispatch["effective_at_s"] <= proposal["expires_at_s"],
                "expired command application",
            )
            require(
                row["authorization"]
                in {"simulation_policy", "scripted_simulation_override"}
                and dispatch["authorization"] == row["authorization"],
                "simulation authorization",
            )
            require(
                row["evidence_kind"] == "simulation"
                and row["physical_hardware_contacted"] is False,
                "evidence scope",
            )
            if run["method"] == "pid":
                require(
                    sensor["sample_id"] in proposal["evidence_ids"]
                    and decision["feedback_sample_id"] == sensor["sample_id"],
                    "causal PID evidence",
                )
                if decision["error_K"] is not None:
                    near(
                        decision["error_K"],
                        target - sensor["value"],
                        "observable PID error",
                    )
                    near(
                        decision["proportional_K"],
                        config["pid"]["kp"] * (target - sensor["value"]),
                        "PID proportional units",
                    )
                require(
                    config["pid"]["integral_min_K"]
                    <= decision["integral_after_K"]
                    <= config["pid"]["integral_max_K"],
                    "integral range",
                )
            else:
                require(
                    decision["feedback_sample_id"] is None
                    and sensor["sample_id"] not in proposal["evidence_ids"],
                    "fixed baseline uses feedback",
                )
            accepted[dispatch["command_id"]] = dispatch
            expected_commands.extend(
                [
                    *applications,
                    dict(
                        event="proposed",
                        time_s=now,
                        proposal=proposal,
                        authorization=row["authorization"],
                    ),
                    dispatch,
                ]
            )
            errors.append(abs(temperature - target))
            total_variation += abs(output - previous_output)
            previous_output, previous_time = output, now
        require(
            commands == expected_commands, "command ledger does not match observations"
        )
        pending = run["accepted_not_yet_applied_at_end"]
        require(
            {p["command_id"] for p in pending} == set(accepted) - applied_ids,
            "end-of-window pending acknowledgments",
        )
        require(
            all(
                p == accepted[p["command_id"]] and p["effective_at_s"] > duration
                for p in pending
            ),
            "unapplied in-window command",
        )
        metrics = run["metrics"]
        require(metrics["samples"] == count, "metric sample count")
        near(metrics["sample_mae_K"], sum(errors) / count, "sample MAE")
        near(
            metrics["sample_rmse_K"],
            math.sqrt(sum(e * e for e in errors) / count),
            "sample RMSE",
        )
        near(
            metrics["trapezoidal_IAE_K_s"],
            sum((a + b) * dt / 2 for a, b in zip(errors, errors[1:])),
            "integrated error",
        )
        near(
            metrics["requested_input_total_variation_K"],
            total_variation,
            "command movement",
        )
        signed_errors = [
            r["plant_truth"]["temperature_K"] - r["setpoint_K"] for r in rows
        ]
        near(
            metrics["maximum_above_target_K"],
            max(0.0, max(signed_errors)),
            "maximum above target",
        )
        near(
            metrics["maximum_below_target_K"],
            max(0.0, -min(signed_errors)),
            "maximum below target",
        )
        for key, predicate in [
            ("fallback_steps", lambda r: r["decision"]["mode"] == "fallback"),
            ("scripted_manual_steps", lambda r: r["decision"]["mode"] == "manual"),
            ("invalid_observation_steps", lambda r: not r["observation"]["valid"]),
            ("saturated_steps", lambda r: r["decision"]["saturated"]),
            ("rate_limited_steps", lambda r: r["decision"]["rate_limited"]),
        ]:
            require(
                metrics[key] == sum(predicate(r) for r in rows),
                "metric counter: " + key,
            )
        reports.append(
            dict(
                run_id=run["run_id"],
                samples=count,
                accepted_commands=len(accepted),
                applied_commands=len(applied_ids),
                pending_commands=len(pending),
            )
        )
    require(sha256(root / "manifest.json") == manifest_hash, "run changed during audit")
    for name, digest in manifest["files"].items():
        require(
            sha256(local(root, name)) == digest,
            "artifact changed during audit: " + name,
        )
    return dict(
        status="pass",
        run_manifest_sha256=manifest_hash,
        files_verified=len(manifest["files"]),
        source_sha256=sha256(Path(__file__)),
        runs=reports,
        scope="Artifact/clock/sensor/command/analytic-plant/metric reconstruction; not physical validation or a proof of every PID trajectory",
    )
