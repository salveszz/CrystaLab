# FILE PURPOSE: Independently reconstruct thermal experiment records and causal provenance.
# Artifact hashes, sensor delays/noise, command acknowledgments, exact plant response
# and reported metrics are checked without calling the controller or plant classes.
"""Independent saved-artifact audit for the temperature software benchmark."""

import json
import hashlib
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


def audit_supervision(root, manifest, config):
    """Reconstruct clock-bound scripted pause eligibility without executing rules."""
    require(
        sha256(root / "supervised_config.json") == manifest["supervised_config_sha256"],
        "supervised configuration identity",
    )
    demo = json.loads((root / "supervised_config.json").read_text())
    require(
        manifest["physical_hardware_contacted"] is False
        and manifest["operator_reviews_created"] == 0,
        "software-only supervision scope",
    )
    require(
        demo["thermal"] == config,
        "thermal configuration differs from the demo declaration",
    )
    require(
        sha256(root / "policy.json") == manifest["policy_sha256"],
        "rule policy identity",
    )
    policy = json.loads((root / "policy.json").read_text())
    mapping = demo["clock_mapping"]
    require(
        mapping == manifest["clock_mapping"]
        and mapping["scope"] == "software_demonstration_only",
        "clock-map scope",
    )
    require(
        mapping["source_origin_s"] == mapping["target_origin_s"] == 0
        and mapping["target_clock_id"] == config["clock_id"],
        "clock-map identities",
    )
    require(mapping["target_seconds_per_source_second"] > 0, "clock-map scale")
    source_root = Path(manifest["source_analysis"]).resolve()
    source = json.loads((root / "candidate_manifest.json").read_text())
    require(
        sha256(root / "candidate_manifest.json")
        == manifest["candidate_manifest_sha256"]
        == sha256(source_root / "manifest.json"),
        "candidate lineage",
    )
    require(
        source["status"] == "complete"
        and source["source_info"] == manifest["source_info"]
        and source["source_info"]["evidence_kind"] in {"fixture", "synthetic"},
        "real evidence cannot become simulated action evidence",
    )
    for name, digest in source["files"].items():
        require(
            sha256(local(source_root, name)) == digest,
            "source candidate artifact identity",
        )
    parent = json.loads((root / "source_replay_manifest.json").read_text())
    require(
        sha256(root / "source_replay_manifest.json")
        == source["source_replay_manifest_sha256"],
        "source observation context",
    )
    require(
        parent["input"].get("sha256") == mapping["expected_source_video_sha256"],
        "mapped video identity",
    )
    source_clock = "source_video:" + (
        parent["input"].get("sha256") or manifest["candidate_manifest_sha256"]
    )
    events = {
        e["event_id"]: e for e in json_rows(source_root / "candidate_events.jsonl")
    }
    observations = json_rows(source_root / "candidate_observations.jsonl")
    proposals = {e["proposal"]["proposal_id"]: e for e in manifest["final_proposals"]}
    decisions = {e["decision_id"]: e for e in manifest["scripted_decisions"]}
    actions = manifest["scripted_executions"]
    require(
        len(decisions) == len(actions) == len(manifest["scripted_decisions"]),
        "unique scripted decision IDs",
    )
    total_pause, previous_end = 0.0, 0.0
    for action in actions:
        entry = proposals[action["proposal_id"]]
        proposal = entry["proposal"]
        digest = hashlib.sha256(
            json.dumps(
                proposal, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        require(
            digest == entry["proposal_sha256"] == action["proposal_sha256"],
            "exact proposal decision binding",
        )
        receipt = decisions[action["decision_id"]]
        require(
            proposal["run_id"] == source["run_id"]
            and proposal["clock_id"] == receipt["clock_id"] == source_clock
            and proposal["evidence_kind"] == source["source_info"]["evidence_kind"]
            and proposal["numerical_actuator_command"] is None,
            "proposal source/clock/action identity",
        )
        require(
            receipt["proposal_id"] == proposal["proposal_id"]
            and receipt["proposal_sha256"] == digest
            and receipt["status"] == entry["status"] == "simulation_authorized",
            "scripted authorization receipt",
        )
        require(
            proposal["kind"] == "ramp_review"
            and proposal["origin"] == "appeared_after_clear_view"
            and proposal["requires_operator_review"] is True
            and proposal["operator_reviewed"] is False,
            "ramp-review semantics",
        )
        require(
            receipt["basis"] == action["basis"] == "scripted_fixture_policy"
            and receipt["operator_reviewed"] is action["operator_reviewed"] is False
            and receipt["physical_hardware_authorized"]
            is action["physical_hardware_authorized"]
            is False,
            "scripted action is not human/hardware approval",
        )
        source_time = action["source_decision_time_s"]
        near(receipt["time_s"], source_time, "decision source clock")
        near(
            action["model_decision_time_s"],
            source_time * mapping["target_seconds_per_source_second"],
            "decision model clock",
        )
        require(
            proposal["proposed_at_s"] + demo["scripted_decision_delay_source_s"]
            <= source_time + 1e-10
            < proposal["expires_at_s"],
            "decision delay/expiry",
        )
        require(
            source_time <= observations[-1]["video_time_s"] + 1e-10,
            "decision after source coverage",
        )
        current = next(
            r
            for r in reversed(observations)
            if r["video_time_s"] <= source_time + 1e-10
        )
        require(
            current["observable"]["crystal_candidate"]
            and source_time - current["video_time_s"]
            <= policy["maximum_evidence_age_s"],
            "fresh decision view",
        )
        event = events[proposal["candidate_event_id"]]
        require(
            event["candidate_type"] == "crystal_candidate"
            and event["origin"] == proposal["origin"]
            and event["track_id"] == proposal["track_id"]
            and event["first_detection_mask_path"]
            == proposal["first_detection_mask_path"],
            "source candidate decision evidence",
        )
        require(
            any(
                t["id"] == event["track_id"]
                and t["state"] == "observed"
                and t["last_frame_index"] == current["frame_index"]
                for t in current["tracks"]
            ),
            "decision candidate no longer observed",
        )
        require(
            action["model_decision_time_s"] >= previous_end
            and 0 < action["pause_duration_model_s"] <= demo["pause_model_duration_s"],
            "pause overlap/duration",
        )
        near(
            action["pause_until_model_s"],
            action["model_decision_time_s"] + action["pause_duration_model_s"],
            "bounded pause end",
        )
        previous_end = action["pause_until_model_s"]
        total_pause += action["pause_duration_model_s"]
    require(
        total_pause <= demo["maximum_total_pause_model_s"], "cumulative pause budget"
    )
    return actions, source_root, source


def audit_control_benchmark(path):
    root = Path(path).resolve()
    manifest_hash = sha256(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest.get("schema")
        in {
            "crystalab_temperature_benchmark_run_v1",
            "crystalab_supervised_control_demo_v1",
            "crystalab_operator_control_demo_v1",
        }
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
    supervised = manifest["schema"] in {
        "crystalab_supervised_control_demo_v1",
        "crystalab_operator_control_demo_v1",
    }
    if manifest["schema"] == "crystalab_operator_control_demo_v1":
        from .operator_audit import audit_operator_supervision

        supervision_auditor = audit_operator_supervision
    else:
        supervision_auditor = audit_supervision
    actions, source_root, source_manifest = (
        supervision_auditor(root, manifest, config) if supervised else ([], None, None)
    )
    methods = (
        ["fixed_recipe", "pid", "rule_supervised"]
        if supervised
        else ["fixed_recipe", "pid"]
    )
    require(
        len(manifest["runs"]) == len(methods) * len(config["scenarios"]),
        "comparison coverage",
    )
    require(
        {(r["scenario"], r["method"]) for r in manifest["runs"]}
        == {(s["id"], m) for s in config["scenarios"] for m in methods},
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
            if run["method"] == "rule_supervised":
                near(row["nominal_recipe_setpoint_K"], target, "nominal recipe target")
                phase = now - sum(
                    max(
                        0.0,
                        min(now, a["pause_until_model_s"]) - a["model_decision_time_s"],
                    )
                    for a in actions
                )
                near(row["supervision"]["recipe_phase_s"], phase, "paused recipe phase")
                target = reference_target(config["recipe"], phase)
                for action in actions:
                    if action["model_decision_time_s"] <= now:
                        require(
                            action["proposal_id"] in row["proposal"]["evidence_ids"]
                            and action["decision_id"]
                            in row["proposal"]["evidence_ids"],
                            "supervised command causal evidence",
                        )
                    if action["model_decision_time_s"] == now:
                        sensor = row["observation"]
                        require(
                            sensor["valid"]
                            and sensor["sample_id"] == action["sensor_sample_id"]
                            and now - sensor["measured_at_s"]
                            <= config["pid"]["maximum_feedback_age_s"],
                            "fresh process evidence at simulation decision",
                        )
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
            if run["method"] in {"pid", "rule_supervised"}:
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
        if supervised:
            nominal_errors = [
                abs(
                    r["plant_truth"]["temperature_K"]
                    - reference_target(config["recipe"], r["time_s"])
                )
                for r in rows
            ]
            near(
                run["nominal_recipe_sample_mae_K"],
                sum(nominal_errors) / len(rows),
                "common nominal recipe error",
            )
            phase = rows[-1].get("supervision", {}).get("recipe_phase_s", duration)
            near(run["final_recipe_phase_s"], phase, "final recipe phase")
            near(run["recipe_phase_lag_s"], duration - phase, "recipe lag cost")
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
    if source_root is not None:
        require(
            sha256(source_root / "manifest.json")
            == manifest["candidate_manifest_sha256"],
            "candidate source changed during control audit",
        )
        for name, digest in source_manifest["files"].items():
            require(
                sha256(local(source_root, name)) == digest,
                "candidate artifact changed during audit",
            )
    return dict(
        status="pass",
        run_manifest_sha256=manifest_hash,
        files_verified=len(manifest["files"]),
        source_sha256=sha256(Path(__file__)),
        runs=reports,
        scope="Artifact/clock/sensor/command/analytic-plant/metric reconstruction; not physical validation or a proof of every PID trajectory",
    )
