# FILE PURPOSE: Reconstruct online sufficient statistics and adaptive pause decisions independently.
# No predictor, recommendation, supervisor, PID or plant implementation is executed.
"""Native-mask, matured-update, source-clock and adaptive-action semantic verification."""

import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .audit import audit_supervision, json_rows, local, near, require
from .benchmark import sha256


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reconstruct_states(replay, states, reference, *, mapping=None, model_step_s=None):
    replay = Path(replay).resolve()
    m = json.loads((replay / "manifest.json").read_text())
    for p, d in m["files"].items():
        require(sha256(local(replay, p)) == d, "adaptive native source artifact")
    native = json_rows(replay / "measurements.jsonl")
    require(len(native) == len(states), "adaptive area coverage")
    config = reference["configuration"]
    gain_reference = reference["positive_relative_gain_reference"]
    xx = config["prior_precision"]
    xy = xx * config["initial_coefficient"]
    coefficient = config["initial_coefficient"]
    error_ratio = reference["initial_error_ratio"]
    updates = 0
    previous = None
    pending = None
    result = {}
    capture = cv2.VideoCapture(str(replay / "overlay.avi"))
    try:
        for row, state in zip(native, states):
            ok, image = capture.read()
            require(ok, "adaptive native preview coverage")
            e = state["estimate"]
            stamp = row["video_time_s"]
            area = row["projected_area_px"]
            require(
                e["run_id"] == m["run_id"]
                and e["clock_id"] == "source_video:" + m["input"]["sha256"]
                and e["view_id"] == "fixed_view_assumption:" + m["run_id"],
                "adaptive stream identity",
            )
            require(
                e["frame_index"] == row["frame_index"]
                and e["valid"] == row["valid"]
                and e["requires_inspection"]
                == row.get("requires_inspection", not row["valid"])
                and e["area_px2"] == area
                and e["image_shape"] == list(image.shape[:2]),
                "adaptive native measurement",
            )
            near(e["observed_at_s"], stamp, "adaptive source clock")
            near(e["received_at_s"], stamp, "adaptive software receipt clock")
            computed = (
                stamp
                if mapping is None
                else (
                    (
                        math.ceil(
                            stamp
                            * mapping["target_seconds_per_source_second"]
                            / model_step_s
                            - 1e-10
                        )
                        * model_step_s
                    )
                    / mapping["target_seconds_per_source_second"]
                )
            )
            near(
                state["computed_at_source_s"],
                computed,
                "adaptive computation source clock",
            )
            if row["mask_path"] is not None:
                path = local(replay, row["mask_path"])
                mask = cv2.imread(str(path), 0)
                require(
                    mask is not None
                    and set(np.unique(mask)) <= {0, 255}
                    and np.count_nonzero(mask) == area
                    and e["mask_sha256"] == sha256(path),
                    "adaptive native mask identity/area",
                )
            else:
                require(e["mask_sha256"] is None, "adaptive invalid mask identity")
            usable = (
                e["valid"]
                and not e["requires_inspection"]
                and computed - stamp <= config["maximum_age_source_s"]
            )
            comparable = (
                previous is not None
                and previous["usable"]
                and stamp - previous["time"] <= config["maximum_gap_source_s"]
                and usable
            )
            before = state["parameter_before"]
            for key, value in dict(
                xx=xx,
                xy=xy,
                coefficient=coefficient,
                error_ratio=error_ratio,
                matured_updates=updates,
            ).items():
                near(before[key], value, "adaptive parameter-before " + key)
            update = None
            if (
                pending is not None
                and usable
                and (
                    previous is None
                    or stamp - previous["time"] <= config["maximum_gap_source_s"]
                )
                and abs(stamp - pending["target"]) < 1e-9
            ):
                scale = pending["area"] * gain_reference
                x = pending["gain"] / scale
                y = (area - pending["area"]) / scale
                residual = (area - pending["future"]) / scale
                xx = config["forgetting_factor"] * xx + x * x
                xy = config["forgetting_factor"] * xy + x * y
                coefficient = min(
                    config["coefficient_max"], max(config["coefficient_min"], xy / xx)
                )
                error_ratio = (1 - config["error_ema_alpha"]) * error_ratio + config[
                    "error_ema_alpha"
                ] * abs(residual)
                updates += 1
                update = state["matured_update"]
                require(update is not None, "omitted matured adaptive update")
                require(
                    update["cut_frame_index"] == pending["frame"]
                    and update["target_frame_index"] == row["frame_index"]
                    and update["target_mask_sha256"] == e["mask_sha256"],
                    "adaptive matured target identity",
                )
                for key, value in dict(
                    x_normalized=x,
                    y_observed_normalized=y,
                    pre_update_residual_normalized=residual,
                    coefficient_used_for_previous_prediction=pending["coefficient"],
                ).items():
                    near(update[key], value, "adaptive observed update " + key)
            if update is None:
                require(
                    state["matured_update"] is None,
                    "adaptive update used unavailable/future target",
                )
            for key, value in dict(
                xx=xx,
                xy=xy,
                coefficient=coefficient,
                error_ratio=error_ratio,
                matured_updates=updates,
            ).items():
                near(
                    state["parameter_after"][key],
                    value,
                    "adaptive parameter-after " + key,
                )
            pending = None
            forecast = None
            if comparable and stamp + config["horizon_source_s"] > computed:
                gain = (
                    (area - previous["area"])
                    * config["horizon_source_s"]
                    / (stamp - previous["time"])
                )
                future = min(
                    image.shape[0] * image.shape[1], max(0.0, area + coefficient * gain)
                )
                forecast = state["forecast"]
                require(forecast is not None, "omitted usable adaptive forecast")
                for key, value in dict(
                    future_area_px2=future,
                    gain_px2=future - area,
                    relative_gain=(future - area) / area,
                    target_time_s=stamp + config["horizon_source_s"],
                ).items():
                    near(forecast[key], value, "adaptive prediction " + key)
                pending = dict(
                    area=area,
                    gain=gain,
                    future=future,
                    target=stamp + config["horizon_source_s"],
                    frame=row["frame_index"],
                    coefficient=coefficient,
                )
            if forecast is None:
                require(
                    state["forecast"] is None,
                    "adaptive forecast used unavailable history",
                )
            payload = {k: v for k, v in state.items() if k != "state_sha256"}
            require(digest(payload) == state["state_sha256"], "adaptive state digest")
            require(state["state_sha256"] not in result, "duplicate adaptive state")
            result[state["state_sha256"]] = state
            previous = dict(usable=usable, time=stamp, area=area)
        require(not capture.read()[0], "extra adaptive source frames")
    finally:
        capture.release()
    return result


def reference_recommendation(state, reference, now, base, step):
    cfg = reference["configuration"]
    reason = None
    forecast = None if state is None else state["forecast"]
    if forecast is None:
        reason = "forecast_unavailable"
    elif now + 1e-12 < state["estimate"]["received_at_s"]:
        reason = "forecast_not_received"
    elif now - state["estimate"]["observed_at_s"] > cfg["maximum_age_source_s"]:
        reason = "stale_area_evidence"
    elif now >= forecast["target_time_s"] - 1e-12:
        reason = "forecast_target_elapsed"
    elif state["parameter_after"]["matured_updates"] < cfg["minimum_matured_updates"]:
        reason = "insufficient_matured_updates"
    elif state["parameter_after"]["error_ratio"] > cfg["maximum_error_ratio"]:
        reason = "observed_forecast_error_too_large"
    elif forecast["relative_gain"] <= 0:
        reason = "no_positive_projected_growth"
    ratio = (
        None
        if forecast is None
        else forecast["relative_gain"] / reference["positive_relative_gain_reference"]
    )
    duration = None
    if reason is None:
        duration = (
            math.floor(
                min(
                    cfg["maximum_pause_model_s"],
                    max(cfg["minimum_pause_model_s"], base * ratio),
                )
                / step
                + 1e-12
            )
            * step
        )
        if duration < cfg["minimum_pause_model_s"]:
            reason = "pause_bounds_do_not_fit_model_step"
            duration = None
    return reason, ratio, duration


def audit_adaptive_supervision(root, manifest, config):
    require(
        manifest["adaptive_policy_implemented"] is True
        and manifest["video_action_responsive"] is False
        and manifest["physical_growth_clock_claimed"] is False,
        "adaptive software/effect scope",
    )
    ref_root = root / "adaptive_reference"
    ref_manifest = json.loads((ref_root / "manifest.json").read_text())
    require(
        sha256(ref_root / "manifest.json") == manifest["reference_manifest_sha256"],
        "adaptive reference identity",
    )
    for p, d in ref_manifest["files"].items():
        require(sha256(local(ref_root, p)) == d, "adaptive reference artifact")
    reference = json.loads((ref_root / "reference.json").read_text())
    surrogate = copy.deepcopy(manifest)
    surrogate.update(
        scripted_executions=manifest["adaptive_executions"],
        scripted_decisions=manifest["adaptive_decisions"],
        final_proposals=manifest["adaptive_final_proposals"],
    )
    actions, source_root, source = audit_supervision(root, surrogate, config)
    mapping = manifest["clock_mapping"]
    replay = Path(source["source_replay"])
    require(
        sha256(replay / "manifest.json") == source["source_replay_manifest_sha256"],
        "adaptive observation/candidate lineage",
    )
    states = json_rows(root / "adaptive_supervised_area_states.jsonl")
    by_hash = reconstruct_states(
        replay, states, reference, mapping=mapping, model_step_s=config["step_s"]
    )
    require(
        manifest["final_adaptive_parameters"] == states[-1]["parameter_after"],
        "adaptive final state",
    )
    entries = json_rows(root / "adaptive_supervised_supervision.jsonl")
    assessments = [r for r in entries if r["event"] == "adaptive_assessment"]
    require(
        len(assessments) == round(config["duration_s"] / config["step_s"]) + 1,
        "adaptive assessment coverage",
    )
    by_time = {r["model_time_s"]: r for r in assessments}
    state_index = 0
    for index, r in enumerate(assessments):
        near(
            r["model_time_s"],
            index * config["step_s"],
            "adaptive assessment model clock",
        )
        now = r["model_time_s"] / mapping["target_seconds_per_source_second"]
        near(r["source_time_s"], now, "adaptive assessment source clock")
        while (
            state_index + 1 < len(states)
            and states[state_index + 1]["computed_at_source_s"] <= now + 1e-12
        ):
            state_index += 1
        state = (
            states[state_index]
            if states[state_index]["computed_at_source_s"] <= now + 1e-12
            else None
        )
        expected_hash = None if state is None else state["state_sha256"]
        q = r["recommendation"]
        require(
            q["state_sha256"] == expected_hash,
            "adaptive policy used future or wrong state",
        )
        reason, ratio, duration = reference_recommendation(
            state,
            reference,
            now,
            json.loads((root / "supervised_config.json").read_text())[
                "pause_model_duration_s"
            ],
            config["step_s"],
        )
        require(
            q["reason"] == reason
            and q["eligible"] == (reason is None)
            and q["pause_model_s"] == duration
            and q["can_authorize_hardware"] is False,
            "adaptive recommendation eligibility/duration",
        )
        require(
            q["forecast_available"]
            == (
                state is not None
                and state["forecast"] is not None
                and reason
                not in {
                    "forecast_unavailable",
                    "forecast_not_received",
                    "stale_area_evidence",
                    "forecast_target_elapsed",
                }
            ),
            "adaptive forecast availability",
        )
        if ratio is not None:
            near(q["forecast_reference_ratio"], ratio, "adaptive reference ratio")
    committed = 0
    for action in actions:
        assessment = by_time[action["model_decision_time_s"]]
        q = assessment["recommendation"]
        require(
            q["eligible"]
            and action["adaptive_state_sha256"] == q["state_sha256"]
            and action["assessment_sha256"] == digest(assessment)
            and action["forecast_informed"] is True,
            "adaptive action forecast binding",
        )
        expected = min(
            q["pause_model_s"],
            json.loads((root / "supervised_config.json").read_text())[
                "maximum_total_pause_model_s"
            ]
            - committed,
        )
        near(
            action["pause_duration_model_s"], expected, "adaptive bounded chosen pause"
        )
        require(
            expected >= reference["configuration"]["minimum_pause_model_s"],
            "adaptive minimum pause",
        )
        committed += expected
    logged_actions = [
        r["execution"] for r in entries if r["event"] == "adaptive_simulation_decision"
    ]
    require(logged_actions == actions, "adaptive action journal coverage")
    # Reconstruct every opportunity, including choices the writer might omit.
    demo = json.loads((root / "supervised_config.json").read_text())
    policy = json.loads((root / "policy.json").read_text())
    candidate_rows = json_rows(source_root / "candidate_observations.jsonl")
    thermal_rows = json_rows(
        root
        / next(
            r["records_path"]
            for r in manifest["runs"]
            if r["method"] == "adaptive_supervised"
        )
    )
    proposals = [e["proposal"] for e in manifest["adaptive_final_proposals"]]
    observed_index = 0
    chosen = set()
    pause_until = 0.0
    reserved = 0.0
    expected_actions = []
    for assessment, row in zip(assessments, thermal_rows):
        now = assessment["model_time_s"]
        source_now = assessment["source_time_s"]
        q = assessment["recommendation"]
        while (
            observed_index + 1 < len(candidate_rows)
            and candidate_rows[observed_index + 1]["video_time_s"] <= source_now + 1e-12
        ):
            observed_index += 1
        observation = candidate_rows[observed_index]
        supervisor_now = max(source_now, observation["video_time_s"])
        packet_gap = (
            observed_index > 0
            and observation["video_time_s"]
            - candidate_rows[observed_index - 1]["video_time_s"]
            > policy["maximum_packet_gap_s"]
        )
        visible = (
            observation["observable"]["crystal_candidate"]
            and not packet_gap
            and supervisor_now - observation["video_time_s"]
            <= policy["maximum_evidence_age_s"]
        )
        sensor = row["observation"]
        feedback = (
            sensor["valid"]
            and sensor["clock_id"] == config["clock_id"]
            and sensor["received_at_s"] <= now
            and now - sensor["measured_at_s"] <= config["pid"]["maximum_feedback_age_s"]
        )
        if (
            now < pause_until
            or source_now > candidate_rows[-1]["video_time_s"] + 1e-12
            or not visible
            or not feedback
            or not q["eligible"]
        ):
            continue
        for proposal in proposals:
            if (
                proposal["proposal_id"] in chosen
                or proposal["kind"] != "ramp_review"
                or not proposal["proposed_at_s"]
                <= supervisor_now
                < proposal["expires_at_s"]
                or supervisor_now - proposal["proposed_at_s"] + 1e-12
                < demo["scripted_decision_delay_source_s"]
            ):
                continue
            track = next(
                (
                    t
                    for t in observation["tracks"]
                    if t["id"] == proposal["track_id"]
                    and t["state"] == "observed"
                    and t["last_frame_index"] == observation["frame_index"]
                ),
                None,
            )
            if track is None:
                continue
            duration = min(
                q["pause_model_s"], demo["maximum_total_pause_model_s"] - reserved
            )
            if duration < reference["configuration"]["minimum_pause_model_s"]:
                continue
            expected_actions.append((proposal["proposal_id"], now, duration))
            chosen.add(proposal["proposal_id"])
            reserved += duration
            pause_until = now + duration
            break
    require(
        len(expected_actions) == len(actions),
        "adaptive omitted or extra policy execution",
    )
    for expected, action in zip(expected_actions, actions):
        require(expected[0] == action["proposal_id"], "adaptive selected proposal")
        near(
            expected[1],
            action["model_decision_time_s"],
            "adaptive first eligible decision time",
        )
        near(expected[2], action["pause_duration_model_s"], "adaptive policy duration")
    return actions
