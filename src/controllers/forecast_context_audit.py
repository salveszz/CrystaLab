# FILE PURPOSE: Reconstruct displayed forecast context and review bindings independently.
"""No online predictor, recommendation implementation or operator store is executed."""

import json
from pathlib import Path

from .adaptive_audit import reconstruct_states, reference_recommendation
from .audit import json_rows, local, require
from .benchmark import sha256
from .operator_audit import digest


def audit_forecast_context(
    root, source_root, reviews=(), *, base_pause=None, step=None
):
    root, source_root = Path(root), Path(source_root)
    manifest = json.loads((root / "manifest.json").read_text())
    source = json.loads((source_root / "manifest.json").read_text())
    require(
        manifest["schema"] == "crystalab_operator_forecast_context_v1"
        and manifest["status"] == "complete",
        "completed forecast context",
    )
    require(
        manifest["source_manifest_sha256"] == sha256(source_root / "manifest.json")
        and manifest["source_replay_manifest_sha256"]
        == source["source_replay_manifest_sha256"],
        "forecast source lineage",
    )
    for path, value in manifest["files"].items():
        require(sha256(local(root, path)) == value, "forecast context artifact")
    require(
        manifest["reference_manifest_sha256"]
        == sha256(root / "reference/manifest.json"),
        "forecast reference identity",
    )
    if base_pause is not None:
        require(
            manifest["base_pause_model_s"] == base_pause
            and manifest["model_step_s"] == step,
            "forecast simulation scale",
        )
    reference = json.loads((root / "reference/reference.json").read_text())
    reference_manifest = json.loads((root / "reference/manifest.json").read_text())
    for path, value in reference_manifest["files"].items():
        require(
            sha256(local(root / "reference", path)) == value,
            "forecast reference artifact",
        )
    require(
        json.loads((root / "reference/policy.json").read_text())
        == reference["configuration"],
        "forecast reference policy",
    )
    replay_manifest = json.loads((root / "replay/manifest.json").read_text())
    require(
        replay_manifest["status"] == "complete"
        and replay_manifest["source_replay_manifest_sha256"]
        == source["source_replay_manifest_sha256"]
        and replay_manifest["reference_manifest_sha256"]
        == manifest["reference_manifest_sha256"]
        and replay_manifest["labels_used_for_state_update"] is False
        and replay_manifest["actuator_commands_issued"] == 0,
        "forecast replay provenance/scope",
    )
    states = json_rows(root / "replay/states.jsonl")
    native = Path(source["source_replay"])
    require(
        sha256(native / "manifest.json") == source["source_replay_manifest_sha256"],
        "forecast native lineage",
    )
    reconstruct_states(native, states, reference)
    contexts = json.loads((root / "contexts.json").read_text())
    observations = json_rows(source_root / "candidate_observations.jsonl")
    require(
        len(contexts) == len(observations) == manifest["frames"],
        "forecast display coverage",
    )
    by_frame = {}
    for context, observation in zip(contexts, observations):
        now = observation["video_time_s"]
        due = [s for s in states if s["computed_at_source_s"] <= now]
        state = due[-1] if due else None
        summary = (
            None
            if state is None
            else dict(
                state_sha256=state["state_sha256"],
                observed_at_s=state["estimate"]["observed_at_s"],
                received_at_s=state["estimate"]["received_at_s"],
                source_frame_index=state["estimate"]["frame_index"],
                current_area_px2=state["estimate"]["area_px2"],
                forecast=state["forecast"],
                **state["parameter_after"],
            )
        )
        require(
            context["state"] == summary,
            "forecast displayed state must be latest received state",
        )
        require(
            context["source_frame_index"] == observation["frame_index"]
            and context["source_time_s"] == now
            and context["source_manifest_sha256"] == manifest["source_manifest_sha256"]
            and context["source_replay_manifest_sha256"]
            == manifest["source_replay_manifest_sha256"]
            and context["reference_manifest_sha256"]
            == manifest["reference_manifest_sha256"],
            "forecast display lineage/clock",
        )
        reason, ratio, duration = reference_recommendation(
            state,
            reference,
            now,
            manifest["base_pause_model_s"],
            manifest["model_step_s"],
        )
        rec = context["recommendation"]
        available = (
            state is not None
            and state["forecast"] is not None
            and reason
            not in {
                "forecast_unavailable",
                "forecast_not_received",
                "stale_area_evidence",
                "forecast_target_elapsed",
            }
        )
        require(
            rec["eligible"] == (reason is None)
            and rec["reason"] == reason
            and rec["forecast_available"] == available
            and rec["forecast_reference_ratio"] == ratio
            and rec["pause_model_s"] == duration
            and rec["state_sha256"]
            == (None if state is None else state["state_sha256"])
            and rec["can_authorize_hardware"] is False,
            "forecast recommendation reconstruction",
        )
        require(
            context["context_sha256"]
            == digest({k: v for k, v in context.items() if k != "context_sha256"}),
            "forecast context digest",
        )
        by_frame[context["source_frame_index"]] = context
    for review in reviews:
        context = by_frame[review["source_frame_index"]]
        require(
            review.get("forecast_context") == context,
            "operator forecast review context",
        )
        basis = review.get("pause_basis", "manual")
        require(basis in {"manual", "adaptive_suggestion"}, "operator pause basis")
        if basis == "adaptive_suggestion":
            require(
                review["action"] == "approve_pause"
                and context["recommendation"]["eligible"]
                and review["pause_duration_model_s"]
                == context["recommendation"]["pause_model_s"],
                "operator suggested pause duration",
            )
    return dict(
        status="pass",
        states=len(states),
        contexts=len(contexts),
        reviews=len(reviews),
        physical_hardware_contacted=False,
    )
