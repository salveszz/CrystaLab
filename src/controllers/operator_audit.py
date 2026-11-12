# FILE PURPOSE: Independently reconstruct operator replay decisions and manual pause endings.
# This verifier does not execute a supervisor, controller, plant or review normalizer.
"""Audit source/receipt/clock/action lineage before the shared numerical plant audit."""

import hashlib
import json
from pathlib import Path

from .audit import local, near, require, json_rows
from .benchmark import sha256


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def audit_operator_supervision(root, manifest, config):
    demo = json.loads((root / "supervised_config.json").read_text())
    require(
        sha256(root / "supervised_config.json") == manifest["supervised_config_sha256"]
        and demo["thermal"] == config,
        "operator thermal configuration",
    )
    require(
        sha256(root / "policy.json") == manifest["policy_sha256"],
        "operator rule policy",
    )
    policy = json.loads((root / "policy.json").read_text())
    mapping = manifest["clock_mapping"]
    require(
        mapping == demo["clock_mapping"]
        and mapping["scope"] == "software_demonstration_only",
        "operator software clock map",
    )
    require(
        mapping["source_origin_s"] == mapping["target_origin_s"] == 0
        and mapping["target_seconds_per_source_second"] > 0
        and mapping["target_clock_id"] == config["clock_id"],
        "operator clock identities",
    )
    source_root = Path(manifest["source_analysis"]).resolve()
    source = json.loads((root / "candidate_manifest.json").read_text())
    require(
        sha256(root / "candidate_manifest.json")
        == manifest["candidate_manifest_sha256"]
        == sha256(source_root / "manifest.json"),
        "operator candidate lineage",
    )
    require(
        source["status"] == "complete"
        and source["source_info"] == manifest["source_info"]
        and source["source_info"]["evidence_kind"] in {"fixture", "synthetic"}
        and source["source_info"].get("split") != "test",
        "operator development simulation scope",
    )
    for name, d in source["files"].items():
        require(sha256(local(source_root, name)) == d, "operator source artifact")
    parent = json.loads((root / "source_replay_manifest.json").read_text())
    require(
        sha256(root / "source_replay_manifest.json")
        == source["source_replay_manifest_sha256"]
        and parent["input"]["sha256"] == mapping["expected_source_video_sha256"],
        "operator source video identity",
    )
    clock = "source_video:" + parent["input"]["sha256"]
    observations = json_rows(source_root / "candidate_observations.jsonl")
    frames = {r["frame_index"]: r for r in observations}
    events = {
        r["event_id"]: r for r in json_rows(source_root / "candidate_events.jsonl")
    }
    require(
        sha256(root / "reviews.json") == manifest["reviews_sha256"],
        "operator review bytes",
    )
    reviews = json.loads((root / "reviews.json").read_text())
    previous_hash, previous_time, ids = None, -1, set()
    proposals = {e["proposal"]["proposal_id"]: e for e in manifest["final_proposals"]}
    for r in reviews:
        require(
            r["schema"] == "crystalab_operator_replay_review_v1"
            and r["acknowledged"] is True
            and r["reviewer"].strip()
            and r["previous_review_sha256"] == previous_hash
            and r["review_id"] not in ids,
            "operator review attestation/ordering",
        )
        require(
            r["reviewer_kind"] in {"human_self_attestation", "automated_fixture_qa"}
            and (
                r["reviewer_kind"] != "automated_fixture_qa"
                or r["evidence_kind"] == "fixture"
            ),
            "operator QA provenance",
        )
        require(
            r["run_id"] == source["run_id"]
            and r["source_manifest_sha256"] == manifest["candidate_manifest_sha256"]
            and r["clock_id"] == clock
            and r["evidence_kind"] == source["source_info"]["evidence_kind"]
            and r["physical_hardware_authorized"] is False,
            "operator review source/scope",
        )
        frame = frames[r["source_frame_index"]]
        near(r["source_time_s"], frame["video_time_s"], "operator review frame clock")
        require(r["source_time_s"] >= previous_time, "operator review clock order")
        p = proposals[r["proposal_id"]]
        require(
            digest(p["proposal"]) == p["proposal_sha256"] == r["proposal_sha256"]
            and p["proposal"]["proposed_at_s"] <= r["source_time_s"],
            "operator exact observed proposal",
        )
        previous_hash, previous_time = digest(r), r["source_time_s"]
        ids.add(r["review_id"])
    require(
        manifest["operator_reviews_created"]
        == sum(r["reviewer_kind"] == "human_self_attestation" for r in reviews)
        and manifest["physical_hardware_contacted"] is False,
        "operator review count/hardware scope",
    )
    review_by_id = {r["review_id"]: r for r in reviews}
    actions = manifest["executions"]
    require(
        {a["decision_id"] for a in actions}
        == {r["review_id"] for r in reviews if r["action"] == "approve_pause"}
        and len({a["decision_id"] for a in actions}) == len(actions),
        "operator approval coverage",
    )
    require(
        {r["review_id"] for r in manifest["manual_resumptions"]}
        == {r["review_id"] for r in reviews if r["action"] == "resume_recipe"},
        "manual resume coverage",
    )
    reserved, previous_end = 0.0, 0.0
    for a in actions:
        r = review_by_id[a["decision_id"]]
        p = proposals[a["proposal_id"]]["proposal"]
        require(
            a["review_receipt_sha256"] == digest(r)
            and a["proposal_sha256"] == r["proposal_sha256"]
            and a["proposal_id"] == r["proposal_id"],
            "operator execution receipt",
        )
        require(
            p["kind"] == "ramp_review"
            and p["origin"] == "appeared_after_clear_view"
            and p["numerical_actuator_command"] is None
            and p["proposed_at_s"] <= r["source_time_s"] < p["expires_at_s"],
            "operator pause eligibility/expiry",
        )
        require(p["candidate_event_id"] in events, "operator source event")
        frame = frames[r["source_frame_index"]]
        require(
            frame["observable"]["crystal_candidate"]
            and any(
                t["id"] == p["track_id"]
                and t["state"] == "observed"
                and t["last_frame_index"] == frame["frame_index"]
                for t in frame["tracks"]
            ),
            "operator fresh observed candidate",
        )
        require(
            a["operator_reviewed"] == (r["reviewer_kind"] == "human_self_attestation")
            and a["physical_hardware_authorized"] is False,
            "operator action provenance",
        )
        near(
            a["source_decision_time_s"],
            r["source_time_s"],
            "operator source decision time",
        )
        near(
            a["model_decision_time_s"],
            r["source_time_s"] * mapping["target_seconds_per_source_second"],
            "operator mapped decision time",
        )
        near(
            a["requested_pause_duration_model_s"],
            r["pause_duration_model_s"],
            "operator requested duration",
        )
        require(
            0 < r["pause_duration_model_s"] <= demo["pause_model_duration_s"]
            and a["model_decision_time_s"] >= previous_end,
            "operator bounded non-overlapping pause",
        )
        requested_end = a["model_decision_time_s"] + r["pause_duration_model_s"]
        near(
            a["requested_pause_until_model_s"], requested_end, "operator requested end"
        )
        resumes = [
            x
            for x in reviews
            if x["action"] == "resume_recipe" and x["pause_review_id"] == r["review_id"]
        ]
        require(len(resumes) <= 1, "operator duplicate resume")
        end = requested_end
        if resumes:
            rr = resumes[0]
            end = rr["source_time_s"] * mapping["target_seconds_per_source_second"]
            require(
                a["model_decision_time_s"] < end < requested_end
                and rr["proposal_id"] == a["proposal_id"],
                "operator manual resume interval",
            )
            saved = next(
                x
                for x in manifest["manual_resumptions"]
                if x["review_id"] == rr["review_id"]
            )
            near(saved["model_time_s"], end, "operator resume model time")
            require(
                saved["review_receipt_sha256"] == digest(rr)
                and saved["physical_hardware_authorized"] is False,
                "operator resume receipt",
            )
        near(a["pause_until_model_s"], end, "operator actual pause end")
        near(
            a["pause_duration_model_s"],
            end - a["model_decision_time_s"],
            "operator actual pause duration",
        )
        reserved += r["pause_duration_model_s"]
        previous_end = end
    require(
        reserved <= demo["maximum_total_pause_model_s"],
        "operator reserved pause budget",
    )
    records = json_rows(
        root
        / next(
            r["records_path"]
            for r in manifest["runs"]
            if r["method"] == "rule_supervised"
        )
    )
    for resume in manifest["manual_resumptions"]:
        row = next(r for r in records if r["time_s"] == resume["model_time_s"])
        require(
            row["observation"]["valid"]
            and row["observation"]["sample_id"] == resume["sensor_sample_id"]
            and resume["review_id"] in row["proposal"]["evidence_ids"],
            "operator resume fresh feedback and command lineage",
        )
    return actions, source_root, source
