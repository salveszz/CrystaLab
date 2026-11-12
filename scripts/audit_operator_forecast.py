# FILE PURPOSE: Independently audit forecast display, review context and preserved operator simulations.
"""Reconstruct a prepared operator forecast session without executing its model or store."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.audit import audit_control_benchmark, json_rows, local, require
from controllers.benchmark import sha256
from controllers.forecast_context_audit import audit_forecast_context
from controllers.operator_audit import digest


def audit_session(session):
    root = Path(session).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest["status"] == "ready" and manifest.get("forecast_available") is True,
        "prepared forecast session",
    )
    for path, value in manifest["files"].items():
        require(sha256(local(root, path)) == value, "operator session artifact")
    source = Path(manifest["source_analysis"])
    require(
        sha256(source / "manifest.json") == manifest["source_manifest_sha256"],
        "operator source identity",
    )
    history = json.loads((root / "history.json").read_text())
    frames = {
        f["frame_index"]: f for f in json.loads((root / "frames.json").read_text())
    }
    proposals = {
        p["proposal"]["proposal_id"]: p
        for p in json_rows(root / "shadow/proposals.jsonl")
    }
    reviews, paths = (
        [],
        [root / "runs/initial"] if manifest["simulation_available"] else [],
    )
    previous = None
    previous_time, ids = -1, set()
    for entry in history["entries"]:
        path = local(root, entry["review_path"])
        review = json.loads(path.read_text())
        require(
            sha256(path) == entry["review_file_sha256"]
            and digest(review) == entry["review_sha256"]
            and review["previous_review_sha256"] == previous
            and review["operator_session_manifest_sha256"]
            == sha256(root / "manifest.json")
            and review["physical_hardware_authorized"] is False,
            "operator session review identity",
        )
        frame = frames[review["source_frame_index"]]
        proposal = proposals[review["proposal_id"]]
        require(
            review["schema"] == "crystalab_operator_replay_review_v1"
            and review["review_id"] not in ids
            and review["acknowledged"] is True
            and bool(review["reviewer"].strip())
            and review["reviewer_kind"] == manifest["reviewer_kind"]
            and review["reviewer_kind"]
            in {"human_self_attestation", "automated_fixture_qa"}
            and (
                review["reviewer_kind"] != "automated_fixture_qa"
                or review["evidence_kind"] == "fixture"
            )
            and review["run_id"] == manifest["run_id"]
            and review["clock_id"] == manifest["clock_id"]
            and review["source_manifest_sha256"] == manifest["source_manifest_sha256"]
            and review["evidence_kind"] == manifest["source_info"]["evidence_kind"],
            "operator review source/attestation",
        )
        require(
            review["source_time_s"] == frame["time_s"] >= previous_time
            and review["proposal_sha256"] == proposal["proposal_sha256"]
            and proposal["proposal"]["proposed_at_s"] <= review["source_time_s"]
            and review["displayed_frame_sha256"]
            == manifest["files"][frame["image_path"]]
            and review["proposal_evidence_image_sha256"]
            == manifest["files"][
                "shadow/" + proposal["artifact_evidence"]["image_path"]
            ],
            "operator displayed pixels/proposal/clock",
        )
        require(
            review["action"]
            in {"acknowledge", "decline", "approve_pause", "resume_recipe"}
            and (
                manifest["simulation_available"]
                or review["action"] in {"acknowledge", "decline"}
            ),
            "operator permitted review action",
        )
        previous = digest(review)
        previous_time = review["source_time_s"]
        ids.add(review["review_id"])
        reviews.append(review)
        require(
            (entry["result_path"] is not None) == manifest["simulation_available"],
            "operator result coverage",
        )
        if entry["result_path"] is not None:
            result = local(root, entry["result_path"])
            require(
                sha256(result / "manifest.json") == entry["result_manifest_sha256"],
                "operator preserved result",
            )
            paths.append(result)
    forecast = audit_forecast_context(root / "forecast", source, reviews)
    results = [audit_control_benchmark(path) for path in paths]
    require(
        manifest["source_info"]["evidence_kind"] != "real" or not paths,
        "real forecast session cannot simulate",
    )
    expected_latest = None if not paths else str(paths[-1].relative_to(root))
    require(history["latest_run"] == expected_latest, "operator latest result identity")
    require(
        history["latest_run_sha256"]
        == (None if not paths else sha256(paths[-1] / "manifest.json")),
        "operator latest result digest",
    )
    require(
        len(list((root / "reviews").glob("*.json"))) == len(reviews),
        "operator review inventory",
    )
    return dict(
        status="pass",
        session_manifest_sha256=sha256(root / "manifest.json"),
        forecast=forecast,
        preserved_results=len(results),
        process_records=sum(r["samples"] for a in results for r in a["runs"]),
        human_reviews=sum(
            r["reviewer_kind"] == "human_self_attestation" for r in reviews
        ),
        physical_hardware_contacted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.resolve().is_relative_to(
        args.session.resolve()
    ):
        parser.error("Preserve the session; write the audit report outside it")
    result = audit_session(args.session)
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
