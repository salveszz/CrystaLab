# FILE PURPOSE: Replay explicit operator decisions through bounded PID and simulated actuation.
# Reviews describe a counterfactual replay, not historical/live hardware approval.
"""Source-bound recipe pause/resume with immutable review provenance."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .benchmark import run_scenario, save_json, sha256, write_row
from .contracts import finite
from .shadow import CandidateEvidence
from .supervised_demo import PauseCoordinator, validate_demo
from .supervision import RuleConfig, value_sha256

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = {"approve_pause", "decline", "acknowledge", "resume_recipe"}


def validate_reviews(reviews, source, *, mapping=None, config=None):
    if not isinstance(reviews, list) or len(reviews) > 128:
        raise ValueError("Provide at most 128 explicit replay reviews")
    packets = {p.frame_index: p for p, _ in source.packets()}
    seen, previous_hash, previous_time = set(), None, -1
    for review in reviews:
        if (
            review.get("schema") != "crystalab_operator_replay_review_v1"
            or review.get("run_id") != source.manifest["run_id"]
            or review.get("clock_id") != source.clock_id
            or review.get("source_manifest_sha256") != source.manifest_sha256
            or review.get("evidence_kind")
            != source.manifest["source_info"]["evidence_kind"]
            or review.get("physical_hardware_authorized") is not False
            or review.get("acknowledged") is not True
            or review.get("action") not in ACTIONS
            or review.get("reviewer_kind")
            not in {"human_self_attestation", "automated_fixture_qa"}
            or not isinstance(review.get("reviewer"), str)
            or not 1 <= len(review["reviewer"].strip()) <= 120
            or not isinstance(review.get("comment"), str)
            or len(review["comment"]) > 2000
            or not isinstance(review.get("recorded_at_utc"), str)
            or not isinstance(review.get("review_id"), str)
            or type(review.get("source_frame_index")) is not int
            or not review["review_id"]
            or review["review_id"] in seen
            or review.get("previous_review_sha256") != previous_hash
        ):
            raise ValueError("Invalid or reordered source-bound operator review")
        for key in (
            "operator_session_manifest_sha256",
            "displayed_frame_sha256",
            "proposal_evidence_image_sha256",
        ):
            value = review.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(
                    "Replay reviews must identify the exact displayed pixels and session"
                )
        if (
            review["reviewer_kind"] == "automated_fixture_qa"
            and review["evidence_kind"] != "fixture"
        ):
            raise ValueError("Automated review QA is restricted to fixture evidence")
        stamp = finite(review["source_time_s"], "review source time", minimum=0)
        packet = packets.get(review["source_frame_index"])
        if packet is None or packet.observed_at_s != stamp or stamp < previous_time:
            raise ValueError(
                "Review frame/clock must match the observed replay in order"
            )
        if mapping is None and review["action"] in {"approve_pause", "resume_recipe"}:
            raise ValueError("Shadow review cannot dispatch a numerical action")
        if mapping is not None:
            model_time = mapping.to_target(stamp)
            step = config["thermal"]["step_s"]
            if abs(model_time / step - round(model_time / step)) > 1e-9:
                raise ValueError(
                    "Review time must align with the declared simulation step"
                )
        seen.add(review["review_id"])
        previous_hash, previous_time = value_sha256(review), stamp


class OperatorPauseCoordinator(PauseCoordinator):
    def __init__(self, source, policy, mapping, config, stream, reviews):
        super().__init__(source, policy, mapping, config, stream)
        self.reviews, self.review_index, self.resumptions = reviews, 0, []

    def __call__(self, now, nominal, observation):
        source_now, feedback, covered = self.advance(now, observation)
        while (
            self.review_index < len(self.reviews)
            and self.reviews[self.review_index]["source_time_s"] <= source_now + 1e-12
        ):
            review = self.reviews[self.review_index]
            if abs(review["source_time_s"] - source_now) > 1e-9:
                raise ValueError("A replay decision cannot be applied late")
            action = review["action"]
            entry = self.supervisor.proposals.get(review["proposal_id"])
            if entry is None or entry["proposal_sha256"] != review["proposal_sha256"]:
                raise ValueError("Review must bind an already observed exact proposal")
            if action in {"approve_pause", "resume_recipe"} and (
                not feedback or not covered
            ):
                raise ValueError(
                    "Simulated action requires source coverage and fresh process feedback"
                )
            if action == "approve_pause":
                duration = finite(
                    review["pause_duration_model_s"], "pause duration", minimum=0
                )
                if (
                    not 0 < duration <= self.config["pause_model_duration_s"]
                    or self.committed_pause_s + duration
                    > self.config["maximum_total_pause_model_s"]
                    or now < self.pause_until_s
                ):
                    raise ValueError(
                        "Pause overlaps or exceeds declared simulation bounds"
                    )
                receipt = self.supervisor.decide_from_operator_replay(review)
                self.decisions.append(receipt)
                self.committed_pause_s += duration
                self.pause_until_s = now + duration
                self.executions.append(
                    dict(
                        proposal_id=review["proposal_id"],
                        proposal_sha256=review["proposal_sha256"],
                        decision_id=review["review_id"],
                        review_receipt_sha256=value_sha256(review),
                        source_decision_time_s=source_now,
                        model_decision_time_s=now,
                        requested_pause_duration_model_s=duration,
                        requested_pause_until_model_s=self.pause_until_s,
                        pause_until_model_s=self.pause_until_s,
                        pause_duration_model_s=duration,
                        held_recipe_phase_s=self.phase_s,
                        sensor_sample_id=observation.sample_id,
                        basis=receipt["basis"],
                        operator_reviewed=receipt["operator_reviewed"],
                        physical_hardware_authorized=False,
                    )
                )
            elif action == "resume_recipe":
                active = next(
                    (
                        a
                        for a in self.executions
                        if a["decision_id"] == review.get("pause_review_id")
                    ),
                    None,
                )
                if (
                    active is None
                    or not active["model_decision_time_s"]
                    < now
                    < active["pause_until_model_s"]
                ):
                    raise ValueError(
                        "Manual resume requires a currently active approved pause"
                    )
                if active["proposal_id"] != review["proposal_id"]:
                    raise ValueError("Resume must reference the active pause proposal")
                active["pause_until_model_s"] = now
                active["pause_duration_model_s"] = now - active["model_decision_time_s"]
                self.pause_until_s = now
                self.resumptions.append(
                    dict(
                        review_id=review["review_id"],
                        pause_review_id=active["decision_id"],
                        model_time_s=now,
                        source_time_s=source_now,
                        sensor_sample_id=observation.sample_id,
                        review_receipt_sha256=value_sha256(review),
                        operator_reviewed=review["reviewer_kind"]
                        == "human_self_attestation",
                        physical_hardware_authorized=False,
                    )
                )
            elif action == "decline" and entry["status"] == "pending_review":
                self.decisions.append(
                    self.supervisor.decide_from_operator_replay(review)
                )
            write_row(
                self.stream,
                dict(event="operator_replay_review", review=review, model_time_s=now),
            )
            self.review_index += 1
        result = self.result(now, nominal, source_now, feedback, covered)
        result["evidence_ids"].extend(r["review_id"] for r in self.resumptions)
        return result


def run_operator_demo(
    analysis, policy_path, config_path, reviews, output, *, forecast_context=None
):
    source = CandidateEvidence(analysis)
    policy_path, config_path, output = [
        Path(p).resolve() for p in (policy_path, config_path, output)
    ]
    config, policy = (
        json.loads(config_path.read_text()),
        RuleConfig(**json.loads(policy_path.read_text())),
    )
    mapping = validate_demo(config, source)
    validate_reviews(reviews, source, mapping=mapping, config=config)
    if forecast_context is not None:
        from .forecast_context import ForecastContext

        context = ForecastContext(forecast_context, source)
        context.validate_reviews(reviews)
    elif any(r.get("forecast_context") is not None for r in reviews):
        raise ValueError("Forecast-bound reviews require their preserved context")
    output.mkdir(parents=True, exist_ok=False)
    if forecast_context is not None:
        shutil.copytree(context.root, output / "forecast_context")
    for path, name in (
        (config_path, "supervised_config.json"),
        (policy_path, "policy.json"),
        (source.root / "manifest.json", "candidate_manifest.json"),
        (source.root / "source_replay_manifest.json", "source_replay_manifest.json"),
    ):
        shutil.copyfile(path, output / name)
    save_json(output / "config.json", config["thermal"])
    save_json(output / "reviews.json", reviews)
    sources = {}
    for path in [*sorted((ROOT / "src/controllers").glob("*.py"))]:
        relative = str(path.relative_to(ROOT))
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[relative] = sha256(target)
    manifest = dict(
        schema="crystalab_operator_control_demo_v1",
        status="running",
        source_analysis=str(source.root),
        candidate_manifest_sha256=source.manifest_sha256,
        supervised_config_sha256=sha256(config_path),
        config_sha256=sha256(output / "config.json"),
        policy_sha256=sha256(policy_path),
        reviews_sha256=sha256(output / "reviews.json"),
        source_sha256=sources,
        clock_mapping=asdict(mapping),
        source_info=source.manifest["source_info"],
        evidence_kind="simulation",
        physical_hardware_contacted=False,
        operator_reviews_created=sum(
            r["reviewer_kind"] == "human_self_attestation" for r in reviews
        ),
        adaptive_policy_implemented=False,
        forecast_context_manifest_sha256=None
        if forecast_context is None
        else context.manifest_hash,
        scope="Counterfactual operator replay and simulated thermal response; named self-attestation is not authenticated identity, historical/live action, physical calibration or defect prevention",
    )
    save_json(output / "manifest.json", manifest)
    reports = []
    try:
        for method in ("fixed_recipe", "pid", "rule_supervised"):
            with (output / f"{method}_supervision.jsonl").open("x") as stream:
                coordinator = (
                    OperatorPauseCoordinator(
                        source, policy, mapping, config, stream, reviews
                    )
                    if method == "rule_supervised"
                    else None
                )
                report, records = run_scenario(
                    config["thermal"],
                    config["thermal"]["scenarios"][0],
                    method,
                    output / f"operator_{method}",
                    target_provider=coordinator,
                )
            report["nominal_recipe_sample_mae_K"] = sum(
                abs(
                    r["plant_truth"]["temperature_K"]
                    - r.get("nominal_recipe_setpoint_K", r["setpoint_K"])
                )
                for r in records
            ) / len(records)
            report["final_recipe_phase_s"] = (
                records[-1]
                .get("supervision", {})
                .get("recipe_phase_s", records[-1]["time_s"])
            )
            report["recipe_phase_lag_s"] = (
                records[-1]["time_s"] - report["final_recipe_phase_s"]
            )
            reports.append(report)
        if coordinator.review_index != len(reviews) or coordinator.index != len(
            coordinator.packets
        ):
            raise ValueError("Simulation does not cover all source packets and reviews")
        source.verify()
        if (
            sha256(config_path) != manifest["supervised_config_sha256"]
            or sha256(policy_path) != manifest["policy_sha256"]
        ):
            raise ValueError("Operator simulation input changed")
        for path, digest in sources.items():
            if sha256(ROOT / path) != digest:
                raise ValueError("Operator simulation producer changed")
        manifest.update(
            status="complete",
            runs=reports,
            executions=coordinator.executions,
            decisions=coordinator.decisions,
            manual_resumptions=coordinator.resumptions,
            final_proposals=list(coordinator.supervisor.proposals.values()),
            files={
                str(p.relative_to(output)): sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p != output / "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            completed_runs=reports,
        )
        save_json(output / "manifest.json", manifest)
        raise
