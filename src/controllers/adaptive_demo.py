# FILE PURPOSE: Connect a changing observed-area model to bounded candidate/PID simulation decisions.
# A fixed video is not an action-responsive growth plant; physical action benefit remains open.
"""Four-way software comparison with matured-only forecast-informed pause durations."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from controllers.benchmark import run_scenario, save_json, sha256, write_row
from controllers.shadow import CandidateEvidence
from controllers.supervised_demo import PauseCoordinator, validate_demo
from controllers.supervision import RuleConfig, value_sha256
from models.adaptive_area import AdaptiveAreaModel
from models.adaptive_area_run import load_reference
from vision.area_forecast_data import ReplayAreas

ROOT = Path(__file__).resolve().parents[2]


class AdaptivePauseCoordinator(PauseCoordinator):
    def __init__(
        self,
        source,
        policy,
        mapping,
        config,
        stream,
        reference,
        adaptive_config,
        state_stream,
    ):
        super().__init__(source, policy, mapping, config, stream)
        self.reader = ReplayAreas(source.manifest["source_replay"])
        if (
            self.reader.manifest_hash
            != source.manifest["source_replay_manifest_sha256"]
            or self.reader.clock_id != source.clock_id
        ):
            raise ValueError(
                "Adaptive observations and candidate source must be identical"
            )
        self.areas = list(self.reader.estimates())
        self.area_index = 0
        self.model = AdaptiveAreaModel(adaptive_config, reference)
        self.state_stream = state_stream

    def __call__(self, now, nominal, observation):
        source_now, feedback, covered = self.advance(now, observation)
        while (
            self.area_index < len(self.areas)
            and self.areas[self.area_index][0].received_at_s <= source_now + 1e-12
        ):
            estimate, _ = self.areas[self.area_index]
            state = self.model.update(
                estimate, now_s=max(source_now, estimate.received_at_s)
            )
            write_row(self.state_stream, state)
            self.area_index += 1
        recommendation = self.model.recommend(
            source_now,
            base_pause_model_s=self.config["pause_model_duration_s"],
            model_step_s=self.config["thermal"]["step_s"],
        )
        assessment = dict(
            event="adaptive_assessment",
            model_time_s=now,
            source_time_s=source_now,
            recommendation=recommendation,
        )
        write_row(self.stream, assessment)
        for key, entry in self.supervisor.proposals.items():
            p = entry["proposal"]
            if (
                entry["status"] != "pending_review"
                or p["kind"] != "ramp_review"
                or self.supervisor.time_s - p["proposed_at_s"] + 1e-12
                < self.config["scripted_decision_delay_source_s"]
                or not self.supervisor.observable["crystal_candidate"]
                or p["track_id"] not in self.supervisor.packet.visible_track_ids
                or not feedback
                or not covered
                or now < self.pause_until_s
                or not recommendation["eligible"]
            ):
                continue
            duration = min(
                recommendation["pause_model_s"],
                self.config["maximum_total_pause_model_s"] - self.committed_pause_s,
            )
            if duration < self.model.config.minimum_pause_model_s:
                continue
            receipt = self.supervisor.decide_for_simulation(
                key,
                proposal_sha256=entry["proposal_sha256"],
                decision="authorize",
                decision_id=f"adaptive_fixture_pause:{len(self.decisions) + 1:06d}",
                at_s=self.supervisor.time_s,
            )
            receipt.update(
                adaptive_state_sha256=recommendation["state_sha256"],
                forecast_informed=True,
            )
            self.decisions.append(receipt)
            self.committed_pause_s += duration
            self.pause_until_s = now + duration
            action = dict(
                proposal_id=key,
                proposal_sha256=entry["proposal_sha256"],
                decision_id=receipt["decision_id"],
                source_decision_time_s=self.supervisor.time_s,
                model_decision_time_s=now,
                pause_until_model_s=self.pause_until_s,
                pause_duration_model_s=duration,
                held_recipe_phase_s=self.phase_s,
                basis="scripted_fixture_policy",
                operator_reviewed=False,
                physical_hardware_authorized=False,
                sensor_sample_id=observation.sample_id,
                adaptive_state_sha256=recommendation["state_sha256"],
                assessment_sha256=value_sha256(assessment),
                forecast_informed=True,
            )
            self.executions.append(action)
            write_row(
                self.stream,
                dict(
                    event="adaptive_simulation_decision",
                    decision=receipt,
                    execution=action,
                ),
            )
        result = self.result(now, nominal, source_now, feedback, covered)
        result.update(
            adaptive_state_sha256=recommendation["state_sha256"],
            adaptive_recommendation=recommendation,
            forecast_available=recommendation["forecast_available"],
            action_response_model=False,
        )
        return result


def run_adaptive_demo(analysis, policy_path, config_path, reference_root, output):
    source = CandidateEvidence(analysis)
    policy_path, config_path, reference_root, output = [
        Path(p).resolve() for p in (policy_path, config_path, reference_root, output)
    ]
    config = json.loads(config_path.read_text())
    mapping = validate_demo(config, source)
    policy = RuleConfig(**json.loads(policy_path.read_text()))
    reference, adaptive_config = load_reference(reference_root)
    if adaptive_config.maximum_pause_model_s > config["pause_model_duration_s"]:
        raise ValueError("Adaptive pause maximum exceeds declared thermal pause bound")
    if source.manifest["source_info"].get("split") == "validation" and source.manifest[
        "source_info"
    ]["family_id"] in {c["family_id"] for c in reference["training_families"]}:
        raise ValueError("Adaptive reference overlaps the validation source")
    # Bind observation lineage before creating outputs, even if no candidate triggers a pause.
    reader = ReplayAreas(source.manifest["source_replay"])
    if reader.manifest_hash != source.manifest["source_replay_manifest_sha256"]:
        raise ValueError("Adaptive source observation changed")
    output.mkdir(parents=True, exist_ok=False)
    for path, name in (
        (config_path, "supervised_config.json"),
        (policy_path, "policy.json"),
        (source.root / "manifest.json", "candidate_manifest.json"),
        (source.root / "source_replay_manifest.json", "source_replay_manifest.json"),
    ):
        shutil.copyfile(path, output / name)
    shutil.copytree(reference_root, output / "adaptive_reference")
    save_json(output / "config.json", config["thermal"])
    source_hashes = {}
    for path in [
        *sorted((ROOT / "src/controllers").glob("*.py")),
        ROOT / "src/models/adaptive_area.py",
        ROOT / "src/models/adaptive_area_run.py",
        ROOT / "src/models/area_forecast.py",
        ROOT / "src/vision/area_forecast_data.py",
    ]:
        relative = str(path.relative_to(ROOT))
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[relative] = sha256(target)
    manifest = dict(
        schema="crystalab_adaptive_control_demo_v1",
        status="running",
        source_analysis=str(source.root),
        candidate_manifest_sha256=source.manifest_sha256,
        supervised_config_sha256=sha256(config_path),
        config_sha256=sha256(output / "config.json"),
        policy_sha256=sha256(policy_path),
        source_sha256=source_hashes,
        reference_manifest_sha256=sha256(reference_root / "manifest.json"),
        clock_mapping=asdict(mapping),
        source_info=source.manifest["source_info"],
        scope="Four-way software comparison: native observed-area adaptation and forecast-scaled candidate pause under explicit fixture authorization. Fixed video has no growth/quality response to thermal commands.",
        evidence_kind="simulation",
        physical_hardware_contacted=False,
        operator_reviews_created=0,
        adaptive_policy_implemented=True,
        video_action_responsive=False,
        physical_growth_clock_claimed=False,
    )
    save_json(output / "manifest.json", manifest)
    reports = []
    coordinators = {}
    try:
        for method in ("fixed_recipe", "pid", "rule_supervised", "adaptive_supervised"):
            with (
                (output / f"{method}_supervision.jsonl").open("x") as stream,
                (output / f"{method}_area_states.jsonl").open("x") as state_stream,
            ):
                coordinator = (
                    AdaptivePauseCoordinator(
                        source,
                        policy,
                        mapping,
                        config,
                        stream,
                        reference,
                        adaptive_config,
                        state_stream,
                    )
                    if method == "adaptive_supervised"
                    else PauseCoordinator(source, policy, mapping, config, stream)
                    if method == "rule_supervised"
                    else None
                )
                report, records = run_scenario(
                    config["thermal"],
                    config["thermal"]["scenarios"][0],
                    method,
                    output / f"run_{method}",
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
            if coordinator is not None:
                if coordinator.index != len(coordinator.packets):
                    raise ValueError("Simulation does not cover source packets")
                coordinators[method] = coordinator
        adaptive = coordinators["adaptive_supervised"]
        rules = coordinators["rule_supervised"]
        if adaptive.area_index != len(adaptive.areas):
            raise ValueError("Simulation does not cover native area observations")
        source.verify()
        reader.verify()
        if (
            sha256(reference_root / "manifest.json")
            != manifest["reference_manifest_sha256"]
            or sha256(config_path) != manifest["supervised_config_sha256"]
            or sha256(policy_path) != manifest["policy_sha256"]
        ):
            raise ValueError("Adaptive control input changed")
        for relative, digest in source_hashes.items():
            if sha256(ROOT / relative) != digest:
                raise ValueError("Adaptive control producer changed")
        manifest.update(
            status="complete",
            runs=reports,
            scripted_executions=rules.executions,
            scripted_decisions=rules.decisions,
            final_proposals=list(rules.supervisor.proposals.values()),
            adaptive_executions=adaptive.executions,
            adaptive_decisions=adaptive.decisions,
            adaptive_final_proposals=list(adaptive.supervisor.proposals.values()),
            final_adaptive_parameters=adaptive.model.latest["parameter_after"],
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
