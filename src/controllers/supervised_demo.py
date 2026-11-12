# FILE PURPOSE: Exercise candidate rules, scripted decisions, PID and simulated actuation together.
# Only fixture/synthetic source clocks may be mapped into this declared software
# thermal timeline. Real videos remain proposal-only; no quality response is modeled.
"""Bounded cooling-recipe pauses in an auditable software control comparison."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .benchmark import run_scenario, save_json, sha256, validate_config, write_row
from .contracts import finite, identifier
from .shadow import CandidateEvidence
from .supervision import RuleConfig, RuleSupervisor

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SimulationClockMap:
    expected_source_video_sha256: str
    source_origin_s: float
    target_origin_s: float
    target_seconds_per_source_second: float
    target_clock_id: str
    provenance: str
    scope: str = "software_demonstration_only"

    def __post_init__(self):
        if (
            self.scope != "software_demonstration_only"
            or len(self.expected_source_video_sha256) != 64
            or any(
                c not in "0123456789abcdef" for c in self.expected_source_video_sha256
            )
        ):
            raise ValueError("Declare a source-bound software-only clock map")
        for key in (
            "source_origin_s",
            "target_origin_s",
            "target_seconds_per_source_second",
        ):
            finite(getattr(self, key), key, minimum=0)
        if self.target_seconds_per_source_second == 0:
            raise ValueError("Software clock scale must be positive")
        identifier(self.target_clock_id, "target clock")
        identifier(self.provenance, "software mapping provenance")

    def to_source(self, target_s):
        return (
            self.source_origin_s
            + (target_s - self.target_origin_s) / self.target_seconds_per_source_second
        )

    def to_target(self, source_s):
        return (
            self.target_origin_s
            + (source_s - self.source_origin_s) * self.target_seconds_per_source_second
        )


class PauseCoordinator:
    """Causally consume due source packets; authorize only a scripted fixture pause.

    The target provider receives only the current process observation, recipe and
    clock. It cannot observe hidden thermal truth or future candidate packets.
    """

    def __init__(self, source, policy, mapping, config, stream):
        if source.manifest["source_info"]["evidence_kind"] not in {
            "fixture",
            "synthetic",
        }:
            raise ValueError(
                "Real source evidence cannot drive the simulated control demo"
            )
        if source.parent["input"].get("sha256") != mapping.expected_source_video_sha256:
            raise ValueError("Software clock map binds a different source video")
        self.source, self.mapping, self.config, self.stream = (
            source,
            mapping,
            config,
            stream,
        )
        self.packets = [packet for packet, _ in source.packets()]
        if not self.packets or mapping.to_target(self.packets[0].received_at_s) != 0:
            raise ValueError(
                "The demonstration requires source coverage beginning at model time zero"
            )
        self.supervisor = RuleSupervisor(
            policy,
            run_id=source.manifest["run_id"],
            clock_id=source.clock_id,
            evidence_kind=source.manifest["source_info"]["evidence_kind"],
        )
        self.index = 0
        self.phase_s = self.previous_time = self.pause_until_s = 0.0
        self.executions, self.decisions, self.created = [], [], []
        self.committed_pause_s = 0.0

    def tick_rules(self, source_now):
        results = []
        while (
            self.index < len(self.packets)
            and self.packets[self.index].received_at_s <= source_now + 1e-12
        ):
            packet = self.packets[self.index]
            results.append(self.supervisor.tick(packet.received_at_s, packet))
            self.index += 1
        if (
            self.supervisor.time_s is None
            or source_now > self.supervisor.time_s + 1e-12
        ):
            results.append(self.supervisor.tick(source_now))
        for result in results:
            self.created.extend(result["created_proposals"])
            write_row(
                self.stream,
                dict(
                    event="rule_tick",
                    source=result,
                    model_time_s=self.mapping.to_target(result["time_s"]),
                ),
            )

    def advance(self, now, observation):
        interval = now - self.previous_time
        if interval < 0:
            raise ValueError("Model clock cannot reverse")
        paused = max(0.0, min(now, self.pause_until_s) - self.previous_time)
        self.phase_s += interval - paused
        self.previous_time = now
        source_now = self.mapping.to_source(now)
        self.tick_rules(source_now)
        current_time = self.supervisor.time_s
        feedback_usable = bool(
            observation.valid
            and observation.clock_id == self.mapping.target_clock_id
            and observation.received_at_s <= now
            and now - observation.measured_at_s
            <= self.config["thermal"]["pid"]["maximum_feedback_age_s"]
        )
        source_available = source_now <= self.packets[-1].observed_at_s + 1e-12
        return source_now, feedback_usable, source_available

    def __call__(self, now, nominal, observation):
        source_now, feedback_usable, source_available = self.advance(now, observation)
        current_time = self.supervisor.time_s
        for key, entry in self.supervisor.proposals.items():
            proposal = entry["proposal"]
            if (
                entry["status"] != "pending_review"
                or proposal["kind"] != "ramp_review"
                or current_time - proposal["proposed_at_s"] + 1e-12
                < self.config["scripted_decision_delay_source_s"]
                or not self.supervisor.observable["crystal_candidate"]
                or proposal["track_id"] not in self.supervisor.packet.visible_track_ids
                or not feedback_usable
                or not source_available
                or now < self.pause_until_s
            ):
                continue
            duration = min(
                self.config["pause_model_duration_s"],
                self.config["maximum_total_pause_model_s"] - self.committed_pause_s,
            )
            if duration <= 0:
                continue
            receipt = self.supervisor.decide_for_simulation(
                key,
                proposal_sha256=entry["proposal_sha256"],
                decision="authorize",
                decision_id=f"scripted_pause:{len(self.decisions) + 1:06d}",
                at_s=current_time,
            )
            self.decisions.append(receipt)
            self.committed_pause_s += duration
            self.pause_until_s = now + duration
            action = dict(
                proposal_id=key,
                proposal_sha256=entry["proposal_sha256"],
                decision_id=receipt["decision_id"],
                source_decision_time_s=current_time,
                model_decision_time_s=now,
                pause_until_model_s=self.pause_until_s,
                pause_duration_model_s=duration,
                held_recipe_phase_s=self.phase_s,
                basis="scripted_fixture_policy",
                operator_reviewed=False,
                physical_hardware_authorized=False,
                sensor_sample_id=observation.sample_id,
            )
            self.executions.append(action)
            write_row(
                self.stream,
                dict(
                    event="scripted_simulation_decision",
                    decision=receipt,
                    execution=action,
                ),
            )
        return self.result(now, nominal, source_now, feedback_usable, source_available)

    def result(self, now, nominal, source_now, feedback_usable, source_available):
        recipe = self.config["thermal"]["recipe"]
        target = float(
            np.interp(
                self.phase_s,
                [p["time_s"] for p in recipe],
                [p["temperature_K"] for p in recipe],
            )
        )
        return dict(
            target_K=target,
            nominal_target_K=nominal,
            recipe_phase_s=self.phase_s,
            source_now_s=source_now,
            source_last_frame_index=self.supervisor.packet.frame_index
            if self.supervisor.packet
            else None,
            source_last_observed_s=self.supervisor.packet.observed_at_s
            if self.supervisor.packet
            else None,
            source_frame_coverage_available=source_available,
            process_feedback_usable=feedback_usable,
            pause_until_model_s=self.pause_until_s,
            pause_active=now < self.pause_until_s,
            committed_pause_model_s=self.committed_pause_s,
            evidence_ids=[
                item
                for action in self.executions
                for item in (action["proposal_id"], action["decision_id"])
            ],
            physical_growth_clock_claimed=False,
            forecast_available=False,
        )


def validate_demo(config, source):
    if config["schema"] != "crystalab_rule_control_demo_v1":
        raise ValueError("Unsupported supervisory demo schema")
    validate_config(config["thermal"])
    if len(config["thermal"]["scenarios"]) != 1:
        raise ValueError("Declare one shared thermal scenario for the three variants")
    for key in (
        "scripted_decision_delay_source_s",
        "pause_model_duration_s",
        "maximum_total_pause_model_s",
    ):
        finite(config[key], key, minimum=0)
    if (
        config["pause_model_duration_s"] <= 0
        or config["maximum_total_pause_model_s"] < config["pause_model_duration_s"]
    ):
        raise ValueError("Pause duration must fit the declared total bound")
    mapping = SimulationClockMap(**config["clock_mapping"])
    if mapping.target_clock_id != config["thermal"]["clock_id"]:
        raise ValueError("Thermal and software mapping clocks differ")
    if source.manifest["source_info"]["evidence_kind"] not in {"fixture", "synthetic"}:
        raise ValueError(
            "Real source evidence is restricted to proposal-only shadow mode"
        )
    if source.parent["input"].get("sha256") != mapping.expected_source_video_sha256:
        raise ValueError("Software mapping source identity differs")
    if mapping.target_origin_s != 0 or mapping.source_origin_s != 0:
        raise ValueError(
            "The first demonstration uses explicit zero-origin software clocks"
        )
    return mapping


def plot_demo(config, all_rows, actions, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for method, color in [
        ("fixed_recipe", "#b47717"),
        ("pid", "#216bc4"),
        ("rule_supervised", "#198759"),
    ]:
        rows = all_rows[method]
        times = [r["time_s"] for r in rows]
        axes[0].plot(
            times,
            [r["plant_truth"]["temperature_K"] for r in rows],
            label=method,
            color=color,
        )
        axes[1].step(
            times,
            [r["plant_truth"]["applied_bath_K"] for r in rows],
            where="post",
            label=method,
            color=color,
        )
        axes[2].plot(
            times,
            [r.get("supervision", {}).get("recipe_phase_s", r["time_s"]) for r in rows],
            label=method,
            color=color,
        )
    axes[0].plot(
        times,
        [r["nominal_recipe_setpoint_K"] for r in rows],
        "k--",
        label="nominal cooling target",
    )
    axes[0].plot(
        times,
        [r["setpoint_K"] for r in rows],
        color="#198759",
        linestyle=":",
        label="supervised target",
    )
    for axis in axes:
        for action in actions:
            axis.axvspan(
                action["model_decision_time_s"],
                action["pause_until_model_s"],
                color="#198759",
                alpha=0.12,
            )
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Simulated temperature (K)")
    axes[1].set_ylabel("Applied bath request (K)")
    axes[2].set_ylabel("Recipe phase (model s)")
    axes[2].set_xlabel("Thermal fixture time (s)")
    fig.suptitle(
        "SOFTWARE / declared source-clock mapping / no modeled defect-response benefit"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def run_demo(analysis, policy_path, config_path, output):
    source = CandidateEvidence(analysis)
    policy_path, config_path, output = (
        Path(policy_path).resolve(),
        Path(config_path).resolve(),
        Path(output).resolve(),
    )
    config_hash, policy_hash = sha256(config_path), sha256(policy_path)
    config = json.loads(config_path.read_text())
    mapping = validate_demo(config, source)
    policy = RuleConfig(**json.loads(policy_path.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "supervised_config.json")
    shutil.copyfile(policy_path, output / "policy.json")
    shutil.copyfile(source.root / "manifest.json", output / "candidate_manifest.json")
    shutil.copyfile(
        source.root / "source_replay_manifest.json",
        output / "source_replay_manifest.json",
    )
    save_json(output / "config.json", config["thermal"])
    source_hashes = {}
    for path in [
        *sorted((ROOT / "src/controllers").glob("*.py")),
        ROOT / "scripts/run_supervised_control.py",
    ]:
        relative = path.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha256(target)
        if sha256(path) != source_hashes[str(relative)]:
            raise ValueError("Demo producer changed during snapshot")
    manifest = dict(
        schema="crystalab_supervised_control_demo_v1",
        status="running",
        source_analysis=str(source.root),
        candidate_manifest_sha256=source.manifest_sha256,
        supervised_config_sha256=config_hash,
        config_sha256=sha256(output / "config.json"),
        policy_sha256=policy_hash,
        source_sha256=source_hashes,
        clock_mapping=asdict(mapping),
        source_info=source.manifest["source_info"],
        scope="Software event-rule/PID integration with scripted fixture decisions; no physical clock, label review or defect-response validation",
        evidence_kind="simulation",
        physical_hardware_contacted=False,
        operator_reviews_created=0,
        adaptive_policy_implemented=False,
    )
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    reports = []
    all_rows = {}
    coordinator = None
    try:
        scenario = config["thermal"]["scenarios"][0]
        for method in ["fixed_recipe", "pid", "rule_supervised"]:
            with (output / (method + "_supervision.jsonl")).open("x") as stream:
                coordinator = (
                    PauseCoordinator(source, policy, mapping, config, stream)
                    if method == "rule_supervised"
                    else None
                )
                report, rows = run_scenario(
                    config["thermal"],
                    scenario,
                    method,
                    output / (scenario["id"] + "_" + method),
                    target_provider=coordinator,
                )
            nominal_errors = [
                r["plant_truth"]["temperature_K"]
                - r.get("nominal_recipe_setpoint_K", r["setpoint_K"])
                for r in rows
            ]
            report["nominal_recipe_sample_mae_K"] = sum(
                abs(e) for e in nominal_errors
            ) / len(rows)
            report["final_recipe_phase_s"] = (
                rows[-1]
                .get("supervision", {})
                .get("recipe_phase_s", rows[-1]["time_s"])
            )
            report["recipe_phase_lag_s"] = (
                rows[-1]["time_s"] - report["final_recipe_phase_s"]
            )
            reports.append(report)
            all_rows[method] = rows
        if coordinator.index != len(coordinator.packets):
            raise ValueError(
                "Thermal window ends before all declared source evidence is consumed"
            )
        plot_demo(config, all_rows, coordinator.executions, output)
        source.verify()
        if sha256(config_path) != config_hash or sha256(policy_path) != policy_hash:
            raise ValueError("Demo inputs changed")
        manifest.update(
            status="complete",
            runs=reports,
            scripted_executions=coordinator.executions,
            scripted_decisions=coordinator.decisions,
            final_proposals=list(coordinator.supervisor.proposals.values()),
            elapsed_seconds=time.perf_counter() - started,
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
            completed_runs=reports,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise
