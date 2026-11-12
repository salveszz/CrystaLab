# FILE PURPOSE: Run frozen-model temporal event/area diagnostics on declared development scenes.
# No model is fitted, no reserved scene is replayed and no physical command or
# human review is produced. Forecast labels stay separate from causal inputs.
"""Reproducible longer-clip diagnostic using existing perception and forecast baselines."""

from collections import Counter
import json
from pathlib import Path
import shutil

from models.area_forecast_run import (
    evaluate_samples,
    load_forecast_model,
    plot_scores,
    snapshot,
)
from models.segmentation import load_run_metadata
from rendering.optical import audit_optical
from vision.area_forecast_data import audit_cohort, json_rows, prepare_cohort
from vision.candidate_replay import analyze_candidates
from vision.candidates import CandidateConfig, KINDS
from vision.event_evaluation import build_optical_annotations, score_candidate_events
from vision.optical_sequence import validate_sequence_codec
from vision.training_data import save_json
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def evaluate_frozen_cohort(cohort, model_path, output):
    """Score existing parameters; source split names do not imply this model fitted them."""
    cohort, model_path, output = [
        Path(p).resolve() for p in (cohort, model_path, output)
    ]
    audit = audit_cohort(cohort)
    cohort_hash = file_sha256(cohort / "manifest.json")
    model = load_forecast_model(model_path)
    model_hash = file_sha256(model_path)
    if json.loads((cohort / "forecast_config.json").read_text()) != model["config"]:
        raise ValueError("Frozen forecast and diagnostic horizon/config differ")
    samples = json_rows(cohort / "samples.jsonl")
    if any(s["split"] not in {"train", "validation"} for s in samples):
        raise ValueError("Frozen development evaluation excludes reserved test")
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(
        output,
        [
            Path(__file__),
            ROOT / "src/models/area_forecast_run.py",
            ROOT / "src/models/area_forecast.py",
            ROOT / "src/vision/area_forecast_data.py",
        ],
    )
    report = dict(
        schema="crystalab_frozen_area_evaluation_v1",
        status="running",
        model_sha256=model_hash,
        cohort_manifest_sha256=cohort_hash,
        cohort_audit=audit,
        source_sha256=source_hashes,
        calibration=model["calibration"],
        calibration_family_overlap=sorted(
            {s["family_id"] for s in samples}
            & set(model["calibration"]["training_family_ids"])
        ),
        fitted_new_parameters=False,
        final_test_scored=False,
        can_authorize_actuation=False,
        scope="Frozen source-clock projected-area diagnostic; report overlapping calibration families explicitly; no physical or adaptive action claim",
    )
    save_json(output / "manifest.json", report)
    try:
        shutil.copyfile(model_path, output / "frozen_model.json")
        shutil.copyfile(cohort / "manifest.json", output / "cohort_manifest.json")
        scores, metrics = evaluate_samples(samples, model["calibration"])
        (output / "scores.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in scores)
        )
        save_json(output / "metrics.json", metrics)
        plot_scores(scores, output)
        if (
            audit_cohort(cohort)["manifest_sha256"] != cohort_hash
            or file_sha256(model_path) != model_hash
        ):
            raise ValueError("Frozen evaluation inputs changed")
        report.update(
            status="complete",
            metrics=metrics,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p != output / "manifest.json"
            },
        )
        save_json(output / "manifest.json", report)
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", report)
        raise


def summarize_events(cases):
    result = {}
    for kind in KINDS:
        tp = sum(c["report"]["per_kind"][kind]["tp"] for c in cases)
        fp = sum(c["report"]["per_kind"][kind]["fp"] for c in cases)
        fn = sum(c["report"]["per_kind"][kind]["fn"] for c in cases)
        eligible = sum(
            c["report"]["observable_deadline_followup"][kind]["eligible"] for c in cases
        )
        matched = sum(
            c["report"]["observable_deadline_followup"][kind]["matched"] for c in cases
        )
        result[kind] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            observable_deadline_eligible=eligible,
            observable_deadline_matched=matched,
            observable_deadline_recall=matched / eligible if eligible else None,
            sum_of_clip_negative_source_seconds=sum(
                c["report"]["negative_coverage"][kind][
                    "usable_annotated_negative_source_seconds"
                ]
                for c in cases
            ),
            false_events_in_negative_intervals=sum(
                c["report"]["negative_coverage"][kind][
                    "false_events_in_negative_intervals"
                ]
                for c in cases
            ),
            physical_negative_seconds=None,
            false_alerts_per_physical_hour=None,
        )
    return result


def split_event_summaries(cases):
    """Keep training clips out of validation metrics even when both are replayed."""
    if any(case.get("split") not in {"train", "validation"} for case in cases):
        raise ValueError(
            "Event summaries require explicit development split membership"
        )
    return {
        split: dict(
            clips=sum(c["split"] == split for c in cases),
            per_kind=summarize_events([c for c in cases if c["split"] == split]),
        )
        for split in ("train", "validation")
    }


def run_temporal_diagnostic(source, experiment_path, output):
    source, experiment_path, output = [
        Path(p).resolve() for p in (source, experiment_path, output)
    ]
    experiment = json.loads(experiment_path.read_text())
    experiment_hash = file_sha256(experiment_path)
    sequence_codec = validate_sequence_codec(experiment.get("sequence_codec", "MJPG"))
    if (
        experiment.get("schema") != "crystalab_optical_temporal_diagnostic_v1"
        or experiment["display_fps"] != 4.0
    ):
        raise ValueError("Declare the supported frozen four-FPS temporal diagnostic")
    protocol_path = ROOT / experiment["protocol_path"]
    if file_sha256(protocol_path) != experiment["protocol_sha256"]:
        raise ValueError("Frozen temporal protocol changed")
    source_audit = audit_optical(source)
    original = json.loads((source / "manifest.json").read_text())
    if original["protocol"] != json.loads(protocol_path.read_text()):
        raise ValueError("Rendered protocol differs from frozen diagnostic")
    scenes = sorted(s["id"] for s in original["scenes"] if s["split"] != "test")
    reserved = sorted(s["id"] for s in original["scenes"] if s["split"] == "test")
    scene_splits = {s["id"]: s["split"] for s in original["scenes"]}
    if (
        scenes != experiment["development_scenes"]
        or reserved != experiment["reserved_test_scenes"]
    ):
        raise ValueError("Development/reserved scene membership changed")
    for key in ("observer", "candidates", "area_forecast"):
        if file_sha256(ROOT / experiment[key]["path"]) != experiment[key]["sha256"]:
            raise ValueError("Frozen diagnostic configuration changed")
    training, weights = load_run_metadata(ROOT / experiment["model_run"])
    if (
        file_sha256(ROOT / experiment["model_run"] / "manifest.json")
        != experiment["model_run_manifest_sha256"]
        or file_sha256(weights) != experiment["weights_sha256"]
    ):
        raise ValueError("Frozen perception model changed")
    model_path = ROOT / experiment["area_forecast"]["existing_model"]
    model = load_forecast_model(model_path)
    if file_sha256(model_path) != experiment["area_forecast"]["model_file_sha256"]:
        raise ValueError("Frozen forecast parameters changed")
    if set(s["family_id"] for s in original["scenes"]) & set(
        model["calibration"]["training_family_ids"]
    ):
        raise ValueError(
            "Temporal diagnostic families overlap forecast calibration families"
        )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(experiment_path, output / "experiment.json")
    paths = [
        Path(__file__),
        ROOT / "scripts/run_temporal_diagnostic.py",
        ROOT / "src/vision/event_evaluation.py",
        ROOT / "src/vision/candidate_replay.py",
        ROOT / "src/vision/candidates.py",
        ROOT / "src/models/area_forecast_run.py",
        ROOT / "src/vision/optical_sequence.py",
        ROOT / "src/vision/area_forecast_data.py",
    ]
    source_hashes = snapshot(output, paths)
    manifest = dict(
        schema="crystalab_temporal_diagnostic_run_v1",
        status="running",
        experiment_sha256=experiment_hash,
        source_manifest_sha256=file_sha256(source / "manifest.json"),
        source_audit=source_audit,
        source_sha256=source_hashes,
        perception_weights_sha256=training["best_weights_sha256"],
        frozen_forecast_sha256=file_sha256(model_path),
        final_test_scored=False,
        fitted_new_parameters=False,
        numerical_actuator_commands=0,
        operator_reviews_created=0,
        physical_time_mapping=None,
        sequence_codec=sequence_codec,
    )
    save_json(output / "manifest.json", manifest)
    try:
        cohort = output / "cohort"
        prepare_cohort(
            source,
            ROOT / experiment["model_run"],
            ROOT / experiment["observer"]["path"],
            ROOT / experiment["area_forecast"]["path"],
            cohort,
            sequence_codec=sequence_codec,
        )
        forecast = evaluate_frozen_cohort(
            cohort, model_path, output / "frozen_forecast"
        )
        policy = CandidateConfig(
            **json.loads((ROOT / experiment["candidates"]["path"]).read_text())
        )
        cases = []
        for scene in scenes:
            replay = cohort / "observations" / scene
            analysis = output / "candidates" / scene
            analyze_candidates(replay, policy, analysis)
            annotations = output / "annotations" / scene
            build_optical_annotations(replay, annotations)
            report = score_candidate_events(
                analysis,
                annotations / "annotations.json",
                output / "event_scores" / scene,
                match_iou=experiment["source_proxy_event_matching"]["minimum_iou"],
                maximum_delay_s=experiment["source_proxy_event_matching"][
                    "maximum_delay_source_s"
                ],
            )
            cases.append(dict(scene_id=scene, split=scene_splits[scene], report=report))
        save_json(
            output / "event_summary.json",
            dict(
                per_kind=summarize_events(cases),
                by_split=split_event_summaries(cases),
                origin_confusion=dict(
                    sum(
                        (Counter(c["report"]["origin_confusion"]) for c in cases),
                        Counter(),
                    )
                ),
                cases=cases,
            ),
        )
        if (
            file_sha256(experiment_path) != experiment_hash
            or file_sha256(source / "manifest.json")
            != manifest["source_manifest_sha256"]
        ):
            raise ValueError("Diagnostic inputs changed during execution")
        for name, digest in source_hashes.items():
            if file_sha256(ROOT / name) != digest:
                raise ValueError("Diagnostic producer changed during execution")
        manifest.update(
            status="complete",
            development_scenes=scenes,
            reserved_test_scenes=reserved,
            forecast_metrics=forecast["metrics"],
            event_metrics=summarize_events(cases),
            event_metrics_by_split=split_event_summaries(cases),
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p != output / "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise
