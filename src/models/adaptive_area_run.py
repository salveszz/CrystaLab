# FILE PURPOSE: Calibrate training-only normalization and evaluate causal online area forecasts.
# State updates consume native observed masks only; independent proxy labels enter scoring later.
"""Audited adaptive projected-area reference, replay and grouped development evaluation."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from models.adaptive_area import AdaptiveAreaConfig, AdaptiveAreaModel
from models.area_forecast_run import (
    evaluate_samples,
    load_forecast_model,
    metric_rows,
    snapshot,
)
from vision.area_forecast_data import ReplayAreas, audit_cohort, json_rows
from vision.training_data import save_json, safe_path
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def sources(output):
    return snapshot(
        output,
        [
            Path(__file__),
            ROOT / "src/models/adaptive_area.py",
            ROOT / "src/models/area_forecast.py",
            ROOT / "src/vision/area_forecast_data.py",
        ],
    )


def fit_reference(cohort, config_path, output):
    cohort, config_path, output = [
        Path(p).resolve() for p in (cohort, config_path, output)
    ]
    audit = audit_cohort(cohort)
    config = AdaptiveAreaConfig(**json.loads(config_path.read_text()))
    cases = [
        c
        for c in json.loads((cohort / "manifest.json").read_text())["cases"]
        if c["split"] == "train"
    ]
    if not cases or len({c["family_id"] for c in cases}) != len(cases):
        raise ValueError("Training reference needs distinct whole training families")
    gains = []
    errors = []
    identities = []
    for case in cases:
        reader = ReplayAreas(cohort / case["replay_path"])
        if reader.manifest["source_info"]["split"] != "train":
            raise ValueError("Reference cannot use validation or test observations")
        stream = [e for e, _ in reader.estimates()]
        local_gains = []
        local_errors = []
        for previous, current in zip(stream, stream[1:]):
            if (
                all(e.valid and not e.requires_inspection for e in (previous, current))
                and abs(
                    current.observed_at_s
                    - previous.observed_at_s
                    - config.horizon_source_s
                )
                < 1e-9
            ):
                gain = (current.area_px2 - previous.area_px2) / previous.area_px2
                if gain > 0:
                    gains.append(gain)
                    local_gains.append(gain)
        for first, cut, target in zip(stream, stream[1:], stream[2:]):
            if all(
                e.valid and not e.requires_inspection for e in (first, cut, target)
            ) and all(
                abs(b.observed_at_s - a.observed_at_s - config.horizon_source_s) < 1e-9
                for a, b in ((first, cut), (cut, target))
            ):
                error = (
                    abs(
                        target.area_px2
                        - (cut.area_px2 + (cut.area_px2 - first.area_px2))
                    )
                    / cut.area_px2
                )
                errors.append(error)
                local_errors.append(error)
        reader.verify()
        identities.append(
            dict(
                family_id=case["family_id"],
                manifest_sha256=reader.manifest_hash,
                source_path=str(reader.root),
                frames=len(stream),
                positive_transitions=len(local_gains),
                error_transitions=len(local_errors),
            )
        )
    if not gains or not errors:
        raise ValueError(
            "Training reference requires positive growth and matured measured transitions"
        )
    reference_value = max(
        config.minimum_reference_relative_gain,
        float(np.quantile(gains, config.reference_positive_gain_quantile)),
    )
    reference = dict(
        schema="crystalab_adaptive_area_reference_v1",
        positive_relative_gain_reference=reference_value,
        initial_error_ratio=float(np.median(errors)) / reference_value,
        training_families=identities,
        gain_samples=len(gains),
        error_samples=len(errors),
        configuration=asdict(config),
        normalization_semantics="Training-only positive adjacent observed mask growth quantile; error initialization is a measured recent-rate residual median. No proxy truth, validation or physical operating bound.",
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "reference.json", reference)
    save_json(output / "policy.json", asdict(config))
    source_hashes = sources(output)
    save_json(
        output / "manifest.json",
        dict(
            schema="crystalab_adaptive_area_reference_run_v1",
            status="complete",
            cohort_manifest_sha256=file_sha256(cohort / "manifest.json"),
            cohort_audit=audit,
            policy_sha256=file_sha256(config_path),
            source_sha256=source_hashes,
            validation_used_for_statistics=False,
            proxy_truth_used_for_statistics=False,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    return reference


def load_reference(root):
    root = Path(root).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema") != "crystalab_adaptive_area_reference_run_v1"
        or m["status"] != "complete"
    ):
        raise ValueError("A complete adaptive training reference is required")
    for p, digest in m["files"].items():
        if file_sha256(safe_path(root, p)) != digest:
            raise ValueError("Adaptive reference artifact changed")
    reference = json.loads((root / "reference.json").read_text())
    config = AdaptiveAreaConfig(**reference["configuration"])
    if json.loads((root / "policy.json").read_text()) != asdict(config):
        raise ValueError("Adaptive reference/policy mismatch")
    return reference, config


def replay_adaptive(replay, reference_root, output):
    reader = ReplayAreas(replay)
    reference_root, output = [Path(p).resolve() for p in (reference_root, output)]
    reference, config = load_reference(reference_root)
    reference_hash = file_sha256(reference_root / "manifest.json")
    model = AdaptiveAreaModel(config, reference)
    output.mkdir(parents=True, exist_ok=False)
    hashes = sources(output)
    records = []
    manifest = dict(
        schema="crystalab_adaptive_area_replay_v1",
        status="running",
        source_replay=str(reader.root),
        source_replay_manifest_sha256=reader.manifest_hash,
        reference_manifest_sha256=reference_hash,
        source_sha256=hashes,
        labels_used_for_state_update=False,
        actuator_commands_issued=0,
        scope="Causal native observed-area state updates and forecasts; no future source masks, quality labels, action-response or actuation",
    )
    save_json(output / "manifest.json", manifest)
    try:
        with (output / "states.jsonl").open("x") as stream:
            for estimate, row in reader.estimates():
                state = model.update(estimate)
                stream.write(json.dumps(state, allow_nan=False) + "\n")
                stream.flush()
                records.append(state)
        reader.verify()
        if file_sha256(reference_root / "manifest.json") != reference_hash:
            raise ValueError("Adaptive reference changed during replay")
        for path, digest in hashes.items():
            if file_sha256(ROOT / path) != digest:
                raise ValueError("Adaptive replay producer changed")
        manifest.update(
            status="complete",
            frames=len(records),
            matured_updates=sum(r["matured_update"] is not None for r in records),
            policy=asdict(config),
            final_parameters=records[-1]["parameter_after"],
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise


def evaluate_adaptive(cohort, reference_root, baseline_model, output):
    cohort, reference_root, baseline_model, output = [
        Path(p).resolve() for p in (cohort, reference_root, baseline_model, output)
    ]
    audit = audit_cohort(cohort)
    reference, config = load_reference(reference_root)
    cohort_manifest = json.loads((cohort / "manifest.json").read_text())
    samples = json_rows(cohort / "samples.jsonl")
    baseline = load_forecast_model(baseline_model)
    if baseline["config"]["horizon_source_s"] != config.horizon_source_s:
        raise ValueError("Comparison horizons differ")
    validation_families = {
        c["family_id"] for c in cohort_manifest["cases"] if c["split"] == "validation"
    }
    adaptive_training_families = {
        c["family_id"] for c in reference["training_families"]
    }
    if validation_families & (
        adaptive_training_families | set(baseline["calibration"]["training_family_ids"])
    ):
        raise ValueError("Forecast calibration/reference overlaps validation families")
    output.mkdir(parents=True, exist_ok=False)
    hashes = sources(output)
    manifest = dict(
        schema="crystalab_adaptive_area_evaluation_v1",
        status="running",
        cohort_audit=audit,
        cohort_manifest_sha256=file_sha256(cohort / "manifest.json"),
        reference_manifest_sha256=file_sha256(reference_root / "manifest.json"),
        baseline_model_sha256=file_sha256(baseline_model),
        source_sha256=hashes,
        parameter_updates_use_only_matured_observed_areas=True,
        final_test_scored=False,
        actuator_commands_issued=0,
    )
    save_json(output / "manifest.json", manifest)
    try:
        states = {}
        replays = []
        for case in cohort_manifest["cases"]:
            if case["split"] not in {"train", "validation"}:
                raise ValueError("Adaptive development evaluation excludes final test")
            replay = cohort / case["replay_path"]
            target = output / "replays" / case["scene_id"]
            report = replay_adaptive(replay, reference_root, target)
            replays.append(
                dict(scene_id=case["scene_id"], split=case["split"], **report)
            )
            states[case["family_id"]] = {
                r["estimate"]["frame_index"]: r
                for r in json_rows(target / "states.jsonl")
            }
        scores, _ = evaluate_samples(samples, baseline["calibration"])
        by_id = {s["sample_id"]: s for s in samples}
        for sample in samples:
            state = states[sample["family_id"]][sample["cut_frame_index"]]
            p = state["forecast"]
            labels = sample["labels"]
            record = dict(
                sample_id=sample["sample_id"],
                family_id=sample["family_id"],
                split=sample["split"],
                method="adaptive_recent_rate",
                prediction=None
                if p is None
                else {
                    key: p[key]
                    for key in (
                        "future_area_px2",
                        "gain_px2",
                        "relative_gain",
                        "clipped_to_image",
                    )
                },
                labels=labels,
                common_history_support=False,
                unavailable_reason=None
                if p is not None
                else state["trend"]["reason"] or "horizon_elapsed_on_receipt",
                state_sha256=state["state_sha256"],
            )
            if p is not None and labels is not None:
                record.update(
                    future_area_error_px2=p["future_area_px2"]
                    - labels["future_proxy_area_px2"],
                    gain_error_px2=p["gain_px2"] - labels["proxy_gain_px2"],
                    relative_gain_error=p["relative_gain"]
                    - labels["relative_proxy_gain"],
                    current_state_error_px2=state["estimate"]["area_px2"]
                    - labels["current_proxy_area_px2"],
                )
            scores.append(record)
        common = {
            sample["sample_id"]
            for sample in samples
            if all(
                r["prediction"] is not None and r["labels"] is not None
                for r in scores
                if r["sample_id"] == sample["sample_id"]
            )
        }
        for r in scores:
            r["common_history_support"] = r["sample_id"] in common
        metrics = {}
        for split in ("train", "validation"):
            metrics[split] = {}
            for method in (*baseline_methods(), "adaptive_recent_rate"):
                selected = [
                    r for r in scores if r["split"] == split and r["method"] == method
                ]
                available = [r for r in selected if "future_area_error_px2" in r]
                common_rows = [r for r in available if r["common_history_support"]]
                metrics[split][method] = dict(
                    labelled_windows=len(selected),
                    available_windows=len(available),
                    all_available=metric_rows(available),
                    common_history_support=metric_rows(common_rows),
                    unavailable_windows=[
                        r["sample_id"] for r in selected if r["prediction"] is None
                    ],
                    per_family={
                        family: metric_rows(
                            [r for r in common_rows if r["family_id"] == family]
                        )
                        for family in sorted({r["family_id"] for r in selected})
                    },
                )
        (output / "scores.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in scores)
        )
        save_json(output / "metrics.json", metrics)
        if (
            file_sha256(cohort / "manifest.json") != manifest["cohort_manifest_sha256"]
            or file_sha256(baseline_model) != manifest["baseline_model_sha256"]
            or file_sha256(reference_root / "manifest.json")
            != manifest["reference_manifest_sha256"]
        ):
            raise ValueError("Adaptive evaluation input changed")
        manifest.update(
            status="complete",
            metrics=metrics,
            replays=replays,
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


def baseline_methods():
    return (
        "persistence",
        "recent_rate",
        "train_mean_fraction",
        "calibrated_recent_rate",
    )
