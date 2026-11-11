# FILE PURPOSE: Fit, compare and replay transparent projected-area forecast baselines.
# Calibration uses training families only; proxy-label evaluation, warmup coverage,
# measurement bias and unvalidated real shadow predictions remain separate.
"""Audited development forecast calibration and source-clock observation replay."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from importlib.metadata import version
import math
from pathlib import Path
import platform
import shutil
import time


from models.area_forecast import (
    AreaForecastConfig,
    AreaTrend,
    FEATURES,
    METHODS,
    fit_calibration,
    predictions,
)
from vision.area_forecast_data import ReplayAreas, audit_cohort, json_rows
from vision.training_data import safe_path, save_json
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def metric_rows(rows):
    if not rows:
        return dict(
            samples=0,
            families=0,
            future_area_mae_px2=None,
            future_area_rmse_px2=None,
            gain_mae_px2=None,
            relative_gain_mae=None,
            future_area_signed_error_px2=None,
            family_macro_area_mae_px2=None,
        )
    errors = [r["future_area_error_px2"] for r in rows]
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(abs(row["future_area_error_px2"]))
    return dict(
        samples=len(rows),
        families=len(families),
        future_area_mae_px2=sum(abs(e) for e in errors) / len(errors),
        future_area_rmse_px2=math.sqrt(sum(e * e for e in errors) / len(errors)),
        gain_mae_px2=sum(abs(r["gain_error_px2"]) for r in rows) / len(rows),
        relative_gain_mae=sum(abs(r["relative_gain_error"]) for r in rows) / len(rows),
        future_area_signed_error_px2=sum(errors) / len(errors),
        family_macro_area_mae_px2=sum(sum(v) / len(v) for v in families.values())
        / len(families),
    )


def evaluate_samples(samples, calibration):
    scores = []
    summaries = {}
    for sample in samples:
        predicted = predictions(
            sample["trend"]["features"],
            coefficient=calibration["coefficient"],
            mean_fraction=calibration["mean_fraction"],
        )
        labels = sample["labels"]
        common = labels is not None and all(
            predicted[method] is not None for method in METHODS
        )
        for method, value in predicted.items():
            record = dict(
                sample_id=sample["sample_id"],
                family_id=sample["family_id"],
                split=sample["split"],
                method=method,
                prediction=value,
                labels=labels,
                common_history_support=common,
                unavailable_reason=sample["trend"]["reason"] if value is None else None,
            )
            if labels is not None and value is not None:
                record.update(
                    future_area_error_px2=value["future_area_px2"]
                    - labels["future_proxy_area_px2"],
                    gain_error_px2=value["gain_px2"] - labels["proxy_gain_px2"],
                    relative_gain_error=value["relative_gain"]
                    - labels["relative_proxy_gain"],
                    current_state_error_px2=sample["trend"]["features"][
                        "current_area_px2"
                    ]
                    - labels["current_proxy_area_px2"],
                )
            scores.append(record)
    for split in ["train", "validation"]:
        labels = sum(s["split"] == split and s["labels"] is not None for s in samples)
        methods = {}
        for method in METHODS:
            selected = [
                r for r in scores if r["split"] == split and r["method"] == method
            ]
            scored = [r for r in selected if "future_area_error_px2" in r]
            common = [r for r in scored if r["common_history_support"]]
            methods[method] = dict(
                labelled_windows=labels,
                prediction_coverage=len(scored) / labels if labels else None,
                unavailable_reasons=dict(
                    Counter(
                        r["unavailable_reason"]
                        for r in selected
                        if r["prediction"] is None
                    )
                ),
                all_available=metric_rows(scored),
                common_history_support=metric_rows(common),
                clipped_predictions=sum(
                    r["prediction"]["clipped_to_image"] for r in scored
                ),
                per_family={
                    family: metric_rows([r for r in common if r["family_id"] == family])
                    for family in sorted({r["family_id"] for r in common})
                },
            )
        summaries[split] = methods
    return scores, summaries


def snapshot(output, paths):
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = file_sha256(target)
        if file_sha256(path) != hashes[str(relative)]:
            raise ValueError("Forecast producer changed during snapshot")
    return hashes


def plot_scores(scores, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    markers = ["o", "s", "^", "D"]
    for method, marker in zip(METHODS, markers):
        rows = [
            r
            for r in scores
            if r["split"] == "validation"
            and r["method"] == method
            and r["common_history_support"]
        ]
        if not rows:
            continue
        axes[0].scatter(
            [r["labels"]["proxy_gain_px2"] for r in rows],
            [r["prediction"]["gain_px2"] for r in rows],
            label=method,
            marker=marker,
            s=45,
        )
        axes[1].scatter(
            [r["labels"]["future_proxy_area_px2"] for r in rows],
            [r["prediction"]["future_area_px2"] for r in rows],
            label=method,
            marker=marker,
            s=45,
        )
    for axis in axes:
        limits = axis.get_xlim() + axis.get_ylim()
        low, high = min(limits), max(limits)
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_xlabel("Original proxy gain (px²)")
    axes[0].set_ylabel("Forecast gain (px²)")
    axes[1].set_xlabel("Original future proxy area (px²)")
    axes[1].set_ylabel("Forecast future area (px²)")
    validation = [
        r for r in scores if r["split"] == "validation" and r["common_history_support"]
    ]
    families = len({r["family_id"] for r in validation})
    windows = len({r["sample_id"] for r in validation})
    fig.suptitle(
        f"Development diagnostic / {families} validation families / {windows} one-step proxy windows"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def train_forecast(cohort, output):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    audit = audit_cohort(cohort)
    cohort_hash = file_sha256(cohort / "manifest.json")
    config = AreaForecastConfig(
        **json.loads((cohort / "forecast_config.json").read_text())
    )
    samples = json_rows(cohort / "samples.jsonl")
    calibration = fit_calibration(samples, config)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(cohort / "manifest.json", output / "cohort_manifest.json")
    source_hashes = snapshot(
        output,
        [
            Path(__file__),
            ROOT / "src/models/area_forecast.py",
            ROOT / "src/vision/area_forecast_data.py",
            ROOT / "scripts/train_area_forecast.py",
        ],
    )
    manifest = dict(
        schema="crystalab_area_forecast_run_v1",
        status="running",
        cohort=str(cohort),
        cohort_manifest_sha256=cohort_hash,
        cohort_audit=audit,
        source_sha256=source_hashes,
        config=asdict(config),
        environment=dict(
            python=platform.python_version(),
            numpy=version("numpy"),
            opencv=version("opencv-python"),
            matplotlib=version("matplotlib"),
        ),
        scope="Synthetic development next-frame projected-area diagnostic; no real, physical-time or action-response performance claim",
        final_test_scored=False,
    )
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        model = dict(
            schema="crystalab_area_forecast_model_v1",
            config=asdict(config),
            calibration=calibration,
            training_samples_sha256=file_sha256(cohort / "samples.jsonl"),
            cohort_manifest_sha256=cohort_hash,
            source_sha256=source_hashes,
            requires_operator_review=True,
            can_authorize_actuation=False,
            physical_horizon_available=False,
            action_conditioned=False,
            selection_policy="report_only_no_automatic_promotion",
        )
        save_json(output / "model.json", model)
        scores, metrics = evaluate_samples(samples, calibration)
        (output / "scores.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in scores)
        )
        save_json(output / "metrics.json", metrics)
        plot_scores(scores, output)
        if audit_cohort(cohort)["manifest_sha256"] != cohort_hash:
            raise ValueError("Forecast cohort changed during fitting")
        manifest.update(
            status="complete",
            calibration=calibration,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
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


def load_forecast_model(path):
    path = Path(path).resolve()
    root = path.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != "crystalab_area_forecast_run_v1"
        or manifest["status"] != "complete"
    ):
        raise ValueError("Completed forecast training run required")
    for name, digest in manifest["files"].items():
        if file_sha256(safe_path(root, name)) != digest:
            raise ValueError("Forecast model run artifact changed")
    if str(path.relative_to(root)) not in manifest["files"]:
        raise ValueError("Forecast model is outside the recorded artifact set")
    model = json.loads(path.read_text())
    if (
        model.get("schema") != "crystalab_area_forecast_model_v1"
        or model["calibration"] != manifest["calibration"]
        or tuple(model["calibration"]["feature_whitelist"]) != FEATURES
    ):
        raise ValueError("Forecast model contract/calibration differs")
    AreaForecastConfig(**model["config"])
    if (
        model["config"] != manifest["config"]
        or model["action_conditioned"] is not False
        or model["can_authorize_actuation"] is not False
        or model["requires_operator_review"] is not True
        or model["physical_horizon_available"] is not False
    ):
        raise ValueError("Forecast model scope changed")
    config = AreaForecastConfig(**model["config"])
    calibration = model["calibration"]
    if not config.coefficient_min <= calibration[
        "coefficient"
    ] <= config.coefficient_max or not math.isfinite(calibration["mean_fraction"]):
        raise ValueError("Forecast model parameters are outside the declared contract")
    return model


def forecast_replay(replay_path, model_path, output):
    reader = ReplayAreas(replay_path)
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    model = load_forecast_model(model_path)
    model_hash = file_sha256(model_path)
    config = AreaForecastConfig(**model["config"])
    trend = AreaTrend(config)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(
        reader.root / "manifest.json", output / "source_replay_manifest.json"
    )
    shutil.copyfile(model_path, output / "model.json")
    source_hashes = snapshot(
        output,
        [
            Path(__file__),
            ROOT / "src/models/area_forecast.py",
            ROOT / "src/vision/area_forecast_data.py",
            ROOT / "scripts/forecast_area.py",
        ],
    )
    manifest = dict(
        schema="crystalab_area_forecast_replay_v1",
        status="running",
        source_replay=str(reader.root),
        source_replay_manifest_sha256=reader.manifest_hash,
        model_sha256=model_hash,
        source_sha256=source_hashes,
        source_info=reader.manifest["source_info"],
        clock_id=reader.clock_id,
        source_experiment_clock=reader.manifest.get("experiment_clock"),
        scope="Unvalidated source-clock projected-area estimates; fixed-view assumption, no face/volume/quality/action-response inference",
        real_accuracy=None,
        physical_horizon_s=None,
        numerical_actuator_commands=0,
        operator_reviews_created=0,
    )
    save_json(output / "manifest.json", manifest)
    count = 0
    coverage = Counter()
    reasons = Counter()
    try:
        with (output / "forecasts.jsonl").open("x") as stream:
            for estimate, record in reader.estimates():
                state = trend.update(estimate)
                predicted = predictions(
                    state["features"],
                    coefficient=model["calibration"]["coefficient"],
                    mean_fraction=model["calibration"]["mean_fraction"],
                )
                row = dict(
                    frame_index=estimate.frame_index,
                    video_time_s=estimate.observed_at_s,
                    target_time_source_s=estimate.observed_at_s
                    + config.horizon_source_s,
                    source_frame=record.get("source_frame"),
                    experiment_time_s=record["experiment_time_s"],
                    estimate=asdict(estimate),
                    trend=state,
                    predictions=predicted,
                    view_comparability="constructed_fixed_pose"
                    if reader.manifest["source_info"].get("source_id")
                    == "crystalab_optical_images_v1"
                    else "assumed_unverified",
                    requires_operator_review=True,
                    can_authorize_actuation=False,
                    physical_horizon_s=None,
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                count += 1
                coverage.update(
                    method for method, value in predicted.items() if value is not None
                )
                if state["reason"]:
                    reasons[state["reason"]] += 1
        reader.verify()
        if file_sha256(model_path) != model_hash:
            raise ValueError("Forecast model changed during replay")
        manifest.update(
            status="complete",
            processed_frames=count,
            predictions_by_method=dict(coverage),
            unavailable_reasons=dict(reasons),
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p != output / "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            processed_frames=count,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise
