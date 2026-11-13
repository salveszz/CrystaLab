# FILE PURPOSE: Run the frozen public observable forecasting comparison using a
# single chronological training prefix, disjoint later blocks and no promotion.
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.observable_forecast import (
    FEATURES,
    disjoint_subset,
    errors,
    fit_ridge,
    partition_samples,
    predict_gains,
)
from vision.video import file_sha256


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def train(protocol_path, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol["schema"] != "crystalab_public_observable_forecast_v1"
        or tuple(protocol["input_whitelist"]) != FEATURES
    ):
        raise ValueError("Require frozen observable task and input whitelist")
    shutil.copyfile(protocol_path, output / "protocol.json")
    for relative in [
        "src/models/observable_forecast.py",
        "scripts/train_crystalcv_observable.py",
    ]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    source = ROOT / protocol["source_export"]
    if (
        file_sha256(source / "manifest.json") != protocol["source_manifest_sha256"]
        or file_sha256(source / "candidate_samples.jsonl")
        != protocol["source_candidates_sha256"]
    ):
        raise ValueError("Audited source export identity changed")
    for relative, digest in json.loads((source / "manifest.json").read_text()).items():
        if file_sha256(source / relative) != digest:
            raise ValueError("Source export content changed")
    audit_path = ROOT / protocol["source_independent_audit"]
    if file_sha256(audit_path) != protocol["source_independent_audit_sha256"]:
        raise ValueError("Source audit identity changed")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["result_manifest_sha256"] != protocol["source_manifest_sha256"]
    ):
        raise ValueError("Require accepted V2 source reconstruction")
    source_protocol = json.loads((source / "protocol.json").read_text())
    if (
        source_protocol["candidate_samples"]["horizon_s"]
        != protocol["nominal_horizon_s"]
        or source_protocol["candidate_samples"]["reported_rate_availability"]
        != "next_source_row"
    ):
        raise ValueError("Source horizon/availability contract changed")
    samples = rows(source / "candidate_samples.jsonl")
    observations = rows(source / "observations.jsonl")
    with (source / "source_series.csv").open() as stream:
        log_seconds = [
            round(float(r["Time (h)"]) * 3600) for r in csv.DictReader(stream)
        ]
    grouped, spans = partition_samples(
        samples, observations, log_seconds, protocol["cohorts"]
    )
    training = grouped["train_prefix"]
    gains = np.array(
        [
            r["target_sqrt_area_px"] - r["inputs"]["current_sqrt_area_px"]
            for r in training
        ]
    )
    models = {
        spec["name"]: fit_ridge(
            [r["inputs"] for r in training],
            gains,
            spec["features"],
            protocol["ridge"]["alpha"],
        )
        for spec in protocol["models"]
    }
    if len(models) != len(protocol["models"]):
        raise ValueError("Model names must be unique")
    mean_gain = float(gains.mean())
    fit_available_at_s = max(r["target_time_s"] for r in training)
    scores, summaries, cohort_info = [], {}, {}
    for cohort in protocol["cohorts"]:
        name = cohort["name"]
        selected = grouped[name]
        if (
            name != "train_prefix"
            and min(r["cutoff_s"] for r in selected) <= fit_available_at_s
        ):
            raise ValueError(
                "Training labels were not available before later predictions"
            )
        subset = disjoint_subset(selected)
        inputs = [r["inputs"] for r in selected]
        truth = np.array([r["target_sqrt_area_px"] for r in selected])
        current = np.array([r["inputs"]["current_sqrt_area_px"] for r in selected])
        predictions = dict(
            persistence=current,
            recent_rate=current
            + np.array(
                [r["inputs"]["trailing_sqrt_area_rate_px_per_h"] for r in selected]
            )
            * protocol["nominal_horizon_s"]
            / 3600,
            train_mean_gain=current + mean_gain,
        )
        predictions.update(
            {
                method: current + predict_gains(model, inputs)
                for method, model in models.items()
            }
        )
        cohort_info[name] = dict(
            samples=len(selected),
            cutoff_range_s=[
                min(r["cutoff_s"] for r in selected),
                max(r["cutoff_s"] for r in selected),
            ],
            support_range_s=[
                min(r["support_start_s"] for r in selected),
                max(r["support_end_s"] for r in selected),
            ],
            disjoint_sample_ids=subset,
        )
        indexes = [i for i, row in enumerate(selected) if row["sample_id"] in subset]
        summaries[name] = {}
        for method, prediction in predictions.items():
            summaries[name][method] = dict(
                all_windows=errors(truth, prediction),
                disjoint_support_subset=errors(truth[indexes], prediction[indexes]),
                negative_future_predictions=int(np.count_nonzero(prediction < 0)),
            )
            for i, row in enumerate(selected):
                scores.append(
                    dict(
                        sample_id=row["sample_id"],
                        cohort=name,
                        evaluation_role="in_sample_fit"
                        if name == "train_prefix"
                        else "within_run_later_block",
                        method=method,
                        cutoff_s=row["cutoff_s"],
                        nominal_target_time_s=row["cutoff_s"]
                        + protocol["nominal_horizon_s"],
                        actual_target_time_s=row["target_time_s"],
                        current_sqrt_area_px=float(current[i]),
                        observed_future_sqrt_area_px=float(truth[i]),
                        predicted_future_sqrt_area_px=float(prediction[i]),
                        predicted_gain_px=float(prediction[i] - current[i]),
                        error_px=float(prediction[i] - truth[i]),
                        disjoint_support_subset=row["sample_id"] in subset,
                    )
                )
    later = [c["name"] for c in protocol["cohorts"] if c["name"] != "train_prefix"]
    descriptive = {
        method: all(
            summaries[name][method][scope]["MAE_px"]
            < summaries[name]["recent_rate"][scope]["MAE_px"]
            for name in later
            for scope in ["all_windows", "disjoint_support_subset"]
        )
        for method in models
    }
    save(
        output / "models.json",
        dict(
            models=models,
            train_mean_gain_px=mean_gain,
            fit_available_at_s=fit_available_at_s,
            training_sample_ids=[r["sample_id"] for r in training],
        ),
    )
    save(
        output / "partitions.json",
        dict(
            cohorts=cohort_info,
            source_assignments=spans,
            purged_sample_ids=[r["sample_id"] for r in grouped["purged"]],
            full_supports_disjoint=True,
        ),
    )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in scores)
    )
    save(
        output / "environment.json",
        dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
    )
    report = dict(
        protocol_sha256=file_sha256(protocol_path),
        source_samples=len(samples),
        assignments=dict(Counter(r["cohort"] for r in spans)),
        fit_available_at_s=fit_available_at_s,
        metrics=summaries,
        model_beats_recent_rate_in_both_later_blocks_and_subsets=descriptive,
        independent_evaluation_groups=0,
        reviewed_ground_truth=False,
        causal_action_effect=None,
        model_promoted=False,
        physical_command=None,
        interpretation="Single-run development on automatic pixel proxies. No independent KDP accuracy or causal response claim.",
    )
    save(output / "report.json", report)
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "assignments",
                    "fit_available_at_s",
                    "model_beats_recent_rate_in_both_later_blocks_and_subsets",
                ]
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/crystalcv_observable_forecast.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.protocol, args.output)
