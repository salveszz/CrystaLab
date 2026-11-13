# FILE PURPOSE: Reconstruct frozen observable fits and scores independently of
# producer/model imports, using augmented least squares and source-row lineage.
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def metrics(truth, predicted):
    residual = [p - y for p, y in zip(predicted, truth)]
    ordered = sorted(abs(e) for e in residual)
    q = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(q), math.ceil(q)
    return dict(
        samples=len(residual),
        MAE_px=math.fsum(abs(e) for e in residual) / len(residual),
        RMSE_px=math.sqrt(math.fsum(e**2 for e in residual) / len(residual)),
        signed_error_px=math.fsum(residual) / len(residual),
        p95_absolute_error_px=ordered[lo] * (hi - q) + ordered[hi] * (q - lo)
        if hi != lo
        else ordered[lo],
    )


def audit(result, protocol_path, output):
    if output.exists() or output.with_suffix(".auditor.py").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), output.with_suffix(".auditor.py"))
    protocol = json.loads(protocol_path.read_text())
    assert protocol["schema"] == "crystalab_public_observable_forecast_v1"
    assert sha(result / "protocol.json") == sha(protocol_path)
    for relative, digest in json.loads((result / "manifest.json").read_text()).items():
        assert sha(result / relative) == digest
    for name in [
        "src/models/observable_forecast.py",
        "scripts/train_crystalcv_observable.py",
    ]:
        assert sha(result / "source" / name) == sha(ROOT / name)
    source = ROOT / protocol["source_export"]
    assert sha(source / "manifest.json") == protocol["source_manifest_sha256"]
    assert (
        sha(source / "candidate_samples.jsonl") == protocol["source_candidates_sha256"]
    )
    for name, digest in json.loads((source / "manifest.json").read_text()).items():
        assert sha(source / name) == digest
    original_audit = ROOT / protocol["source_independent_audit"]
    assert sha(original_audit) == protocol["source_independent_audit_sha256"]
    upstream = json.loads(original_audit.read_text())
    assert (
        upstream["passed"]
        and upstream["result_manifest_sha256"] == protocol["source_manifest_sha256"]
    )
    assert sha(original_audit.with_suffix(".auditor.py")) == upstream["auditor_sha256"]
    samples = jsonl(source / "candidate_samples.jsonl")
    observation_times = {
        r["sample_id"]: r["filename_elapsed_s"]
        for r in jsonl(source / "observations.jsonl")
    }
    with (source / "source_series.csv").open() as stream:
        seconds = [
            int((Decimal(r["Time (h)"]) * 3600).to_integral_value())
            for r in csv.DictReader(stream)
        ]
    assignments, cohort_rows = [], {r["name"]: [] for r in protocol["cohorts"]}
    support_image_ids = {name: set() for name in cohort_rows}
    support_log_ids = {name: set() for name in cohort_rows}
    for row in samples:
        assert list(row["inputs"]) == protocol["input_whitelist"]
        assert (
            not row["independent_ground_truth"]
            and row["split"] == "development"
            and row["physical_command"] is None
        )
        assert row["maximum_input_time_s"] <= row["cutoff_s"]
        first, last = row["input_log_row_range"]
        assert seconds[last + 1] <= row["cutoff_s"]
        support_start = min(
            seconds[first],
            *(observation_times[k] for k in row["input_image_sample_ids"]),
        )
        matches = [
            c["name"]
            for c in protocol["cohorts"]
            if c["minimum_cutoff_s"] <= row["cutoff_s"] <= c["maximum_cutoff_s"]
        ]
        assert len(matches) <= 1
        name = matches[0] if matches else "purged"
        assignments.append(
            dict(
                sample_id=row["sample_id"],
                cohort=name,
                support_start_s=float(support_start),
                support_end_s=float(row["target_time_s"]),
            )
        )
        if matches:
            cohort_rows[name].append(row)
            support_image_ids[name].update(
                row["input_image_sample_ids"] + [row["target_sample_id"]]
            )
            support_log_ids[name].update(range(first, last + 1))
    saved_parts = json.loads((result / "partitions.json").read_text())
    assert saved_parts["source_assignments"] == assignments
    assert saved_parts["purged_sample_ids"] == [
        r["sample_id"] for r in assignments if r["cohort"] == "purged"
    ]
    spans = {r["sample_id"]: r for r in assignments}
    names = [r["name"] for r in protocol["cohorts"]]
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            assert not support_image_ids[name] & support_image_ids[other]
            assert not support_log_ids[name] & support_log_ids[other]
            assert max(
                spans[r["sample_id"]]["support_end_s"] for r in cohort_rows[name]
            ) < min(
                spans[r["sample_id"]]["support_start_s"] for r in cohort_rows[other]
            )
    train = cohort_rows["train_prefix"]
    y = np.array(
        [r["target_sqrt_area_px"] - r["inputs"]["current_sqrt_area_px"] for r in train]
    )
    saved_models = json.loads((result / "models.json").read_text())
    assert saved_models["training_sample_ids"] == [r["sample_id"] for r in train]
    assert saved_models["fit_available_at_s"] == max(r["target_time_s"] for r in train)
    mean_gain = math.fsum(y) / len(y)
    assert math.isclose(saved_models["train_mean_gain_px"], mean_gain, abs_tol=1e-12)
    fits, max_coefficient_error = {}, 0.0
    for spec in protocol["models"]:
        features = spec["features"]
        x = np.array([[r["inputs"][k] for k in features] for r in train])
        mean = np.array([math.fsum(x[:, i]) / len(x) for i in range(x.shape[1])])
        scale = np.array(
            [
                math.sqrt(math.fsum((x[:, i] - mean[i]) ** 2) / len(x))
                for i in range(x.shape[1])
            ]
        )
        scale[scale == 0] = 1
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        penalty = np.zeros((len(features), len(features) + 1))
        penalty[:, 1:] = np.eye(len(features)) * math.sqrt(protocol["ridge"]["alpha"])
        coefficients = np.linalg.lstsq(
            np.vstack([design, penalty]),
            np.concatenate([y, np.zeros(len(features))]),
            rcond=None,
        )[0]
        stored = saved_models["models"][spec["name"]]
        assert (
            stored["schema"] == "crystalab_observable_gain_ridge_v1"
            and stored["features"] == features
        )
        np.testing.assert_allclose(stored["mean"], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stored["scale"], scale, rtol=1e-12, atol=1e-12)
        difference = float(
            abs(coefficients - [stored["intercept"], *stored["coefficients"]]).max()
        )
        assert difference < 1e-10
        max_coefficient_error = max(max_coefficient_error, difference)
        assert (
            stored["training_samples"] == len(train)
            and stored["alpha"] == protocol["ridge"]["alpha"]
        )
        assert not stored["model_promoted"] and stored["physical_command"] is None
        fits[spec["name"]] = (features, mean, scale, coefficients)
    predictions = jsonl(result / "predictions.jsonl")
    lookup = {(r["cohort"], r["sample_id"], r["method"]): r for r in predictions}
    assert len(lookup) == len(predictions)
    report = json.loads((result / "report.json").read_text())
    assert report["assignments"] == dict(Counter(r["cohort"] for r in assignments))
    maximum_prediction_difference, checked = 0.0, 0
    subset_counts = {}
    for name in names:
        selected = cohort_rows[name]
        if name != "train_prefix":
            assert (
                min(r["cutoff_s"] for r in selected)
                > saved_models["fit_available_at_s"]
            )
        subset, previous_end = [], -math.inf
        for row in selected:
            support = spans[row["sample_id"]]
            if support["support_start_s"] > previous_end:
                subset.append(row["sample_id"])
                previous_end = support["support_end_s"]
        info = saved_parts["cohorts"][name]
        assert (
            info["samples"] == len(selected) and info["disjoint_sample_ids"] == subset
        )
        assert info["cutoff_range_s"] == [
            min(r["cutoff_s"] for r in selected),
            max(r["cutoff_s"] for r in selected),
        ]
        assert info["support_range_s"] == [
            min(spans[r["sample_id"]]["support_start_s"] for r in selected),
            max(spans[r["sample_id"]]["support_end_s"] for r in selected),
        ]
        subset_counts[name] = len(subset)
        true = [r["target_sqrt_area_px"] for r in selected]
        for method in protocol["baselines"] + [r["name"] for r in protocol["models"]]:
            values = []
            for row in selected:
                current = row["inputs"]["current_sqrt_area_px"]
                if method == "persistence":
                    predicted = current
                elif method == "recent_rate":
                    predicted = (
                        current
                        + row["inputs"]["trailing_sqrt_area_rate_px_per_h"]
                        * protocol["nominal_horizon_s"]
                        / 3600
                    )
                elif method == "train_mean_gain":
                    predicted = current + mean_gain
                else:
                    features, mean, scale, coefficients = fits[method]
                    vector = [1.0] + [
                        (row["inputs"][feature] - mean[j]) / scale[j]
                        for j, feature in enumerate(features)
                    ]
                    predicted = current + math.fsum(
                        a * b for a, b in zip(vector, coefficients)
                    )
                record = lookup[name, row["sample_id"], method]
                difference = abs(predicted - record["predicted_future_sqrt_area_px"])
                assert difference < 1e-10
                maximum_prediction_difference = max(
                    maximum_prediction_difference, difference
                )
                assert (
                    record["current_sqrt_area_px"] == current
                    and record["observed_future_sqrt_area_px"]
                    == row["target_sqrt_area_px"]
                )
                assert (
                    record["nominal_target_time_s"]
                    == row["cutoff_s"] + protocol["nominal_horizon_s"]
                )
                assert (
                    record["actual_target_time_s"] == row["target_time_s"]
                    and record["cutoff_s"] == row["cutoff_s"]
                )
                assert math.isclose(
                    record["predicted_gain_px"], predicted - current, abs_tol=1e-10
                )
                assert math.isclose(
                    record["error_px"],
                    predicted - row["target_sqrt_area_px"],
                    abs_tol=1e-10,
                )
                assert record["disjoint_support_subset"] == (row["sample_id"] in subset)
                assert record["evaluation_role"] == (
                    "in_sample_fit"
                    if name == "train_prefix"
                    else "within_run_later_block"
                )
                values.append(predicted)
                checked += 1
            indexes = [
                i for i, row in enumerate(selected) if row["sample_id"] in subset
            ]
            for scope, indices in [
                ("all_windows", range(len(selected))),
                ("disjoint_support_subset", indexes),
            ]:
                independent = metrics(
                    [true[i] for i in indices], [values[i] for i in indices]
                )
                for key, value in independent.items():
                    assert math.isclose(
                        report["metrics"][name][method][scope][key],
                        value,
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    )
            assert report["metrics"][name][method][
                "negative_future_predictions"
            ] == sum(v < 0 for v in values)
    assert checked == len(predictions)
    for model in fits:
        improved = all(
            report["metrics"][name][model][scope]["MAE_px"]
            < report["metrics"][name]["recent_rate"][scope]["MAE_px"]
            for name in names[1:]
            for scope in ["all_windows", "disjoint_support_subset"]
        )
        assert (
            report["model_beats_recent_rate_in_both_later_blocks_and_subsets"][model]
            == improved
        )
    assert (
        report["independent_evaluation_groups"] == 0
        and not report["reviewed_ground_truth"]
        and not report["model_promoted"]
    )
    assert report["causal_action_effect"] is None and report["physical_command"] is None
    record = dict(
        passed=True,
        protocol_sha256=sha(protocol_path),
        result_manifest_sha256=sha(result / "manifest.json"),
        auditor_sha256=sha(Path(__file__)),
        source_candidates=len(samples),
        assignments=report["assignments"],
        disjoint_subset_counts=subset_counts,
        models_reconstructed=len(fits),
        prediction_records_reconstructed=checked,
        maximum_coefficient_difference=max_coefficient_error,
        maximum_prediction_difference_px=maximum_prediction_difference,
        full_cohort_supports_disjoint=True,
        training_preprocessing_prefix_only=True,
        source_independent_audit_sha256=sha(original_audit),
        independent_evaluation_groups=0,
        model_promoted=False,
        physical_command=None,
    )
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/crystalcv_observable_forecast.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result, args.protocol, args.output)
