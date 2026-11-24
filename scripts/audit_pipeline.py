# FILE PURPOSE: Independently verify a full simulation and synthetic XGBoost run.
# Reconstruct conservation/geometry, raw-data labels, baselines and serialized
# tree predictions without importing the training implementation or XGBoost.
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from environment.dataset import FEATURES, audit_dataset, read_json, read_rows


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def tree_predictions(path, rows, columns):
    document = read_json(path)["learner"]
    assert document["feature_names"] == columns
    assert document["objective"]["name"] == "reg:squarederror"
    raw = document["learner_model_param"]["base_score"]
    base = json.loads(raw) if isinstance(raw, str) else raw
    base = base[0] if isinstance(base, list) else base
    trees = document["gradient_booster"]["model"]["trees"]
    result = []
    for row in rows:
        vector = np.array(
            [
                np.nan if row["features"][name] is None else row["features"][name]
                for name in FEATURES
            ]
            + [int(row["missing"][name]) for name in FEATURES],
            dtype=np.float32,
        )
        prediction = np.float32(base)
        for tree in trees:
            node = 0
            while tree["left_children"][node] != -1:
                assert tree["split_type"][node] == 0
                value = vector[tree["split_indices"][node]]
                go_left = (
                    bool(tree["default_left"][node])
                    if np.isnan(value)
                    else value < np.float32(tree["split_conditions"][node])
                )
                node = tree["left_children" if go_left else "right_children"][node]
            prediction = np.float32(
                prediction + np.float32(tree["split_conditions"][node])
            )
        result.append(float(prediction))
    return np.array(result)


def independent_metrics(rows, values):
    if not rows:
        return dict(
            samples=0,
            groups=0,
            mae=None,
            rmse=None,
            signed_error=None,
            group_macro_mae=None,
        )
    errors = [p - r["labels"]["volume_gain"] for r, p in zip(rows, values)]
    groups = defaultdict(list)
    for row, error in zip(rows, errors):
        groups[row["group_id"]].append(abs(error))
    return dict(
        samples=len(rows),
        groups=len(groups),
        mae=math.fsum(abs(v) for v in errors) / len(errors),
        rmse=math.sqrt(math.fsum(v * v for v in errors) / len(errors)),
        signed_error=math.fsum(errors) / len(errors),
        group_macro_mae=math.fsum(math.fsum(v) / len(v) for v in groups.values())
        / len(groups),
    )


def audit_training(dataset, training):
    dataset_audit = audit_dataset(dataset)
    source = read_json(dataset / "manifest.json")
    manifest = read_json(training / "manifest.json")
    assert (
        manifest["status"] == "complete"
        and manifest["schema"] == "crystalab_growth_xgboost_v1"
    )
    assert manifest["dataset_manifest_sha256"] == sha(
        dataset / "manifest.json"
    ) and manifest["samples_sha256"] == sha(dataset / "samples.jsonl")
    for name, digest in manifest["files"].items():
        assert sha(training / name) == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert sha(training / "source_snapshot" / name) == digest, name
    config = read_json(training / "config.json")
    assert sha(training / "config.json") == manifest["config_sha256"]
    columns = list(FEATURES) + ["missing__" + name for name in FEATURES]
    assert (
        manifest["input_columns"]
        == columns
        == read_json(training / "fit_inputs.json")["input_columns"]
    )
    rows = read_rows(dataset / "samples.jsonl")
    for row in rows:
        assert set(row["features"]) == set(FEATURES) == set(row["missing"])
        assert all(
            row["missing"][name] == (row["features"][name] is None) for name in FEATURES
        )
    splits = {
        name: [r for r in rows if r["split"] == name]
        for name in ["train", "validation", "test"]
    }
    for key in ["family_id", "group_id"]:
        group_sets = [{r[key] for r in subset} for subset in splits.values()]
        assert all(
            not a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :]
        )
    fit = read_json(training / "fit_inputs.json")
    assert fit["train_sample_ids"] == [r["sample_id"] for r in splits["train"]]
    assert fit["validation_sample_ids"] == [
        r["sample_id"] for r in splits["validation"]
    ]
    mean = math.fsum(r["labels"]["volume_gain"] for r in splits["train"]) / len(
        splits["train"]
    )
    assert math.isclose(mean, fit["mean_training_gain"], abs_tol=1e-14)
    selection = read_json(training / "selection.json")
    release = read_json(training / "test_release.json")
    assert (
        not selection["test_used_for_selection"] and release["test_scoring_passes"] == 1
    )
    assert release["selection_sha256"] == sha(training / "selection.json")
    assert (
        release["model_sha256"]
        == selection["selected_model_sha256"]
        == sha(training / "model.json")
    )
    candidate_scores = []
    for candidate, definition in zip(selection["candidates"], config["candidates"]):
        assert (
            candidate["params"] == definition
            and sha(training / candidate["model"]) == candidate["model_sha256"]
        )
        predicted = tree_predictions(
            training / candidate["model"], splits["validation"], columns
        )
        score = independent_metrics(splits["validation"], predicted)["mae"]
        assert (
            abs(score - candidate["validation_mae"])
            < config["manual_prediction_audit_atol"]
        )
        candidate_scores.append((candidate["validation_mae"], candidate["index"]))
    assert len(candidate_scores) == len(config["candidates"])
    assert selection["selected"]["index"] == min(candidate_scores)[1]
    assert sha(training / selection["selected"]["model"]) == sha(
        training / "model.json"
    )
    actual = read_rows(training / "predictions.jsonl")
    assert [r["sample_id"] for r in actual] == [
        r["sample_id"] for subset in splits.values() for r in subset
    ]
    report = read_json(training / "report.json")
    maximum = 0.0
    for split, subset in splits.items():
        stored = [r for r in actual if r["split"] == split]
        manually = tree_predictions(training / "model.json", subset, columns)
        error = max(
            abs(float(a) - r["predictions"]["xgboost"])
            for a, r in zip(manually, stored)
        )
        assert error <= config["manual_prediction_audit_atol"]
        maximum = max(maximum, error)
        for original, predicted in zip(subset, stored):
            assert (
                predicted["target_volume_gain"] == original["labels"]["volume_gain"]
                and predicted["group_id"] == original["group_id"]
            )
            assert predicted["startup"] == original["missing"]["volume"]
            rate = original["features"]["recent_volume_rate"]
            expected = {
                "persistence": 0.0,
                "train_mean": mean,
                "recent_rate": mean
                if rate is None
                else rate * original["features"]["horizon"],
            }
            assert predicted["recent_rate_fallback"] == (rate is None)
            for method, value in expected.items():
                assert abs(predicted["predictions"][method] - value) < 1e-14
        for method in ["persistence", "train_mean", "recent_rate", "xgboost"]:
            for cohort, indices in [
                ("all", range(len(subset))),
                (
                    "startup",
                    [i for i, r in enumerate(subset) if r["missing"]["volume"]],
                ),
                (
                    "observed",
                    [i for i, r in enumerate(subset) if not r["missing"]["volume"]],
                ),
            ]:
                expected = independent_metrics(
                    [subset[i] for i in indices],
                    [stored[i]["predictions"][method] for i in indices],
                )
                for key, value in expected.items():
                    recorded = report["metrics"][split][method][cohort][key]
                    assert (
                        recorded is None
                        if value is None
                        else math.isclose(recorded, value, rel_tol=1e-12, abs_tol=1e-12)
                    )
            for recipe, recorded in report["metrics"][split][method][
                "per_recipe"
            ].items():
                indices = [
                    i
                    for i, r in enumerate(subset)
                    if r["episode_id"].split("__", 1)[1] == recipe
                ]
                expected = independent_metrics(
                    [subset[i] for i in indices],
                    [stored[i]["predictions"][method] for i in indices],
                )
                for key, value in expected.items():
                    assert math.isclose(
                        recorded[key], value, rel_tol=1e-12, abs_tol=1e-12
                    )
    group_order = sorted(
        {r["group_id"] for r in splits["train"]},
        key=lambda value: hashlib.sha256(
            (config["learning_curve_seed"] + value).encode()
        ).hexdigest(),
    )
    curves = read_json(training / "learning_curve.json")
    assert [r["training_groups"] for r in curves] == sorted(
        set(min(n, len(group_order)) for n in config["learning_curve_groups"])
    )
    for curve in curves:
        assert curve["group_ids"] == group_order[: curve["training_groups"]]
        chosen = [r for r in splits["train"] if r["group_id"] in curve["group_ids"]]
        assert curve["training_sample_ids"] == [r["sample_id"] for r in chosen]
        assert (
            curve["training_samples"] == len(chosen)
            and sha(training / curve["model"]) == curve["model_sha256"]
        )
        prediction = tree_predictions(
            training / curve["model"], splits["validation"], columns
        )
        expected = independent_metrics(splits["validation"], prediction)
        assert (
            abs(curve["validation"]["mae"] - expected["mae"])
            <= config["manual_prediction_audit_atol"]
        )
    assert (
        not report["quality_target_available"] and not report["physical_model_promoted"]
    )
    return dict(
        passed=True,
        dataset=dataset_audit,
        model_sha256=sha(training / "model.json"),
        prediction_rows=len(actual),
        maximum_manual_tree_prediction_difference=maximum,
        train_validation_test_families_disjoint=True,
        test_selection_frozen=True,
        feature_columns=columns,
    )


def audit_simulation(simulation, expected_duration):
    config = read_json(simulation / "config.json")
    with (simulation / "measurements.csv").open() as stream:
        rows = [
            {key: float(v) for key, v in row.items()} for row in csv.DictReader(stream)
        ]
    assert (
        config["duration"] == expected_duration == rows[-1]["time"]
        and rows[0]["time"] == 0.0
    )
    assert all(
        0 < b["time"] - a["time"] <= config["dt"] + 1e-12
        for a, b in zip(rows, rows[1:])
    )
    assert len(rows) == math.ceil(expected_duration / config["dt"]) + 1
    assert all(math.isfinite(v) for row in rows for v in row.values())
    mass = max(
        abs(r["total_mass"] - rows[0]["total_mass"]) / rows[0]["total_mass"]
        for r in rows
    )
    carrier = max(
        abs(r["liquid_volume"] - rows[0]["liquid_volume"]) / rows[0]["liquid_volume"]
        for r in rows
    )
    assert mass <= 1e-10 and carrier <= 1e-10
    with np.load(simulation / "fields.npz", allow_pickle=False) as fields:
        assert all(np.isfinite(fields[key]).all() for key in fields.files)
        assert fields["temperature"].shape == (config["nz"], config["ny"], config["nx"])
        assert (fields["concentration"] >= 0).all() and (
            fields["liquid_volumes"] >= 0
        ).all()
        volume = float(ConvexHull(fields["vertices"]).volume)
        assert math.isclose(volume, rows[-1]["volume"], rel_tol=1e-10, abs_tol=1e-10)
        dissolved = float(np.sum(fields["concentration"] * fields["liquid_volumes"]))
        assert math.isclose(
            dissolved + volume * config["crystal_density"],
            rows[-1]["total_mass"],
            rel_tol=1e-10,
        )
        assert math.isclose(
            float(fields["liquid_volumes"].sum()),
            rows[-1]["liquid_volume"],
            rel_tol=1e-10,
        )
        assert (fields["vertices"] >= -1e-10).all() and (
            fields["vertices"]
            <= np.array([config["width"], config["depth"], config["height"]]) + 1e-10
        ).all()
    with Image.open(simulation / "growth.gif") as image:
        assert image.n_frames >= 2 and image.info.get("loop") is None
        frames = image.n_frames
    with Image.open(simulation / "final_tank.png") as image:
        image.verify()
    return dict(
        passed=True,
        final_time=rows[-1]["time"],
        history_rows=len(rows),
        gif_frames=frames,
        final_volume=volume,
        maximum_relative_mass_drift=mass,
        maximum_relative_carrier_drift=carrier,
    )


def audit(run, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(run / "manifest.json")
    config = manifest["config"]
    for name, digest in manifest["source_sha256"].items():
        assert sha(run / "source_snapshot" / name) == digest
    for name in ["dataset", "simulation", "training"]:
        assert (
            manifest["stages"][name]["status"] == "complete"
            and manifest["stages"][name]["return_code"] == 0
        )
    model = audit_training(run / "dataset", run / "training")
    assert (
        model["dataset"]["episodes"] == config["expected_episodes"]
        and model["dataset"]["samples"] == config["expected_samples"]
    )
    simulation = audit_simulation(
        run / "simulation", config["expected_simulation_duration"]
    )
    if (run / "files.json").exists():
        for name, record in read_json(run / "files.json").items():
            assert (run / name).stat().st_size == record["bytes"] and sha(
                run / name
            ) == record["sha256"]
    result = dict(
        passed=True,
        simulation=simulation,
        training=model,
        auditor_sha256=sha(Path(__file__)),
        scope=config["scope"],
        physical_model_promoted=False,
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.output)
