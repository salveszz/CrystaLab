# FILE PURPOSE: Verify causal features, family isolation, missing-value fallbacks,
# genuine test-label independence, saved inference and independent tree auditing.
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import test_dataset as dataset_fixture
from environment.dataset import FEATURES, generate, read_rows, read_json, file_hash
from models.growth_xgboost import (
    INPUTS,
    baseline_values,
    feature_matrix,
    metrics,
    predict_saved,
    train,
    validate_config,
    validate_splits,
)
from audit_pipeline import audit_training


def sample():
    features = {name: 1.0 for name in FEATURES}
    features.update(volume=None, recent_volume_rate=None, horizon=10.0)
    return dict(
        features=features,
        missing={name: value is None for name, value in features.items()},
        labels=dict(volume_gain=0.2),
        group_id="fixture",
    )


def test_only_causal_features_and_explicit_missing_flags_enter_matrix():
    row = sample()
    before = feature_matrix([row])
    changed = copy.deepcopy(row)
    changed.update(sample_id="future-secret", decision_time=999, exact_truth=1e20)
    changed["labels"]["volume_gain"] = 1e20
    np.testing.assert_array_equal(before, feature_matrix([changed]))
    assert before.shape == (1, 28) and len(INPUTS) == 28
    assert np.isnan(before[0, FEATURES.index("volume")])
    assert before[0, 14 + FEATURES.index("volume")] == 1
    changed["features"]["future_volume"] = 99
    with pytest.raises(ValueError, match="whitelist"):
        feature_matrix([changed])


def test_missing_mismatch_and_nonfinite_available_values_are_rejected():
    row = sample()
    row["missing"]["volume"] = False
    with pytest.raises(ValueError, match="Missing"):
        feature_matrix([row])
    row = sample()
    row["features"]["temperature"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        feature_matrix([row])


def test_recent_rate_fallback_is_training_mean_not_zero_or_future_truth():
    a, b = sample(), sample()
    b["features"]["recent_volume_rate"] = -0.01
    b["missing"]["recent_volume_rate"] = False
    values, fallback = baseline_values([a, b], 0.15)
    np.testing.assert_allclose(values["recent_rate"], [0.15, -0.1])
    assert fallback == [True, False]
    assert metrics([a, b], [0.3, 0.3])["signed_error"] == pytest.approx(0.1)


def test_related_family_crossing_splits_is_rejected():
    rows = []
    for i, split in enumerate(["train", "validation", "test"]):
        rows.append(
            dict(
                sample(),
                sample_id=str(i),
                group_id=str(i),
                family_id=str(i),
                split=split,
            )
        )
    validate_splits(rows)
    rows[-1]["family_id"] = rows[0]["family_id"]
    with pytest.raises(ValueError, match="families"):
        validate_splits(rows)


def test_group_macro_metric_does_not_overweight_longer_groups():
    rows = [dict(sample(), group_id=group) for group in ["a", "a", "a", "b"]]
    result = metrics(rows, [0.2, 0.2, 0.2, 1.2])
    assert result["mae"] == pytest.approx(0.25)
    assert result["group_macro_mae"] == pytest.approx(0.5)


def test_invalid_task_or_zero_rounds_fail_before_fitting():
    cfg = read_json(ROOT / "configs/xgboost_growth.json")
    cfg["num_boost_round"] = 0
    with pytest.raises(ValueError, match="round budgets"):
        validate_config(cfg)
    cfg = read_json(ROOT / "configs/xgboost_growth.json")
    cfg["candidates"][0]["objective"] = "reg:logistic"
    with pytest.raises(ValueError, match="only valid depth"):
        validate_config(cfg)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    if importlib.util.find_spec("xgboost") is None:
        pytest.skip("Install the growth-training extra for XGBoost integration checks")
    root = tmp_path_factory.mktemp("growth-model")
    cfg = read_json(ROOT / "configs/xgboost_growth.json")
    cfg.update(
        num_boost_round=12,
        early_stopping_rounds=3,
        learning_curve_groups=[1],
        candidates=[dict(max_depth=2, eta=0.1), dict(max_depth=3, eta=0.1)],
    )
    cfg["params"]["nthread"] = 1
    path = root / "training.json"
    path.write_text(json.dumps(cfg))
    protocol = dataset_fixture.protocol()
    generate(protocol, root / "dataset", jobs=1)
    train(root / "dataset", path, root / "fit")
    return root, path, protocol


def test_full_small_training_saved_reload_and_manual_tree_audit(fitted):
    root, _, _ = fitted
    rows = read_rows(root / "dataset/samples.jsonl")
    saved = {r["sample_id"]: r for r in read_rows(root / "fit/predictions.jsonl")}
    prediction = predict_saved(root / "fit", rows)
    np.testing.assert_array_equal(
        prediction, [saved[r["sample_id"]]["predictions"]["xgboost"] for r in rows]
    )
    audit = audit_training(root / "dataset", root / "fit")
    assert audit["passed"] and audit["prediction_rows"] == len(rows)
    assert audit["maximum_manual_tree_prediction_difference"] <= 0.000002


def test_changing_only_test_physics_cannot_change_selected_model(fitted):
    root, config, protocol = fitted
    modified = copy.deepcopy(protocol)
    modified["groups"][-1]["overrides"]["concentration"] = 1.55
    generate(modified, root / "changed_test_dataset", jobs=1)
    train(root / "changed_test_dataset", config, root / "changed_test_fit")
    assert file_hash(root / "fit/model.json") == file_hash(
        root / "changed_test_fit/model.json"
    )
    a = read_rows(root / "dataset/samples.jsonl")
    b = read_rows(root / "changed_test_dataset/samples.jsonl")
    assert [r for r in a if r["split"] != "test"] == [
        r for r in b if r["split"] != "test"
    ]
    assert [r["labels"] for r in a if r["split"] == "test"] != [
        r["labels"] for r in b if r["split"] == "test"
    ]


def test_rehashed_prediction_tamper_fails_semantic_audit(fitted):
    root, _, _ = fitted
    changed = root / "tampered"
    shutil.copytree(root / "fit", changed)
    rows = read_rows(changed / "predictions.jsonl")
    rows[0]["predictions"]["xgboost"] += 0.01
    (changed / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    manifest = read_json(changed / "manifest.json")
    manifest["files"]["predictions.jsonl"] = file_hash(changed / "predictions.jsonl")
    (changed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit_training(root / "dataset", changed)


def test_changed_model_is_refused_before_inference(fitted):
    root, _, _ = fitted
    changed = root / "bad_weights"
    shutil.copytree(root / "fit", changed)
    with (changed / "model.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="identity"):
        predict_saved(changed, [sample()])
