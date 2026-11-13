# FILE PURPOSE: Check temporal support separation, frozen prefix preprocessing
# and forecast input boundaries on constructed data, not real model accuracy.
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.observable_forecast import (
    FEATURES,
    disjoint_subset,
    errors,
    fit_ridge,
    input_matrix,
    partition_samples,
    predict_gains,
)


def inputs(size, rate=1.0, infusion=2.0):
    return dict(zip(FEATURES, [size**2, size, rate, infusion, infusion]))


def cohort_fixture():
    observations = [
        dict(sample_id=f"frame_{t}", filename_elapsed_s=t) for t in range(0, 20401, 300)
    ]
    seconds = list(range(0, 20401, 5))
    samples = []
    for t in [3900, 4200, 10500, 10800, 17100, 17400]:
        samples.append(
            dict(
                sample_id=f"frame_{t}",
                cutoff_s=t,
                inputs=inputs(10 + t / 3600),
                input_image_sample_ids=[
                    f"frame_{p}" for p in range(t - 3600, t + 1, 300)
                ],
                input_log_row_range=[(t - 3600) // 5 - 1, t // 5 - 1],
                maximum_input_time_s=t,
                last_input_availability_bound_s=t,
                target_sample_id=f"frame_{t + 1800}",
                target_time_s=t + 1800,
                target_sqrt_area_px=10 + (t + 1800) / 3600,
            )
        )
    cohorts = [
        dict(name="train_prefix", minimum_cutoff_s=3900, maximum_cutoff_s=4200),
        dict(name="later_block_1", minimum_cutoff_s=10500, maximum_cutoff_s=10800),
        dict(name="later_block_2", minimum_cutoff_s=17100, maximum_cutoff_s=17400),
    ]
    return samples, observations, seconds, cohorts


def test_partition_separates_full_supports_and_preserves_purge():
    samples, observations, seconds, cohorts = cohort_fixture()
    cohorts[1]["maximum_cutoff_s"] = 10500
    grouped, spans = partition_samples(samples, observations, seconds, cohorts)
    assert [r["sample_id"] for r in grouped["purged"]] == ["frame_10800"]
    assert grouped["train_prefix"][0]["support_start_s"] == 295
    assert grouped["later_block_1"][0]["support_start_s"] == 6895
    assert len(spans) == len(samples)


def test_shared_log_rows_cannot_hide_behind_disjoint_cutoff_times():
    samples, observations, seconds, cohorts = cohort_fixture()
    samples[2]["input_log_row_range"][0] = samples[0]["input_log_row_range"][0]
    with pytest.raises(ValueError, match="share source"):
        partition_samples(samples, observations, seconds, cohorts)


def test_future_input_and_same_cycle_availability_are_rejected():
    samples, observations, seconds, cohorts = cohort_fixture()
    bad = copy.deepcopy(samples)
    bad[0]["input_image_sample_ids"].append(bad[0]["target_sample_id"])
    with pytest.raises(ValueError, match="Future or unavailable"):
        partition_samples(bad, observations, seconds, cohorts)
    bad = copy.deepcopy(samples)
    bad[0]["input_log_row_range"][1] += 1
    with pytest.raises(ValueError, match="availability"):
        partition_samples(bad, observations, seconds, cohorts)


def test_support_subset_skips_nearby_correlated_windows():
    rows = [
        dict(sample_id="a", support_start_s=0, support_end_s=10),
        dict(sample_id="b", support_start_s=5, support_end_s=15),
        dict(sample_id="c", support_start_s=10, support_end_s=20),
        dict(sample_id="d", support_start_s=11, support_end_s=21),
    ]
    assert disjoint_subset(rows) == ["a", "d"]


def test_fit_uses_training_prefix_statistics_and_no_later_values():
    train = [inputs(2.0), inputs(4.0), inputs(6.0)]
    model = fit_ridge(train, [0.3, 0.5, 0.8], FEATURES[:3], 1.0)
    np.testing.assert_array_equal(model["mean"], [56 / 3, 4.0, 1.0])
    assert model["scale"][-1] == 1
    before = copy.deepcopy(model)
    predict_gains(model, [inputs(1000.0, infusion=10000.0)])
    assert model == before
    assert model["training_samples"] == 3
    assert not model["model_promoted"] and model["physical_command"] is None


def test_vision_ablation_ignores_process_values():
    model = fit_ridge(
        [inputs(2), inputs(4), inputs(6)], [0.1, 0.4, 0.9], FEATURES[:3], 1.0
    )
    a, b = inputs(8, infusion=2), inputs(8, infusion=100)
    np.testing.assert_array_equal(predict_gains(model, [a]), predict_gains(model, [b]))


def test_unvarying_training_feature_cannot_learn_an_action_effect():
    model = fit_ridge([inputs(2), inputs(4)], [0.2, 0.4], [FEATURES[3]], 1.0)
    assert model["coefficients"] == [0.0]
    assert predict_gains(model, [inputs(3, infusion=2000)])[0] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "extra", ["target_sqrt_area_px", "actual_horizon_s", "cutoff_s"]
)
def test_prediction_rejects_non_whitelisted_metadata(extra):
    row = inputs(2)
    row[extra] = 999.0
    with pytest.raises(ValueError, match="whitelist"):
        input_matrix([row], FEATURES)


def test_metrics_retain_signed_error_and_near_zero_gains():
    result = errors([0, 0.001, 1], [1, 1.001, 2])
    assert result["MAE_px"] == pytest.approx(1)
    assert result["signed_error_px"] == pytest.approx(1)
    assert result["RMSE_px"] == pytest.approx(1)
    assert result["p95_absolute_error_px"] == pytest.approx(1)
