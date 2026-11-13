# FILE PURPOSE: Verify past-only process inputs, explicit time quantization and
# native threshold tracking on constructed fixtures, not real accuracy labels.
import copy
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vision.process_series import (
    INPUT_FIELDS,
    asof_index,
    candidate_samples,
    integer_log_seconds,
    primary_contour,
    trailing_reported_mean,
)
from check_crystalcv_process import read_stored_workbook
from audit_crystalcv_process import polygon_moments, reconstruct_smoothing


def series():
    times = np.arange(0, 10801, 300)
    observations = [
        dict(
            sample_id=f"image_{t}",
            filename_elapsed_s=int(t),
            integrity_valid=True,
            contour_area_px2=float((10 + t / 3600) ** 2),
        )
        for t in times
    ]
    settings = dict(
        history_s=3600,
        horizon_s=1800,
        target_lateness_s=30,
        minimum_history_images=4,
        reported_rate_availability="next_source_row",
    )
    seconds = np.arange(0, 10801, 5)
    rates = np.where(seconds < 3900, 2.0, 4.0)
    return observations, seconds, rates, settings


def test_declared_source_time_precision_restores_same_second():
    seconds = np.arange(0, 144777, 5)
    hours = np.array([float(f"{v / 3600:.6f}") for v in seconds])
    restored, error = integer_log_seconds(hours, 0.00180001)
    np.testing.assert_array_equal(restored, seconds)
    assert error <= 0.00180001
    # Six decimal hours can look slightly future/past until the documented
    # integer-second representation has been inverted.
    assert hours[1] * 3600 > seconds[1]
    assert asof_index(restored, 5, 6) == 1


@pytest.mark.parametrize(
    "hours,tolerance",
    [
        ([0, 0], 0.002),
        ([0, 0.001], 0.002),
        ([0, float("nan")], 0.002),
        ([0, 1], float("nan")),
        ([0, 1], 0.5),
    ],
)
def test_invalid_source_clock_is_not_silently_rounded(hours, tolerance):
    with pytest.raises(ValueError):
        integer_log_seconds(hours, tolerance)


def test_asof_rejects_future_rows_and_stale_state():
    seconds = np.array([5, 10, 20])
    assert asof_index(seconds, 4, 6) is None
    assert asof_index(seconds, 19, 6) is None
    assert asof_index(seconds, 16, 6) == 1
    assert asof_index(seconds, 20, 6) == 2


def test_reported_mean_integrates_irregular_past_intervals():
    seconds = np.array([0, 7, 19, 27])
    rates = np.array([1.0, 4.0, 2.0, 999.0])
    mean, first, last = trailing_reported_mean(seconds, rates, 25, 20)
    assert mean == pytest.approx((2 * 1 + 12 * 4 + 6 * 2) / 20)
    assert (first, last) == (0, 2)
    assert trailing_reported_mean(seconds, rates, 10, 20) is None


def test_candidate_inputs_ignore_future_images_and_actions():
    observations, seconds, rates, settings = series()
    before, _ = candidate_samples(observations, seconds, rates, settings, 6)
    reference = next(r for r in before if r["cutoff_s"] == 3900)
    mutated = copy.deepcopy(observations)
    for row in mutated:
        if row["filename_elapsed_s"] > 3900:
            row["contour_area_px2"] *= 4
    altered_rates = rates.copy()
    altered_rates[seconds >= 3900] = 1000  # Includes the same-cycle command.
    after, _ = candidate_samples(mutated, seconds, altered_rates, settings, 6)
    changed = next(r for r in after if r["cutoff_s"] == 3900)
    assert reference["inputs"] == changed["inputs"]
    assert changed["target_sqrt_area_px"] == 2 * reference["target_sqrt_area_px"]
    assert tuple(reference["inputs"]) == INPUT_FIELDS
    assert reference["inputs"]["trailing_sqrt_area_rate_px_per_h"] == pytest.approx(1)
    assert reference["inputs"]["trailing_reported_infusion_mean_ml_per_h"] == 2
    assert reference["inputs"]["reported_infusion_rate_ml_per_h"] == 2
    assert reference["last_input_log_time_s"] == 3895
    assert reference["last_input_availability_bound_s"] == 3900
    assert reference["maximum_input_time_s"] <= reference["cutoff_s"]
    assert reference["physical_command"] is None
    assert reference["independent_ground_truth"] is False


def test_candidate_requires_history_fresh_log_and_comparable_target():
    observations, seconds, rates, settings = series()
    observations[19]["integrity_valid"] = False  # target at cutoff 3900
    rows, rejected = candidate_samples(observations, seconds, rates, settings, 6)
    reasons = {r["cutoff_s"]: r["reason"] for r in rejected}
    assert reasons[0] == "insufficient_image_history"
    assert reasons[3600] == "missing_or_stale_past_process_state"
    assert reasons[3900] == "missing_comparable_future_proxy"
    assert reasons[5700] == "invalid_current_observation"
    assert reasons[10800] == "missing_comparable_future_proxy"
    assert all(r["cutoff_s"] != 3900 for r in rows)
    rows, rejected = candidate_samples(
        observations, seconds[seconds != 3900], rates[seconds != 3900], settings, 0
    )
    assert (
        next(r["reason"] for r in rejected if r["cutoff_s"] == 3900)
        == "missing_or_stale_past_process_state"
    )


def threshold_settings():
    return dict(
        lower_RGB=[50, 0, 0],
        upper_RGB=[255, 60, 255],
        analysis_circle_xy=[50, 50],
        analysis_radius_px=47,
        maximum_centroid_step_px=10,
    )


def test_native_threshold_tracker_keeps_nearby_primary_over_larger_distractor():
    image = np.full((100, 100, 3), 150, np.uint8)
    image[40:50, 40:50] = [0, 0, 180]  # BGR red, selected by source RGB bounds
    first, _ = primary_contour(image, threshold_settings())
    assert first["contour_area_px2"] == 81
    image[60:80, 60:80] = [0, 0, 180]
    tracked, mask = primary_contour(image, threshold_settings(), first["centroid_xy"])
    assert tracked["integrity_valid"]
    assert tracked["contour_area_px2"] == 81
    assert not tracked["largest_and_tracked_agree"]
    assert np.count_nonzero(mask) == 100
    assert not tracked["accepted_mask_ground_truth"]


def test_tracker_rejects_roi_contact_jump_and_missing_contour():
    settings = threshold_settings()
    image = np.full((100, 100, 3), 150, np.uint8)
    empty, _ = primary_contour(image, settings)
    assert empty["reason"] == "no_positive_contour"
    image[40:50, 40:50] = [0, 0, 180]
    jumped, _ = primary_contour(image, settings, [70, 70])
    assert not jumped["integrity_valid"]
    image[:] = 150
    image[45:55, 92:100] = [0, 0, 180]
    clipped, _ = primary_contour(image, settings)
    assert not clipped["integrity_valid"]


def test_workbook_reference_reader_refuses_formula_cache(tmp_path):
    source = tmp_path / "source.xlsx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>""",
        )
    with pytest.raises(ValueError, match="formula"):
        read_stored_workbook(source)


def test_independent_smoothing_preserves_quadratic_and_exposes_future_dependence():
    time = np.arange(101, dtype=float)
    values = 3 + 2 * time + 0.25 * time**2
    smooth = reconstruct_smoothing(values, 7)
    np.testing.assert_allclose(smooth, values, atol=1e-10, rtol=0)
    changed = values.copy()
    changed[51:] += 0.5
    altered = reconstruct_smoothing(changed, 7)
    assert abs(altered[50] - smooth[50]) > 0.1
    np.testing.assert_allclose(altered[:48], smooth[:48], atol=1e-10, rtol=0)


def test_independent_polygon_moments_are_orientation_invariant():
    points = [[2, 3], [6, 3], [6, 5], [2, 5]]
    for polygon in (points, points[::-1]):
        area, center = polygon_moments(polygon)
        assert area == 8
        np.testing.assert_array_equal(center, [4, 4])
