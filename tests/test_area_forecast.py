# FILE PURPOSE: Verify causal area forecasts, missing-history coverage and training-only calibration.
# Analytic pixel areas test source clocks and reset boundaries without assuming
# optical visibility, physical growth units, action response or reviewed real masks.

import copy
from dataclasses import replace
import unittest

from models.area_forecast import (
    AreaEstimate,
    AreaForecastConfig,
    AreaTrend,
    fit_calibration,
    predictions,
)


def estimate(time=0, area=100, frame=0, **changes):
    fields = dict(
        run_id="run",
        clock_id="source-clock",
        view_id="view",
        frame_index=frame,
        observed_at_s=time,
        received_at_s=time,
        image_shape=(100, 100),
        area_px2=area,
        valid=True,
        requires_inspection=False,
        evidence_kind="fixture",
        mask_sha256="a" * 64,
    )
    fields.update(changes)
    return AreaEstimate(**fields)


def sample(family, split, rate=80, label=0.2):
    return dict(
        sample_id=family + ":window",
        family_id=family,
        split=split,
        trend=dict(
            history_ready=True,
            features=dict(
                current_area_px2=100.0,
                recent_rate_px2_per_source_s=rate,
                history_duration_source_s=0.25,
                horizon_source_s=0.25,
                frame_pixels=10000,
            ),
        ),
        labels=dict(relative_proxy_gain=label),
    )


class AreaForecastTests(unittest.TestCase):
    def test_startup_preserves_baselines_without_inventing_zero_recent_rate(self):
        result = AreaTrend(AreaForecastConfig()).update(estimate())
        self.assertFalse(result["history_ready"])
        self.assertIsNone(result["features"]["recent_rate_px2_per_source_s"])
        output = predictions(result["features"], coefficient=1.0, mean_fraction=0.1)
        self.assertEqual(output["persistence"]["future_area_px2"], 100)
        self.assertAlmostEqual(output["train_mean_fraction"]["future_area_px2"], 110)
        self.assertIsNone(output["recent_rate"])
        self.assertIsNone(output["calibrated_recent_rate"])

    def test_linear_history_forecasts_actual_source_time_and_allows_shrinking_projection(
        self,
    ):
        trend = AreaTrend(AreaForecastConfig())
        trend.update(estimate())
        result = trend.update(estimate(0.25, 120, 1))
        self.assertEqual(result["features"]["recent_rate_px2_per_source_s"], 80)
        output = predictions(result["features"], coefficient=0.5, mean_fraction=0.1)
        self.assertEqual(output["recent_rate"]["future_area_px2"], 140)
        self.assertEqual(output["calibrated_recent_rate"]["future_area_px2"], 130)
        shrinking = trend.update(estimate(0.5, 110, 2))
        output = predictions(shrinking["features"], coefficient=1, mean_fraction=0)
        self.assertEqual(output["recent_rate"]["future_area_px2"], 100)

    def test_invalid_inspection_staleness_and_long_gap_reset_history(self):
        for middle, now, reason in [
            (
                estimate(0.25, None, 1, valid=False, mask_sha256=None),
                0.25,
                "invalid_measurement",
            ),
            (
                estimate(0.25, 110, 1, requires_inspection=True),
                0.25,
                "inspection_required",
            ),
            (estimate(0.25, 110, 1), 1.0, "stale_measurement"),
        ]:
            with self.subTest(reason=reason):
                trend = AreaTrend(AreaForecastConfig())
                trend.update(estimate())
                result = trend.update(middle, now_s=now)
                self.assertEqual(result["reason"], reason)
                self.assertTrue(
                    all(
                        v is None
                        for v in predictions(
                            result["features"], coefficient=1, mean_fraction=0
                        ).values()
                    )
                )
                next_time = max(now + 0.25, 0.5)
                self.assertFalse(
                    trend.update(estimate(next_time, 120, 2))["history_ready"]
                )
        trend = AreaTrend(AreaForecastConfig())
        trend.update(estimate())
        gap = trend.update(estimate(1, 120, 1))
        self.assertEqual(gap["reset_reason"], "observation_gap")
        self.assertFalse(gap["history_ready"])

    def test_run_view_and_shape_contexts_cannot_mix_or_reverse_same_clock(self):
        for change in [
            dict(run_id="other"),
            dict(view_id="rotated"),
            dict(image_shape=(200, 200)),
        ]:
            with self.subTest(change=change):
                trend = AreaTrend(AreaForecastConfig())
                trend.update(estimate())
                result = trend.update(estimate(0.25, 120, 1, **change))
                self.assertEqual(result["reset_reason"], "stream_context_changed")
                self.assertFalse(result["history_ready"])
        trend = AreaTrend(AreaForecastConfig())
        trend.update(estimate(0.5, 110, 2))
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            trend.update(estimate(0.25, 120, 1, view_id="other"))
        new = trend.update(estimate(0, 100, 0, run_id="new", clock_id="new-clock"))
        self.assertFalse(new["history_ready"])

    def test_future_data_does_not_change_saved_prefix_features(self):
        trend = AreaTrend(AreaForecastConfig())
        trend.update(estimate())
        prefix = trend.update(estimate(0.25, 120, 1))
        saved = copy.deepcopy(prefix)
        trend.update(estimate(0.5, 500, 2))
        self.assertEqual(prefix, saved)
        with self.assertRaisesRegex(ValueError, "not been received"):
            AreaTrend(AreaForecastConfig()).update(estimate(0.5, 200, 2), now_s=0.25)

    def test_training_calibration_ignores_validation_labels_and_uses_no_hidden_features(
        self,
    ):
        data = [
            sample("train-a", "train"),
            sample("train-b", "train"),
            sample("val", "validation", label=1000),
        ]
        first = fit_calibration(data, AreaForecastConfig())
        data[-1]["labels"]["relative_proxy_gain"] = -1000
        second = fit_calibration(data, AreaForecastConfig())
        self.assertEqual(first, second)
        self.assertEqual(first["training_family_ids"], ["train-a", "train-b"])
        self.assertAlmostEqual(first["coefficient"], 0.08 / (0.08 + 0.0001))
        self.assertEqual(first["mean_fraction"], 0.2)
        data[0]["trend"]["features"]["hidden_concentration"] = 1.4
        with self.assertRaisesRegex(ValueError, "whitelist"):
            fit_calibration(data, AreaForecastConfig())
        with self.assertRaisesRegex(ValueError, "at least two"):
            fit_calibration([sample("only", "train")], AreaForecastConfig())
        with self.assertRaisesRegex(ValueError, "rate signal"):
            fit_calibration(
                [sample("a", "train", rate=0), sample("b", "train", rate=0)],
                AreaForecastConfig(),
            )

    def test_image_bounds_and_bad_records_fail_explicitly(self):
        trend = AreaTrend(AreaForecastConfig())
        trend.update(estimate(area=9000))
        result = trend.update(estimate(0.25, 9990, 1))
        output = predictions(result["features"], coefficient=3, mean_fraction=1)
        self.assertEqual(output["calibrated_recent_rate"]["future_area_px2"], 10000)
        self.assertTrue(output["calibrated_recent_rate"]["clipped_to_image"])
        for change in [
            dict(area_px2=0),
            dict(area_px2=10001),
            dict(mask_sha256=None),
            dict(valid=True, area_px2=None),
            dict(received_at_s=-1),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                estimate(**change)
        for change in [
            dict(horizon_source_s=0),
            dict(ridge=0),
            dict(coefficient_min=4),
            dict(maximum_age_source_s=True),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(AreaForecastConfig(), **change)
