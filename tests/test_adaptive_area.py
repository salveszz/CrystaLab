# FILE PURPOSE: Check matured-only online learning, stream isolation and bounded policy changes.
from dataclasses import replace
import copy
import unittest

from models.adaptive_area import AdaptiveAreaConfig, AdaptiveAreaModel
from models.area_forecast import AreaEstimate


def estimate(frame, area, **change):
    return replace(
        AreaEstimate(
            run_id="fixture",
            clock_id="source_clock",
            view_id="fixed",
            frame_index=frame,
            observed_at_s=frame * 0.25,
            received_at_s=frame * 0.25,
            image_shape=(256, 256),
            area_px2=area,
            valid=area is not None,
            requires_inspection=area is None,
            evidence_kind="fixture",
            mask_sha256=None if area is None else f"{frame + 1:064x}",
        ),
        **change,
    )


def reference(gain=0.1):
    return dict(
        schema="crystalab_adaptive_area_reference_v1",
        positive_relative_gain_reference=gain,
        initial_error_ratio=0.0,
    )


class AdaptiveAreaTests(unittest.TestCase):
    def test_coefficients_change_only_when_observed_target_matures(self):
        model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
        first = model.update(estimate(0, 100))
        self.assertIsNone(first["forecast"])
        second = model.update(estimate(1, 110))
        saved = copy.deepcopy(second)
        self.assertEqual(second["parameter_after"]["coefficient"], 1)
        self.assertEqual(second["forecast"]["future_area_px2"], 120)
        third = model.update(estimate(2, 121))
        x = 10 / 11
        y = 1
        expected = (0.95 * 3 + x * y) / (0.95 * 3 + x * x)
        self.assertAlmostEqual(third["parameter_after"]["coefficient"], expected)
        self.assertEqual(third["matured_update"]["target_frame_index"], 2)
        self.assertEqual(second, saved)
        self.assertAlmostEqual(
            third["forecast"]["future_area_px2"], 121 + expected * 11
        )

    def test_identical_prefix_is_independent_of_future_values(self):
        records = []
        for values in ([100, 110, 121, 130], [100, 110, 121, 600]):
            model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
            records.append([model.update(estimate(i, a)) for i, a in enumerate(values)])
        self.assertEqual(records[0][:3], records[1][:3])
        self.assertNotEqual(records[0][3], records[1][3])

    def test_missing_inspection_and_gap_cancel_pending_learning(self):
        for failed in (
            estimate(2, None),
            estimate(2, 121, requires_inspection=True),
            estimate(4, 121),
        ):
            with self.subTest(failed=failed):
                model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
                model.update(estimate(0, 100))
                model.update(estimate(1, 110))
                state = model.update(failed)
                self.assertIsNone(state["matured_update"])
                self.assertEqual(state["parameter_after"]["matured_updates"], 0)
                self.assertEqual(
                    state["cancelled_update"], "unusable_or_unaligned_target"
                )

    def test_context_change_resets_parameters_and_late_receipt_is_not_a_forecast(self):
        model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
        for i, a in enumerate([100, 110, 121]):
            model.update(estimate(i, a))
        state = model.update(estimate(0, 300, run_id="another", clock_id="other"))
        self.assertTrue(state["context_parameters_reset"])
        self.assertEqual(state["parameter_after"]["matured_updates"], 0)
        self.assertEqual(state["parameter_after"]["coefficient"], 1)
        model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
        model.update(estimate(0, 100))
        late = model.update(estimate(1, 110, received_at_s=0.55))
        self.assertTrue(late["trend"]["history_ready"])
        self.assertIsNone(late["forecast"])

    def test_recommendation_changes_with_forecast_and_never_authorizes_hardware(self):
        model = AdaptiveAreaModel(
            AdaptiveAreaConfig(minimum_matured_updates=1), reference()
        )
        for i, a in enumerate([100, 110, 121]):
            model.update(estimate(i, a))
        first = model.recommend(0.5, base_pause_model_s=60, model_step_s=1)
        model.update(estimate(3, 122))
        second = model.recommend(0.75, base_pause_model_s=60, model_step_s=1)
        self.assertTrue(first["eligible"])
        self.assertTrue(second["eligible"])
        self.assertGreater(first["pause_model_s"], second["pause_model_s"])
        self.assertEqual(second["pause_model_s"], 15)
        self.assertFalse(second["can_authorize_hardware"])
        expired = model.recommend(1.0, base_pause_model_s=60, model_step_s=1)
        self.assertFalse(expired["eligible"])
        self.assertEqual(expired["reason"], "forecast_target_elapsed")

    def test_invalid_config_and_non_native_inputs_are_refused(self):
        for key, value in [
            ("forgetting_factor", 0),
            ("error_ema_alpha", 2),
            ("minimum_matured_updates", True),
            ("prior_precision", 0),
            ("minimum_pause_model_s", 61),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                AdaptiveAreaConfig(**{key: value})
        model = AdaptiveAreaModel(AdaptiveAreaConfig(), reference())
        with self.assertRaises(ValueError):
            model.update({"future_truth": 100})
