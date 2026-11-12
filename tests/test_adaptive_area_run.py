# FILE PURPOSE: Verify training-only normalization, grouped evaluation and reference integrity.
import json
from pathlib import Path
import tempfile
import unittest

from test_area_forecast_pipeline import fixture_cohort
from models.adaptive_area import AdaptiveAreaConfig
from dataclasses import asdict
from models.adaptive_area_run import fit_reference, load_reference, evaluate_adaptive
from models.area_forecast_run import train_forecast


class AdaptiveAreaRunTests(unittest.TestCase):
    def test_only_training_observations_set_reference_and_validation_scores_remain_separate(
        self,
    ):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            cohort = fixture_cohort(root)
            config = root / "adaptive.json"
            config.write_text(json.dumps(asdict(AdaptiveAreaConfig())))
            ref = fit_reference(cohort, config, root / "reference")
            self.assertEqual(
                {r["family_id"] for r in ref["training_families"]},
                {"fixture_0", "fixture_1"},
            )
            self.assertEqual(sum(r["frames"] for r in ref["training_families"]), 6)
            self.assertEqual(ref["gain_samples"], 4)
            self.assertEqual(ref["error_samples"], 2)
            train_forecast(cohort, root / "baseline")
            result = evaluate_adaptive(
                cohort,
                root / "reference",
                root / "baseline/model.json",
                root / "evaluation",
            )
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["final_test_scored"])
            metrics = result["metrics"]["validation"]["adaptive_recent_rate"]
            self.assertEqual(metrics["labelled_windows"], 2)
            self.assertEqual(metrics["available_windows"], 1)
            self.assertEqual(metrics["common_history_support"]["samples"], 1)
            path = root / "reference/reference.json"
            value = json.loads(path.read_text())
            value["positive_relative_gain_reference"] *= 2
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_reference(root / "reference")
