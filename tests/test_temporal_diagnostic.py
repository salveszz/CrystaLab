# FILE PURPOSE: Verify frozen forecast scoring cannot silently fit or hide training-family overlap.
"""Use artificial mask cohorts; actual optical/model execution has a separate evidence gate."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_area_forecast_pipeline import fixture_cohort
from models.area_forecast_run import train_forecast
from vision.temporal_diagnostic import evaluate_frozen_cohort
from vision.video import file_sha256


class FrozenTemporalDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cohort = fixture_cohort(cls.root)
        cls.fit = cls.root / "fit"
        cls.training = train_forecast(cls.cohort, cls.fit)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_existing_parameters_are_scored_without_refitting_and_overlap_is_explicit(
        self,
    ):
        before = file_sha256(self.fit / "model.json")
        with patch(
            "models.area_forecast_run.fit_calibration",
            side_effect=AssertionError("No new fit allowed"),
        ):
            result = evaluate_frozen_cohort(
                self.cohort, self.fit / "model.json", self.root / "frozen"
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["calibration"], self.training["calibration"])
        self.assertEqual(result["metrics"], self.training["metrics"])
        self.assertEqual(
            result["calibration_family_overlap"], ["fixture_0", "fixture_1"]
        )
        self.assertFalse(result["fitted_new_parameters"])
        self.assertFalse(result["can_authorize_actuation"])
        self.assertEqual(before, file_sha256(self.fit / "model.json"))
        self.assertEqual(before, file_sha256(self.root / "frozen/frozen_model.json"))
        self.assertFalse((self.root / "frozen/model.json").exists())

    def test_source_model_tampering_is_refused_before_creating_evaluation(self):
        import shutil

        edited = self.root / "edited_model"
        shutil.copytree(self.fit, edited)
        model = json.loads((edited / "model.json").read_text())
        model["calibration"]["coefficient"] += 0.1
        (edited / "model.json").write_text(json.dumps(model))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            evaluate_frozen_cohort(
                self.cohort, edited / "model.json", self.root / "blocked"
            )
        self.assertFalse((self.root / "blocked").exists())


if __name__ == "__main__":
    unittest.main()
