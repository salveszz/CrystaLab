# FILE PURPOSE: Verify displayed causal forecast provenance and explicit suggested/manual pauses.
# Generated fixtures and analytic normalization never represent human or real training labels.
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_candidate_fixture
from audit_operator_forecast import audit_session
from test_operator_console import prepare_operator_fixture, review_payload
from controllers.audit import audit_control_benchmark
from controllers.benchmark import sha256
from controllers.forecast_context_audit import audit_forecast_context
from controllers.supervision import value_sha256
from dashboard.operator import OperatorStore
from dashboard.review import ReviewConflict
from models.adaptive_area import AdaptiveAreaConfig


def prepare_forecast_fixture(root):
    reference = root / "reference"
    reference.mkdir()
    config = asdict(AdaptiveAreaConfig(horizon_source_s=0.1))
    (reference / "policy.json").write_text(json.dumps(config))
    (reference / "reference.json").write_text(
        json.dumps(
            dict(
                schema="crystalab_adaptive_area_reference_v1",
                configuration=config,
                positive_relative_gain_reference=0.05,
                initial_error_ratio=0.0,
                training_families=[],
                scope="Analytic generated-fixture normalization; no fitted real-data claim",
            )
        )
    )
    (reference / "manifest.json").write_text(
        json.dumps(
            dict(
                schema="crystalab_adaptive_area_reference_run_v1",
                status="complete",
                files={p.name: sha256(p) for p in reference.iterdir()},
            )
        )
    )
    original = make_candidate_fixture.geometry

    def geometry(index):
        values = original(index)
        if not values:
            return values
        seed = np.zeros_like(values[0][2])
        seed[60:100, 40 : 90 + index + index * index // 20] = 255
        return [("seed", 0, seed), *values[1:]]

    with patch.object(make_candidate_fixture, "geometry", geometry):
        return prepare_operator_fixture(root, adaptive_reference=reference)


class OperatorForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = prepare_forecast_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.session = Path(tempfile.mkdtemp(dir=self.root)) / "session"
        shutil.copytree(self.prepared, self.session)
        self.store = OperatorStore(self.session)

    def payload(self, frame=13, *, basis="adaptive_suggestion"):
        context = self.store.frame(frame)["forecast_context"]
        payload = review_payload(self.store, frame=frame)
        payload.update(
            forecast_context_sha256=context["context_sha256"],
            pause_basis=basis,
            pause_duration_model_s=context["recommendation"]["pause_model_s"],
        )
        return payload

    def test_source_frame_forecast_and_withheld_unknown_evidence_are_reconstructed(
        self,
    ):
        startup = self.store.frame(0)["forecast_context"]
        self.assertIsNone(startup["state"]["forecast"])
        self.assertFalse(startup["recommendation"]["eligible"])
        context = self.store.frame(13)["forecast_context"]
        self.assertEqual(context["state"]["source_frame_index"], 13)
        self.assertNotEqual(context["state"]["coefficient"], 1)
        self.assertTrue(context["recommendation"]["eligible"])
        self.assertGreaterEqual(context["recommendation"]["pause_model_s"], 15)
        self.assertLess(context["recommendation"]["pause_model_s"], 60)
        unknown = self.store.frame(23)["forecast_context"]
        self.assertFalse(unknown["recommendation"]["eligible"])
        self.assertFalse(unknown["recommendation"]["forecast_available"])
        self.assertEqual(
            audit_forecast_context(self.session / "forecast", self.store.source.root)[
                "status"
            ],
            "pass",
        )

    def test_suggested_pause_review_and_manual_resume_preserve_exact_context(self):
        payload = self.payload()
        duration = payload["pause_duration_model_s"]
        saved = self.store.review(payload)
        self.assertEqual(saved["reviews"][0]["pause_basis"], "adaptive_suggestion")
        self.assertEqual(saved["runs"][2]["recipe_phase_lag_s"], duration)
        first_history, _ = self.store.history()
        first = self.session / first_history["latest_run"]
        resume = review_payload(self.store, frame=14, action="resume_recipe")
        resume.update(
            forecast_context_sha256=self.store.frame(14)["forecast_context"][
                "context_sha256"
            ],
            pause_review_id=saved["reviews"][0]["review_id"],
        )
        saved = self.store.review(resume)
        self.assertEqual(saved["runs"][2]["recipe_phase_lag_s"], 10)
        history, _ = self.store.history()
        for path in (first, self.session / history["latest_run"]):
            self.assertEqual(audit_control_benchmark(path)["status"], "pass")
        self.assertTrue(
            all(r["reviewer_kind"] == "automated_fixture_qa" for r in saved["reviews"])
        )
        audit = audit_session(self.session)
        self.assertEqual(audit["preserved_results"], 3)
        self.assertEqual(audit["human_reviews"], 0)

    def test_stale_context_or_changed_suggestion_cannot_silently_be_recorded(self):
        for change in (
            dict(forecast_context_sha256=None),
            dict(
                forecast_context_sha256=self.store.frame(14)["forecast_context"][
                    "context_sha256"
                ]
            ),
            dict(pause_duration_model_s=60),
        ):
            with self.subTest(change=change):
                payload = self.payload()
                payload.update(change)
                with self.assertRaises((ValueError, ReviewConflict)):
                    self.store.review(payload)
                self.assertEqual(self.store.summary()["reviews"], [])
        payload = self.payload(basis="manual")
        payload["pause_duration_model_s"] = 20
        saved = self.store.review(payload)
        self.assertEqual(saved["runs"][2]["recipe_phase_lag_s"], 20)
        self.assertEqual(saved["reviews"][0]["pause_basis"], "manual")

    def test_resealed_future_context_and_changed_context_artifact_are_refused(self):
        copied = self.session.parent / "future_context"
        shutil.copytree(self.session / "forecast", copied)
        path = copied / "contexts.json"
        contexts = json.loads(path.read_text())
        contexts[13]["state"] = contexts[14]["state"]
        contexts[13]["context_sha256"] = value_sha256(
            {k: v for k, v in contexts[13].items() if k != "context_sha256"}
        )
        path.write_text(json.dumps(contexts))
        manifest = json.loads((copied / "manifest.json").read_text())
        manifest["files"]["contexts.json"] = sha256(path)
        (copied / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "latest received"):
            audit_forecast_context(copied, self.store.source.root)
        path = self.session / "forecast/contexts.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.store.frame(13)
