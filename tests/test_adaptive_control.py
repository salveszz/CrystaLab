# FILE PURPOSE: Verify actual adaptive state/action/PID integration using imperfect generated masks.
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_candidate_fixture
from controllers.adaptive_demo import run_adaptive_demo
from controllers.audit import audit_control_benchmark
from controllers.benchmark import sha256
from models.adaptive_area import AdaptiveAreaConfig
from vision.candidates import CandidateConfig

ROOT = Path(__file__).resolve().parents[1]


class AdaptiveControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        original = make_candidate_fixture.geometry

        def geometry(i):
            values = original(i)
            if not values:
                return values
            seed = np.zeros_like(values[0][2])
            seed[60:100, 40 : 90 + i + i * i // 20] = 255
            return [("seed", 0, seed), *values[1:]]

        with patch.object(make_candidate_fixture, "geometry", geometry):
            make_candidate_fixture.generate_candidate_fixture(
                cls.root / "fixture", candidate_config=CandidateConfig()
            )
        cls.analysis = cls.root / "fixture_candidates"
        cls.policy = ROOT / "configs/event_rule_policy.json"
        config = json.loads(
            (ROOT / "configs/supervised_control_fixture.json").read_text()
        )
        config["clock_mapping"]["expected_source_video_sha256"] = json.loads(
            (cls.analysis / "source_replay_manifest.json").read_text()
        )["input"]["sha256"]
        cls.config = cls.root / "config.json"
        cls.config.write_text(json.dumps(config))
        cls.reference = cls.root / "reference"
        cls.reference.mkdir()
        adaptive = AdaptiveAreaConfig(horizon_source_s=0.1)
        reference = dict(
            schema="crystalab_adaptive_area_reference_v1",
            positive_relative_gain_reference=0.05,
            initial_error_ratio=0.0,
            configuration=asdict(adaptive),
            training_families=[],
            scope="Explicit analytic fixture normalization; not a fitted real/training-data claim",
        )
        (cls.reference / "reference.json").write_text(json.dumps(reference))
        (cls.reference / "policy.json").write_text(json.dumps(asdict(adaptive)))
        (cls.reference / "manifest.json").write_text(
            json.dumps(
                dict(
                    schema="crystalab_adaptive_area_reference_run_v1",
                    status="complete",
                    files={
                        p.name: sha256(p)
                        for p in cls.reference.iterdir()
                        if p.is_file()
                    },
                )
            )
        )
        cls.output = cls.root / "comparison"
        cls.result = run_adaptive_demo(
            cls.analysis, cls.policy, cls.config, cls.reference, cls.output
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_four_variants_changing_parameters_and_bounded_forecast_scaled_pause(self):
        self.assertEqual(len(self.result["runs"]), 4)
        self.assertTrue(self.result["adaptive_policy_implemented"])
        self.assertFalse(self.result["video_action_responsive"])
        self.assertEqual(self.result["operator_reviews_created"], 0)
        self.assertEqual(len(self.result["adaptive_executions"]), 1)
        action = self.result["adaptive_executions"][0]
        self.assertGreaterEqual(action["pause_duration_model_s"], 15)
        self.assertLess(action["pause_duration_model_s"], 60)
        self.assertFalse(action["operator_reviewed"])
        self.assertFalse(action["physical_hardware_authorized"])
        states = [
            json.loads(v)
            for v in (self.output / "adaptive_supervised_area_states.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertTrue(any(s["parameter_after"]["coefficient"] != 1 for s in states))
        self.assertEqual(audit_control_benchmark(self.output)["status"], "pass")
        report = next(
            r for r in self.result["runs"] if r["method"] == "adaptive_supervised"
        )
        rows = [
            json.loads(v)
            for v in (self.output / report["records_path"]).read_text().splitlines()
        ]
        self.assertFalse(rows[-1]["supervision"]["forecast_available"])

    def test_resealed_future_state_or_wrong_adaptive_duration_fails_audit(self):
        edited = self.root / "edited"
        shutil.copytree(self.output, edited)
        p = edited / "adaptive_supervised_supervision.jsonl"
        rows = [json.loads(v) for v in p.read_text().splitlines()]
        states = [
            json.loads(v)
            for v in (edited / "adaptive_supervised_area_states.jsonl")
            .read_text()
            .splitlines()
        ]
        rows[1]["recommendation"]["state_sha256"] = states[-1]["state_sha256"]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        m = json.loads((edited / "manifest.json").read_text())
        m["files"][p.name] = sha256(p)
        (edited / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "future or wrong state"):
            audit_control_benchmark(edited)
        edited = self.root / "wrong_duration"
        shutil.copytree(self.output, edited)
        p = edited / "manifest.json"
        m = json.loads(p.read_text())
        m["adaptive_executions"][0]["pause_duration_model_s"] -= 1
        p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            audit_control_benchmark(edited)

    def test_mapping_mismatch_is_rejected_before_output(self):
        config = json.loads(self.config.read_text())
        config["clock_mapping"]["expected_source_video_sha256"] = "0" * 64
        path = self.root / "wrong.json"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "source identity"):
            run_adaptive_demo(
                self.analysis, self.policy, path, self.reference, self.root / "blocked"
            )
        self.assertFalse((self.root / "blocked").exists())
