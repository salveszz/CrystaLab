# FILE PURPOSE: Verify source-bound shadow evidence and the rule/PID simulated control chain.
# Analytic supplied-mask fixtures test causality and integration, not learned-model
# performance. Real-data eligibility and forged phase progress are explicitly rejected.

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_candidate_fixture import generate_candidate_fixture
from controllers.audit import audit_control_benchmark
from controllers.benchmark import sha256
from controllers.shadow import CandidateEvidence, run_shadow
from controllers.supervised_demo import run_demo, validate_demo
from controllers.supervision import value_sha256
from vision.candidates import CandidateConfig

ROOT = Path(__file__).resolve().parents[1]


class SupervisoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        generate_candidate_fixture(
            cls.root / "fixture", candidate_config=CandidateConfig()
        )
        cls.analysis = cls.root / "fixture_candidates"
        cls.policy = ROOT / "configs/event_rule_policy.json"
        cls.shadow = run_shadow(cls.analysis, cls.policy, cls.root / "shadow")
        config = json.loads(
            (ROOT / "configs/supervised_control_fixture.json").read_text()
        )
        config["clock_mapping"]["expected_source_video_sha256"] = json.loads(
            (cls.analysis / "source_replay_manifest.json").read_text()
        )["input"]["sha256"]
        cls.config = cls.root / "demo.json"
        cls.config.write_text(json.dumps(config, indent=2))
        cls.demo = run_demo(cls.analysis, cls.policy, cls.config, cls.root / "demo")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_shadow_keeps_all_inspection_load_and_localized_evidence_without_commands(
        self,
    ):
        self.assertEqual(self.shadow["processed_frames"], 48)
        self.assertEqual(
            self.shadow["proposals_by_kind"],
            dict(
                candidate_inspection=2,
                ramp_review=1,
                view_inspection=1,
                inclusion_alert=1,
            ),
        )
        self.assertEqual(self.shadow["numerical_actuator_commands"], 0)
        self.assertFalse(self.shadow["temperature_feedback_available"])
        output = self.root / "shadow"
        proposals = [
            json.loads(line)
            for line in (output / "proposals.jsonl").read_text().splitlines()
        ]
        inclusion = next(
            p for p in proposals if p["proposal"]["kind"] == "inclusion_alert"
        )
        self.assertEqual(
            inclusion["proposal_sha256"], value_sha256(inclusion["proposal"])
        )
        self.assertEqual(
            sha256(
                output / inclusion["artifact_evidence"]["first_detection_mask_path"]
            ),
            inclusion["artifact_evidence"]["source_mask_sha256"],
        )
        self.assertEqual(
            cv2.imread(
                str(output / inclusion["artifact_evidence"]["image_path"])
            ).shape[:2],
            (288, 320),
        )
        for name, digest in self.shadow["files"].items():
            self.assertEqual(sha256(output / name), digest)

    def test_three_variants_and_exact_sixty_second_pause_audit_without_quality_claim(
        self,
    ):
        audit = audit_control_benchmark(self.root / "demo")
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(len(audit["runs"]), 3)
        self.assertEqual(sum(r["samples"] for r in audit["runs"]), 1803)
        self.assertEqual(len(self.demo["scripted_executions"]), 1)
        action = self.demo["scripted_executions"][0]
        self.assertAlmostEqual(action["model_decision_time_s"], 130)
        self.assertEqual(action["pause_duration_model_s"], 60)
        self.assertFalse(action["operator_reviewed"])
        supervised = next(
            r for r in self.demo["runs"] if r["method"] == "rule_supervised"
        )
        self.assertEqual(supervised["recipe_phase_lag_s"], 60)
        self.assertEqual(supervised["final_recipe_phase_s"], 540)
        self.assertFalse(self.demo["adaptive_policy_implemented"])
        self.assertFalse(self.demo["physical_hardware_contacted"])

    def test_resealed_future_phase_or_clock_mapping_fails_semantic_audit(self):
        edited = self.root / "edited_demo"
        shutil.copytree(self.root / "demo", edited)
        manifest = json.loads((edited / "manifest.json").read_text())
        report = next(r for r in manifest["runs"] if r["method"] == "rule_supervised")
        path = edited / report["records_path"]
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[150]["supervision"]["recipe_phase_s"] = 150
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["files"][report["records_path"]] = sha256(path)
        (edited / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "paused recipe phase"):
            audit_control_benchmark(edited)

    def test_candidate_tampering_final_test_and_mapping_mismatch_fail_before_outputs(
        self,
    ):
        edited = self.root / "edited_analysis"
        shutil.copytree(self.analysis, edited)
        path = edited / "candidate_events.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            run_shadow(edited, self.policy, self.root / "bad_shadow")
        self.assertFalse((self.root / "bad_shadow").exists())
        source = CandidateEvidence(self.analysis)
        config = json.loads(self.config.read_text())
        config["clock_mapping"]["expected_source_video_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source identity"):
            validate_demo(config, source)
        source.manifest["source_info"]["evidence_kind"] = "real"
        config = json.loads(self.config.read_text())
        with self.assertRaisesRegex(ValueError, "proposal-only"):
            validate_demo(config, source)
        manifest = json.loads((edited / "manifest.json").read_text())
        manifest["source_info"]["split"] = "test"
        (edited / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "final test"):
            CandidateEvidence(edited)
