# FILE PURPOSE: Verify saved control experiments, reproducibility and independent semantic audit.
# Temporary software fixtures exercise missing/held sensors and scripted overrides;
# edited artifacts cannot become accepted evidence merely by recomputing their hash.

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from controllers.audit import audit_control_benchmark
from controllers.benchmark import run_benchmark, run_scenario, sha256, validate_config

ROOT = Path(__file__).resolve().parents[1]


class ControlBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.run_directory = cls.root / "run"
        cls.config = json.loads(
            (ROOT / "configs/temperature_control_baseline.json").read_text()
        )
        cls.manifest = run_benchmark(
            ROOT / "configs/temperature_control_baseline.json", cls.run_directory
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_declared_comparisons_audit_with_realistic_pending_and_fault_states(
        self,
    ):
        audit = audit_control_benchmark(self.run_directory)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(len(audit["runs"]), 8)
        self.assertEqual(sum(r["samples"] for r in audit["runs"]), 4808)
        self.assertTrue(all(r["pending_commands"] > 0 for r in audit["runs"]))
        fault = next(
            r for r in self.manifest["runs"] if r["run_id"] == "feedback_faults:pid"
        )
        self.assertEqual(fault["metrics"]["scripted_manual_steps"], 20)
        self.assertEqual(fault["metrics"]["invalid_observation_steps"], 22)
        self.assertGreater(fault["metrics"]["fallback_steps"], 22)
        saturated = next(
            r
            for r in self.manifest["runs"]
            if r["run_id"] == "unreachable_then_recover:pid"
        )
        self.assertGreater(saturated["metrics"]["saturated_steps"], 0)
        self.assertFalse(self.manifest["physical_hardware_contacted"])
        with self.assertRaises(FileExistsError):
            run_benchmark(
                ROOT / "configs/temperature_control_baseline.json", self.run_directory
            )

    def test_repeated_scenario_preserves_observations_and_command_ledger_exactly(self):
        scenario = next(
            s for s in self.config["scenarios"] if s["id"] == "feedback_faults"
        )
        rerun = self.root / "repeat"
        run_scenario(self.config, scenario, "pid", rerun)
        for name in ["observations.jsonl", "commands.jsonl"]:
            self.assertEqual(
                (rerun / name).read_bytes(),
                (self.run_directory / "feedback_faults_pid" / name).read_bytes(),
            )

    def test_hash_tampering_and_resealed_future_feedback_fail_independent_audit(self):
        edited = self.root / "edited"
        shutil.copytree(self.run_directory, edited)
        manifest = json.loads((edited / "manifest.json").read_text())
        relative = manifest["runs"][1]["records_path"]
        records = [json.loads(s) for s in (edited / relative).read_text().splitlines()]
        records[10]["observation"]["measured_at_s"] = 11
        (edited / relative).write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            audit_control_benchmark(edited)
        manifest["files"][relative] = sha256(edited / relative)
        (edited / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "sensor delay/held clock"):
            audit_control_benchmark(edited)

    def test_invalid_experiment_definition_fails_before_output_creation(self):
        with self.assertRaisesRegex(ValueError, "Unsupported control"):
            run_scenario(
                self.config,
                self.config["scenarios"][0],
                "unknown",
                self.root / "unsupported",
            )
        self.assertFalse((self.root / "unsupported").exists())
        cases = []
        config = copy.deepcopy(self.config)
        config["step_s"] = 0
        cases.append(config)
        config = copy.deepcopy(self.config)
        config["scenarios"][0]["sensor_delay_s"] = -1
        cases.append(config)
        config = copy.deepcopy(self.config)
        config["scenarios"][0]["id"] = "../escape"
        cases.append(config)
        config = copy.deepcopy(self.config)
        config["recipe"][-1]["time_s"] = 500
        cases.append(config)
        config = copy.deepcopy(self.config)
        config["scenarios"][0]["held_intervals"] = [dict(start_s=30, end_s=20)]
        cases.append(config)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)
