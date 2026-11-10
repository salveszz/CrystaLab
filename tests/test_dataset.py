# FILE PURPOSE: Verify controlled recording and synthetic data integrity without ML.
# Exercise causal delays/labels, independent RNG streams, exact clocks, grouped
# splits, serial/parallel replay, guarded resume, retained failures and tampering.
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from environment.dataset import (
    FEATURES,
    audit_dataset,
    build_cases,
    canonical,
    file_hash,
    generate,
    read_json,
    read_rows,
    write_json,
)
from environment.kdp_env import (
    RecordingConfig,
    expand_schedule,
    make_observations,
    record_episode,
)
from simulation.engine import Simulation
from simulation.tank import SimulationConfig


def protocol():
    return dict(
        description="small reproducible test protocol",
        simulation=dict(
            nx=5, ny=5, nz=5, duration=2.5, dt=0.7, concentration_noise=0.001
        ),
        recording=dict(
            sample_interval=0.5,
            decision_interval=1,
            forecast_horizon=1,
            sensor_delay=0.5,
            relative_observation_noise=0.01,
            observation_seed=16,
        ),
        groups=[
            dict(id=name, family_id=name, split=split, overrides=dict(concentration=c))
            for name, split, c in [
                ("a", "train", 1.25),
                ("b", "validation", 1.3),
                ("c", "test", 1.4),
            ]
        ],
        recipes=[
            dict(id="baseline", schedule=[dict(time=0, temperature=25, agitation=1)]),
            dict(
                id="change",
                schedule=[
                    dict(time=0, temperature=25, agitation=1),
                    dict(time=1, temperature=22, agitation=0),
                ],
            ),
        ],
    )


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            nx=5, ny=5, nz=5, duration=2.5, dt=0.7, concentration_noise=0
        )
        self.recording = RecordingConfig(
            sample_interval=0.5,
            decision_interval=1,
            forecast_horizon=1,
            sensor_delay=0.5,
        )
        self.schedule = [dict(time=0, temperature=25, agitation=1)]

    def test_controls_preserve_state_validate_atomically_and_relax(self):
        sim = Simulation(self.config)
        old_config = sim.config
        temperature = sim.tank.temperature.copy()
        concentration = sim.tank.concentration.copy()
        rng_state = copy.deepcopy(sim.rng.bit_generator.state)
        vertices = sim.crystal.vertices.copy()
        with self.assertRaises(ValueError):
            sim.set_controls(temperature=22, agitation=float("nan"))
        self.assertIs(sim.config, old_config)
        self.assertIs(sim.tank.config, old_config)
        sim.set_controls(temperature=22, agitation=0)
        self.assertIs(sim.tank.config, sim.config)
        self.assertEqual(sim.rng.bit_generator.state, rng_state)
        np.testing.assert_array_equal(sim.tank.temperature, temperature)
        np.testing.assert_array_equal(sim.tank.concentration, concentration)
        np.testing.assert_array_equal(sim.crystal.vertices, vertices)
        sim.step(0.2)
        self.assertGreater(sim.measurements()["temperature"], 22)
        self.assertLess(sim.measurements()["temperature"], 25)
        sim.run()
        with self.assertRaises(ValueError):
            sim.set_controls(agitation=1)

    def test_noisy_delayed_observations_do_not_change_truth(self):
        exact = record_episode(self.config, self.recording, self.schedule)
        noisy_config = replace(
            self.recording,
            relative_observation_noise=0.2,
            temperature_observation_noise=0.5,
            observation_seed=99,
        )
        noisy = record_episode(self.config, noisy_config, self.schedule)
        self.assertEqual(exact.truth, noisy.truth)
        for key in exact.final_fields:
            np.testing.assert_array_equal(
                exact.final_fields[key], noisy.final_fields[key]
            )
        self.assertEqual(noisy.observations[0]["sensor_time"], None)
        self.assertTrue(noisy.observations[0]["missing"]["volume"])
        self.assertEqual(noisy.observations[1]["sensor_time"], 0)
        self.assertNotEqual(noisy.observations[1]["volume"], exact.truth[0]["volume"])
        self.assertEqual(
            noisy.observations, make_observations(noisy.truth, noisy_config)
        )
        prefix = make_observations(noisy.truth[:3], noisy_config)
        self.assertEqual(prefix, noisy.observations[:3])

    def test_action_boundary_cannot_change_past_or_exact_truth_at_decision(self):
        a = record_episode(self.config, self.recording, self.schedule)
        b = record_episode(
            self.config,
            self.recording,
            self.schedule + [dict(time=1, temperature=22, agitation=0)],
        )
        self.assertEqual(a.truth[:3], b.truth[:3])
        self.assertNotEqual(a.truth[3]["volume"], b.truth[3]["volume"])
        self.assertEqual([r["time"] for r in b.truth], [0, 0.5, 1, 1.5, 2, 2.5])
        self.assertEqual(b.diagnostics["final_time"], 2.5)
        self.assertTrue(b.diagnostics["exact_final_time"])
        self.assertEqual([r["time"] for r in b.controls], [0, 1, 2])
        self.assertEqual(b.controls[-1]["applies_until"], 2.5)

    def test_bad_clocks_actions_and_limits_rejected(self):
        for settings in (
            dict(sensor_delay=0.1),
            dict(forecast_horizon=2),
            dict(sample_interval=True),
            dict(history_samples=0),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(self.recording, **settings)
        for row in (
            dict(time=0.5, temperature=25, agitation=1),
            dict(time=0, temperature=21, agitation=1),
            dict(time=0, temperature=25, agitation=3),
            dict(time=0, temperature=True, agitation=1),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                expand_schedule(self.config, self.recording, [row])
        with self.assertRaises(ValueError):
            expand_schedule(
                self.config,
                replace(self.recording, max_temperature_change=1),
                [dict(time=0, temperature=22, agitation=1)],
            )


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.design = protocol()
        cls.report = generate(cls.design, cls.root / "serial")
        generate(cls.design, cls.root / "parallel", jobs=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_end_to_end_labels_splits_missingness_and_parallel_replay(self):
        report = audit_dataset(self.root / "serial")
        self.assertEqual(report["episodes"], 6)
        self.assertEqual(report["samples"], 12)
        self.assertEqual(report["omitted_incomplete_horizons"], 6)
        self.assertEqual(
            report["samples_by_split"], dict(train=4, validation=4, test=4)
        )
        samples = read_rows(self.root / "serial/samples.jsonl")
        self.assertEqual(samples, read_rows(self.root / "parallel/samples.jsonl"))
        for sample in samples:
            self.assertEqual(set(sample["features"]), set(FEATURES))
            self.assertNotIn("total_mass", sample["features"])
            self.assertNotIn("seed", sample["features"])
            self.assertIsNone(sample["labels"]["quality"])
            self.assertEqual(sample["label_end_time"] - sample["decision_time"], 1)
            if sample["sensor_time"] is not None:
                self.assertLessEqual(sample["sensor_time"], sample["decision_time"])
        folder = self.root / "serial/episodes/a__baseline"
        events = read_rows(folder / "attempt_1/events.jsonl")
        truth = {r["time"]: r for r in events if r["kind"] == "truth"}
        self.assertEqual(
            samples[0]["labels"]["volume_gain"], truth[1]["volume"] - truth[0]["volume"]
        )
        self.assertTrue(samples[0]["missing"]["volume"])
        self.assertNotEqual(
            build_cases(self.design)[0]["recording"]["observation_seed"],
            build_cases(self.design)[2]["recording"]["observation_seed"],
        )

    def test_output_refuses_overwrite_resume_is_identity_checked(self):
        with self.assertRaises(FileExistsError):
            generate(self.design, self.root / "serial")
        before = (self.root / "serial/samples.jsonl").read_bytes()
        generate(self.design, self.root / "serial", resume=True, jobs=2)
        self.assertEqual(before, (self.root / "serial/samples.jsonl").read_bytes())
        changed = copy.deepcopy(self.design)
        changed["simulation"]["dt"] = 0.5
        with self.assertRaisesRegex(ValueError, "Resume requires"):
            generate(changed, self.root / "serial", resume=True)

    def test_related_groups_or_seed_only_copies_cannot_cross_splits(self):
        changed = copy.deepcopy(self.design)
        changed["groups"][1]["family_id"] = "a"
        with self.assertRaisesRegex(ValueError, "family"):
            build_cases(changed)
        changed = copy.deepcopy(self.design)
        changed["groups"][1]["overrides"] = dict(concentration=1.25, seed=999)
        with self.assertRaisesRegex(ValueError, "Identical physical"):
            build_cases(changed)
        changed["groups"][1]["id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "identifier"):
            build_cases(changed)

    def clone(self, name):
        import shutil

        target = self.root / name
        shutil.copytree(self.root / "serial", target)
        return target

    def test_sloped_vertex_reordering_does_not_change_geometry_audit(self):
        from environment.dataset import _episode, verify_episode

        design = read_json("configs/synthetic_growth.json")
        design["simulation"].update(
            nx=6, ny=6, nz=6, duration=2, dt=1, boundary_layer_cells=0.45
        )
        design["recording"].update(
            sample_interval=0.5,
            decision_interval=1,
            forecast_horizon=1,
            sensor_delay=0.5,
        )
        design["recipes"] = design["recipes"][:1]
        case = build_cases(design)[0]
        root = self.root / "sloped"
        _episode((case, str(root)))
        folder = root / "episodes" / case["id"]
        path = folder / "attempt_1/final_fields.npz"
        with np.load(path) as archive:
            fields = {key: archive[key] for key in archive.files}
        fields["vertices"] = fields["vertices"][::-1]
        np.savez_compressed(path, **fields)
        metadata = read_json(folder / "episode.json")
        metadata["files"]["final_fields.npz"] = file_hash(path)
        write_json(folder / "episode.json", metadata)
        self.assertEqual(len(verify_episode(folder, case)["samples"]), 2)
        fields["vertices"][0, 0] += 0.1
        np.savez_compressed(path, **fields)
        metadata["files"]["final_fields.npz"] = file_hash(path)
        write_json(folder / "episode.json", metadata)
        with self.assertRaisesRegex(ValueError, "Final geometry"):
            verify_episode(folder, case)

    def test_updated_sample_hash_cannot_hide_future_feature_or_wrong_label(self):
        folder = self.clone("bad_samples")
        samples = read_rows(folder / "samples.jsonl")
        samples[0]["features"]["volume"] = 999
        (folder / "samples.jsonl").write_text(
            "".join(canonical(r) + "\n" for r in samples)
        )
        manifest = read_json(folder / "manifest.json")
        manifest["samples_hash"] = file_hash(folder / "samples.jsonl")
        write_json(folder / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "raw-record reconstruction"):
            audit_dataset(folder)

    def test_updated_raw_hash_cannot_hide_a_future_sensor_timestamp(self):
        folder = self.clone("bad_observations")
        episode = folder / "episodes/a__baseline"
        path = episode / "attempt_1/events.jsonl"
        rows = read_rows(path)
        observation = next(r for r in rows if r["kind"] == "observation")
        observation["sensor_time"] = 2
        path.write_text("".join(canonical(r) + "\n" for r in rows))
        metadata = read_json(episode / "episode.json")
        metadata["files"]["events.jsonl"] = file_hash(path)
        write_json(episode / "episode.json", metadata)
        with self.assertRaisesRegex(ValueError, "causal source"):
            audit_dataset(folder)

    def test_final_fields_and_source_snapshot_are_actually_checked(self):
        folder = self.clone("bad_fields")
        episode = folder / "episodes/a__baseline"
        path = episode / "attempt_1/final_fields.npz"
        with np.load(path) as archive:
            values = {k: archive[k] for k in archive.files}
        values["concentration"] += 0.1
        np.savez_compressed(path, **values)
        metadata = read_json(episode / "episode.json")
        metadata["files"]["final_fields.npz"] = file_hash(path)
        write_json(episode / "episode.json", metadata)
        with self.assertRaisesRegex(ValueError, "dissolved mass"):
            audit_dataset(folder)
        folder = self.clone("bad_source")
        (folder / "source_snapshot/src/simulation/engine.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
            audit_dataset(folder)

    def test_failed_partial_episode_is_retained_and_resume_restarts_it(self):
        folder = self.root / "failure"

        def fail(*args, **kwargs):
            kwargs["on_event"](dict(kind="truth", time=0))
            raise ArithmeticError("deliberate test failure")

        with patch("environment.dataset.record_episode", side_effect=fail):
            with self.assertRaisesRegex(ArithmeticError, "deliberate"):
                generate(self.design, folder)
        partial = folder / "episodes/a__baseline/attempt_1"
        self.assertTrue((partial / "events.jsonl").exists())
        self.assertEqual(read_json(partial / "failure.json")["status"], "failed")
        result = generate(self.design, folder, resume=True)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            read_json(folder / "episodes/a__baseline/episode.json")["attempt"], 2
        )
        self.assertTrue((partial / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
