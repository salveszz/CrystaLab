# FILE PURPOSE: Verify that candidate evidence is emitted before EOF and matches saved-replay decisions.
# Supplied fixture masks test causal delivery, original clocks, sampled frames,
# failure retention and independent event scoring without opening a camera.

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_candidate_fixture import generate_candidate_fixture, geometry
from vision.baseline import BaselineConfig
from vision.candidates import CandidateConfig
from vision.candidate_replay import analyze_candidates
from vision.event_evaluation import score_candidate_events
from vision.learned import LearnedObservationConfig, LearnedSeedTracker
from vision.replay import replay
from vision.video import file_sha256


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CandidateStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root / "fixture"
        generate_candidate_fixture(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        self.parent = self.case / "observation"
        self.candidates = self.case / "candidates"
        self.config = LearnedObservationConfig((65, 80))
        self.info = json.loads((self.fixture / "source_info.json").read_text())

    def factory(self, before_frame=None):
        config = self.config

        class Observer(LearnedSeedTracker):
            index = 0

            def measure(self, image, timestamp):
                self.index = round(timestamp * 10)
                if before_frame is not None:
                    before_frame(self.index)
                record, mask = super().measure(image, timestamp)
                record["method"] = "fixture_supplied_candidate_masks"
                record["display_method"] = (
                    "FIXTURE / supplied masks / no model accuracy"
                )
                return record, mask

        def factory(output):
            def predict(image):
                return [
                    dict(
                        class_id=class_id, score=0.9, mask=mask, fixture_object_id=name
                    )
                    for name, class_id, mask in geometry(observer.index)
                ]

            observer = Observer(config, predict)
            observer.info = dict(
                method="fixture_supplied_candidate_masks", model_accuracy_claim=False
            )
            observer.source_files = [Path(__file__).resolve()]
            return observer

        return factory

    def run_stream(self, **kwargs):
        options = dict(
            run_id="stream_fixture",
            source_info=self.info,
            observer_factory=self.factory(),
            frame_map=self.fixture / "frame_map.jsonl",
            candidate_config=CandidateConfig(),
            candidate_output=self.candidates,
        )
        options.update(kwargs)
        return replay(
            self.fixture / "sequence.avi", self.config, self.parent, **options
        )

    def test_events_are_flushed_before_next_prediction_and_match_saved_analysis(self):
        seen = []

        def before_frame(index):
            if index in {2, 12}:
                alerts = rows(self.candidates / "candidate_events.jsonl")
                seen.append((index, [e["alert_frame_index"] for e in alerts]))
                self.assertEqual(
                    json.loads((self.candidates / "manifest.json").read_text())[
                        "status"
                    ],
                    "running",
                )
                self.assertEqual(
                    json.loads((self.parent / "manifest.json").read_text())["status"],
                    "running",
                )

        parent = self.run_stream(observer_factory=self.factory(before_frame))
        self.assertEqual(seen, [(2, [1]), (12, [1, 11])])
        stream = json.loads((self.candidates / "manifest.json").read_text())
        offline = self.case / "offline"
        saved = analyze_candidates(self.parent, CandidateConfig(), offline)
        for name in ["candidate_events.jsonl", "candidate_observations.jsonl"]:
            self.assertEqual(rows(self.candidates / name), rows(offline / name))
        for key in ["event_kinds", "issues", "usable_source_seconds", "final_tracks"]:
            self.assertEqual(stream[key], saved[key])
        for path, digest in stream["files"].items():
            self.assertEqual(file_sha256(self.candidates / path), digest)
        self.assertEqual(
            stream["source_replay_manifest_sha256"],
            file_sha256(self.parent / "manifest.json"),
        )
        self.assertEqual(parent["candidate_analysis"]["persistent_candidate_events"], 4)
        self.assertEqual(stream["mode"], "during_observation")
        self.assertEqual(
            parent["pipeline_latency"]["scope"].split(";")[0],
            "Decoded frame through observer, masks, overlays and flushed observation/candidate records",
        )
        report = score_candidate_events(
            self.candidates, self.fixture / "annotations.json", self.case / "evaluation"
        )
        self.assertEqual(report["per_kind"]["crystal_candidate"]["tp"], 2)
        self.assertEqual(report["per_kind"]["crystal_candidate"]["fp"], 1)
        self.assertEqual(report["per_kind"]["inclusion_candidate"]["tp"], 1)
        self.assertFalse(report["physical_clock_scored"])

    def test_sampled_frames_keep_original_clock_and_confirmation_spacing(self):
        parent = self.run_stream(sample_every=2, max_frames=13)
        observations = rows(self.candidates / "candidate_observations.jsonl")
        self.assertEqual(
            [r["frame_index"] for r in observations], list(range(0, 13, 2))
        )
        self.assertEqual(parent["decoded_frames"], 13)
        self.assertEqual(
            [
                e["alert_frame_index"]
                for e in rows(self.candidates / "candidate_events.jsonl")
            ],
            [2, 12],
        )
        event = rows(self.candidates / "candidate_events.jsonl")[-1]
        self.assertEqual(event["onset_interval_source_s"], [0.8, 1.0])
        self.assertAlmostEqual(event["confirmation_delay_s"], 0.2)
        self.assertEqual(observations[-1]["source_frame"]["output_frame_index"], 12)
        self.assertIsNone(observations[-1]["experiment_time_s"])

    def test_failed_observer_retains_partial_alerts_and_both_failed_manifests(self):
        def before_frame(index):
            if index == 12:
                raise OSError("fixture observer interrupted")

        with self.assertRaisesRegex(OSError, "interrupted"):
            self.run_stream(observer_factory=self.factory(before_frame))
        self.assertEqual(
            [
                e["alert_frame_index"]
                for e in rows(self.candidates / "candidate_events.jsonl")
            ],
            [1, 11],
        )
        for folder in [self.parent, self.candidates]:
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["processed_frames"], 12)
        with self.assertRaisesRegex(ValueError, "Completed candidate analysis"):
            score_candidate_events(
                self.candidates, self.fixture / "annotations.json", self.case / "score"
            )

    def test_unsupported_inputs_fail_before_capture_or_output_creation(self):
        with patch("vision.replay.VideoReader") as reader:
            for extra in [
                dict(candidate_output=None),
                dict(candidate_config=None),
                dict(observer_factory=None),
                dict(source_info=dict(split="test")),
                dict(candidate_output=self.parent / "nested"),
                dict(candidate_output=self.case),
            ]:
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    self.run_stream(**extra)
            self.assertFalse(self.parent.exists())
            reader.assert_not_called()
        self.candidates.mkdir()
        marker = self.candidates / "keep.txt"
        marker.write_text("existing evidence")
        with self.assertRaises(FileExistsError):
            self.run_stream()
        self.assertEqual(marker.read_text(), "existing evidence")
        with self.assertRaisesRegex(ValueError, "learned observer"):
            replay(
                self.fixture / "sequence.avi",
                BaselineConfig((65, 80)),
                self.parent,
                run_id="baseline",
                candidate_config=CandidateConfig(),
                candidate_output=self.case / "other",
            )
