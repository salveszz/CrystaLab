# FILE PURPOSE: Verify video clocks, seed identity, invalid states and replay evidence.
# Independent pixel geometry and a real encoded clip exercise the complete path;
# tests do not establish crystal optics, quality detection or real-world accuracy.

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_vision_fixture import generate_fixture
from vision.baseline import BaselineConfig, SeedTracker
from vision.replay import replay
from vision.video import VideoReader


def image(seed=True):
    pixels = np.full((120, 160, 3), 220, np.uint8)
    if seed:
        pixels[40:60, 30:60] = 40
    pixels[65:105, 95:145] = 20
    return pixels


class SeedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = BaselineConfig((45, 50), threshold=110, morphology_size=1)

    def test_analytic_area_and_identity_ignore_larger_distractor(self):
        tracker = SeedTracker(self.config)
        result, mask = tracker.measure(image(), 0)
        self.assertEqual(result["projected_area_px"], 600)
        self.assertEqual(result["bbox_px"], [30, 40, 30, 20])
        self.assertEqual(result["centroid_px"], [44.5, 49.5])
        self.assertEqual(np.count_nonzero(mask), 600)
        self.assertFalse(result["quality_target_available"])
        result, mask = tracker.measure(image(False), 0.1)
        self.assertFalse(result["valid"])
        self.assertIsNone(mask)
        self.assertIsNone(result["projected_area_px"])

    def test_recovery_is_bounded_and_never_reinitializes_after_long_gap(self):
        tracker = SeedTracker(replace(self.config, maximum_gap_s=0.5))
        tracker.measure(image(), 0)
        tracker.measure(image(False), 0.1)
        self.assertTrue(tracker.measure(image(), 0.2)[0]["valid"])
        tracker.measure(image(False), 0.3)
        record, _ = tracker.measure(image(), 0.8)
        self.assertEqual(record["reason"], "identity_requires_reinitialization")
        self.assertFalse(tracker.measure(image(), 0.9)[0]["valid"])

    def test_uniform_and_clipped_views_are_invalid(self):
        tracker = SeedTracker(self.config)
        record, _ = tracker.measure(np.zeros((120, 160, 3), np.uint8), 0)
        self.assertEqual(record["reason"], "low_contrast")
        tracker = SeedTracker(replace(self.config, roi=(30, 30, 40, 50)))
        record, _ = tracker.measure(image(), 0)
        self.assertFalse(record["valid"])

    def test_ambiguity_rejects_split_seed_instead_of_selecting_one_piece(self):
        tracker = SeedTracker(replace(self.config, ambiguity_margin=0.1))
        tracker.measure(image(), 0)
        split = image()
        split[40:60, 44:46] = 220
        self.assertEqual(
            tracker.measure(split, 0.1)[0]["reason"], "ambiguous_seed_identity"
        )

    def test_bad_inputs_clocks_and_geometry_rejected(self):
        for settings in (
            dict(seed_point=(True, 2)),
            dict(morphology_size=2),
            dict(minimum_iou=0),
            dict(threshold=256),
            dict(maximum_gap_s=float("nan")),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(self.config, **settings)
        tracker = SeedTracker(self.config)
        tracker.measure(image(), 0)
        with self.assertRaises(ValueError):
            tracker.measure(image(), 0)
        with self.assertRaises(ValueError):
            SeedTracker(replace(self.config, seed_point=(500, 2))).measure(image(), 0)


class VideoReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = generate_fixture(cls.root / "input")
        cls.video = cls.fixture / "fixture.avi"
        cls.config = BaselineConfig(
            **json.loads((cls.fixture / "baseline.json").read_text())
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_encoded_clip_container_clocks_and_explicit_nominal_mode(self):
        with VideoReader(self.video) as reader:
            frames = list(reader)
        self.assertEqual(len(frames), 24)
        np.testing.assert_allclose(
            [f.video_time_s for f in frames], np.arange(24) / 10, atol=1e-8
        )
        self.assertEqual(frames[1].timestamp_basis, "container")
        with VideoReader(self.video, clock="nominal_fps", fps=5) as reader:
            nominal = list(reader)
        self.assertAlmostEqual(nominal[-1].video_time_s, 4.6)
        with self.assertRaises(ValueError):
            VideoReader(self.video, fps=5)

    def test_bad_pts_fail_without_silent_fps_fallback(self):
        original = cv2.VideoCapture

        class NoPTS:
            def __init__(self, path):
                self.inner = original(path)

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def get(self, key):
                return 0 if key == cv2.CAP_PROP_POS_MSEC else self.inner.get(key)

        with (
            patch("vision.video.cv2.VideoCapture", NoPTS),
            VideoReader(self.video) as reader,
        ):
            next(reader)
            with self.assertRaisesRegex(ValueError, "No clock fallback"):
                next(reader)

    def test_complete_replay_masks_events_hashes_and_experiment_clock(self):
        output = self.root / "replay"
        report = replay(
            self.video,
            self.config,
            output,
            run_id="fixture",
            source_info=dict(
                evidence_kind="fixture",
                experiment_clock_provenance="test mapping, not growth data",
            ),
            experiment_origin_s=60,
            experiment_seconds_per_video_second=2,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["valid_observations"], 20)
        self.assertEqual(report["invalid_observations"], 4)
        rows = [
            json.loads(line)
            for line in (output / "measurements.jsonl").read_text().splitlines()
        ]
        self.assertAlmostEqual(rows[-1]["experiment_time_s"], 64.6)
        events = [
            json.loads(line)
            for line in (output / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [r["kind"] for r in events],
            ["seed_acquired", "inspection_required", "seed_acquired"],
        )
        for row in rows:
            if not row["valid"]:
                self.assertIsNone(row["mask_path"])
                continue
            truth = (
                cv2.imread(
                    str(
                        self.fixture / f"truth_masks/frame_{row['frame_index']:08d}.png"
                    ),
                    0,
                )
                > 0
            )
            found = cv2.imread(str(output / row["mask_path"]), 0) > 0
            self.assertGreater(
                np.count_nonzero(truth & found) / np.count_nonzero(truth | found), 0.98
            )
        for name, expected in report["files"].items():
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), expected
            )
        with VideoReader(output / "overlay.avi") as reader:
            self.assertEqual(len(list(reader)), 24)
        with self.assertRaises(FileExistsError):
            replay(self.video, self.config, output, run_id="fixture")

    def test_sampling_keeps_source_indices_and_does_not_invent_growth_time(self):
        output = self.root / "sampled"
        report = replay(
            self.video,
            self.config,
            output,
            run_id="sampled",
            sample_every=3,
            max_frames=10,
        )
        rows = [
            json.loads(line)
            for line in (output / "measurements.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["frame_index"] for r in rows], [0, 3, 6, 9])
        self.assertTrue(all(r["experiment_time_s"] is None for r in rows))
        self.assertEqual(report["decoded_frames"], 10)
        self.assertTrue(report["stopped_by_frame_limit"])

    def test_failures_are_retained_and_camera_needs_explicit_limit(self):
        output = self.root / "failed"
        with self.assertRaises(ValueError):
            replay(
                self.video,
                replace(self.config, seed_point=(9999, 3)),
                output,
                run_id="failed",
            )
        self.assertEqual(
            json.loads((output / "manifest.json").read_text())["status"], "failed"
        )
        with self.assertRaises(ValueError):
            replay(
                0,
                self.config,
                self.root / "camera",
                run_id="camera",
                camera=True,
                clock="arrival",
            )
        self.assertFalse((self.root / "camera").exists())
        with self.assertRaises(ValueError):
            replay(
                self.video,
                self.config,
                self.root / "mapping",
                run_id="mapping",
                experiment_origin_s=0,
                experiment_seconds_per_video_second=1,
            )


if __name__ == "__main__":
    unittest.main()
