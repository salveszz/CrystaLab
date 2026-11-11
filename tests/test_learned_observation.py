# FILE PURPOSE: Verify learned-mask identity, gap/ambiguity handling and replay provenance.
# Predictor masks are explicit artificial fixtures, not model accuracy evidence.
# Real model executions are separately recorded in development replay reports.

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_vision_fixture import generate_fixture
from models.segmentation import load_run_metadata
from vision.learned import LearnedObservationConfig, LearnedSeedTracker
from vision.replay import replay
from vision.video import file_sha256


def pixels():
    image = np.full((120, 160, 3), 220, np.uint8)
    image[40:60, 30:60] = 40
    return image


def seed_mask():
    mask = np.zeros((120, 160), np.uint8)
    mask[40:60, 30:60] = 255
    return mask


def instance(mask, class_id=0, score=0.9):
    return dict(
        mask=mask,
        class_id=class_id,
        class_name=["seed", "secondary_crystal", "inclusion_candidate"][class_id],
        score=score,
        score_is_calibrated=False,
    )


class LearnedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = LearnedObservationConfig((45, 50))

    def test_seed_component_and_identity_ignore_larger_object_and_reflection(self):
        mask = seed_mask()
        mask[10:12, 10:12] = 255
        distractor = np.zeros_like(mask)
        distractor[65:105, 95:145] = 255
        tracker = LearnedSeedTracker(
            self.config, lambda image: [instance(mask), instance(distractor)]
        )
        record, selected = tracker.measure(pixels(), 0)
        self.assertTrue(record["valid"])
        self.assertEqual(record["projected_area_px"], 600)
        self.assertEqual(record["excluded_seed_instance_pixels"], 4)
        self.assertEqual(record["seed_instance_index"], 0)
        self.assertFalse(record["confidence_is_calibrated"])
        self.assertFalse(record["target_absence_is_healthy"])
        np.testing.assert_array_equal(selected, seed_mask())
        self.assertEqual(len(tracker.instances), 2)
        tracker.predictor = lambda image: [instance(distractor)]
        record, mask = tracker.measure(pixels(), 0.1)
        self.assertFalse(record["valid"])
        self.assertIsNone(mask)
        self.assertIsNone(record["projected_area_px"])

    def test_split_and_duplicate_seed_candidates_remain_ambiguous(self):
        tracker = LearnedSeedTracker(self.config, lambda image: [instance(seed_mask())])
        tracker.measure(pixels(), 0)
        split = seed_mask()
        split[:, 44:46] = 0
        tracker.predictor = lambda image: [instance(split)]
        self.assertEqual(
            tracker.measure(pixels(), 0.1)[0]["reason"], "ambiguous_seed_identity"
        )
        tracker = LearnedSeedTracker(
            self.config, lambda image: [instance(seed_mask()), instance(seed_mask())]
        )
        self.assertEqual(
            tracker.measure(pixels(), 0)[0]["reason"], "ambiguous_seed_identity"
        )

    def test_seed_role_survives_raw_visual_class_change(self):
        tracker = LearnedSeedTracker(
            self.config, lambda image: [instance(seed_mask(), class_id=1)]
        )
        record, _ = tracker.measure(pixels(), 0)
        self.assertTrue(record["valid"])
        self.assertEqual(record["seed_raw_class_id"], 1)
        tracker.predictor = lambda image: [instance(seed_mask(), class_id=0)]
        record, _ = tracker.measure(pixels(), 0.1)
        self.assertTrue(record["valid"])
        self.assertEqual(record["seed_raw_class_id"], 0)
        self.assertEqual(record["overlap_iou"], 1.0)
        self.assertTrue(record["requires_inspection"])
        self.assertIn("raw_seed_class_transition", record["inspection_reasons"])
        strict = LearnedSeedTracker(
            replace(self.config, seed_candidate_class_ids=(0,)),
            lambda image: [instance(seed_mask(), class_id=1)],
        )
        self.assertFalse(strict.measure(pixels(), 0)[0]["valid"])

    def test_compatible_cross_class_outlines_retain_alternatives_and_request_review(
        self,
    ):
        larger = seed_mask()
        larger[40:60, 60] = 255
        tracker = LearnedSeedTracker(
            self.config,
            lambda image: [
                instance(seed_mask(), class_id=0, score=0.8),
                instance(larger, class_id=1, score=0.9),
            ],
        )
        record, mask = tracker.measure(pixels(), 0)
        self.assertTrue(record["valid"])
        self.assertTrue(record["requires_inspection"])
        self.assertEqual(record["status"], "tracked_with_alternatives")
        self.assertEqual(record["projected_area_range_px"], [600, 620])
        self.assertEqual(record["seed_raw_class_id"], 1)
        self.assertEqual(record["seed_alternatives"][0]["raw_class_id"], 0)
        self.assertGreater(record["seed_alternatives"][0]["mask_iou_to_selected"], 0.9)
        np.testing.assert_array_equal(mask, larger)

    def test_bounded_recovery_does_not_auto_reinitialize_after_long_gap(self):
        tracker = LearnedSeedTracker(
            replace(self.config, maximum_gap_s=0.5),
            lambda image: [instance(seed_mask())],
        )
        tracker.measure(pixels(), 0)
        tracker.predictor = lambda image: []
        self.assertFalse(tracker.measure(pixels(), 0.1)[0]["valid"])
        self.assertEqual(tracker.instances, [])
        tracker.predictor = lambda image: [instance(seed_mask())]
        self.assertTrue(tracker.measure(pixels(), 0.2)[0]["valid"])
        self.assertEqual(
            tracker.measure(pixels(), 0.8)[0]["reason"],
            "identity_requires_reinitialization",
        )
        self.assertEqual(
            tracker.measure(pixels(), 0.9)[0]["reason"],
            "identity_requires_reinitialization",
        )
        self.assertEqual(len(tracker.instances), 1)

    def test_clipped_low_score_wrong_class_and_unreadable_views_are_invalid(self):
        for config, predictions in [
            (replace(self.config, roi=(30, 30, 40, 50)), [instance(seed_mask())]),
            (self.config, [instance(seed_mask(), score=0.1)]),
            (self.config, [instance(seed_mask(), class_id=2)]),
            (self.config, [instance(np.full((120, 160), 255, np.uint8))]),
        ]:
            with self.subTest(config=config):
                self.assertFalse(
                    LearnedSeedTracker(config, lambda image: predictions).measure(
                        pixels(), 0
                    )[0]["valid"]
                )
        tracker = LearnedSeedTracker(self.config, lambda image: [])
        self.assertEqual(
            tracker.measure(np.zeros((120, 160, 3), np.uint8), 0)[0]["reason"],
            "low_contrast",
        )

    def test_invalid_config_predictions_clock_and_dimensions_fail(self):
        for settings in [
            dict(confidence=float("nan")),
            dict(confidence=True),
            dict(threads=0),
            dict(minimum_iou=0),
            dict(seed_point=(True, 2)),
            dict(seed_candidate_class_ids=(2,)),
            dict(seed_candidate_class_ids=(0, 0)),
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(self.config, **settings)
        tracker = LearnedSeedTracker(self.config, lambda image: [instance(seed_mask())])
        tracker.measure(pixels(), 0)
        with self.assertRaises(ValueError):
            tracker.measure(pixels(), 0)
        with self.assertRaises(ValueError):
            tracker.measure(pixels()[:100], 0.1)
        for bad in [
            instance(seed_mask().astype(float)),
            instance(seed_mask(), score=float("nan")),
        ]:
            with self.assertRaisesRegex(ValueError, "Invalid learned instance"):
                LearnedSeedTracker(self.config, lambda image: [bad]).measure(
                    pixels(), 0
                )


class LearnedReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = generate_fixture(self.root / "fixture")
        self.config = LearnedObservationConfig((85, 110))
        self.video = self.fixture / "fixture.avi"

    def observer(self, output):
        index = 0

        def predict(image):
            nonlocal index
            seed = cv2.imread(
                str(self.fixture / f"truth_masks/frame_{index:08d}.png"), 0
            )
            distractor = np.zeros_like(seed)
            distractor[140:160, 240:260] = 255
            index += 1
            return ([instance(seed)] if np.any(seed) else []) + [
                instance(distractor, class_id=1)
            ]

        tracker = LearnedSeedTracker(self.config, predict)
        tracker.info = dict(
            method="fixture_supplied_predictions_not_real_model", accuracy_claim=False
        )
        return tracker

    def test_encoded_replay_saves_raw_candidates_and_observation_events_only(self):
        output = self.root / "replay"
        result = replay(
            self.video,
            self.config,
            output,
            run_id="fixture",
            observer_factory=self.observer,
            source_info=dict(evidence_kind="fixture"),
        )
        self.assertEqual(result["valid_observations"], 20)
        self.assertEqual(result["invalid_observations"], 4)
        self.assertEqual(result["raw_candidate_instances"], 44)
        rows = [
            json.loads(line)
            for line in (output / "measurements.jsonl").read_text().splitlines()
        ]
        self.assertEqual(rows[10]["objects"][0]["class_id"], 1)
        self.assertFalse(rows[10]["valid"])
        self.assertIsNone(rows[10]["mask_path"])
        self.assertIsNone(rows[-1]["experiment_time_s"])
        self.assertTrue((output / rows[-1]["objects"][0]["mask_path"]).exists())
        events = [
            json.loads(line)["kind"]
            for line in (output / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            events, ["seed_acquired", "inspection_required", "seed_acquired"]
        )
        for path, digest in result["files"].items():
            self.assertEqual(file_sha256(output / path), digest)
        for path, digest in result["source_sha256"].items():
            self.assertEqual(file_sha256(output / "source_snapshot" / path), digest)

    def frame_map(self):
        path = self.root / "map.jsonl"
        rows = [
            dict(
                output_frame_index=i,
                display_time_s=i / 10,
                source_sample_id=f"fixture:{i}",
                acquisition_time_s=None,
                nominal_elapsed_s=i * 300,
            )
            for i in range(24)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path, rows

    def test_original_frame_mapping_preserves_nominal_time_without_claiming_growth_clock(
        self,
    ):
        path, _ = self.frame_map()
        output = self.root / "mapped"
        report = replay(
            self.video,
            self.config,
            output,
            run_id="mapped",
            observer_factory=self.observer,
            source_info=dict(
                evidence_kind="fixture", frame_map_sha256=file_sha256(path)
            ),
            frame_map=path,
            sample_every=3,
            max_frames=10,
        )
        rows = [
            json.loads(line)
            for line in (output / "measurements.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [r["source_frame"]["source_sample_id"] for r in rows],
            ["fixture:0", "fixture:3", "fixture:6", "fixture:9"],
        )
        self.assertEqual(rows[-1]["source_frame"]["nominal_elapsed_s"], 2700)
        self.assertIsNone(rows[-1]["experiment_time_s"])
        self.assertEqual(file_sha256(output / "frame_map.jsonl"), file_sha256(path))
        self.assertEqual(report["decoded_frames"], 10)

    def test_wrong_map_identity_and_display_timing_are_rejected(self):
        path, rows = self.frame_map()
        with self.assertRaisesRegex(ValueError, "frame-map identity"):
            replay(
                self.video,
                self.config,
                self.root / "wrong-hash",
                run_id="bad",
                observer_factory=self.observer,
                frame_map=path,
            )
        self.assertFalse((self.root / "wrong-hash").exists())
        rows[0]["display_time_s"] = 0.01
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output = self.root / "bad-timing"
        with self.assertRaisesRegex(ValueError, "display timing"):
            replay(
                self.video,
                self.config,
                output,
                run_id="bad",
                observer_factory=self.observer,
                frame_map=path,
                source_info=dict(frame_map_sha256=file_sha256(path)),
            )
        self.assertEqual(
            json.loads((output / "manifest.json").read_text())["status"], "failed"
        )

    def test_missing_model_or_changed_model_never_falls_back_to_baseline(self):
        def missing(output):
            raise FileNotFoundError("missing model fixture")

        output = self.root / "missing-model"
        with self.assertRaises(FileNotFoundError):
            replay(
                self.video,
                self.config,
                output,
                run_id="missing",
                observer_factory=missing,
            )
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["processed_frames"], 0)

        def changed(output):
            tracker = self.observer(output)

            def verify():
                raise ValueError("model changed fixture")

            tracker.verify = verify
            return tracker

        output = self.root / "changed-model"
        with self.assertRaisesRegex(ValueError, "model changed"):
            replay(
                self.video,
                self.config,
                output,
                run_id="changed",
                observer_factory=changed,
            )
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["processed_frames"], 24)


class TrainedRunMetadataTests(unittest.TestCase):
    def test_weight_source_and_path_checks_before_optional_runtime_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.pt").write_bytes(b"TEST FIXTURE ONLY, NEVER LOADED")
            snapshot = root / "source_snapshot/src/model.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("# FILE PURPOSE: Metadata test fixture only.\n")
            manifest = dict(
                schema="crystalab_yolov8_seg_run_v1",
                status="complete",
                best_weights="weights.pt",
                best_weights_sha256=file_sha256(root / "weights.pt"),
                source_sha256={"src/model.py": file_sha256(snapshot)},
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_run_metadata(root)[1], root / "weights.pt")
            (root / "weights.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "weight identity"):
                load_run_metadata(root)
            manifest["best_weights_sha256"] = file_sha256(root / "weights.pt")
            (root / "manifest.json").write_text(json.dumps(manifest))
            snapshot.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source snapshot"):
                load_run_metadata(root)
            manifest["best_weights"] = "../outside.pt"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                load_run_metadata(root)
