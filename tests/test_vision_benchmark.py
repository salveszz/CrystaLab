# FILE PURPOSE: Ensure seed comparisons retain missing observations and avoid truth-based selection.
# Artificial masks cover conditional area error, negative cases, class identity,
# clipping and explicit exclusion of final-test scoring.

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from models.vision_benchmark import (
    benchmark_seed,
    calibrate_seed_baseline,
    measurement_error,
    select_learned_seed,
    summarize_errors,
)
from vision.training_data import polygon_masks
from vision.training_data import save_json


class SeedBenchmarkTests(unittest.TestCase):
    def test_missing_measurement_stays_in_overlap_denominator(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[3:9, 3:9] = 255
        result = summarize_errors(
            [measurement_error(mask, mask), measurement_error(mask, None)]
        )
        self.assertEqual(result["mean_iou_all_truth"], 0.5)
        self.assertEqual(result["mean_dice_all_truth"], 0.5)
        self.assertEqual(result["valid_coverage_on_truth"], 0.5)
        self.assertEqual(result["area_error_valid_count"], 1)
        self.assertEqual(result["area_mae_px_valid"], 0.0)

    def test_false_measurement_on_negative_is_separate_from_area_accuracy(self):
        mask = np.ones((8, 8), np.uint8) * 255
        result = summarize_errors(
            [measurement_error(None, mask), measurement_error(None, None)]
        )
        self.assertEqual(result["false_valid_on_negative"], 1)
        self.assertEqual(result["truth_images"], 0)
        self.assertIsNone(result["mean_iou_all_truth"])
        self.assertIsNone(result["area_mae_px_valid"])
        with self.assertRaisesRegex(ValueError, "shape"):
            measurement_error(mask, np.ones((4, 4), np.uint8))

    def test_learned_identity_is_not_chosen_using_truth_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((12, 12), np.uint8)
            mask[3:8, 3:8] = 255
            cv2.imwrite(str(root / "mask.png"), mask)
            candidate = dict(class_id=0, mask_path="mask.png")
            selected, status = select_learned_seed([candidate], mask.shape, root)
            self.assertEqual(status, "valid")
            np.testing.assert_array_equal(selected, mask)
            self.assertEqual(
                select_learned_seed([candidate, candidate], mask.shape, root),
                (None, "ambiguous_seed"),
            )
            self.assertEqual(
                select_learned_seed(
                    [dict(class_id=1, mask_path="mask.png")], mask.shape, root
                ),
                (None, "missing_seed"),
            )
            mask[0, 4] = 255
            cv2.imwrite(str(root / "mask.png"), mask)
            self.assertEqual(
                select_learned_seed([candidate], mask.shape, root),
                (None, "clipped_seed"),
            )

    def test_polygon_rasterization_uses_declared_nearest_pixel_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text("0 0.2 0.2 0.7 0.2 0.7 0.7 0.2 0.7\n")
            target = polygon_masks(path, (10, 10))[0]
            self.assertEqual(target["class_id"], 0)
            self.assertEqual(np.count_nonzero(target["mask"]), 36)
            path.write_text("")
            self.assertEqual(polygon_masks(path, (10, 10)), [])

    def test_test_split_is_rejected_before_data_or_predictions_are_loaded(self):
        with self.assertRaisesRegex(ValueError, "final test"):
            benchmark_seed("missing", {}, "missing", split="test")

    def test_calibration_reads_only_training_pixels_and_retains_all_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((32, 32, 3), 220, np.uint8)
            image[10:22, 10:22] = 64
            cv2.imwrite(str(root / "train.png"), image)
            (root / "train.txt").write_text(
                "0 0.3125 0.3125 0.65625 0.3125 0.65625 0.65625 0.3125 0.65625\n"
            )
            save_json(
                root / "manifest.json",
                dict(
                    records=[
                        dict(
                            split="train",
                            sample_id="train",
                            family_key="train-family",
                            image_path="train.png",
                            label_path="train.txt",
                        ),
                        dict(
                            split="validation",
                            sample_id="do-not-read",
                            family_key="validation-family",
                            image_path="absent.png",
                            label_path="absent.txt",
                        ),
                    ]
                ),
            )
            config = dict(
                schema="crystalab_seed_benchmark_v1",
                mode="single_frame_fixed_seed_point",
                image_shape=[32, 32],
                baseline=dict(seed_point=[16, 16], morphology_size=1),
            )
            protocol = dict(
                schema="crystalab_seed_calibration_v1",
                polarities=["dark"],
                thresholds=[32, 128],
            )
            # Isolate selection from the separately tested source reconstruction audit.
            with patch(
                "models.vision_benchmark.audit_segmentation",
                return_value=dict(manifest_sha256="fixture"),
            ):
                result = calibrate_seed_baseline(root, config, protocol, root / "out")
            self.assertEqual(result["selected_index"], 1)
            self.assertEqual(result["training_sample_ids"], ["train"])
            self.assertEqual(len(result["candidates"]), 2)
            self.assertEqual(
                result["candidates"][0]["report"]["mean_iou_all_truth"], 0.0
            )
            self.assertEqual(
                result["candidates"][1]["report"]["mean_iou_all_truth"], 1.0
            )
