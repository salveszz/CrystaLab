# FILE PURPOSE: Verify that segmentation diagnostics count misses, duplicates and class errors.
# Artificial masks test global assignment without loading optional model weights.

import unittest
import numpy as np

from models.segmentation_eval import evaluate_development, match_instances


class SegmentationMatchingTests(unittest.TestCase):
    def test_global_assignment_preserves_both_possible_matches(self):
        left = np.zeros((10, 20), np.uint8)
        left[:, :10] = 255
        right = np.zeros_like(left)
        right[:, 10:] = 255
        wide = np.full_like(left, 255)
        truth = [dict(class_id=0, mask=left), dict(class_id=0, mask=right)]
        predictions = [dict(class_id=0, mask=wide), dict(class_id=0, mask=left)]
        matches = match_instances(truth, predictions, 0.5)
        self.assertEqual(
            {(m["truth_index"], m["prediction_index"]) for m in matches},
            {(0, 1), (1, 0)},
        )
        self.assertEqual(sorted(m["iou"] for m in matches), [0.5, 1.0])

    def test_wrong_class_and_duplicate_masks_do_not_add_true_positives(self):
        mask = np.ones((10, 10), np.uint8) * 255
        truth = [dict(class_id=0, mask=mask)]
        self.assertEqual(match_instances(truth, [dict(class_id=1, mask=mask)]), [])
        self.assertEqual(len(match_instances(truth, [truth[0], truth[0]])), 1)
        self.assertEqual(match_instances(truth, []), [])
        self.assertEqual(match_instances([], truth), [])

    def test_development_command_rejects_final_test_before_loading_any_model(self):
        with self.assertRaisesRegex(ValueError, "final test"):
            evaluate_development("missing", "missing", "missing", split="test")
