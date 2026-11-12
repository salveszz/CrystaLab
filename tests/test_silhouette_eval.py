# FILE PURPOSE: Keep misses and forbid target-guided prediction selection in the silhouette diagnostic.
import unittest
import numpy as np
from models.silhouette_eval import choose_instance, mask_score, summarize


class SilhouetteEvaluationTests(unittest.TestCase):
    def test_prediction_selection_uses_score_even_when_a_lower_score_matches_truth(
        self,
    ):
        truth = np.zeros((20, 20), np.uint8)
        truth[2:8, 2:8] = 255
        wrong = np.zeros_like(truth)
        wrong[12:18, 12:18] = 255
        chosen, score, extra = choose_instance(
            [dict(mask=truth, score=0.5), dict(mask=wrong, score=0.9)], truth.shape
        )
        self.assertEqual(score, 0.9)
        self.assertEqual(extra, 1)
        self.assertEqual(mask_score(chosen, truth)["proxy_mask_iou"], 0)

    def test_missing_masks_are_zero_iou_and_remain_in_family_macro(self):
        truth = np.zeros((10, 10), np.uint8)
        truth[2:4, 2:4] = 255
        missing, _, _ = choose_instance([], truth.shape)
        rows = [
            dict(
                family_key=family,
                evidence_kind="real",
                extra_predictions=0,
                **mask_score(predicted, truth),
            )
            for family, predicted in [("a", truth), ("a", truth), ("b", missing)]
        ]
        report = summarize(rows)["all"]
        self.assertEqual(report["missing_predictions"], 1)
        self.assertAlmostEqual(report["proxy_mask_iou_mean"], 2 / 3)
        self.assertEqual(report["proxy_mask_iou_family_macro"], 0.5)
