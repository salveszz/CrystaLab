# FILE PURPOSE: Verify reference registration, causal windows and explicit observability gates.
# Pixels and assertions below are artificial unit fixtures, not reviewed real evidence.
import copy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from vision.projected_edges import ProjectedEdgeTracker, similarity_registration

ROOT = Path(__file__).resolve().parents[1]


class ProjectedEdgeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/projected_edge_diagnostic.json").read_text())
        self.config = self.protocol["measurement"]

    def tracker(self, kind="fixture"):
        return ProjectedEdgeTracker(self.config, self.protocol["marker_BGR"], run_id="test", clock_id="source:test", evidence_kind=kind)

    def sample(self, step, **changes):
        width, height = self.protocol["image_size"]
        image = np.full((height, width, 3), 235, np.uint8)
        mask = np.zeros((height, width), np.uint8)
        cv2.rectangle(mask, (172, 114 - step), (212 + 4 * step, 174 + step), 255, -1)
        image[mask > 0] = 50
        for (x, y), color in zip(self.protocol["marker_centers"], self.protocol["marker_BGR"]):
            cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), color, -1)
        context = dict(run_id="test", evidence_kind="fixture", frame_index=step,
            observed_at_source_s=float(step), available_at_source_s=float(step), reference_id="mount", pose_id="pose",
            attachment_supported=True, pose_supported=True, full_visibility_supported=True, provenance="UNIT FIXTURE ONLY")
        context.update(changes)
        return image, mask, context

    def measure(self, tracker, step, **changes):
        image, mask, context = self.sample(step, **changes)
        return tracker.measure(image, mask, timestamp=float(step), frame_index=step, context=context)

    def test_similarity_reconstructs_known_rotation_scale_translation(self):
        reference = np.array([[0., 0.], [10., 0.], [0., 20.]])
        angle = .3
        forward = 1.1 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        current = reference @ forward.T + [30, -20]
        matrix, rms = similarity_registration(current, reference)
        np.testing.assert_allclose(current @ matrix[:, :2].T + matrix[:, 2], reference, atol=1e-11)
        self.assertLess(rms, 1e-11)

    def test_persistent_anchored_advance_uses_only_matured_history(self):
        tracker = self.tracker()
        rows = [self.measure(tracker, i) for i in range(12)]
        self.assertIsNone(rows[1]["comparison"])
        self.assertEqual(rows[3]["comparison"]["baseline_frame_index"], 1)
        self.assertEqual(rows[3]["comparison"]["horizon_source_s"], 2)
        self.assertTrue(all(row["comparison"]["axes"][1]["comparable_growth"] for row in rows[2:]))
        np.testing.assert_allclose(rows[3]["comparison"]["axes"][0]["positive_advance_px"], 8, atol=1e-10)
        self.assertEqual(len(tracker.proposals), 1)
        self.assertEqual(tracker.proposals[0]["source_frame_index"], 3)
        self.assertIsNone(tracker.proposals[0]["physical_command"])
        self.assertTrue(tracker.proposals[0]["requires_operator_review"])
        self.assertFalse(rows[3]["physical_face_rate_available"])

    def test_unknown_real_reference_never_produces_a_number_or_review(self):
        tracker = self.tracker("real")
        image, mask, _ = self.sample(0)
        row = tracker.measure(image, mask, timestamp=0., frame_index=0, context=None)
        self.assertEqual(row["reason"], "missing_external_view_reference")
        self.assertIsNone(row["projected_supports_px"])
        self.assertEqual(row["physical_commands"], [])

    def test_attachment_pose_and_visibility_assertions_are_separate_requirements(self):
        for field, reason in [("attachment_supported", "attachment_not_established"),
                              ("pose_supported", "object_pose_not_comparable"),
                              ("full_visibility_supported", "incomplete_silhouette_view")]:
            with self.subTest(field=field):
                row = self.measure(self.tracker(), 0, **{field:False})
                self.assertFalse(row["valid"])
                self.assertEqual(row["reason"], reason)

    def test_occlusion_breaks_history_and_requires_new_persistence(self):
        tracker = self.tracker()
        for i in range(3): self.measure(tracker, i)
        self.measure(tracker, 3, full_visibility_supported=False)
        rows = [self.measure(tracker, i) for i in range(4, 8)]
        self.assertIsNone(rows[1]["comparison"])
        self.assertEqual([p["source_frame_index"] for p in tracker.proposals], [7])

    def test_changed_pose_and_observation_gap_do_not_bridge_windows(self):
        tracker = self.tracker()
        for i in range(3): self.measure(tracker, i)
        changed = self.measure(tracker, 3, pose_id="new-pose")
        self.assertIsNone(changed["comparison"])
        self.assertIsNone(self.measure(tracker, 7, pose_id="new-pose")["comparison"])
        self.assertEqual(tracker.proposals, [])

    def test_missing_marker_and_non_similarity_are_unavailable(self):
        tracker = self.tracker();self.measure(tracker, 0)
        image, mask, context = self.sample(1)
        image[46:55, 56:65] = 235
        row = tracker.measure(image, mask, timestamp=1., frame_index=1, context=context)
        self.assertEqual(row["reason"], "reference_markers_missing_or_ambiguous")
        image, mask, context = self.sample(2)
        matrix = np.array([[1., .12, -17.28], [0, 1, 0]])
        image = cv2.warpAffine(image, matrix, self.protocol["image_size"], flags=cv2.INTER_NEAREST, borderValue=(235, 235, 235))
        row = tracker.measure(image, mask, timestamp=2., frame_index=2, context=context)
        self.assertEqual(row["reason"], "reference_is_not_a_supported_similarity")

    def test_clipped_and_disconnected_seed_masks_are_unavailable(self):
        for reason in ("seed_silhouette_clipped", "seed_mask_identity_not_unique"):
            with self.subTest(reason=reason):
                image, mask, context = self.sample(0)
                if reason == "seed_silhouette_clipped": mask[0, 192] = 255
                else: mask[20:23, 20:23] = 255
                row = self.tracker().measure(image, mask, timestamp=0., frame_index=0, context=context)
                self.assertEqual(row["reason"], reason)

    def test_future_or_wrong_source_assertions_and_nonmonotonic_clock_are_refused(self):
        for changes in (dict(available_at_source_s=1.), dict(frame_index=2), dict(evidence_kind="real")):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): self.measure(self.tracker(), 0, **changes)
        tracker = self.tracker();self.measure(tracker, 0)
        with self.assertRaises(ValueError): self.measure(tracker, 0)

    def test_invalid_anchor_or_configuration_is_refused(self):
        bad = copy.deepcopy(self.config);bad["anchor_barycentric_weights"] = [-1., 1., 1.]
        with self.assertRaises(ValueError):
            ProjectedEdgeTracker(bad, self.protocol["marker_BGR"], run_id="test", clock_id="clock", evidence_kind="fixture")
        image, mask, context = self.sample(0)
        mask[:, :180] = 0
        # Move the remaining silhouette beyond the externally defined anchor.
        mask[:, :200] = 0
        row = self.tracker().measure(image, mask, timestamp=0., frame_index=0, context=context)
        self.assertEqual(row["reason"], "independent_anchor_outside_seed")
