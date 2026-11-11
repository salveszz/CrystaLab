# FILE PURPOSE: Verify causal candidate identity, persistence and unknown-interval semantics.
# Analytic masks independently exercise startup/new/gap distinctions, false
# persistence through missing views, split ambiguity and bounded old-ID memory.

from dataclasses import replace
import unittest

import numpy as np

from vision.candidates import (
    CandidateConfig,
    CandidateTracker,
    extract_candidates,
    footprint,
    footprint_iou,
    KINDS,
)


def mask(x=50, y=50, w=20, h=20):
    result = np.zeros((120, 160), np.uint8)
    result[y : y + h, x : x + w] = 255
    return result


def detection(x=50, y=50, w=20, h=20, kind="crystal_candidate", raw=1):
    return dict(
        candidate_type=kind,
        footprint=footprint(mask(x, y, w, h)),
        score=0.9,
        raw_class_ids=[raw],
        source_instances=[0],
    )


def usable(value=True):
    return {kind: value for kind in KINDS}


class CandidateTrackingTests(unittest.TestCase):
    def setUp(self):
        self.config = CandidateConfig()
        self.tracker = CandidateTracker(self.config, "fixture")

    def test_cropped_footprints_preserve_exact_iou(self):
        self.assertEqual(
            footprint_iou(footprint(mask(w=40)), footprint(mask(w=20))), 0.5
        )
        self.assertEqual(footprint_iou(footprint(mask()), footprint(mask(x=100))), 0.0)

    def test_startup_and_new_appearance_have_separate_persistent_events(self):
        first = self.tracker.update(0, 0, [detection()], usable())
        self.assertEqual(first["events"], [])
        second = self.tracker.update(1, 0.25, [detection(), detection(x=100)], usable())
        self.assertEqual([e["kind"] for e in second["events"]], ["startup_candidate"])
        third = self.tracker.update(2, 0.5, [detection(), detection(x=100)], usable())
        self.assertEqual(
            [e["kind"] for e in third["events"]], ["new_visible_candidate"]
        )
        event = third["events"][0]
        self.assertEqual(event["onset_interval_source_s"], [0, 0.25])
        self.assertEqual(event["confirmation_delay_s"], 0.25)
        self.assertTrue(event["requires_operator_review"])
        self.assertFalse(event["physical_defect_confirmed"])
        self.assertEqual(third["assignments"][0], first["assignments"][0])

    def test_transient_and_missed_detections_do_not_accumulate_confirmations(self):
        self.tracker.update(0, 0, [], usable())
        self.assertEqual(
            self.tracker.update(1, 0.1, [detection()], usable())["events"], []
        )
        self.tracker.update(2, 0.2, [], usable())
        self.assertEqual(
            self.tracker.update(3, 0.3, [detection()], usable())["events"], []
        )
        self.assertEqual(
            len(self.tracker.update(4, 0.4, [detection()], usable())["events"]), 1
        )
        self.assertEqual(
            self.tracker.update(5, 0.5, [detection()], usable())["events"], []
        )

    def test_unknown_view_breaks_streak_and_unknown_interval_is_not_new(self):
        self.tracker.update(0, 0, [], usable())
        gap = self.tracker.update(1, 0.1, [], usable(False))
        self.assertEqual(gap["usable_intervals"], [])
        self.tracker.update(2, 0.2, [detection()], usable())
        result = self.tracker.update(3, 0.3, [detection()], usable())
        self.assertEqual(
            result["events"][0]["kind"], "candidate_after_unknown_interval"
        )
        self.assertEqual(result["events"][0]["onset_interval_source_s"], [0, 0.2])
        self.assertEqual(result["usable_intervals"][0]["start_s"], 0.2)

    def test_first_usable_view_after_initial_occlusion_is_still_startup(self):
        self.tracker.update(0, 0, [], usable(False))
        self.tracker.update(1, 0.1, [detection()], usable())
        result = self.tracker.update(2, 0.2, [detection()], usable())
        self.assertEqual(result["events"][0]["kind"], "startup_candidate")
        self.assertEqual(result["events"][0]["onset_interval_source_s"], [None, 0.1])

    def test_old_identity_is_not_silently_reused_or_called_new_after_long_gap(self):
        old = self.tracker.update(0, 0, [detection()], usable())["assignments"][0]
        self.tracker.update(1, 0.1, [detection()], usable())
        first = self.tracker.update(2, 1.5, [detection()], usable())
        new = first["assignments"][0]
        self.assertNotEqual(old, new)
        result = self.tracker.update(3, 1.6, [detection()], usable())
        self.assertEqual(result["events"][0]["kind"], "candidate_identity_unresolved")
        self.assertEqual(result["events"][0]["possible_prior_track_ids"], [old])

    def test_split_association_is_unknown_and_does_not_spawn_false_new_ids(self):
        self.tracker.update(0, 0, [detection(w=40)], usable())
        split = self.tracker.update(
            1, 0.1, [detection(w=20), detection(x=70, w=20)], usable()
        )
        self.assertFalse(split["observable"]["crystal_candidate"])
        self.assertEqual(len(split["tracks"]), 1)
        self.assertEqual(split["assignments"], {})
        self.tracker.update(2, 0.2, [detection(w=40)], usable())
        result = self.tracker.update(3, 0.3, [detection(w=40)], usable())
        self.assertEqual(result["events"][0]["kind"], "startup_candidate")

    def test_raw_crystal_role_label_changes_do_not_change_candidate_identity(self):
        first = self.tracker.update(0, 0, [detection(raw=0)], usable())
        second = self.tracker.update(1, 0.1, [detection(raw=1)], usable())
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(second["events"][0]["raw_class_ids"], [0, 1])

    def test_capacity_and_bad_time_fail_instead_of_forgetting_identity(self):
        tracker = CandidateTracker(replace(self.config, maximum_tracks=1), "bounded")
        tracker.update(0, 0, [detection()], usable())
        with self.assertRaisesRegex(ValueError, "capacity"):
            tracker.update(1, 0.1, [detection(x=100)], usable())
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.tracker.update(0, 0, [], usable())
            self.tracker.update(0, 0, [], usable())
        for setting in [
            dict(minimum_confirmations=True),
            dict(maximum_gap_s=0),
            dict(minimum_duration_s=float("nan")),
            dict(fixed_view_assumed=False),
            dict(duplicate_iou=0.1),
        ]:
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                replace(self.config, **setting)

    def test_extraction_excludes_seed_roles_localizes_inclusions_and_retains_aliases(
        self,
    ):
        seed = mask(x=20, y=20, w=40, h=40)
        outside = mask(x=100, y=50, w=15, h=15)
        inside = mask(x=30, y=30, w=6, h=6)
        objects = [
            dict(
                instance_index=0,
                class_id=0,
                assigned_role="seed_component",
                score=0.9,
                mask=seed,
            ),
            dict(
                instance_index=1,
                class_id=1,
                assigned_role="unassigned_crystal_candidate",
                score=0.9,
                mask=outside,
            ),
            dict(
                instance_index=2,
                class_id=0,
                assigned_role="unassigned_crystal_candidate",
                score=0.8,
                mask=outside,
            ),
            dict(
                instance_index=3,
                class_id=2,
                assigned_role="unverified_inclusion_candidate",
                score=0.9,
                mask=inside,
            ),
        ]
        candidates, observable, issues = extract_candidates(
            dict(valid=True, requires_inspection=False), objects, seed, self.config
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(observable.values()))
        self.assertEqual(issues, [])
        self.assertEqual(candidates[0]["source_instances"], [1, 2])
        self.assertEqual(candidates[0]["raw_class_ids"], [0, 1])
        objects[-1]["mask"] = outside
        _, observable, issues = extract_candidates(
            dict(valid=True), objects, seed, self.config
        )
        self.assertFalse(observable["inclusion_candidate"])
        self.assertEqual(issues[-1]["reason"], "inclusion_not_localized_to_seed")
        _, observable, _ = extract_candidates(
            dict(valid=True, requires_inspection=True), objects, seed, self.config
        )
        self.assertFalse(any(observable.values()))
