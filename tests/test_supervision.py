# FILE PURPOSE: Verify origin-aware review rules, clock validity, bounded state and simulation decisions.
# Constructed candidates are unreviewed fixtures; no test creates a real annotation
# receipt or claims a human/equipment approval.

from dataclasses import replace
import unittest

from controllers.supervision import RuleConfig, RuleEvidence, RuleSupervisor


def event(
    identifier="event:1",
    *,
    kind="crystal_candidate",
    origin="appeared_after_clear_view",
):
    return dict(
        event_id=identifier,
        track_id="track:1",
        candidate_type=kind,
        origin=origin,
        alert_time_s=0.0,
        alert_frame_index=0,
        clock="source_video_time_s",
        requires_operator_review=True,
        physical_defect_confirmed=False,
        first_detection_mask_path="masks/first.png",
        evidence_path="evidence/frame.png",
        bbox_px=[10, 20, 8, 9],
    )


def packet(time=0, frame=0, events=(), **changes):
    fields = dict(
        run_id="run",
        clock_id="clock",
        packet_id=f"frame:{frame}",
        frame_index=frame,
        observed_at_s=time,
        received_at_s=time,
        evidence_kind="fixture",
        crystal_observable=True,
        inclusion_observable=True,
        events=tuple(events),
        visible_track_ids=("track:1",),
    )
    fields.update(changes)
    return RuleEvidence(**fields)


class SupervisoryRulesTests(unittest.TestCase):
    def supervisor(self, *, evidence_kind="fixture", **changes):
        return RuleSupervisor(
            replace(RuleConfig(), **changes),
            run_id="run",
            clock_id="clock",
            evidence_kind=evidence_kind,
        )

    def test_startup_gap_and_new_candidates_have_different_operator_proposals(self):
        cases = [
            ("present_at_first_usable_frame", "candidate_inspection"),
            ("appeared_after_unknown_interval", "candidate_inspection"),
            ("identity_unresolved_after_gap", "candidate_inspection"),
            ("appeared_after_clear_view", "ramp_review"),
        ]
        for origin, kind in cases:
            with self.subTest(origin=origin):
                result = self.supervisor().tick(
                    0, packet(events=[event(origin=origin)])
                )
                proposal = result["created_proposals"][0]["proposal"]
                self.assertEqual(proposal["kind"], kind)
                self.assertTrue(proposal["requires_operator_review"])
                self.assertFalse(proposal["operator_reviewed"])
                self.assertIsNone(proposal["numerical_actuator_command"])
                self.assertEqual(result["numerical_actuator_commands"], [])

    def test_inclusion_alert_retains_location_and_never_proposes_temperature(self):
        result = self.supervisor().tick(
            0, packet(events=[event(kind="inclusion_candidate")])
        )
        proposal = result["created_proposals"][0]["proposal"]
        self.assertEqual(proposal["kind"], "inclusion_alert")
        self.assertEqual(proposal["bbox_px"], [10, 20, 8, 9])
        self.assertEqual(proposal["recommendation"], "inspect_and_mark_inclusion")
        self.assertIsNone(proposal["numerical_actuator_command"])

    def test_repeated_packet_or_event_cannot_duplicate_proposals_or_change_its_evidence(
        self,
    ):
        supervisor = self.supervisor()
        first = packet(events=[event()])
        supervisor.tick(0, first)
        self.assertEqual(supervisor.tick(0.1, first)["created_proposals"], [])
        self.assertEqual(
            supervisor.tick(0.2, packet(0.2, 1, [event()]))["created_proposals"], []
        )
        changed = event()
        changed["bbox_px"][0] = 11
        with self.assertRaisesRegex(ValueError, "reused with different"):
            supervisor.tick(0.3, packet(0.3, 2, [changed]))

    def test_unknown_view_emits_one_inspection_and_restoration_does_not_certify_quality(
        self,
    ):
        supervisor = self.supervisor()
        first = supervisor.tick(
            0,
            packet(
                crystal_observable=False,
                issues=("seed_or_view_uncertain",),
                events=[event()],
            ),
        )
        self.assertEqual(
            first["created_proposals"][0]["proposal"]["kind"], "view_inspection"
        )
        self.assertEqual(len(first["ignored_events"]), 1)
        second = supervisor.tick(0.1, packet(0.1, 1, crystal_observable=False))
        self.assertEqual(second["created_proposals"], [])
        restored = supervisor.tick(0.2, packet(0.2, 2))
        self.assertEqual(restored["transitions"][0]["status"], "visibility_restored")
        self.assertIsNone(restored["physical_quality_healthy"])
        again = supervisor.tick(0.3, packet(0.3, 3, inclusion_observable=False))
        self.assertEqual(len(again["created_proposals"]), 1)

    def test_stale_missing_and_packet_gaps_require_a_new_usable_view(self):
        supervisor = self.supervisor(maximum_packet_gap_s=0.2)
        supervisor.tick(0, packet(events=[event()]))
        gap = supervisor.tick(0.5, packet(0.5, 1))
        self.assertIn("vision_packet_gap", gap["reasons"])
        self.assertFalse(supervisor.tick(0.6)["observable"]["crystal_candidate"])
        self.assertTrue(
            supervisor.tick(0.7, packet(0.7, 2))["observable"]["crystal_candidate"]
        )
        self.assertFalse(supervisor.tick(1.8)["observable"]["crystal_candidate"])
        missing = self.supervisor().tick(0)
        self.assertEqual(missing["reasons"], ["missing_vision_evidence"])

    def test_simulated_authorization_binds_exact_proposal_and_keeps_human_review_false(
        self,
    ):
        supervisor = self.supervisor()
        entry = supervisor.tick(0, packet(events=[event()]))["created_proposals"][0]
        proposal_id = entry["proposal"]["proposal_id"]
        with self.assertRaisesRegex(ValueError, "exact proposal"):
            supervisor.decide_for_simulation(
                proposal_id,
                proposal_sha256="wrong",
                decision="authorize",
                decision_id="decision:0",
                at_s=0,
            )
        receipt = supervisor.decide_for_simulation(
            proposal_id,
            proposal_sha256=entry["proposal_sha256"],
            decision="authorize",
            decision_id="decision:1",
            at_s=0,
        )
        self.assertEqual(receipt["action"], "pause_recipe")
        self.assertFalse(receipt["operator_reviewed"])
        self.assertFalse(receipt["physical_hardware_authorized"])
        with self.assertRaisesRegex(ValueError, "unique"):
            supervisor.decide_for_simulation(
                proposal_id,
                proposal_sha256=entry["proposal_sha256"],
                decision="authorize",
                decision_id="decision:1",
                at_s=0,
            )

    def test_expired_disappeared_and_real_evidence_cannot_authorize_simulated_pause(
        self,
    ):
        for kind in ["fixture", "real"]:
            supervisor = self.supervisor(evidence_kind=kind)
            entry = supervisor.tick(0, packet(events=[event()], evidence_kind=kind))[
                "created_proposals"
            ][0]
            if kind == "fixture":
                supervisor.tick(0.1, packet(0.1, 1, visible_track_ids=()))
            with self.assertRaises(ValueError):
                supervisor.decide_for_simulation(
                    entry["proposal"]["proposal_id"],
                    proposal_sha256=entry["proposal_sha256"],
                    decision="authorize",
                    decision_id="no",
                    at_s=supervisor.time_s,
                )
        supervisor = self.supervisor(maximum_evidence_age_s=5, maximum_packet_gap_s=5)
        entry = supervisor.tick(0, packet(events=[event()]))["created_proposals"][0]
        expired = supervisor.tick(2.1, packet(2.1, 1))
        self.assertEqual(expired["transitions"][0]["status"], "expired")
        with self.assertRaisesRegex(ValueError, "pending"):
            supervisor.decide_for_simulation(
                entry["proposal"]["proposal_id"],
                proposal_sha256=entry["proposal_sha256"],
                decision="authorize",
                decision_id="late",
                at_s=2.1,
            )

    def test_clock_identity_future_data_and_capacity_fail_explicitly(self):
        for change in [
            dict(run_id="other"),
            dict(clock_id="other"),
            dict(evidence_kind="real"),
            dict(received_at_s=1),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.supervisor().tick(0, packet(**change))
        with self.assertRaisesRegex(ValueError, "Future candidate"):
            future = event()
            future["alert_time_s"] = 1
            packet(events=[future])
        supervisor = self.supervisor(maximum_proposals=1)
        supervisor.tick(0, packet(events=[event()]))
        with self.assertRaisesRegex(ValueError, "capacity"):
            supervisor.tick(0.1, packet(0.1, 1, [event("event:2")]))
        for change in [dict(maximum_evidence_age_s=0), dict(maximum_proposals=True)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(RuleConfig(), **change)
