# FILE PURPOSE: Turn causal visible candidates into bounded, evidence-linked review proposals.
# Startup/new/gap origins, unknown views, expiry and scripted simulation decisions
# remain explicit. This baseline has no hardware driver or adaptive forecast policy.
"""Stateful event-rule baseline; proposals do not constitute operator approval."""

import copy
from dataclasses import asdict, dataclass
import hashlib
import json

from .contracts import finite, identifier

KINDS = ("crystal_candidate", "inclusion_candidate")
ORIGINS = {
    "present_at_first_usable_frame",
    "appeared_after_clear_view",
    "appeared_after_unknown_interval",
    "identity_unresolved_after_gap",
}


def value_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RuleConfig:
    schema: str = "crystalab_event_rule_policy_v1"
    maximum_evidence_age_s: float = 1.0
    maximum_packet_gap_s: float = 1.0
    proposal_lifetime_s: float = 2.0
    maximum_proposals: int = 128

    def __post_init__(self):
        if self.schema != "crystalab_event_rule_policy_v1":
            raise ValueError("Unsupported event-rule policy")
        for key in (
            "maximum_evidence_age_s",
            "maximum_packet_gap_s",
            "proposal_lifetime_s",
        ):
            if finite(getattr(self, key), key, minimum=0) == 0:
                raise ValueError("Rule clock limits must be positive")
        if type(self.maximum_proposals) is not int or self.maximum_proposals < 1:
            raise ValueError("Rule proposal capacity must be a positive integer")


@dataclass(frozen=True)
class RuleEvidence:
    run_id: str
    clock_id: str
    packet_id: str
    frame_index: int
    observed_at_s: float
    received_at_s: float
    evidence_kind: str
    crystal_observable: bool
    inclusion_observable: bool
    events: tuple[dict, ...] = ()
    issues: tuple[str, ...] = ()
    visible_track_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("run_id", "clock_id", "packet_id"):
            identifier(getattr(self, key), key)
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("Rule evidence needs an original frame index")
        finite(self.observed_at_s, "evidence time", minimum=0)
        finite(self.received_at_s, "evidence receipt", minimum=0)
        if self.observed_at_s > self.received_at_s:
            raise ValueError("Evidence cannot be received before observation")
        if self.evidence_kind not in {"fixture", "synthetic", "real"}:
            raise ValueError("Declare the rule evidence kind")
        if (
            type(self.crystal_observable) is not bool
            or type(self.inclusion_observable) is not bool
        ):
            raise ValueError("Candidate visibility must be explicit")
        if not isinstance(self.events, tuple) or not isinstance(self.issues, tuple):
            raise ValueError("Rule events/issues must be immutable tuples")
        for event in self.events:
            validate_event(event, self.observed_at_s, self.frame_index)
        if not isinstance(self.visible_track_ids, tuple):
            raise ValueError("Visible track IDs must be an immutable tuple")
        for track in self.visible_track_ids:
            identifier(track, "visible track ID")
        value_sha256(asdict(self))


def validate_event(event, observed, frame):
    for key in ("event_id", "track_id", "first_detection_mask_path", "evidence_path"):
        identifier(event[key], key)
    if event["candidate_type"] not in KINDS or event["origin"] not in ORIGINS:
        raise ValueError("Unsupported candidate type/origin")
    finite(event["alert_time_s"], "candidate alert clock", minimum=0)
    if (
        event["alert_time_s"] > observed
        or type(event["alert_frame_index"]) is not int
        or not 0 <= event["alert_frame_index"] <= frame
    ):
        raise ValueError("Future candidate evidence cannot drive rules")
    if (
        event["clock"] != "source_video_time_s"
        or event["requires_operator_review"] is not True
        or event["physical_defect_confirmed"] is not False
    ):
        raise ValueError(
            "Candidate protocol must preserve unreviewed source-clock semantics"
        )
    box = event["bbox_px"]
    if (
        len(box) != 4
        or any(type(v) is not int for v in box)
        or min(box[:2]) < 0
        or min(box[2:]) <= 0
    ):
        raise ValueError("Candidate localization is invalid")


class RuleSupervisor:
    """Emit one proposal per candidate and one inspection per unknown-view episode.

    Tick and evidence times use the explicitly bound source clock. Pending actions
    expire; source visibility returning closes its inspection episode only, never
    certifies healthy quality. No action is automatically approved.
    """

    def __init__(self, config, *, run_id, clock_id, evidence_kind):
        identifier(run_id, "run_id")
        identifier(clock_id, "clock_id")
        if evidence_kind not in {"fixture", "synthetic", "real"}:
            raise ValueError("Unsupported supervisory evidence kind")
        self.config, self.run_id, self.clock_id = config, run_id, clock_id
        self.evidence_kind = evidence_kind
        self.time_s = None
        self.packet = None
        self.packet_signature = None
        self.packet_gap = False
        self.proposals, self.seen_events = {}, {}
        self.decision_ids = set()
        self.active_inspection = None
        self.unknown_episode = False
        self.observable = {kind: False for kind in KINDS}

    def _propose(self, kind, now, *, event=None, reasons=()):
        if len(self.proposals) >= self.config.maximum_proposals:
            raise ValueError(
                "Supervisory proposal capacity reached; explicitly review/reinitialize"
            )
        identifier = f"{self.run_id}:proposal:{len(self.proposals) + 1:06d}"
        payload = dict(
            proposal_id=identifier,
            run_id=self.run_id,
            clock_id=self.clock_id,
            kind=kind,
            proposed_at_s=now,
            expires_at_s=now + self.config.proposal_lifetime_s,
            evidence_kind=self.evidence_kind,
            packet_id=None if self.packet is None else self.packet.packet_id,
            observed_at_s=None if self.packet is None else self.packet.observed_at_s,
            source_frame_index=None if self.packet is None else self.packet.frame_index,
            candidate_event_id=None if event is None else event["event_id"],
            track_id=None if event is None else event["track_id"],
            origin=None if event is None else event["origin"],
            bbox_px=None if event is None else list(event["bbox_px"]),
            first_detection_mask_path=None
            if event is None
            else event["first_detection_mask_path"],
            source_evidence_path=None if event is None else event["evidence_path"],
            recommendation={
                "ramp_review": "review_ramp_pause",
                "candidate_inspection": "inspect_candidate_identity",
                "inclusion_alert": "inspect_and_mark_inclusion",
                "view_inspection": "inspect_view",
            }[kind],
            reasons=list(reasons),
            requires_operator_review=True,
            operator_reviewed=False,
            physical_defect_confirmed=False,
            numerical_actuator_command=None,
        )
        entry = dict(
            proposal=payload,
            proposal_sha256=value_sha256(payload),
            status="pending_review",
        )
        self.proposals[identifier] = entry
        return copy.deepcopy(entry)

    def tick(self, now_s, packet=None):
        now = finite(now_s, "supervisor clock", minimum=0)
        if self.time_s is not None and now <= self.time_s:
            raise ValueError("Supervisor clock must strictly increase")
        gap = False
        new_packet = False
        if packet is not None:
            if not isinstance(packet, RuleEvidence):
                raise ValueError("Use RuleEvidence packets")
            if (
                packet.run_id != self.run_id
                or packet.clock_id != self.clock_id
                or packet.evidence_kind != self.evidence_kind
            ):
                raise ValueError("Rule evidence run/clock/kind identity differs")
            if packet.received_at_s > now:
                raise ValueError("Rule evidence has not yet arrived")
            signature = value_sha256(asdict(packet))
            if self.packet is not None:
                if packet.frame_index == self.packet.frame_index:
                    if signature != self.packet_signature:
                        raise ValueError("Repeated evidence frame changed")
                elif (
                    packet.frame_index < self.packet.frame_index
                    or packet.observed_at_s <= self.packet.observed_at_s
                ):
                    raise ValueError("Rule evidence frames/clocks must increase")
                else:
                    gap = (
                        packet.observed_at_s - self.packet.observed_at_s
                        > self.config.maximum_packet_gap_s
                    )
                    new_packet = True
            else:
                new_packet = True
            if new_packet:
                self.packet, self.packet_signature = copy.deepcopy(packet), signature
                self.packet_gap = gap
        self.time_s = now
        created, transitions, ignored = [], [], []
        for entry in self.proposals.values():
            if (
                entry["status"] == "pending_review"
                and now >= entry["proposal"]["expires_at_s"]
            ):
                entry["status"] = "expired"
                transitions.append(
                    dict(
                        proposal_id=entry["proposal"]["proposal_id"],
                        status="expired",
                        time_s=now,
                    )
                )
        reasons = []
        current = self.packet
        if current is None:
            reasons = ["missing_vision_evidence"]
            self.observable = {kind: False for kind in KINDS}
        elif now - current.observed_at_s > self.config.maximum_evidence_age_s:
            reasons = ["stale_vision_evidence"]
            self.observable = {kind: False for kind in KINDS}
        else:
            self.observable = dict(
                crystal_candidate=current.crystal_observable,
                inclusion_candidate=current.inclusion_observable,
            )
            if not all(self.observable.values()):
                reasons = list(current.issues) or ["candidate_view_uncertain"]
            if self.packet_gap:
                reasons.append("vision_packet_gap")
                self.observable = {kind: False for kind in KINDS}
        unknown = not all(self.observable.values())
        if unknown and not self.unknown_episode:
            entry = self._propose("view_inspection", now, reasons=reasons)
            self.active_inspection = entry["proposal"]["proposal_id"]
            created.append(entry)
        elif not unknown and self.unknown_episode:
            entry = self.proposals.get(self.active_inspection)
            if entry is not None and entry["status"] == "pending_review":
                entry["status"] = "visibility_restored"
                transitions.append(
                    dict(
                        proposal_id=self.active_inspection,
                        status="visibility_restored",
                        time_s=now,
                        physical_quality_healthy=None,
                    )
                )
            self.active_inspection = None
        self.unknown_episode = unknown
        if new_packet:
            for event in current.events:
                event_hash = value_sha256(event)
                if event["event_id"] in self.seen_events:
                    if self.seen_events[event["event_id"]] != event_hash:
                        raise ValueError(
                            "Candidate event ID reused with different evidence"
                        )
                    continue
                if len(self.seen_events) >= self.config.maximum_proposals:
                    raise ValueError(
                        "Supervisory event capacity reached; explicitly review/reinitialize"
                    )
                self.seen_events[event["event_id"]] = event_hash
                if (
                    not self.observable[event["candidate_type"]]
                    or now - event["alert_time_s"] > self.config.maximum_evidence_age_s
                ):
                    ignored.append(
                        dict(
                            event_id=event["event_id"],
                            reason="uncertain_or_stale_candidate_evidence",
                        )
                    )
                    continue
                kind = (
                    "inclusion_alert"
                    if event["candidate_type"] == "inclusion_candidate"
                    else "ramp_review"
                    if event["origin"] == "appeared_after_clear_view"
                    else "candidate_inspection"
                )
                created.append(self._propose(kind, now, event=event))
        return dict(
            time_s=now,
            clock_id=self.clock_id,
            observable=dict(self.observable),
            reasons=reasons,
            created_proposals=created,
            transitions=transitions,
            ignored_events=ignored,
            pending_proposal_ids=[
                key
                for key, entry in self.proposals.items()
                if entry["status"] == "pending_review"
            ],
            numerical_actuator_commands=[],
            physical_quality_healthy=None,
        )

    def decide_for_simulation(
        self, proposal_id, *, proposal_sha256, decision, decision_id, at_s
    ):
        """Record an explicit fixture policy decision, never a human label review.

        Only fixture/synthetic evidence is eligible. A fresh, observable crystal
        view is required to authorize a ramp proposal. Real shadow proposals cannot
        enter this simulation dispatch path.
        """
        identifier(decision_id, "simulation decision ID")
        if decision_id in self.decision_ids:
            raise ValueError("Simulation decision IDs must be unique")
        finite(at_s, "simulation decision time", minimum=0)
        if self.evidence_kind == "real":
            raise ValueError(
                "Real shadow proposals cannot be authorized for simulated actuation"
            )
        if at_s != self.time_s:
            raise ValueError("Decide only after the current supervisor tick")
        if decision not in {"authorize", "decline"}:
            raise ValueError("Unsupported scripted simulation decision")
        entry = self.proposals[proposal_id]
        if proposal_sha256 != entry["proposal_sha256"]:
            raise ValueError("Simulation decision does not bind the exact proposal")
        if entry["status"] != "pending_review":
            raise ValueError(
                "Only pending proposals can receive a new simulation decision"
            )
        if decision == "authorize" and (
            entry["proposal"]["kind"] != "ramp_review"
            or not self.observable["crystal_candidate"]
            or entry["proposal"]["track_id"] not in self.packet.visible_track_ids
        ):
            raise ValueError(
                "Simulation pause needs a ramp proposal and fresh usable crystal evidence"
            )
        entry["status"] = (
            "simulation_authorized"
            if decision == "authorize"
            else "simulation_declined"
        )
        self.decision_ids.add(decision_id)
        return dict(
            decision_id=decision_id,
            proposal_id=proposal_id,
            proposal_sha256=proposal_sha256,
            time_s=at_s,
            clock_id=self.clock_id,
            status=entry["status"],
            action="pause_recipe" if decision == "authorize" else None,
            basis="scripted_fixture_policy",
            operator_reviewed=False,
            physical_hardware_authorized=False,
        )
