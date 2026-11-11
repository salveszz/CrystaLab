# FILE PURPOSE: Track visible candidate masks causally and emit persistent inspection events.
# Startup, later appearance, unknown intervals and unresolved old identities remain
# distinct. Detector absence is never a physical healthy/resolved-state claim.
"""Bounded fixed-view candidate association using retained observation masks."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

KINDS = ("crystal_candidate", "inclusion_candidate")


@dataclass(frozen=True)
class CandidateConfig:
    schema: str = "crystalab_candidate_policy_v1"
    minimum_confirmations: int = 2
    minimum_duration_s: float = 0.0
    maximum_gap_s: float = 1.0
    minimum_iou: float = 0.2
    duplicate_iou: float = 0.9
    minimum_area_px: int = 12
    maximum_seed_overlap_fraction: float = 0.25
    minimum_inclusion_seed_fraction: float = 0.5
    maximum_tracks: int = 128
    fixed_view_assumed: bool = True

    def __post_init__(self):
        if (
            self.schema != "crystalab_candidate_policy_v1"
            or self.fixed_view_assumed is not True
        ):
            raise ValueError("Declare the supported fixed-view candidate policy")
        for key in ("minimum_confirmations", "minimum_area_px", "maximum_tracks"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        for key in (
            "minimum_duration_s",
            "maximum_gap_s",
            "minimum_iou",
            "duplicate_iou",
            "maximum_seed_overlap_fraction",
            "minimum_inclusion_seed_fraction",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if (
            self.maximum_gap_s <= 0
            or not 0 < self.minimum_iou <= 1
            or not self.minimum_iou <= self.duplicate_iou <= 1
        ):
            raise ValueError("Invalid gap/IoU policy")
        if (
            self.maximum_seed_overlap_fraction > 1
            or self.minimum_inclusion_seed_fraction > 1
        ):
            raise ValueError("Invalid seed-overlap policy")


def footprint(mask):
    x, y, w, h = cv2.boundingRect(mask)
    if not w or not h:
        raise ValueError("Empty candidate footprint")
    crop = (mask[y : y + h, x : x + w] > 0).copy()
    return dict(bbox=[x, y, w, h], crop=crop, area=int(np.count_nonzero(crop)))


def footprint_iou(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    x, y = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= x or bottom <= y:
        return 0.0
    intersection = int(
        np.count_nonzero(
            a["crop"][y - ay : bottom - ay, x - ax : right - ax]
            & b["crop"][y - by : bottom - by, x - bx : right - bx]
        )
    )
    return intersection / (a["area"] + b["area"] - intersection)


def footprint_mask(value, shape):
    mask = np.zeros(shape, np.uint8)
    x, y, w, h = value["bbox"]
    mask[y : y + h, x : x + w] = value["crop"].astype(np.uint8) * 255
    return mask


def extract_candidates(record, instances, seed_mask, config):
    usable = record["valid"] and not record.get("requires_inspection", False)
    observable = {kind: bool(usable) for kind in KINDS}
    issues = []
    detections = []
    if not usable:
        return (
            detections,
            observable,
            [
                dict(
                    reason="seed_or_view_uncertain",
                    detail=record.get("inspection_reason") or record.get("reason"),
                )
            ],
        )
    if seed_mask is None:
        raise ValueError("Usable seed observation omitted its mask")
    for obj in instances:
        if obj["assigned_role"] in {"seed_component", "compatible_seed_hypothesis"}:
            continue
        kind = "inclusion_candidate" if obj["class_id"] == 2 else "crystal_candidate"
        mask = obj["mask"]
        if (
            mask.shape != seed_mask.shape
            or mask.dtype != np.uint8
            or not set(np.unique(mask)) <= {0, 255}
        ):
            raise ValueError("Candidate mask does not match the original frame")
        if not math.isfinite(obj["score"]) or not 0 <= obj["score"] <= 1:
            raise ValueError("Invalid detector score")
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        eligible = [
            i
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= config.minimum_area_px
        ]
        if not eligible:
            continue
        reason = None
        if len(eligible) > 1:
            reason = "multiple_candidate_components"
        component = eligible[0]
        x, y, w, h, area = (int(v) for v in stats[component])
        selected = (components == component).astype(np.uint8) * 255
        if x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]:
            reason = "clipped_candidate"
        fraction = float(np.count_nonzero((selected > 0) & (seed_mask > 0)) / area)
        if (
            kind == "crystal_candidate"
            and fraction > config.maximum_seed_overlap_fraction
        ):
            reason = "candidate_identity_near_seed"
        if (
            kind == "inclusion_candidate"
            and fraction < config.minimum_inclusion_seed_fraction
        ):
            reason = "inclusion_not_localized_to_seed"
        if reason:
            observable[kind] = False
            issues.append(
                dict(
                    reason=reason,
                    candidate_type=kind,
                    instance_index=obj["instance_index"],
                )
            )
            continue
        candidate = dict(
            candidate_type=kind,
            footprint=footprint(selected),
            score=float(obj["score"]),
            source_instances=[obj["instance_index"]],
            raw_class_ids=[obj["class_id"]],
            component_id=component,
            seed_overlap_fraction=fraction,
        )
        alias = next(
            (
                d
                for d in detections
                if d["candidate_type"] == kind
                and footprint_iou(d["footprint"], candidate["footprint"])
                >= config.duplicate_iou
            ),
            None,
        )
        if alias is not None:
            alias["source_instances"].extend(candidate["source_instances"])
            alias["raw_class_ids"] = sorted(
                set(alias["raw_class_ids"] + candidate["raw_class_ids"])
            )
            if candidate["score"] > alias["score"]:
                alias.update(
                    footprint=candidate["footprint"],
                    score=candidate["score"],
                    component_id=component,
                    seed_overlap_fraction=fraction,
                )
        else:
            detections.append(candidate)
    return detections, observable, issues


class CandidateTracker:
    """Require unique admissible mask associations and consecutive usable confirmations.

    Retired footprints stay bounded by maximum_tracks; reaching capacity fails
    instead of forgetting old identities and falsely calling a return 'new'.
    """

    def __init__(self, config, run_id):
        self.config = config
        self.run_id = run_id
        self.tracks = {}
        self.step = -1
        self.previous_time = None
        self.previous_frame = None
        self.event_count = 0
        self.previous_usable = {kind: False for kind in KINDS}
        self.last_usable = {kind: None for kind in KINDS}

    def update(self, frame_index, timestamp, detections, observable):
        if (
            type(frame_index) is not int
            or frame_index < 0
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            raise ValueError(
                "Candidate observation needs finite time and a frame index"
            )
        if self.previous_time is not None and (
            timestamp <= self.previous_time or frame_index <= self.previous_frame
        ):
            raise ValueError("Candidate source frames/times must strictly increase")
        self.step += 1
        config = self.config
        events = []
        issues = []
        assignments = {}
        usable = {kind: bool(observable[kind]) for kind in KINDS}
        for track in self.tracks.values():
            if timestamp - track["last_time_s"] > config.maximum_gap_s:
                track.update(state="expired", streak=0)
        for kind in KINDS:
            indices = [
                i for i, d in enumerate(detections) if d["candidate_type"] == kind
            ]
            active = [
                t
                for t in self.tracks.values()
                if t["candidate_type"] == kind and t["state"] != "expired"
            ]
            edges = {
                (track["id"], i)
                for track in active
                for i in indices
                if footprint_iou(track["footprint"], detections[i]["footprint"])
                >= config.minimum_iou
            }
            if any(
                sum(t == track["id"] for t, _ in edges) > 1 for track in active
            ) or any(sum(j == i for _, j in edges) > 1 for i in indices):
                usable[kind] = False
                issues.append(
                    dict(reason="ambiguous_candidate_association", candidate_type=kind)
                )
            if not usable[kind]:
                for track in active:
                    track.update(state="unobserved", streak=0)
                continue
            matched = {i: t for t, i in edges}
            for i in indices:
                detection = detections[i]
                if i in matched:
                    track = self.tracks[matched[i]]
                    contiguous = (
                        track["last_step"] == self.step - 1
                        and self.previous_usable[kind]
                    )
                    track["streak"] = track["streak"] + 1 if contiguous else 1
                    if not contiguous:
                        track["streak_start_s"] = timestamp
                    track["observations"] += 1
                else:
                    if len(self.tracks) >= config.maximum_tracks:
                        raise ValueError(
                            "Candidate track capacity reached; review/reinitialize explicitly"
                        )
                    previous = self.last_usable[kind]
                    prior = [
                        t["id"]
                        for t in self.tracks.values()
                        if t["candidate_type"] == kind
                        and t["state"] == "expired"
                        and footprint_iou(t["footprint"], detection["footprint"])
                        >= config.minimum_iou
                    ]
                    origin = (
                        "identity_unresolved_after_gap"
                        if prior
                        else "present_at_first_usable_frame"
                        if previous is None
                        else "appeared_after_clear_view"
                        if self.previous_usable[kind]
                        and timestamp - self.previous_time <= config.maximum_gap_s
                        else "appeared_after_unknown_interval"
                    )
                    identifier = f"{self.run_id}:candidate:{len(self.tracks) + 1:06d}"
                    track = dict(
                        id=identifier,
                        candidate_type=kind,
                        origin=origin,
                        possible_prior_track_ids=prior,
                        first_frame_index=frame_index,
                        first_time_s=timestamp,
                        first_footprint=detection["footprint"],
                        first_detection_ref=detection.get("observation_ref"),
                        onset_lower_s=None if previous is None else previous["time_s"],
                        onset_lower_frame_index=None
                        if previous is None
                        else previous["frame_index"],
                        streak=1,
                        streak_start_s=timestamp,
                        observations=1,
                        event_id=None,
                        raw_class_ids=[],
                        score_min=detection["score"],
                        score_max=detection["score"],
                    )
                    self.tracks[identifier] = track
                track.update(
                    last_frame_index=frame_index,
                    last_time_s=timestamp,
                    last_step=self.step,
                    footprint=detection["footprint"],
                    state="observed",
                    last_detection_ref=detection.get("observation_ref"),
                )
                track["raw_class_ids"] = sorted(
                    set(track["raw_class_ids"] + detection["raw_class_ids"])
                )
                track["score_min"] = min(track["score_min"], detection["score"])
                track["score_max"] = max(track["score_max"], detection["score"])
                assignments[i] = track["id"]
                if (
                    track["event_id"] is None
                    and track["streak"] >= config.minimum_confirmations
                    and timestamp - track["streak_start_s"] >= config.minimum_duration_s
                ):
                    self.event_count += 1
                    track["event_id"] = (
                        f"{self.run_id}:candidate_event:{self.event_count:06d}"
                    )
                    event = dict(
                        event_id=track["event_id"],
                        track_id=track["id"],
                        candidate_type=kind,
                        origin=track["origin"],
                        kind={
                            "present_at_first_usable_frame": "startup_candidate",
                            "appeared_after_clear_view": "new_visible_candidate",
                            "appeared_after_unknown_interval": "candidate_after_unknown_interval",
                            "identity_unresolved_after_gap": "candidate_identity_unresolved",
                        }[track["origin"]],
                        first_frame_index=track["first_frame_index"],
                        first_detection_time_s=track["first_time_s"],
                        onset_interval_source_s=[
                            track["onset_lower_s"],
                            track["first_time_s"],
                        ],
                        onset_lower_frame_index=track["onset_lower_frame_index"],
                        alert_frame_index=frame_index,
                        alert_time_s=timestamp,
                        confirmation_delay_s=timestamp - track["first_time_s"],
                        consecutive_observations=track["streak"],
                        confirmation_window_start_s=track["streak_start_s"],
                        possible_prior_track_ids=track["possible_prior_track_ids"],
                        raw_class_ids=track["raw_class_ids"],
                        score_range=[track["score_min"], track["score_max"]],
                        first_detection_ref=track["first_detection_ref"],
                        confirmation_ref=track["last_detection_ref"],
                        bbox_px=track["footprint"]["bbox"],
                        requires_operator_review=True,
                        physical_defect_confirmed=False,
                        score_is_calibrated=False,
                        clock="source_video_time_s",
                        first_footprint=track["first_footprint"],
                    )
                    events.append(event)
            matched_tracks = set(assignments.values())
            for track in active:
                if track["id"] not in matched_tracks:
                    track.update(state="not_detected", streak=0)
        intervals = []
        for kind in KINDS:
            if (
                usable[kind]
                and self.previous_usable[kind]
                and timestamp - self.previous_time <= config.maximum_gap_s
            ):
                intervals.append(
                    dict(
                        candidate_type=kind, start_s=self.previous_time, end_s=timestamp
                    )
                )
            if usable[kind]:
                self.last_usable[kind] = dict(frame_index=frame_index, time_s=timestamp)
        self.previous_usable = usable
        self.previous_time = timestamp
        self.previous_frame = frame_index
        states = [
            {
                k: v
                for k, v in track.items()
                if k not in {"footprint", "first_footprint"}
            }
            for track in self.tracks.values()
        ]
        return dict(
            assignments=assignments,
            tracks=states,
            events=events,
            observable=usable,
            issues=issues,
            usable_intervals=intervals,
        )
