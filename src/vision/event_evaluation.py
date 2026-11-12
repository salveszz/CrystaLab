# FILE PURPOSE: Score candidate inspection events against independent visible-onset annotations.
# Constructed fixtures/proxies stay distinct from real labels, which require an
# explicit exact-byte review receipt. Source-video seconds never imply physical hours.
"""One-to-one event matching, origin diagnostics, misses and negative coverage."""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.candidates import KINDS, footprint, footprint_iou
from vision.training_data import safe_path, save_json
from vision.video import file_sha256


def json_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def load_annotations(path, video_hash, *, require_review=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("uncertain", False):
        raise ValueError("Uncertain annotation set cannot be scored as reviewed truth")
    if (
        data.get("schema") != "crystalab_visible_event_annotations_v1"
        or data.get("clock") != "source_video_time_s"
        or data.get("evidence_kind") not in {"fixture", "synthetic", "real"}
        or data["source_video_sha256"] != video_hash
    ):
        raise ValueError("Annotation schema/evidence/source clock identity differs")
    masks = {}
    ids = set()
    for event in data["events"]:
        if (
            event["event_id"] in ids
            or event["candidate_type"] not in KINDS
            or event.get("uncertain", False)
        ):
            raise ValueError("Duplicate, unsupported or uncertain event annotation")
        ids.add(event["event_id"])
        low, high = event["onset_interval_source_s"]
        if (
            not math.isfinite(high)
            or high < 0
            or low is not None
            and (not math.isfinite(low) or low < 0 or low > high)
        ):
            raise ValueError("Invalid annotated onset interval")
        if event["origin"] not in {
            "present_at_first_usable_frame",
            "appeared_after_clear_view",
            "appeared_after_unknown_interval",
            "identity_unresolved_after_gap",
        }:
            raise ValueError("Unknown annotated visible-origin category")
        mask_path = safe_path(path.parent, event["first_visible_mask_path"])
        if file_sha256(mask_path) != event["mask_sha256"]:
            raise ValueError("Annotation mask identity changed")
        mask = cv2.imread(str(mask_path), 0)
        if mask is None or not set(np.unique(mask)) <= {0, 255} or not np.any(mask):
            raise ValueError("Invalid visible-event mask")
        masks[event["event_id"]] = mask
    for interval in data["negative_intervals"]:
        if interval.get("uncertain", False):
            raise ValueError("Uncertain interval cannot supply negative truth")
        start, end = interval["start_s"], interval["end_s"]
        if (
            interval["candidate_type"] not in KINDS
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
        ):
            raise ValueError("Invalid negative observation interval")
    receipt = None
    if data["evidence_kind"] == "real" and require_review:
        receipt_path = path.with_suffix(".review.json")
        if not receipt_path.exists():
            raise ValueError(
                "Real event annotations need explicit source/negative-period review"
            )
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("annotation_sha256") != file_sha256(path)
            or receipt.get("source_video_sha256") != video_hash
            or not receipt.get("reviewer")
            or receipt.get("ack_event_source_reviewed") is not True
            or receipt.get("mask_sha256")
            != {
                event["first_visible_mask_path"]: event["mask_sha256"]
                for event in data["events"]
            }
        ):
            raise ValueError(
                "Real event review receipt no longer matches exact annotations/source"
            )
    return data, masks, receipt


def review_annotations(
    path,
    source_video,
    reviewer,
    *,
    ack_event_source_reviewed=False,
    ack_physical_clock_reviewed=False,
):
    if not reviewer.strip() or ack_event_source_reviewed is not True:
        raise ValueError(
            "A named reviewer must explicitly acknowledge source events and negative periods"
        )
    path = Path(path).resolve()
    video_hash = file_sha256(source_video)
    data, _, _ = load_annotations(path, video_hash, require_review=False)
    if data["evidence_kind"] != "real":
        raise ValueError("This operator receipt is for actual real annotations only")
    destination = path.with_suffix(".review.json")
    if destination.exists():
        raise FileExistsError("Retain prior review; use a new annotation version")
    receipt = dict(
        reviewer=reviewer.strip(),
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        ack_event_source_reviewed=True,
        physical_clock_reviewed=ack_physical_clock_reviewed is True,
        annotation_sha256=file_sha256(path),
        source_video_sha256=video_hash,
        mask_sha256={
            event["first_visible_mask_path"]: event["mask_sha256"]
            for event in data["events"]
        },
        scope="Operator attestation for exact visible labels/negative intervals; not identity authentication or physical-defect proof",
    )
    save_json(destination, receipt)
    return receipt


def merged_intervals(values):
    merged = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def observation_coverage(rows, maximum_gap_s):
    """Reconstruct class-specific usable intervals instead of trusting saved duration claims."""
    if not math.isfinite(maximum_gap_s) or maximum_gap_s <= 0:
        raise ValueError("Declare a finite positive observation-gap bound")
    observations = []
    intervals = {kind: [] for kind in KINDS}
    previous = None
    for row in rows:
        when = row["video_time_s"]
        observable = row["observable"]
        if (
            not math.isfinite(when)
            or when < 0
            or set(observable) != set(KINDS)
            or any(type(v) is not bool for v in observable.values())
            or previous is not None
            and when <= previous["video_time_s"]
        ):
            raise ValueError("Candidate observation clock/observability is invalid")
        expected = []
        if previous is not None and when - previous["video_time_s"] <= maximum_gap_s:
            expected = [
                dict(candidate_type=kind, start_s=previous["video_time_s"], end_s=when)
                for kind in KINDS
                if observable[kind] and previous["observable"][kind]
            ]
        if row["usable_intervals"] != expected:
            raise ValueError(
                "Usable intervals differ from adjacent class-specific observations"
            )
        for interval in expected:
            intervals[interval["candidate_type"]].append(
                (interval["start_s"], interval["end_s"])
            )
        observations.append(dict(time_s=when, observable=observable))
        previous = row
    return observations, {
        kind: merged_intervals(values) for kind, values in intervals.items()
    }


def observable_deadline_followup(
    targets, matched_ids, observations, intervals, maximum_delay_s
):
    """Separate available clip duration from uninterrupted usable evidence through the deadline."""
    output = {}
    last = observations[-1]["time_s"]
    for kind in KINDS:
        eligible, excluded = [], []
        for event in targets:
            if event["candidate_type"] != kind:
                continue
            start = event["onset_interval_source_s"][1]
            end = start + maximum_delay_s
            onset_usable = any(
                a - 1e-9 <= start <= b + 1e-9 for a, b in intervals[kind]
            ) or any(
                math.isclose(row["time_s"], start, abs_tol=1e-9, rel_tol=0)
                and row["observable"][kind]
                for row in observations
            )
            duration = sum(
                max(0, min(b, end) - max(a, start)) for a, b in intervals[kind]
            )
            reason = (
                "clip_ends_before_deadline"
                if end > last + 1e-9
                else "onset_frame_unobservable"
                if not onset_usable
                else "unobservable_or_gap_within_deadline"
                if duration + 1e-9 < maximum_delay_s
                else None
            )
            if reason is None:
                eligible.append(event["event_id"])
            else:
                excluded.append(
                    dict(
                        event_id=event["event_id"],
                        reason=reason,
                        required_window_source_s=[start, end],
                        usable_source_seconds=duration,
                    )
                )
        matched = sum(event_id in matched_ids for event_id in eligible)
        output[kind] = dict(
            eligible=len(eligible),
            matched=matched,
            recall=matched / len(eligible) if eligible else None,
            eligible_event_ids=eligible,
            excluded=excluded,
        )
    return output


def score_candidate_events(
    analysis, annotations, output, *, match_iou=0.5, maximum_delay_s=1.0
):
    if (
        not 0 < match_iou <= 1
        or not math.isfinite(maximum_delay_s)
        or maximum_delay_s < 0
    ):
        raise ValueError("Invalid event matching gates")
    analysis, annotations, output = (
        Path(analysis).resolve(),
        Path(annotations).resolve(),
        Path(output).resolve(),
    )
    analysis_hash = file_sha256(analysis / "manifest.json")
    annotation_hash = file_sha256(annotations)
    manifest = json.loads((analysis / "manifest.json").read_text())
    if (
        manifest.get("schema") != "crystalab_candidate_analysis_v1"
        or manifest["status"] != "complete"
    ):
        raise ValueError("Completed candidate analysis required")
    for path, digest in manifest["files"].items():
        if file_sha256(safe_path(analysis, path)) != digest:
            raise ValueError("Candidate analysis artifact changed")
    source = json.loads((analysis / "source_replay_manifest.json").read_text())
    if (
        file_sha256(analysis / "source_replay_manifest.json")
        != manifest["source_replay_manifest_sha256"]
    ):
        raise ValueError("Copied source replay manifest identity differs")
    truth, masks, receipt = load_annotations(annotations, source["input"]["sha256"])
    if truth["evidence_kind"] != source["source_info"]["evidence_kind"]:
        raise ValueError(
            "Constructed annotations cannot masquerade as real event evidence"
        )
    predictions = [
        json.loads(line)
        for line in (analysis / "candidate_events.jsonl").read_text().splitlines()
    ]
    observations, usable_intervals = observation_coverage(
        json_rows(analysis / "candidate_observations.jsonl"),
        manifest["config"]["maximum_gap_s"],
    )
    if not observations:
        raise ValueError("No candidate observation window")
    first_time, last_time = observations[0]["time_s"], observations[-1]["time_s"]
    targets = [
        event
        for event in truth["events"]
        if first_time <= event["onset_interval_source_s"][1] <= last_time
    ]
    excluded = [event["event_id"] for event in truth["events"] if event not in targets]
    scores = np.zeros((len(targets), len(predictions)))
    ious = np.zeros_like(scores)
    for i, target in enumerate(targets):
        target_fp = footprint(masks[target["event_id"]])
        low, high = target["onset_interval_source_s"]
        for j, prediction in enumerate(predictions):
            if prediction["candidate_type"] != target["candidate_type"]:
                continue
            when = prediction["alert_time_s"]
            if when < (low if low is not None else 0) or when > high + maximum_delay_s:
                continue
            if (
                low is not None
                and low < high
                and prediction["first_detection_time_s"] <= low
            ):
                continue
            mask = cv2.imread(
                str(safe_path(analysis, prediction["first_detection_mask_path"])), 0
            )
            if mask is None or mask.shape != masks[target["event_id"]].shape:
                raise ValueError("Event/annotation mask shape differs")
            ious[i, j] = footprint_iou(target_fp, footprint(mask))
            if ious[i, j] >= match_iou:
                scores[i, j] = 1 + ious[i, j] / (max(scores.shape) + 1)
    matches = []
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        for i, j in zip(rows, columns):
            if scores[i, j] <= 0:
                continue
            target, prediction = targets[i], predictions[j]
            low, high = target["onset_interval_source_s"]
            matches.append(
                dict(
                    truth_event_id=target["event_id"],
                    prediction_event_id=prediction["event_id"],
                    candidate_type=target["candidate_type"],
                    mask_iou=float(ious[i, j]),
                    origin_correct=target["origin"] == prediction["origin"],
                    truth_origin=target["origin"],
                    prediction_origin=prediction["origin"],
                    alert_delay_after_first_visible_s=prediction["alert_time_s"] - high,
                    alert_delay_interval_s=[
                        prediction["alert_time_s"] - high,
                        None if low is None else prediction["alert_time_s"] - low,
                    ],
                )
            )
    matched_gt = {m["truth_event_id"] for m in matches}
    matched_pred = {m["prediction_event_id"] for m in matches}
    duplicates = [
        predictions[j]["event_id"]
        for j in range(len(predictions))
        if predictions[j]["event_id"] not in matched_pred and scores[:, j].any()
    ]
    per_kind = {}
    for kind in KINDS:
        tp = sum(m["candidate_type"] == kind for m in matches)
        gt = sum(e["candidate_type"] == kind for e in targets)
        pred = sum(e["candidate_type"] == kind for e in predictions)
        per_kind[kind] = dict(
            tp=tp,
            fp=pred - tp,
            fn=gt - tp,
            truth=gt,
            predictions=pred,
            precision=tp / pred if pred else None,
            recall=tp / gt if gt else None,
        )
    negative = {kind: [] for kind in KINDS}
    for kind, usable in usable_intervals.items():
        for usable_start, usable_end in usable:
            for interval in truth["negative_intervals"]:
                if interval["candidate_type"] != kind:
                    continue
                start = max(usable_start, interval["start_s"])
                end = min(usable_end, interval["end_s"])
                if start < end:
                    negative[kind].append((start, end))
    negative = {kind: merged_intervals(values) for kind, values in negative.items()}
    clock = source.get("experiment_clock", {})
    physical = bool(
        truth["evidence_kind"] == "real"
        and receipt
        and receipt.get("physical_clock_reviewed")
        and clock.get("origin_s") is not None
        and clock.get("seconds_per_video_second") is not None
        and source["source_info"].get("experiment_clock_provenance")
    )
    rates = {}
    for kind in KINDS:
        duration = sum(end - start for start, end in negative[kind])
        false = sum(
            e["candidate_type"] == kind
            and e["event_id"] not in matched_pred
            and any(start <= e["alert_time_s"] <= end for start, end in negative[kind])
            for e in predictions
        )
        physical_seconds = (
            duration * clock["seconds_per_video_second"] if physical else None
        )
        rates[kind] = dict(
            usable_annotated_negative_source_seconds=duration,
            false_events_in_negative_intervals=false,
            physical_negative_seconds=physical_seconds,
            false_alerts_per_observed_negative_hour=false / physical_seconds * 3600
            if physical_seconds
            else None,
        )
    new_truth = {
        e["event_id"] for e in targets if e["origin"] == "appeared_after_clear_view"
    }
    new_pred = {
        e["event_id"] for e in predictions if e["origin"] == "appeared_after_clear_view"
    }
    new_tp = sum(
        m["truth_event_id"] in new_truth and m["prediction_event_id"] in new_pred
        for m in matches
    )
    report = dict(
        schema="crystalab_candidate_event_evaluation_v1",
        status="complete",
        analysis_manifest_sha256=analysis_hash,
        annotation_sha256=annotation_hash,
        annotation_evidence_kind=truth["evidence_kind"],
        review_receipt=receipt,
        match_iou=match_iou,
        maximum_delay_source_s=maximum_delay_s,
        events=len(predictions),
        truth_events=len(targets),
        analysis_source_window_s=[first_time, last_time],
        excluded_outside_window_event_ids=excluded,
        metric_scope="Visible-episode matching in available clip; elapsed-window and class-observable deadline coverage are reported separately",
        usable_source_intervals=usable_intervals,
        observable_deadline_followup=observable_deadline_followup(
            targets, matched_gt, observations, usable_intervals, maximum_delay_s
        ),
        complete_followup={
            kind: dict(
                eligible=sum(
                    e["candidate_type"] == kind
                    and e["onset_interval_source_s"][1] + maximum_delay_s <= last_time
                    for e in targets
                ),
                matched=sum(
                    e["candidate_type"] == kind
                    and e["onset_interval_source_s"][1] + maximum_delay_s <= last_time
                    and e["event_id"] in matched_gt
                    for e in targets
                ),
                censored_event_ids=[
                    e["event_id"]
                    for e in targets
                    if e["candidate_type"] == kind
                    and e["onset_interval_source_s"][1] + maximum_delay_s > last_time
                ],
            )
            for kind in KINDS
        },
        per_kind=per_kind,
        matches=matches,
        duplicate_alert_ids=duplicates,
        missed_event_ids=[
            e["event_id"] for e in targets if e["event_id"] not in matched_gt
        ],
        unmatched_prediction_ids=[
            e["event_id"] for e in predictions if e["event_id"] not in matched_pred
        ],
        origin_confusion=dict(
            Counter(
                m["truth_origin"] + " -> " + m["prediction_origin"] for m in matches
            )
        ),
        new_visibility_claims=dict(
            tp=new_tp, fp=len(new_pred) - new_tp, fn=len(new_truth) - new_tp
        ),
        negative_coverage=rates,
        physical_clock_scored=physical,
        first_hour_claim_evaluated=False,
        source_sha256={"src/vision/event_evaluation.py": file_sha256(Path(__file__))},
        scope="Visible candidate inspection events under declared source clock; no microscopic nucleation, quality or control benefit claim",
    )
    if (
        file_sha256(analysis / "manifest.json") != analysis_hash
        or file_sha256(annotations) != annotation_hash
    ):
        raise ValueError("Evaluation input identity changed during scoring")
    for path, digest in manifest["files"].items():
        if file_sha256(safe_path(analysis, path)) != digest:
            raise ValueError("Analysis artifact changed during scoring")
    for event in truth["events"]:
        if (
            file_sha256(safe_path(annotations.parent, event["first_visible_mask_path"]))
            != event["mask_sha256"]
        ):
            raise ValueError("Annotation mask changed during scoring")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", report)
    return report


def build_optical_annotations(replay, output):
    """Derive construction/proxy visibility from original scene masks, never model outputs."""
    from rendering.optical import audit_optical
    import shutil

    replay, output = Path(replay).resolve(), Path(output).resolve()
    source = json.loads((replay / "manifest.json").read_text())
    if (
        source["status"] != "complete"
        or source["source_info"].get("evidence_kind") != "synthetic"
    ):
        raise ValueError(
            "Optical construction annotations require a completed synthetic replay"
        )
    if source["source_info"].get("split") == "test":
        raise ValueError("Development construction labels exclude final test")
    dataset = Path(source["source_info"]["source_dataset"])
    audit_optical(dataset)
    if (
        file_sha256(dataset / "manifest.json")
        != source["source_info"]["source_manifest_sha256"]
    ):
        raise ValueError("Original optical construction changed")
    if (
        file_sha256(replay / "frame_map.jsonl")
        != source["source_info"]["frame_map_sha256"]
    ):
        raise ValueError("Optical source frame map changed")
    mapping = [
        json.loads(line)
        for line in (replay / "frame_map.jsonl").read_text().splitlines()
    ]
    mapping = [
        row for row in mapping if row["output_frame_index"] < source["decoded_frames"]
    ]
    rows = {
        row["sample_id"]: row
        for row in [
            json.loads(line)
            for line in (dataset / "samples.jsonl").read_text().splitlines()
        ]
    }
    events = []
    seen = set()
    negative = []
    previous = None
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    for index, frame in enumerate(mapping):
        row = rows[frame["source_sample_id"]]
        present = set()
        for obj in row["objects"]:
            if obj["class_id"] == 0 or not obj["label_eligible"]:
                continue
            kind = (
                "crystal_candidate" if obj["class_id"] == 1 else "inclusion_candidate"
            )
            present.add(kind)
            key = row["scene_id"] + ":" + obj["id"]
            if key in seen:
                continue
            seen.add(key)
            name = f"masks/{len(events):04d}.png"
            shutil.copyfile(dataset / obj["mask_path"], output / name)
            events.append(
                dict(
                    event_id=key,
                    candidate_type=kind,
                    origin="present_at_first_usable_frame"
                    if index == 0
                    else "appeared_after_clear_view",
                    onset_interval_source_s=[
                        None if index == 0 else mapping[index - 1]["display_time_s"],
                        frame["display_time_s"],
                    ],
                    first_visible_frame_index=frame["output_frame_index"],
                    source_sample_id=frame["source_sample_id"],
                    first_visible_mask_path=name,
                    mask_sha256=file_sha256(output / name),
                    uncertain=False,
                )
            )
        if previous is not None:
            for kind in KINDS:
                if kind not in present and kind not in previous["present"]:
                    negative.append(
                        dict(
                            candidate_type=kind,
                            start_s=previous["time_s"],
                            end_s=frame["display_time_s"],
                        )
                    )
        previous = dict(present=present, time_s=frame["display_time_s"])
    truth = dict(
        schema="crystalab_visible_event_annotations_v1",
        evidence_kind="synthetic",
        clock="source_video_time_s",
        source_video_sha256=source["input"]["sha256"],
        source_manifest_sha256=source["source_info"]["source_manifest_sha256"],
        label_semantics="Original optical emission-proxy eligibility and constructed scene identities; not exact empirical visibility",
        events=events,
        negative_intervals=negative,
        review_status="construction_not_real_human_review",
    )
    save_json(output / "annotations.json", truth)
    return truth
