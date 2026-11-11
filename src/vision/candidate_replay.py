# FILE PURPOSE: Record causal candidate decisions during observation or on saved replays.
# Per-frame identities, unknown coverage, startup/new/gap events and inspection
# evidence are retained without dispatching commands or inferring physical defects.
"""Traceable streaming and saved-replay candidate inspection artifacts."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import cv2

from vision.candidates import (
    CandidateTracker,
    extract_candidates,
    footprint_mask,
    KINDS,
)
from vision.training_data import safe_path, save_json
from vision.video import file_sha256, VideoReader

ROOT = Path(__file__).resolve().parents[2]


def candidate_overlay(image, detections, result):
    overlay = image.copy()
    states = {t["id"]: t for t in result["tracks"]}
    for i, detection in enumerate(detections):
        x, y, w, h = detection["footprint"]["bbox"]
        identifier = result["assignments"].get(i)
        label = (
            "unassigned"
            if identifier is None
            else identifier.rsplit(":", 1)[-1]
            + (" review" if states[identifier]["event_id"] else " pending")
        )
        color = (
            (200, 80, 220)
            if detection["candidate_type"] == "crystal_candidate"
            else (240, 180, 40)
        )
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, 1)
        cv2.putText(
            overlay,
            label,
            (max(0, x), min(overlay.shape[0] - 5, y + h + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )
    unknown = [
        kind.replace("_candidate", "")
        for kind in KINDS
        if not result["observable"][kind]
    ]
    text = (
        "Candidate observation unknown: " + ",".join(unknown)
        if unknown
        else "Candidate events require operator review"
    )
    scale = min(
        0.4,
        max(
            0.18,
            0.4
            * (overlay.shape[1] - 12)
            / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]),
        ),
    )
    cv2.putText(
        overlay,
        text,
        (5, overlay.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        overlay,
        text,
        (5, overlay.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 220, 255),
        1,
    )
    return overlay


class CandidateRecorder:
    """Flush decisions per frame; bind the completed parent only after it closes.

    Keeping outputs outside the parent avoids a circular manifest hash. The same
    recorder consumes native masks during observation or immutable saved masks.
    """

    def __init__(
        self,
        replay,
        config,
        output,
        *,
        run_id,
        source_info,
        overlay_timing,
        mode="saved_replay",
        source_hash=None,
    ):
        self.replay, self.output = Path(replay).resolve(), Path(output).resolve()
        if self.output.is_relative_to(self.replay) or self.replay.is_relative_to(
            self.output
        ):
            raise ValueError(
                "Candidate and observation outputs must be separate directories"
            )
        if source_info.get("split") == "test":
            raise ValueError("Development candidate analysis excludes final test")
        self.output.mkdir(parents=True, exist_ok=False)
        for name in ["candidate_masks", "event_masks", "evidence"]:
            (self.output / name).mkdir()
        source_hashes = {}
        for path in [
            Path(__file__),
            ROOT / "src/vision/candidates.py",
            ROOT / "src/vision/video.py",
            ROOT / "src/vision/training_data.py",
        ]:
            relative = path.relative_to(ROOT)
            target = self.output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hashes[str(relative)] = file_sha256(target)
            if file_sha256(path) != source_hashes[str(relative)]:
                raise ValueError("Candidate source changed during snapshot")
        self.manifest = dict(
            schema="crystalab_candidate_analysis_v1",
            status="running",
            run_id=run_id,
            source_replay=str(self.replay),
            source_replay_manifest_sha256=source_hash,
            source_sha256=source_hashes,
            config=asdict(config),
            source_info=source_info,
            clock="source_video_time_s",
            mode=mode,
            scope="Experimental candidate inspection events, not physical defects or action-response evidence",
            interval_semantics="Intervals between adjacent usable processed observations within configured gap; not verified continuous physical coverage",
            overlay_timing=overlay_timing,
            actuator_commands_issued=False,
        )
        save_json(self.output / "manifest.json", self.manifest)
        self.tracker = CandidateTracker(config, run_id)
        self.started = time.perf_counter()
        self.processed = self.event_count = 0
        self.issues_count, self.event_kinds, self.candidate_types = (
            Counter(),
            Counter(),
            Counter(),
        )
        self.coverage = {kind: 0.0 for kind in KINDS}
        self.writer = None
        self.records = (self.output / "candidate_observations.jsonl").open("x")
        self.events = (self.output / "candidate_events.jsonl").open("x")

    def record(self, record, instances, seed, image, display_fps):
        if "objects" not in record:
            raise ValueError(
                "Seed-only baseline records cannot supply candidate absence evidence"
            )
        shape = image.shape[:2]
        detections, observable, issues = extract_candidates(
            record, instances, seed, self.tracker.config
        )
        public = []
        for i, detection in enumerate(detections):
            path = f"candidate_masks/frame_{record['frame_index']:08d}_{i:03d}.png"
            if not cv2.imwrite(
                str(self.output / path), footprint_mask(detection["footprint"], shape)
            ):
                raise OSError("Cannot save candidate mask")
            detection["observation_ref"] = dict(
                frame_index=record["frame_index"],
                video_time_s=record["video_time_s"],
                mask_path=path,
                source_instances=detection["source_instances"],
                source_frame=record.get("source_frame"),
                experiment_time_s=record["experiment_time_s"],
            )
            public.append({k: v for k, v in detection.items() if k != "footprint"})
            public[-1].update(
                mask_path=path,
                bbox_px=detection["footprint"]["bbox"],
                projected_area_px=detection["footprint"]["area"],
            )
        result = self.tracker.update(
            record["frame_index"], record["video_time_s"], detections, observable
        )
        issues += result["issues"]
        self.issues_count.update(issue["reason"] for issue in issues)
        for i, detection in enumerate(public):
            detection["track_id"] = result["assignments"].get(i)
        overlay = candidate_overlay(image, detections, result)
        if self.writer is None:
            height, width = shape
            self.writer = cv2.VideoWriter(
                str(self.output / "overlay.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                display_fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise OSError("Candidate overlay writer unavailable")
        self.writer.write(overlay)
        for event in result["events"]:
            self.event_count += 1
            first = event.pop("first_footprint")
            mask_path = f"event_masks/event_{self.event_count:06d}_first.png"
            if not cv2.imwrite(
                str(self.output / mask_path), footprint_mask(first, shape)
            ):
                raise OSError("Cannot save first-detection evidence mask")
            evidence = f"evidence/frame_{record['frame_index']:08d}.png"
            if not cv2.imwrite(str(self.output / evidence), overlay):
                raise OSError("Cannot save candidate evidence")
            event.update(
                first_detection_mask_path=mask_path,
                evidence_path=evidence,
                source_frame=record.get("source_frame"),
                experiment_time_s=record["experiment_time_s"],
            )
            self.events.write(json.dumps(event, allow_nan=False) + "\n")
            self.events.flush()
            self.event_kinds[event["kind"]] += 1
            self.candidate_types[event["candidate_type"]] += 1
        for interval in result["usable_intervals"]:
            self.coverage[interval["candidate_type"]] += (
                interval["end_s"] - interval["start_s"]
            )
        row = dict(
            frame_index=record["frame_index"],
            video_time_s=record["video_time_s"],
            source_frame=record.get("source_frame"),
            experiment_time_s=record["experiment_time_s"],
            detections=public,
            tracks=result["tracks"],
            observable=result["observable"],
            issues=issues,
            usable_intervals=result["usable_intervals"],
        )
        self.records.write(json.dumps(row, allow_nan=False) + "\n")
        self.records.flush()
        self.processed += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.records.close()
        self.events.close()

    def fail(self, error):
        self.close()
        self.manifest.update(
            status="failed",
            processed_frames=self.processed,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(self.output / "manifest.json", self.manifest)

    def complete(self):
        self.close()
        source_hash = file_sha256(self.replay / "manifest.json")
        expected = self.manifest["source_replay_manifest_sha256"]
        if expected is not None and source_hash != expected:
            raise ValueError("Source replay changed during candidate analysis")
        source = json.loads((self.replay / "manifest.json").read_text())
        if (
            source["status"] != "complete"
            or self.processed != source["processed_frames"]
        ):
            raise ValueError(
                "Completed parent observation with matching frame count required"
            )
        if (
            source["run_id"] != self.manifest["run_id"]
            or source["source_info"] != self.manifest["source_info"]
        ):
            raise ValueError("Parent observation identity differs")
        for path, digest in source["files"].items():
            if file_sha256(safe_path(self.replay, path)) != digest:
                raise ValueError("Source replay artifact changed during analysis")
        shutil.copyfile(
            self.replay / "manifest.json", self.output / "source_replay_manifest.json"
        )
        if file_sha256(self.output / "source_replay_manifest.json") != source_hash:
            raise ValueError("Source replay changed during completion")
        self.manifest.update(
            status="complete",
            source_replay_manifest_sha256=source_hash,
            processed_frames=self.processed,
            persistent_candidate_events=self.event_count,
            event_kinds=dict(self.event_kinds),
            candidate_types=dict(self.candidate_types),
            issues=dict(self.issues_count),
            usable_source_seconds=self.coverage,
            elapsed_seconds=time.perf_counter() - self.started,
            final_tracks=[
                {
                    k: v
                    for k, v in t.items()
                    if k not in {"footprint", "first_footprint"}
                }
                for t in self.tracker.tracks.values()
            ],
            files={
                str(p.relative_to(self.output)): file_sha256(p)
                for p in self.output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(self.output / "manifest.json", self.manifest)
        return self.manifest


def analyze_candidates(replay, config, output):
    replay = Path(replay).resolve()
    source = json.loads((replay / "manifest.json").read_text())
    if (
        source.get("schema") != "crystalab_video_observation_v1"
        or source["status"] != "complete"
    ):
        raise ValueError("A completed learned observation replay is required")
    for path, digest in source["files"].items():
        if file_sha256(safe_path(replay, path)) != digest:
            raise ValueError("Original replay artifact identity changed")
    recorder = CandidateRecorder(
        replay,
        config,
        output,
        run_id=source["run_id"],
        source_info=source["source_info"],
        overlay_timing=source["overlay_timing"],
        source_hash=file_sha256(replay / "manifest.json"),
    )
    try:
        with (
            VideoReader(replay / "overlay.avi") as reader,
            (replay / "measurements.jsonl").open() as observations,
        ):
            for ordinal, line in enumerate(observations):
                record = json.loads(line)
                if "objects" not in record:
                    raise ValueError(
                        "Seed-only baseline records cannot supply candidate absence evidence"
                    )
                frame = next(reader)
                if frame.index != ordinal:
                    raise ValueError("Parent overlay frame order differs")
                seed = (
                    None
                    if record["mask_path"] is None
                    else cv2.imread(str(safe_path(replay, record["mask_path"])), 0)
                )
                instances = []
                for obj in record["objects"]:
                    mask = cv2.imread(str(safe_path(replay, obj["mask_path"])), 0)
                    if mask is None or mask.shape != frame.image.shape[:2]:
                        raise ValueError("Invalid original-frame instance mask")
                    instances.append(dict(obj, mask=mask))
                recorder.record(
                    record, instances, seed, frame.image, source["overlay_display_fps"]
                )
            if next(reader, None) is not None:
                raise ValueError("Parent overlay contains unmatched frames")
        return recorder.complete()
    except Exception as error:
        recorder.fail(error)
        raise
    finally:
        recorder.close()
