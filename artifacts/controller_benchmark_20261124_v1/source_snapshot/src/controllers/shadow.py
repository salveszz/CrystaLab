# FILE PURPOSE: Bind saved candidate evidence to auditable supervisory shadow proposals.
# Original source clocks and uncertainty remain intact. Evidence images/masks are
# copied with their identities; no process readings or numerical actions are invented.
"""Inspect event-rule behavior on immutable development candidate observations."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import cv2

from .benchmark import save_json, sha256, write_row
from .supervision import RuleConfig, RuleEvidence, RuleSupervisor
from vision.training_data import safe_path
from vision.video import VideoReader

ROOT = Path(__file__).resolve().parents[2]


class CandidateEvidence:
    """Verify and expose a finite saved source without converting it to physical time."""

    def __init__(self, path):
        self.root = Path(path).resolve()
        self.manifest_sha256 = sha256(self.root / "manifest.json")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("schema") != "crystalab_candidate_analysis_v1"
            or m["status"] != "complete"
            or m["clock"] != "source_video_time_s"
        ):
            raise ValueError("A complete source-clock candidate analysis is required")
        if m["source_info"].get("split") == "test":
            raise ValueError("Development supervision excludes final test")
        self.parent = json.loads(
            (self.root / "source_replay_manifest.json").read_text()
        )
        if (
            sha256(self.root / "source_replay_manifest.json")
            != m["source_replay_manifest_sha256"]
        ):
            raise ValueError("Candidate parent identity differs")
        if (
            self.parent["status"] != "complete"
            or self.parent["run_id"] != m["run_id"]
            or self.parent["source_info"] != m["source_info"]
        ):
            raise ValueError("Source observation context differs")
        self.clock_id = "source_video:" + (
            self.parent["input"].get("sha256") or self.manifest_sha256
        )
        self.events = defaultdict(list)
        seen = set()
        for line in (self.root / "candidate_events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event_id"] in seen:
                raise ValueError("Duplicate saved candidate event ID")
            seen.add(event["event_id"])
            self.events[event["alert_frame_index"]].append(event)
        if len(seen) != m["persistent_candidate_events"]:
            raise ValueError("Saved candidate event count differs")
        self.verify()

    def verify(self):
        if sha256(self.root / "manifest.json") != self.manifest_sha256:
            raise ValueError("Candidate source manifest changed")
        for name, digest in self.manifest["files"].items():
            if sha256(safe_path(self.root, name)) != digest:
                raise ValueError("Candidate source artifact identity changed")

    def packets(self):
        count = 0
        used_frames = set()
        previous_frame = previous_time = -1
        with (self.root / "candidate_observations.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                frame, stamp = row["frame_index"], row["video_time_s"]
                if frame <= previous_frame or stamp <= previous_time:
                    raise ValueError(
                        "Candidate records must preserve increasing original frames/clocks"
                    )
                events = tuple(self.events.get(frame, []))
                if any(e["alert_time_s"] != stamp for e in events):
                    raise ValueError(
                        "Candidate event does not match its source observation clock"
                    )
                yield (
                    RuleEvidence(
                        run_id=self.manifest["run_id"],
                        clock_id=self.clock_id,
                        packet_id=f"{self.manifest['run_id']}:frame:{frame:08d}",
                        frame_index=frame,
                        observed_at_s=stamp,
                        received_at_s=stamp,
                        evidence_kind=self.manifest["source_info"]["evidence_kind"],
                        crystal_observable=row["observable"]["crystal_candidate"],
                        inclusion_observable=row["observable"]["inclusion_candidate"],
                        events=events,
                        visible_track_ids=tuple(
                            t["id"]
                            for t in row["tracks"]
                            if t["state"] == "observed"
                            and t["last_frame_index"] == frame
                        ),
                        issues=tuple(
                            sorted({issue["reason"] for issue in row["issues"]})
                        ),
                    ),
                    row,
                )
                count += 1
                previous_frame, previous_time = frame, stamp
                used_frames.add(frame)
        if (
            count != self.manifest["processed_frames"]
            or not set(self.events) <= used_frames
        ):
            raise ValueError("Candidate event/frame coverage differs")


def shadow_overlay(image, result):
    output = cv2.copyMakeBorder(
        image, 0, 48, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )
    for index, text in enumerate(
        [
            "SHADOW / review proposal only",
            f"Pending: {len(result['pending_proposal_ids'])} / source: {result['time_s']:.3f}s",
        ]
    ):
        scale = min(
            0.45,
            0.45
            * (image.shape[1] - 10)
            / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]),
        )
        cv2.putText(
            output,
            text,
            (5, image.shape[0] + 18 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 220, 255),
            1,
        )
    return output


def run_shadow(analysis, policy_path, output):
    source = CandidateEvidence(analysis)
    policy_path, output = Path(policy_path).resolve(), Path(output).resolve()
    policy_hash = sha256(policy_path)
    config = RuleConfig(**json.loads(policy_path.read_text()))
    supervisor = RuleSupervisor(
        config,
        run_id=source.manifest["run_id"],
        clock_id=source.clock_id,
        evidence_kind=source.manifest["source_info"]["evidence_kind"],
    )
    output.mkdir(parents=True, exist_ok=False)
    for folder in ["evidence", "masks"]:
        (output / folder).mkdir()
    shutil.copyfile(policy_path, output / "policy.json")
    shutil.copyfile(source.root / "manifest.json", output / "candidate_manifest.json")
    shutil.copyfile(
        source.root / "source_replay_manifest.json",
        output / "source_replay_manifest.json",
    )
    source_hashes = {}
    for path in [
        Path(__file__),
        ROOT / "src/controllers/supervision.py",
        ROOT / "src/controllers/contracts.py",
        ROOT / "src/controllers/benchmark.py",
        ROOT / "src/vision/video.py",
        ROOT / "src/vision/training_data.py",
    ]:
        relative = path.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = sha256(target)
        if sha256(path) != source_hashes[str(relative)]:
            raise ValueError("Shadow producer changed during snapshot")
    manifest = dict(
        schema="crystalab_supervisory_shadow_v1",
        status="running",
        source_analysis=str(source.root),
        candidate_manifest_sha256=source.manifest_sha256,
        source_sha256=source_hashes,
        policy_sha256=policy_hash,
        config=asdict(config),
        source_info=source.manifest["source_info"],
        source_input=source.parent["input"],
        source_experiment_clock=source.parent.get("experiment_clock"),
        clock_id=source.clock_id,
        clock_semantics="Logical original source-video clock; receipt equals replay time, not measured processing latency or physical growth time",
        scope="Event-rule review proposals on development observations; no defect confirmation or action-response evidence",
        temperature_feedback_available=False,
        numerical_actuator_commands=0,
        operator_reviews_created=0,
        physical_hardware_contacted=False,
    )
    save_json(output / "manifest.json", manifest)
    processed = 0
    counts = Counter()
    writer, last = None, None
    started = time.perf_counter()
    try:
        with (
            VideoReader(source.root / "overlay.avi") as reader,
            (output / "decisions.jsonl").open("x") as decisions,
            (output / "proposals.jsonl").open("x") as proposals,
        ):
            for packet, original in source.packets():
                frame = next(reader)
                if frame.index != processed:
                    raise ValueError("Candidate overlay order differs")
                result = supervisor.tick(packet.observed_at_s, packet)
                overlay = shadow_overlay(frame.image, result)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / "overlay.avi"),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        source.parent["overlay_display_fps"],
                        (overlay.shape[1], overlay.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError("Shadow overlay writer unavailable")
                writer.write(overlay)
                for entry in result["created_proposals"]:
                    proposal = entry["proposal"]
                    number = proposal["proposal_id"].rsplit(":", 1)[-1]
                    evidence = f"evidence/proposal_{number}.png"
                    if not cv2.imwrite(str(output / evidence), overlay):
                        raise OSError("Cannot save supervisory proposal evidence")
                    refs = dict(
                        image_path=evidence,
                        native_image_height=frame.image.shape[0],
                        footer_height=48,
                    )
                    if proposal["first_detection_mask_path"] is not None:
                        original_mask = safe_path(
                            source.root, proposal["first_detection_mask_path"]
                        )
                        relative = f"masks/proposal_{number}_first.png"
                        shutil.copyfile(original_mask, output / relative)
                        refs.update(
                            first_detection_mask_path=relative,
                            source_mask_sha256=sha256(original_mask),
                        )
                    write_row(proposals, dict(**entry, artifact_evidence=refs))
                    counts[proposal["kind"]] += 1
                write_row(
                    decisions,
                    dict(
                        **result,
                        source_frame_index=packet.frame_index,
                        source_frame=original.get("source_frame"),
                        experiment_time_s=original["experiment_time_s"],
                    ),
                )
                processed += 1
                last = overlay
            if next(reader, None) is not None:
                raise ValueError("Candidate overlay has unmatched frames")
        if not processed:
            raise ValueError("No source evidence processed")
        writer.release()
        writer = None
        if not cv2.imwrite(str(output / "final_shadow.png"), last):
            raise OSError("Cannot save final shadow view")
        source.verify()
        if (
            sha256(policy_path) != policy_hash
            or sha256(output / "policy.json") != policy_hash
        ):
            raise ValueError("Shadow rule policy changed during processing")
        manifest.update(
            status="complete",
            processed_frames=processed,
            proposals_by_kind=dict(counts),
            final_proposals=list(supervisor.proposals.values()),
            elapsed_seconds=time.perf_counter() - started,
            files={
                str(p.relative_to(output)): sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            processed_frames=processed,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise
    finally:
        if writer is not None:
            writer.release()
