# FILE PURPOSE: Run timestamped seed observation and preserve auditable replay artifacts.
# Raw file identity, configuration, masks, validity events and source-clock basis
# are saved separately. Partial/failed outputs remain visible and are not reused.
"""File/camera observation runner without model training or hardware commands."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from .baseline import SeedTracker
from .video import VideoReader, file_sha256
from rendering.observation import observation_overlay


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_row(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def replay(
    source,
    config,
    output,
    *,
    run_id,
    source_info=None,
    camera=False,
    clock="container",
    fps=None,
    max_frames=None,
    sample_every=1,
    experiment_origin_s=None,
    experiment_seconds_per_video_second=None,
):
    """Export masks, JSONL and a convenience constant-FPS overlay; never overwrite.

    JSONL frame timestamps are authoritative for variable-rate inputs. The AVI
    preserves frame order and labels each source time, not variable display delays.
    Experiment-time mapping is optional, explicit and requires provenance.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A nonempty run ID is required")
    if type(sample_every) is not int or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if camera and max_frames is None:
        raise ValueError("Camera capture requires an explicit frame limit")
    if (experiment_origin_s is None) != (experiment_seconds_per_video_second is None):
        raise ValueError("Experiment time needs both an origin and a scale")
    for value in (experiment_origin_s, experiment_seconds_per_video_second):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("Experiment-clock parameters must be finite")
    if (
        experiment_seconds_per_video_second is not None
        and experiment_seconds_per_video_second <= 0
    ):
        raise ValueError("Experiment time scale must be positive")
    source_info = dict(source_info or {})
    if source_info.get("evidence_kind", "unknown") not in {
        "real",
        "synthetic",
        "fixture",
        "unknown",
    }:
        raise ValueError("Unknown source evidence kind")
    if experiment_origin_s is not None and not source_info.get(
        "experiment_clock_provenance"
    ):
        raise ValueError("Experiment-clock mapping requires its source evidence")
    tracker = SeedTracker(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    (output / "evidence").mkdir()
    started = time.perf_counter()
    manifest = dict(
        schema="crystalab_video_observation_v1",
        run_id=run_id,
        status="running",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        config=asdict(config),
        source_info=source_info,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            opencv=cv2.__version__,
        ),
        sampling=dict(every_source_frame=sample_every, max_decoded_frames=max_frames),
        experiment_clock=dict(
            origin_s=experiment_origin_s,
            seconds_per_video_second=experiment_seconds_per_video_second,
        ),
        scope="Unvalidated projected seed baseline; no defect/quality or physical-control claim",
        overlay_timing="Constant display FPS; original frame timestamps in measurements.jsonl are authoritative",
    )
    write_json(output / "manifest.json", manifest)
    writer = None
    processed = valid = decoded = 0
    previous_status, last_overlay = None, None
    try:
        with (
            VideoReader(source, camera=camera, clock=clock, fps=fps) as reader,
            (output / "measurements.jsonl").open("x") as measurements,
            (output / "events.jsonl").open("x") as events,
        ):
            manifest["input"] = reader.metadata
            write_json(output / "manifest.json", manifest)
            for frame in reader:
                decoded += 1
                if frame.index % sample_every == 0:
                    record, mask = tracker.measure(frame.image, frame.video_time_s)
                    record.update(
                        run_id=run_id,
                        frame_index=frame.index,
                        video_time_s=frame.video_time_s,
                        timestamp_basis=frame.timestamp_basis,
                        experiment_time_s=None,
                        mask_path=None,
                    )
                    if experiment_origin_s is not None:
                        record["experiment_time_s"] = (
                            experiment_origin_s
                            + frame.video_time_s * experiment_seconds_per_video_second
                        )
                    if mask is not None:
                        record["mask_path"] = f"masks/frame_{frame.index:08d}.png"
                        if not cv2.imwrite(str(output / record["mask_path"]), mask):
                            raise OSError("Could not save observation mask")
                    overlay = observation_overlay(frame.image, record, mask)
                    if writer is None:
                        height, width = overlay.shape[:2]
                        if width % 2 or height % 2:
                            raise ValueError(
                                "AVI overlay requires even image dimensions; input was not resized"
                            )
                        display_fps = (reader.fps or 10.0) / sample_every
                        writer = cv2.VideoWriter(
                            str(output / "overlay.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            display_fps,
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise OSError("MJPG overlay writer is unavailable")
                        manifest["overlay_display_fps"] = display_fps
                    writer.write(overlay)
                    status = (record["status"], record["reason"])
                    if status != previous_status:
                        evidence = f"evidence/frame_{frame.index:08d}.png"
                        if not cv2.imwrite(str(output / evidence), overlay):
                            raise OSError("Could not save event evidence")
                        write_row(
                            events,
                            dict(
                                run_id=run_id,
                                frame_index=frame.index,
                                video_time_s=frame.video_time_s,
                                kind="seed_acquired"
                                if record["valid"]
                                else "inspection_required",
                                reason=record["reason"],
                                evidence_path=evidence,
                            ),
                        )
                    previous_status, last_overlay = status, overlay
                    write_row(measurements, record)
                    processed += 1
                    valid += int(record["valid"])
                if max_frames is not None and decoded >= max_frames:
                    break
            if processed == 0:
                raise ValueError("No frames were processed")
            manifest["stopped_by_frame_limit"] = (
                max_frames is not None and decoded >= max_frames
            )
            if not camera and file_sha256(source) != manifest["input"]["sha256"]:
                raise ValueError("Input changed during observation")
        writer.release()
        writer = None
        if not cv2.imwrite(str(output / "final_observation.png"), last_overlay):
            raise OSError("Could not save final observation")
        manifest.update(
            status="complete",
            decoded_frames=decoded,
            processed_frames=processed,
            valid_observations=valid,
            invalid_observations=processed - valid,
            elapsed_seconds=time.perf_counter() - started,
        )
        manifest["files"] = {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            decoded_frames=decoded,
            processed_frames=processed,
            error=f"{type(error).__name__}: {error}",
        )
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if writer is not None:
            writer.release()
