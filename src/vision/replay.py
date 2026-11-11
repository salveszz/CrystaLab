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
import shutil
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
    observer_factory=None,
    frame_map=None,
    candidate_config=None,
    candidate_output=None,
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
    if (candidate_config is None) != (candidate_output is None):
        raise ValueError(
            "Candidate observation needs both a policy and a separate output"
        )
    if candidate_config is not None:
        from .candidates import CandidateConfig

        if (
            not isinstance(candidate_config, CandidateConfig)
            or observer_factory is None
        ):
            raise ValueError(
                "Candidate observation requires a learned observer and CandidateConfig"
            )
        if source_info.get("split") == "test":
            raise ValueError("Development candidate observation excludes final test")
        parent_path, candidate_path = (
            Path(output).resolve(),
            Path(candidate_output).resolve(),
        )
        if parent_path.is_relative_to(candidate_path) or candidate_path.is_relative_to(
            parent_path
        ):
            raise ValueError(
                "Candidate and observation outputs must be separate directories"
            )
        if candidate_path.exists():
            raise FileExistsError("Candidate output already exists")
    mapped_frames = None
    map_path = None
    if frame_map is not None:
        if camera:
            raise ValueError(
                "An archived frame map cannot be attached to a live camera"
            )
        map_path = Path(frame_map).resolve()
        if file_sha256(map_path) != source_info.get("frame_map_sha256"):
            raise ValueError("Source frame-map identity does not match source metadata")
        mapped_frames = [json.loads(line) for line in map_path.read_text().splitlines()]
        if not mapped_frames or any(
            type(row.get("output_frame_index")) is not int
            or row["output_frame_index"] != i
            or type(row.get("display_time_s")) not in {int, float}
            or not math.isfinite(row.get("display_time_s", float("nan")))
            or row["display_time_s"] < 0
            for i, row in enumerate(mapped_frames)
        ):
            raise ValueError(
                "Source frame map needs contiguous indices and finite display timestamps"
            )
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
    candidate_recorder = None
    processed = valid = decoded = instance_count = inspection_count = 0
    observe_times = []
    pipeline_times = []
    previous_status, last_overlay = None, None
    try:
        if candidate_config is not None:
            from .candidate_replay import CandidateRecorder

            candidate_recorder = CandidateRecorder(
                output,
                candidate_config,
                candidate_output,
                run_id=run_id,
                source_info=source_info,
                overlay_timing=manifest["overlay_timing"],
                mode="during_observation",
            )
            manifest["candidate_analysis"] = dict(
                output=str(Path(candidate_output).resolve()),
                mode="during_observation",
                config=asdict(candidate_config),
            )
        setup = time.perf_counter()
        tracker = (
            SeedTracker(config)
            if observer_factory is None
            else observer_factory(output)
        )
        manifest["observer_setup_seconds"] = time.perf_counter() - setup
        manifest["observer"] = getattr(
            tracker, "info", dict(method="opencv_threshold_seed_overlap_v1")
        )
        if observer_factory is not None:
            manifest["scope"] = (
                "Experimental learned projected observation; candidate classes/scores unvalidated for real events/quality/control"
            )
            (output / "instances").mkdir()
        root = Path(__file__).resolve().parents[2]
        sources = [
            Path(__file__),
            root / "src/vision/video.py",
            root / "src/vision/baseline.py",
            root / "src/rendering/observation.py",
            *getattr(tracker, "source_files", []),
        ]
        if candidate_recorder is not None:
            sources += [root / p for p in candidate_recorder.manifest["source_sha256"]]
        source_hashes = {}
        for source_path in sources:
            relative = source_path.relative_to(root)
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            source_hashes[str(relative)] = file_sha256(target)
            if file_sha256(source_path) != source_hashes[str(relative)]:
                raise ValueError("Observer source changed during snapshot")
        manifest["source_sha256"] = source_hashes
        if map_path is not None:
            shutil.copyfile(map_path, output / "frame_map.jsonl")
            manifest["source_frame_map"] = dict(
                original_path=str(map_path), sha256=file_sha256(map_path)
            )
        with (
            VideoReader(source, camera=camera, clock=clock, fps=fps) as reader,
            (output / "measurements.jsonl").open("x") as measurements,
            (output / "events.jsonl").open("x") as events,
        ):
            manifest["input"] = reader.metadata
            write_json(output / "manifest.json", manifest)
            for frame in reader:
                decoded += 1
                source_frame = None
                if mapped_frames is not None:
                    if frame.index >= len(mapped_frames):
                        raise ValueError(
                            "Video contains frames absent from source frame map"
                        )
                    source_frame = mapped_frames[frame.index]
                    if not math.isclose(
                        source_frame["display_time_s"],
                        frame.video_time_s,
                        abs_tol=0.001,
                        rel_tol=0,
                    ):
                        raise ValueError(
                            "Video clock differs from declared derivative display timing"
                        )
                if frame.index % sample_every == 0:
                    observe_started = time.perf_counter()
                    record, mask = tracker.measure(frame.image, frame.video_time_s)
                    observe_seconds = time.perf_counter() - observe_started
                    observe_times.append(observe_seconds)
                    record.update(
                        run_id=run_id,
                        frame_index=frame.index,
                        video_time_s=frame.video_time_s,
                        timestamp_basis=frame.timestamp_basis,
                        experiment_time_s=None,
                        mask_path=None,
                        observe_wall_seconds=observe_seconds,
                    )
                    if experiment_origin_s is not None:
                        record["experiment_time_s"] = (
                            experiment_origin_s
                            + frame.video_time_s * experiment_seconds_per_video_second
                        )
                    if source_frame is not None:
                        record["source_frame"] = source_frame
                    instances = getattr(tracker, "instances", [])
                    if observer_factory is not None:
                        record["objects"] = []
                        for index, instance in enumerate(instances):
                            instance_mask = instance["mask"]
                            path = f"instances/frame_{frame.index:08d}_{index:03d}.png"
                            if not cv2.imwrite(str(output / path), instance_mask):
                                raise OSError(
                                    "Could not save raw candidate instance mask"
                                )
                            metadata = {
                                k: v for k, v in instance.items() if k != "mask"
                            }
                            alternatives = {
                                item["instance_index"]
                                for item in record.get("seed_alternatives", [])
                            }
                            metadata.update(
                                instance_index=index,
                                mask_path=path,
                                projected_area_px=int(np.count_nonzero(instance_mask)),
                                assigned_role="seed_component"
                                if index == record.get("seed_instance_index")
                                else (
                                    "compatible_seed_hypothesis"
                                    if index in alternatives
                                    else (
                                        "unassigned_crystal_candidate"
                                        if instance["class_id"] in {0, 1}
                                        else "unverified_inclusion_candidate"
                                    )
                                ),
                            )
                            record["objects"].append(metadata)
                            instance_count += 1
                    if mask is not None:
                        record["mask_path"] = f"masks/frame_{frame.index:08d}.png"
                        if not cv2.imwrite(str(output / record["mask_path"]), mask):
                            raise OSError("Could not save observation mask")
                    overlay = observation_overlay(
                        frame.image, record, mask, instances=instances
                    )
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
                    if candidate_recorder is not None:
                        candidate_recorder.record(
                            record,
                            [
                                dict(obj, mask=instance["mask"])
                                for obj, instance in zip(record["objects"], instances)
                            ],
                            mask,
                            overlay,
                            manifest["overlay_display_fps"],
                        )
                    needs_inspection = record.get(
                        "requires_inspection", not record["valid"]
                    )
                    status = (
                        record["status"],
                        record["reason"],
                        record.get("inspection_reason"),
                    )
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
                                kind="inspection_required"
                                if needs_inspection
                                else "seed_acquired",
                                reason=record.get("inspection_reason")
                                or record["reason"],
                                evidence_path=evidence,
                            ),
                        )
                    previous_status, last_overlay = status, overlay
                    write_row(measurements, record)
                    pipeline_times.append(time.perf_counter() - observe_started)
                    processed += 1
                    valid += int(record["valid"])
                    inspection_count += int(needs_inspection)
                if max_frames is not None and decoded >= max_frames:
                    break
            if processed == 0:
                raise ValueError("No frames were processed")
            manifest["stopped_by_frame_limit"] = (
                max_frames is not None and decoded >= max_frames
            )
            if not camera and file_sha256(source) != manifest["input"]["sha256"]:
                raise ValueError("Input changed during observation")
            if mapped_frames is not None:
                if not manifest["stopped_by_frame_limit"] and decoded != len(
                    mapped_frames
                ):
                    raise ValueError(
                        "Source frame map contains unmatched extra video frames"
                    )
                if (
                    file_sha256(map_path) != source_info["frame_map_sha256"]
                    or file_sha256(output / "frame_map.jsonl")
                    != source_info["frame_map_sha256"]
                ):
                    raise ValueError("Source frame map changed during replay")
            if hasattr(tracker, "verify"):
                tracker.verify()
        writer.release()
        writer = None
        if candidate_recorder is not None:
            candidate_recorder.close()
            manifest["candidate_analysis"].update(
                processed_frames=candidate_recorder.processed,
                persistent_candidate_events=candidate_recorder.event_count,
            )
        if not cv2.imwrite(str(output / "final_observation.png"), last_overlay):
            raise OSError("Could not save final observation")
        manifest.update(
            status="complete",
            decoded_frames=decoded,
            processed_frames=processed,
            valid_observations=valid,
            invalid_observations=processed - valid,
            elapsed_seconds=time.perf_counter() - started,
            raw_candidate_instances=instance_count,
            inspection_observations=inspection_count,
            observation_latency=dict(
                first_seconds=observe_times[0],
                subsequent_median_seconds=float(np.median(observe_times[1:]))
                if len(observe_times) > 1
                else None,
                all_p95_seconds=float(np.percentile(observe_times, 95)),
                scope="Tracker measure call after decoding; excludes model setup, overlay and disk output",
            ),
            pipeline_latency=dict(
                first_seconds=pipeline_times[0],
                subsequent_median_seconds=float(np.median(pipeline_times[1:]))
                if len(pipeline_times) > 1
                else None,
                all_p95_seconds=float(np.percentile(pipeline_times, 95)),
                scope="Decoded frame through observer, masks, overlays and flushed observation/candidate records; excludes decoding, setup, final hashing and camera exposure/transport",
            ),
        )
        manifest["files"] = {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        if candidate_recorder is not None:
            candidate_recorder.complete()
        return manifest
    except Exception as error:
        if candidate_recorder is not None:
            candidate_recorder.fail(error)
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
        if candidate_recorder is not None:
            candidate_recorder.close()
