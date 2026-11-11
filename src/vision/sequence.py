# FILE PURPOSE: Export a development image sequence as a traceable convenience video.
# Display timing/resampling stay separate from original indices, nominal capture
# cadence and unknown actual acquisition times. Test runs are not used for tuning.
"""Pilot replay adapters for an indexed real microscope sequence."""

import json
import math
from pathlib import Path
import zipfile

import cv2

from .data import SCHEMA, load_index_image, read_json
from .video import file_sha256


def export_pilot(
    index, run_id, output, *, display_fps=4.0, max_frames=30, max_width=960
):
    if (
        isinstance(display_fps, bool)
        or not math.isfinite(display_fps)
        or display_fps <= 0
    ):
        raise ValueError("Display FPS must be finite and positive")
    if (
        type(max_frames) is not int
        or max_frames < 1
        or type(max_width) is not int
        or max_width < 2
    ):
        raise ValueError("Frame/width limits must be positive integers")
    index = Path(index)
    manifest = read_json(index / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
        raise ValueError("Incomplete/unsupported real-image index")
    if file_sha256(index / "images.jsonl") != manifest["index_sha256"]:
        raise ValueError("Image index hash changed")
    rows = [
        json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["run_id"] == run_id]
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Pilot inspection is restricted to a declared training run")
    archive_path = manifest["archive_path"]
    protocol = manifest["protocol"]
    if file_sha256(archive_path) != protocol["archive_sha256"]:
        raise ValueError("Source archive changed")
    rows = rows[:max_frames]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="running",
        run_id=run_id,
        source_index_sha256=manifest["index_sha256"],
        display_fps=display_fps,
        scope="Development replay; video display time is not acquisition time",
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    writer = None
    try:
        maps = []
        with zipfile.ZipFile(archive_path) as archive:
            for output_index, row in enumerate(rows):
                pixels = load_index_image(archive, row)
                original_h, original_w = pixels.shape[:2]
                width = min(max_width, original_w) // 2 * 2
                height = int(round(original_h * width / original_w)) // 2 * 2
                if height < 2:
                    raise ValueError("Resampling would produce an empty frame")
                resized = cv2.resize(
                    pixels, (width, height), interpolation=cv2.INTER_AREA
                )
                if writer is None:
                    shape = (width, height)
                    writer = cv2.VideoWriter(
                        str(output / "sequence.avi"),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        display_fps,
                        shape,
                    )
                    if not writer.isOpened():
                        raise OSError("MJPG writer unavailable")
                    if not cv2.imwrite(str(output / "first_frame.png"), resized):
                        raise OSError("Cannot save pilot inspection frame")
                elif shape != (width, height):
                    raise ValueError("Frame shape changed within the sequence")
                writer.write(resized)
                maps.append(
                    dict(
                        output_frame_index=output_index,
                        display_time_s=output_index / display_fps,
                        source_sample_id=row["sample_id"],
                        source_frame_index=row["frame_index"],
                        archive_member=row["archive_member"],
                        image_sha256=row["image_sha256"],
                        acquisition_time_s=None,
                        nominal_elapsed_s=row["nominal_elapsed_s"],
                        scale_x=width / original_w,
                        scale_y=height / original_h,
                    )
                )
        writer.release()
        writer = None
        with (output / "frame_map.jsonl").open("x") as stream:
            for row in maps:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        source_info = dict(
            evidence_kind="real",
            material=protocol["material"],
            source_id=protocol["source_id"],
            experiment_id=run_id,
            split="train",
            source_url=protocol["source_url"],
            citation=protocol["citation"],
            license=protocol["license"],
            license_url=protocol["license_url"],
            source_archive_sha256=protocol["archive_sha256"],
            source_index_sha256=manifest["index_sha256"],
            derivative="JPEG sequence resized and MJPG encoded; original source retained",
            frame_map_sha256=file_sha256(output / "frame_map.jsonl"),
            timing="Display FPS only. Nominal acquisition cadence is in frame_map; actual capture times are unverified.",
        )
        (output / "source_info.json").write_text(
            json.dumps(source_info, indent=2) + "\n"
        )
        result.update(
            status="complete",
            frames=len(maps),
            output_width=shape[0],
            output_height=shape[1],
            files={
                p.name: file_sha256(p)
                for p in output.iterdir()
                if p.name != "manifest.json"
            },
        )
        save()
        return result
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if writer is not None:
            writer.release()
