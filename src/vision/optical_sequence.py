# FILE PURPOSE: Encode one development optical scene with traceable source-frame mapping.
# Display seconds never become physical growth time. Whole-scene membership is
# retained, and reserved test scenes are excluded from development replay export.
"""Convenience synthetic video export for the timestamped learned observation path."""

import json
import math
from pathlib import Path

import cv2

from rendering.optical import audit_optical
from vision.training_data import save_json
from vision.video import file_sha256


def export_optical_sequence(source, scene_id, output, *, display_fps=4.0):
    if (
        isinstance(display_fps, bool)
        or not math.isfinite(display_fps)
        or display_fps <= 0
    ):
        raise ValueError("Display FPS must be positive and finite")
    source, output = Path(source).resolve(), Path(output).resolve()
    audit_optical(source)
    original = json.loads((source / "manifest.json").read_text())
    source_hash = file_sha256(source / "manifest.json")
    rows = [
        json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines()
    ]
    rows = sorted(
        [r for r in rows if r["scene_id"] == scene_id], key=lambda r: r["frame_index"]
    )
    if not rows or any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError(
            "Development export requires one training/validation scene; test excluded"
        )
    if len({r["family_id"] for r in rows}) != 1 or len({r["split"] for r in rows}) != 1:
        raise ValueError("Optical sequence family/split changed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema="crystalab_optical_sequence_v1",
        status="running",
        source_manifest_sha256=source_hash,
        source_dataset=str(source),
        scene_id=scene_id,
        family_id=rows[0]["family_id"],
        split=rows[0]["split"],
        display_fps=display_fps,
        scope="Synthetic development video; no physical-time conversion or event/control truth",
    )
    save_json(output / "manifest.json", manifest)
    writer = None
    try:
        mapping = []
        shape = None
        for index, row in enumerate(rows):
            path = source / row["image_path"]
            if file_sha256(path) != original["artifact_sha256"][row["image_path"]]:
                raise ValueError("Original optical image changed")
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            if width % 2 or height % 2:
                raise ValueError("Encoded overlay needs even image dimensions")
            if writer is None:
                shape = (width, height)
                writer = cv2.VideoWriter(
                    str(output / "sequence.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    display_fps,
                    shape,
                )
                if not writer.isOpened():
                    raise OSError("MJPG video writer unavailable")
                if not cv2.imwrite(str(output / "first_frame.png"), image):
                    raise OSError("Cannot save source preview")
            elif shape != (width, height):
                raise ValueError("Optical image shape changed")
            writer.write(image)
            mapping.append(
                dict(
                    output_frame_index=index,
                    display_time_s=index / display_fps,
                    source_sample_id=row["sample_id"],
                    source_frame_index=row["frame_index"],
                    source_image_path=row["image_path"],
                    image_sha256=file_sha256(path),
                    acquisition_time_s=None,
                    synthetic_time=row["synthetic_time"],
                    time_semantics="illustrative_geometry_step_not_physical_time",
                )
            )
        writer.release()
        writer = None
        (output / "frame_map.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in mapping)
        )
        source_info = dict(
            evidence_kind="synthetic",
            material="illustrative_optical_geometry",
            source_id="crystalab_optical_images_v1",
            scene_id=scene_id,
            family_id=rows[0]["family_id"],
            split=rows[0]["split"],
            source_dataset=str(source),
            source_manifest_sha256=source_hash,
            frame_map_sha256=file_sha256(output / "frame_map.jsonl"),
            derivative="PNG frames MJPG encoded at original dimensions; no source frame duplication",
            timing="Display FPS only. Synthetic geometry steps are not physical seconds.",
            optical_quality=None,
            physical_defect_response=None,
        )
        save_json(output / "source_info.json", source_info)
        if file_sha256(source / "manifest.json") != source_hash or any(
            file_sha256(source / r["source_image_path"]) != r["image_sha256"]
            for r in mapping
        ):
            raise ValueError("Original optical source changed during export")
        manifest.update(
            status="complete",
            frames=len(rows),
            width=shape[0],
            height=shape[1],
            files={
                p.name: file_sha256(p)
                for p in output.iterdir()
                if p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise
    finally:
        if writer is not None:
            writer.release()
