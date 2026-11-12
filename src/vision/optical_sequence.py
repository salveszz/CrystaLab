# FILE PURPOSE: Encode one development optical scene with traceable source-frame mapping.
# Display seconds never become physical growth time. Whole-scene membership is
# retained, and reserved test scenes are excluded from development replay export.
"""Convenience synthetic video export for the timestamped learned observation path."""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from rendering.optical import audit_optical
from vision.training_data import save_json
from vision.video import file_sha256


def validate_sequence_codec(codec):
    if type(codec) is not str or codec not in {"MJPG", "FFV1"}:
        raise ValueError("Declare MJPG or lossless FFV1 sequence encoding")
    return codec


def verify_sequence_pixels(source, mapping, video, *, codec):
    """Decode every frame; FFV1 must reproduce original BGR pixels exactly."""
    validate_sequence_codec(codec)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise OSError("Encoded sequence cannot be decoded")
        tag = int(capture.get(cv2.CAP_PROP_FOURCC))
        reported_fourcc = "".join(chr((tag >> (8 * i)) & 255) for i in range(4))
        actual_codec = reported_fourcc.upper()
        if actual_codec != codec:
            raise ValueError("Encoded sequence codec differs from requested codec")
        total_error, values, maximum, changed = 0, 0, 0, 0
        for row in mapping:
            ok, decoded = capture.read()
            original = cv2.imread(str(source / row["source_image_path"]))
            if not ok or original is None or decoded.shape != original.shape:
                raise ValueError("Encoded sequence frame count or dimensions differ")
            difference = np.abs(decoded.astype(np.int16) - original.astype(np.int16))
            frame_maximum = int(difference.max())
            if codec == "FFV1" and frame_maximum:
                raise ValueError("Lossless FFV1 sequence changed original pixels")
            total_error += int(difference.sum())
            values += difference.size
            maximum = max(maximum, frame_maximum)
            changed += int(frame_maximum > 0)
        if capture.read()[0]:
            raise ValueError("Encoded sequence has extra frames")
        return dict(
            decoded_frames=len(mapping),
            actual_codec=actual_codec,
            reported_fourcc=reported_fourcc,
            exact_pixel_match=changed == 0,
            frames_with_pixel_changes=changed,
            maximum_absolute_channel_difference=maximum,
            mean_absolute_channel_difference=total_error / values if values else None,
        )
    finally:
        capture.release()


def export_optical_sequence(source, scene_id, output, *, display_fps=4.0, codec="MJPG"):
    codec = validate_sequence_codec(codec)
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
        codec=codec,
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
                    cv2.VideoWriter_fourcc(*codec),
                    display_fps,
                    shape,
                )
                if not writer.isOpened():
                    raise OSError(f"{codec} video writer unavailable")
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
        pixel_verification = verify_sequence_pixels(
            source, mapping, output / "sequence.avi", codec=codec
        )
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
            derivative=f"PNG frames {codec} encoded at original dimensions; no source frame duplication",
            sequence_codec=codec,
            sequence_pixel_verification=pixel_verification,
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
            pixel_verification=pixel_verification,
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
