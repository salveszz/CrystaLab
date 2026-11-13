# FILE PURPOSE: Index public perovskite growth images and published measurements.
# Native bytes, display overlays, clocks and automatic/manual labels stay distinct.
# No source class, measurement, tracking output or action becomes KDP truth.
"""Bounded, streaming acquisition index for the two versioned CrystalCV archives."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import zipfile

import cv2
import numpy as np
import PIL
from PIL import Image

from .training_data import save_json
from .video import file_sha256

SCHEMA = "crystalab_crystalcv_index_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def filename_time(name):
    """The recorded filename clock, never an inferred growth-start timestamp."""
    match = re.fullmatch(
        r"(\d+)h ([0-5]\d)m ([0-5]\d)s", Path(name).stem.removesuffix("-1")
    )
    return (
        None
        if match is None
        else sum(int(v) * scale for v, scale in zip(match.groups(), (3600, 60, 1)))
    )


def _checked_archives(cache, protocol):
    if (
        protocol.get("schema") != "crystalab_crystalcv_protocol_v1"
        or protocol.get("pitch_class_mapping") is not None
    ):
        raise ValueError("Preserve the separate source task and unmapped pitch classes")
    if protocol.get("split") != "development" or not protocol.get(
        "dependency_family_id"
    ):
        raise ValueError(
            "Source/run grouping needs review before independent evaluation"
        )
    if protocol.get("license_id") != "cc-by-4.0":
        raise ValueError("Require inspected versioned source license")
    for key in (
        "maximum_member_bytes",
        "maximum_expanded_bytes",
        "maximum_image_pixels",
    ):
        if type(protocol.get(key)) is not int or protocol[key] <= 0:
            raise ValueError("Declare positive bounded import sizes")
    assets = protocol.get("archives", [])
    if not assets or len({a["archive_id"] for a in assets}) != len(assets):
        raise ValueError("Declare distinct archive identities")
    for asset in assets:
        for key in ("filename", "metadata_filename"):
            if Path(asset[key]).name != asset[key] or "\\" in asset[key]:
                raise ValueError("Source assets must use plain filenames")
        metadata_path = cache / asset["metadata_filename"]
        if file_sha256(metadata_path) != asset["metadata_sha256"]:
            raise ValueError("Source license metadata changed")
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("metadata", {}).get("license", {}).get("id")
            != protocol["license_id"]
        ):
            raise ValueError("Versioned metadata does not supply the declared license")
        published = next(
            (v for v in metadata["files"] if v["key"] == asset["filename"]), None
        )
        if (
            published is None
            or published["size"] != asset["bytes"]
            or published["checksum"] != "md5:" + asset["md5"]
        ):
            raise ValueError("Archive identity differs from publisher metadata")
        path = cache / asset["filename"]
        if path.stat().st_size != asset["bytes"]:
            raise ValueError("Source archive size changed")
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                sha.update(block)
                md5.update(block)
        if sha.hexdigest() != asset["sha256"] or md5.hexdigest() != asset["md5"]:
            raise ValueError("Source archive checksum changed")
        yield asset, path


def _members(archive, protocol):
    entries = archive.infolist()
    seen = set()
    if sum(i.file_size for i in entries) > protocol["maximum_expanded_bytes"]:
        raise ValueError("Expanded archive exceeds protocol bound")
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in entry.filename
            or entry.filename in seen
            or ((entry.external_attr >> 16) & 0o170000) == 0o120000
        ):
            raise ValueError("Unsafe, duplicate or linked archive member")
        seen.add(entry.filename)
        if entry.file_size > protocol["maximum_member_bytes"]:
            raise ValueError("Archive member exceeds protocol bound")
        if not entry.is_dir():
            yield entry


def _image(payload, maximum_pixels):
    with Image.open(io.BytesIO(payload)) as header:
        if header.width * header.height > maximum_pixels or header.mode not in {
            "RGB",
            "L",
        }:
            raise ValueError(
                "Source image exceeds pixel bound or has unsupported pixel type"
            )
        image_format = header.format
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8:
        raise ValueError("Native image decode failed")
    return image, image_format


def scan(cache, protocol):
    """Yield index records without keeping archives or all decoded images in RAM."""
    for asset, path in _checked_archives(Path(cache), protocol):
        archive_id = asset["archive_id"]
        with zipfile.ZipFile(path) as archive:
            for entry in _members(archive, protocol):
                payload = archive.read(entry)  # zipfile checks this member's CRC.
                member = entry.filename
                row = dict(
                    archive_id=archive_id,
                    member=member,
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
                yield "members", row
                if PurePosixPath(member).suffix.lower() in IMAGE_SUFFIXES:
                    image, image_format = _image(
                        payload, protocol["maximum_image_pixels"]
                    )
                    role = (
                        "published_annotated_display"
                        if "/Manual Verification Annotated/" in member
                        else "automatic_annotated_display"
                        if "annotated_frames" in PurePosixPath(member).parts
                        else "source_growth_image"
                        if any(
                            member.startswith(prefix)
                            for prefix in asset["raw_image_path_prefixes"]
                        )
                        else "source_reference_image"
                    )
                    folder = str(PurePosixPath(member).parent)
                    material = next(
                        (
                            rule["material"]
                            for rule in asset["material_rules"]
                            if rule["path_component"] in PurePosixPath(member).parts
                        ),
                        None,
                    )
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    record = dict(
                        **row,
                        sample_id=f"{archive_id}:{member}",
                        sequence_id=f"{archive_id}:{folder}",
                        source_folder=folder,
                        source_id=protocol["source_id"],
                        dependency_family_id=protocol["dependency_family_id"],
                        split=protocol["split"],
                        material=material,
                        role=role,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        decoded_format=image_format,
                        native_BGR_sha256=pixel_hash,
                        pixel_family_id=pixel_hash,
                        filename_elapsed_s=filename_time(member),
                        clock_kind="source_filename_elapsed_not_verified_growth_start",
                        growth_run_start=None,
                        first_hour_membership=None,
                        actual_acquisition_timestamp=None,
                        segmentation_labels=None,
                        event_labels=None,
                        sensor_values=None,
                        applied_commands=None,
                        quality_labels=None,
                        local_human_review=False,
                        license=protocol["license_id"],
                        attribution=protocol["attribution"],
                    )
                    yield "images", record
                if member.lower().endswith(".csv"):
                    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
                    count, frame_values, object_ids = 0, set(), set()
                    for item in reader:
                        count += 1
                        if "frame" in item:
                            frame_values.add(item["frame"])
                        if "object_id" in item:
                            object_ids.add(item["object_id"])
                    yield (
                        "tables",
                        dict(
                            **row,
                            columns=reader.fieldnames,
                            rows=count,
                            distinct_frames=len(frame_values),
                            distinct_object_ids=len(object_ids),
                            evidence_kind="published_manual_rectangle_measurements"
                            if member == protocol["manual_table_member"]
                            else asset.get("table_roles", {}).get(
                                member,
                                "author_automatic_tracking_or_processed_output"
                                if "tracking_data" in member
                                or "/Pre_Processed/" in member
                                else "unclassified_source_table",
                            ),
                            pixel_segmentation_ground_truth=False,
                            feature_causality_audited=False,
                        ),
                    )


def _manual_measurements(cache, protocol, images):
    asset = next(
        v
        for v in protocol["archives"]
        if v["archive_id"] == protocol["manual_archive_id"]
    )
    by_member = {
        r["member"]: r for r in images if r["archive_id"] == asset["archive_id"]
    }
    raw = sorted(
        (
            r
            for r in by_member.values()
            if r["source_folder"] == protocol["thermal_image_directory"]
        ),
        key=lambda r: r["filename_elapsed_s"],
    )
    with zipfile.ZipFile(Path(cache) / asset["filename"]) as archive:
        reader = csv.DictReader(
            io.StringIO(
                archive.read(protocol["manual_table_member"]).decode("utf-8-sig")
            )
        )
        seen = set()
        for values in reader:
            frame = int(values["Frame"])
            if frame in seen or not 0 <= frame < len(raw):
                raise ValueError("Manual table repeats or exceeds source frame indexes")
            seen.add(frame)
            image = raw[frame]
            annotation_member = (
                protocol["manual_image_directory"]
                + "/"
                + Path(image["member"]).stem
                + "-1.jpg"
            )
            annotation = by_member.get(annotation_member)
            if (
                annotation is None
                or annotation["filename_elapsed_s"] != image["filename_elapsed_s"]
            ):
                raise ValueError(
                    "Manual table cannot be bound to its source annotation/image"
                )
            numeric = {k: float(v) for k, v in values.items() if k != "Frame"}
            if (
                not np.isfinite(list(numeric.values())).all()
                or numeric["Manual Box FIJI"] <= 0
            ):
                raise ValueError("Invalid published manual measurements")
            yield dict(
                source_frame_index=frame,
                raw_sample_id=image["sample_id"],
                raw_native_BGR_sha256=image["native_BGR_sha256"],
                annotated_sample_id=annotation["sample_id"],
                source_table_member=protocol["manual_table_member"],
                original_values=values,
                manual_rectangle_area_px2=numeric["Manual Box FIJI"],
                source_reported_rectangle_area_px2=numeric["Rect"],
                author_scale_px2_per_mm2=protocol["author_scale_px2_per_mm2"],
                filename_elapsed_s=image["filename_elapsed_s"],
                nominal_tracking_elapsed_s=frame
                * protocol["thermal_nominal_frame_interval_s"],
                target_kind="author_selected_primary_crystal_manual_rectangle_area_not_silhouette_mask",
                local_human_review=False,
                observed_object_identity="authors_manual_verification_target",
                segmentation_polygon=None,
            )


def import_data(cache, protocol_path, output):
    cache, protocol_path, output = (
        Path(cache).resolve(),
        Path(protocol_path).resolve(),
        Path(output).resolve(),
    )
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    shutil.copyfile(__file__, output / "importer_snapshot.py")
    manifest = dict(
        schema=SCHEMA,
        status="writing",
        cache_path=str(cache),
        protocol=protocol,
        protocol_sha256=file_sha256(protocol_path),
        source_sha256=file_sha256(__file__),
        environment=dict(
            python=sys.version,
            opencv=cv2.__version__,
            numpy=np.__version__,
            pillow=PIL.__version__,
            platform=platform.platform(),
        ),
    )
    save_json(output / "manifest.json", manifest)
    images, counts, formats, roles, folders, duplicates = (
        [],
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        defaultdict(list),
    )
    streams = {
        kind: (output / f"{kind}.jsonl").open("x")
        for kind in ("members", "images", "tables", "manual_measurements")
    }
    try:
        for kind, record in scan(cache, protocol):
            streams[kind].write(json.dumps(record, allow_nan=False) + "\n")
            counts[kind] += 1
            if kind == "images":
                images.append(record)
                formats[record["decoded_format"]] += 1
                roles[record["role"]] += 1
                folders[record["sequence_id"]] += 1
                duplicates[record["native_BGR_sha256"]].append(record["sample_id"])
        for record in _manual_measurements(cache, protocol, images):
            streams["manual_measurements"].write(
                json.dumps(record, allow_nan=False) + "\n"
            )
            counts["manual_measurements"] += 1
        for stream in streams.values():
            stream.close()
        save_json(
            output / "duplicate_pixels.json",
            {key: value for key, value in duplicates.items() if len(value) > 1},
        )
        save_json(output / "reported_actions.json", protocol["reported_actions"])
        manifest.update(
            status="complete",
            counts=dict(counts),
            formats=dict(formats),
            roles=dict(roles),
            source_folders=dict(folders),
            duplicate_pixel_groups=sum(len(v) > 1 for v in duplicates.values()),
            independent_evaluation_groups_assigned=0,
            real_pitch_segmentation_labels=0,
            local_human_reviews=0,
            file_sha256={
                p.name: file_sha256(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise
    finally:
        for stream in streams.values():
            stream.close()


def load_image(index, sample_id):
    """Load one indexed source image; preserve raw pixels and reject source edits."""
    index = Path(index)
    manifest = json.loads((index / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "complete"
        or file_sha256(index / "images.jsonl")
        != manifest["file_sha256"]["images.jsonl"]
    ):
        raise ValueError("Completed, unchanged CrystalCV index required")
    record = None
    with (index / "images.jsonl").open() as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["sample_id"] == sample_id:
                record = candidate
                break
    if record is None:
        raise ValueError("Unknown source sample")
    asset = next(
        v
        for v in manifest["protocol"]["archives"]
        if v["archive_id"] == record["archive_id"]
    )
    with zipfile.ZipFile(Path(manifest["cache_path"]) / asset["filename"]) as archive:
        payload = archive.read(record["member"])
    if hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("Native source image bytes changed")
    image, image_format = _image(payload, manifest["protocol"]["maximum_image_pixels"])
    if (
        image_format != record["decoded_format"]
        or hashlib.sha256(image.tobytes()).hexdigest() != record["native_BGR_sha256"]
    ):
        raise ValueError("Native decoded pixels changed")
    return image, record
