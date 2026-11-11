# FILE PURPOSE: Import and audit real Leeds microscope images without inventing labels.
# Whole run/condition-family splits are frozen in a protocol. Original JPEGs and
# manual crystal-fit sidecars stay in the checked source ZIP; masks/events and
# actual acquisition times remain unavailable until their own annotation review.
"""Hash-pinned real-image indexing and lazy loading for the first vision dataset."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from .video import file_sha256

SCHEMA = "crystalab_real_vision_index_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
    )


def _protocol(protocol):
    if protocol.get("schema") != "crystalab_leeds_import_v1":
        raise ValueError("Unsupported real-image import protocol")
    if not protocol.get("source_id") or not protocol.get("material"):
        raise ValueError("Source identity/material required")
    if not protocol.get("license") or not protocol.get("license_url"):
        raise ValueError("Source license and attribution are required")
    interval = protocol.get("nominal_frame_interval_s")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("Invalid nominal frame interval")
    if not protocol.get("clock_provenance"):
        raise ValueError("Nominal timing requires source evidence and limits")
    families, aliases = {}, set()
    for run, settings in protocol["experiments"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
            raise ValueError("Unsafe run identifier")
        if settings["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        family = settings["family_id"]
        if families.setdefault(family, settings["split"]) != settings["split"]:
            raise ValueError("Condition family cannot cross splits")
        alias = settings["annotation_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", alias) or alias in aliases:
            raise ValueError("Unsafe/duplicate annotation identifier")
        aliases.add(alias)
        if (
            type(settings["expected_frames"]) is not int
            or settings["expected_frames"] < 1
        ):
            raise ValueError("Invalid expected frame count")
    if set(families.values()) != {"train", "validation", "test"}:
        raise ValueError("Declare nonempty train, validation and test families")


def _scan(archive_path, protocol):
    _protocol(protocol)
    if file_sha256(archive_path) != protocol["archive_sha256"]:
        raise ValueError("Source archive hash differs from the reviewed protocol")
    rows, seen_hashes = [], {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Archive has duplicate member names")
        if any(
            Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n for n in names
        ):
            raise ValueError("Unsafe archive member")
        grouped = {}
        for name in names:
            match = re.fullmatch(
                r"experiments/img/([^/]+)/(?:LGA_alpha_)?(\d+)\.jpg", name
            )
            if match:
                run, number = match.groups()
                if run not in protocol["experiments"]:
                    raise ValueError(f"Undeclared real experiment: {run}")
                grouped.setdefault(run, []).append((int(number), name))
            elif name.startswith("experiments/img/") and not name.endswith("/"):
                raise ValueError(f"Unrecognized original image filename: {name}")
        if set(grouped) != set(protocol["experiments"]):
            raise ValueError("Source experiments differ from the frozen protocol")
        for run, settings in protocol["experiments"].items():
            members = sorted(grouped[run])
            if [i for i, _ in members] != list(range(settings["expected_frames"])):
                raise ValueError(
                    f"Missing/duplicate/nonconsecutive source frames for {run}"
                )
            alias = settings["annotation_id"]
            for index, member in members:
                payload = archive.read(member)
                fingerprint = hashlib.sha256(payload).hexdigest()
                if (
                    seen_hashes.setdefault(fingerprint, settings["split"])
                    != settings["split"]
                ):
                    raise ValueError("Identical real images cannot cross splits")
                pixels = cv2.imdecode(
                    np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR
                )
                if pixels is None or pixels.ndim != 3:
                    raise ValueError(f"Undecodable image: {member}")
                label_member = f"experiments/manual/{alias}/{index:04d}.json"
                label_hash = None
                if label_member in names:
                    raw = archive.read(label_member)
                    fit = json.loads(
                        raw,
                        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
                    )
                    if not isinstance(fit, dict) or not {
                        "distances",
                        "miller_indices",
                        "rotation",
                        "origin",
                        "scale",
                    } <= set(fit):
                        raise ValueError(
                            f"Unsupported manual-fit sidecar: {label_member}"
                        )
                    label_hash = hashlib.sha256(raw).hexdigest()
                else:
                    label_member = None
                rows.append(
                    dict(
                        sample_id=f"{protocol['source_id']}:{run}:{index:04d}",
                        source_id=protocol["source_id"],
                        material=protocol["material"],
                        evidence_kind="real",
                        run_id=run,
                        family_id=settings["family_id"],
                        split=settings["split"],
                        frame_index=index,
                        archive_member=member,
                        image_sha256=fingerprint,
                        image_width=pixels.shape[1],
                        image_height=pixels.shape[0],
                        acquisition_time_s=None,
                        nominal_elapsed_s=index * protocol["nominal_frame_interval_s"],
                        timestamp_basis="paper_reported_cadence_not_verified_per_frame",
                        labels=dict(
                            manual_crystal_fit=label_member,
                            manual_fit_sha256=label_hash,
                            manual_fit_semantics="source_manual_3d_shape_parameters_not_pixel_masks"
                            if label_member
                            else None,
                            segmentation_mask=None,
                            visible_event=None,
                            optical_quality=None,
                        ),
                        segmentation_training_ready=False,
                    )
                )
    return rows


def import_leeds(archive_path, protocol, output):
    """Index actual checked images; refuse existing outputs and preserve failures."""
    _protocol(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema=SCHEMA,
        status="running",
        protocol=protocol,
        archive_path=str(Path(archive_path).resolve()),
        importer_sha256=file_sha256(__file__),
    )
    path = output / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    save()
    try:
        rows = _scan(archive_path, protocol)
        with (output / "images.jsonl").open("x") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        summary = dict(
            images=len(rows),
            runs=len({r["run_id"] for r in rows}),
            images_by_split=dict(Counter(r["split"] for r in rows)),
            images_by_run=dict(Counter(r["run_id"] for r in rows)),
            images_with_manual_shape_fit=sum(
                r["labels"]["manual_crystal_fit"] is not None for r in rows
            ),
            reviewed_segmentation_masks=0,
            reviewed_visible_events=0,
        )
        manifest.update(
            status="complete",
            index_sha256=file_sha256(output / "images.jsonl"),
            summary=summary,
        )
        save()
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


def audit_image_index(output, *, archive_path=None):
    """Re-read every original image/sidecar and reconstruct the stored index."""
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
        raise ValueError("Unsupported/incomplete real-image index")
    if file_sha256(output / "images.jsonl") != manifest["index_sha256"]:
        raise ValueError("Image index hash mismatch")
    rows = [
        json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()
    ]
    expected = _scan(archive_path or manifest["archive_path"], manifest["protocol"])
    if rows != expected:
        raise ValueError(
            "Image records/labels differ from original archive reconstruction"
        )
    return dict(
        status="pass", **manifest["summary"], index_sha256=manifest["index_sha256"]
    )


def load_index_image(archive, record):
    """Decode one indexed original from an open ZipFile, verifying its bytes."""
    payload = archive.read(record["archive_member"])
    if hashlib.sha256(payload).hexdigest() != record["image_sha256"]:
        raise ValueError("Image identity changed")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (
        record["image_height"],
        record["image_width"],
    ):
        raise ValueError("Image shape/decode differs from its record")
    return image
