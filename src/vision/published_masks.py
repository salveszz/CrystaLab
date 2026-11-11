# FILE PURPOSE: Index published real protein-crystal masks without relabeling them as KDP truth.
# Source splits, selected-object semantics and original bytes stay intact. Boolean
# mask stacks are streamed one plane at a time, with NumPy pickle loading disabled.
"""Audited CHiMP image/mask acquisition; no automatic pitch-class training export."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np

from vision.training_data import save_json
from vision.video import file_sha256

SCHEMA = "crystalab_published_mask_index_v1"
CLASSES = ("crystal", "drop")
SCOPE = "Published selected/mountable protein-crystal and drop masks; not KDP, exhaustive generic crystal, seed-role, event or quality truth"


def validate_protocol(protocol):
    if protocol.get("schema") != "crystalab_chimp_import_v1":
        raise ValueError("Unsupported published-mask import protocol")
    if (
        not protocol.get("source_id")
        or not protocol.get("citation")
        or protocol.get("license_id") != "cc-by-4.0"
        or protocol.get("source_classes") != list(CLASSES)
        or protocol.get("pitch_class_mapping") is not None
    ):
        raise ValueError("Preserve source annotation identity and unmapped classes")
    for key in ("archive_sha256", "metadata_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", protocol.get(key, "")):
            raise ValueError("Pin the original archive and license metadata")
    for key in ("maximum_instances", "maximum_pixels", "maximum_stack_bytes"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError("Declare positive import bounds")
    if not protocol["datasets"]:
        raise ValueError("Declare source dataset membership")
    seen = set()
    for name, dataset in protocol["datasets"].items():
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise ValueError("Invalid source dataset ID")
        for key in ("directory", "training_csv", "validation_csv"):
            path = Path(dataset[key])
            if path.is_absolute() or ".." in path.parts or "\\" in str(path):
                raise ValueError("Unsafe source member path")
        if dataset["directory"] in seen:
            raise ValueError("Duplicate source dataset directory")
        seen.add(dataset["directory"])
        if (
            type(dataset["expected_images"]) is not int
            or dataset["expected_images"] < 1
        ):
            raise ValueError("Declare expected image counts")


def published_masks(payload, shape, protocol):
    """Yield (label, native bool plane); never materialize the full mask tensor."""
    height, width = shape
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        names = inner.namelist()
        if len(names) != 2 or set(names) != {"masks.npy", "class_labels.npy"}:
            raise ValueError("Unexpected or duplicate arrays in source NPZ")
        if sum(i.file_size for i in inner.infolist()) > protocol["maximum_stack_bytes"]:
            raise ValueError("Source mask stack exceeds declared expanded bound")
        if inner.getinfo("class_labels.npy").file_size > 1_000_000:
            raise ValueError("Source class array is too large")
        labels = np.load(io.BytesIO(inner.read("class_labels.npy")), allow_pickle=False)
        if labels.ndim != 1 or len(labels) > protocol["maximum_instances"]:
            raise ValueError("Invalid source class array")
        if len(labels) and (
            labels.dtype.kind != "U" or any(v not in CLASSES for v in labels)
        ):
            raise ValueError("Unknown source class; no pitch label may be inferred")
        with inner.open("masks.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            if reader is None:
                raise ValueError("Unsupported NPY format")
            dimensions, fortran, dtype = reader(stream)
            allowed = {(len(labels), height, width)}
            if len(labels) == 0:
                allowed.add((0,))  # Empty stacks retain the publisher's original shape.
            if dimensions not in allowed or fortran or dtype != np.dtype(bool):
                raise ValueError("Source mask axes/dtype differ from native image")
            size = height * width
            for label in labels:
                raw = stream.read(size)
                if len(raw) != size:
                    raise ValueError("Truncated source mask plane")
                mask = np.frombuffer(raw, np.uint8).reshape(shape)
                if np.any(mask > 1) or not np.any(mask):
                    raise ValueError(
                        "Source masks must be nonempty canonical boolean planes"
                    )
                yield str(label), mask.view(bool)
            if stream.read(1):
                raise ValueError("Unexpected trailing mask array bytes")


def image_bytes(archive, member, protocol):
    raw = archive.read(member)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[0] * image.shape[1] > protocol["maximum_pixels"]:
        raise ValueError("Undecodable or oversized source image")
    return raw, image


def scan(archive_path, metadata_path, protocol):
    validate_protocol(protocol)
    if (
        file_sha256(archive_path) != protocol["archive_sha256"]
        or file_sha256(metadata_path) != protocol["metadata_sha256"]
    ):
        raise ValueError("Published archive or license metadata identity changed")
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata["metadata"]["license"]["id"] != protocol["license_id"] or str(
        metadata["id"]
    ) != str(protocol["zenodo_record_id"]):
        raise ValueError("Source license/record differs from protocol")
    declared = [f for f in metadata["files"] if f["key"] == protocol["archive_name"]]
    if (
        len(declared) != 1
        or declared[0]["checksum"] != "md5:" + protocol["publisher_md5"]
    ):
        raise ValueError("Publisher file checksum declaration differs")
    # SHA256 pins exact bytes; the published MD5 and byte count independently bind acquisition.
    md5 = hashlib.md5()
    with Path(archive_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            md5.update(chunk)
    if (
        md5.hexdigest() != protocol["publisher_md5"]
        or Path(archive_path).stat().st_size != declared[0]["size"]
    ):
        raise ValueError("Published file size/MD5 differs")
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n for n in names
        ):
            raise ValueError("Duplicate or unsafe ZIP members")
        for dataset_id, dataset in protocol["datasets"].items():
            directory = dataset["directory"]
            membership = {}
            for split in ("training", "validation"):
                member = f"{directory}/{dataset[split + '_csv']}"
                reader = csv.DictReader(
                    io.StringIO(archive.read(member).decode("utf-8-sig"))
                )
                if reader.fieldnames != ["filename"]:
                    raise ValueError("Unexpected source split CSV columns")
                for entry in reader:
                    name = entry["filename"]
                    if (
                        not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name)
                        or name in membership
                    ):
                        raise ValueError(
                            "Duplicate/cross-split or unsafe source filename"
                        )
                    membership[name] = "train" if split == "training" else "validation"
            images = {
                n.rsplit("/", 1)[-1]
                for n in names
                if n.startswith(directory + "/images/") and n.endswith(".jpg")
            }
            masks = {
                Path(n).stem + ".jpg"
                for n in names
                if n.startswith(directory + "/masks/") and n.endswith(".npz")
            }
            if (
                set(membership) != images
                or images != masks
                or len(images) != dataset["expected_images"]
            ):
                raise ValueError("Source image/mask/CSV coverage differs")
            for name, split in sorted(membership.items()):
                image_member = f"{directory}/images/{name}"
                mask_member = f"{directory}/masks/{Path(name).stem}.npz"
                raw, image = image_bytes(archive, image_member, protocol)
                payload = archive.read(mask_member)
                objects = []
                for index, (label, mask) in enumerate(
                    published_masks(payload, image.shape[:2], protocol)
                ):
                    ys, xs = (
                        np.flatnonzero(np.any(mask, axis=1)),
                        np.flatnonzero(np.any(mask, axis=0)),
                    )
                    x, y, right, bottom = (
                        int(xs[0]),
                        int(ys[0]),
                        int(xs[-1]),
                        int(ys[-1]),
                    )
                    objects.append(
                        dict(
                            index=index,
                            source_class=label,
                            pixels=int(np.count_nonzero(mask)),
                            bbox_xywh=[x, y, right - x + 1, bottom - y + 1],
                            touches_frame=bool(
                                x == 0
                                or y == 0
                                or right == image.shape[1] - 1
                                or bottom == image.shape[0] - 1
                            ),
                            bool_plane_sha256=hashlib.sha256(
                                mask.tobytes()
                            ).hexdigest(),
                        )
                    )
                rows.append(
                    dict(
                        sample_id=f"{protocol['source_id']}:{dataset_id}:{Path(name).stem}",
                        source_dataset=dataset_id,
                        source_split=split,
                        evidence_kind="real",
                        material="protein_crystals_in_sitting_drops",
                        image_member=image_member,
                        image_sha256=hashlib.sha256(raw).hexdigest(),
                        decoded_pixels_sha256=hashlib.sha256(
                            image.tobytes()
                        ).hexdigest(),
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        mask_member=mask_member,
                        mask_archive_sha256=hashlib.sha256(payload).hexdigest(),
                        source_objects=objects,
                        annotation_scope=SCOPE,
                        physical_run_id=None,
                        acquisition_time_s=None,
                        pitch_class_mapping=None,
                        pitch_training_ready=False,
                        local_human_review_receipt=None,
                    )
                )
    covered = {r["image_member"] for r in rows} | {r["mask_member"] for r in rows}
    if any(n.endswith((".jpg", ".npz")) and n not in covered for n in names):
        raise ValueError("Unaccounted image/mask outside declared source datasets")
    return rows


def summarize(rows):
    duplicates = defaultdict(list)
    classes = Counter()
    for row in rows:
        duplicates[row["decoded_pixels_sha256"]].append(row)
        classes.update(obj["source_class"] for obj in row["source_objects"])
    repeated = [group for group in duplicates.values() if len(group) > 1]
    return dict(
        images=len(rows),
        source_instance_masks=sum(classes.values()),
        masks_by_source_class=dict(classes),
        images_by_source_dataset=dict(Counter(r["source_dataset"] for r in rows)),
        images_by_source_split=dict(Counter(r["source_split"] for r in rows)),
        source_split_by_dataset={
            d: dict(
                Counter(r["source_split"] for r in rows if r["source_dataset"] == d)
            )
            for d in sorted({r["source_dataset"] for r in rows})
        },
        images_without_selected_crystal=sum(
            not any(o["source_class"] == "crystal" for o in r["source_objects"])
            for r in rows
        ),
        empty_source_annotation_images=sum(not r["source_objects"] for r in rows),
        duplicate_pixel_groups=[[r["sample_id"] for r in group] for group in repeated],
        cross_split_duplicate_groups=[
            [r["sample_id"] for r in group]
            for group in repeated
            if len({r["source_split"] for r in group}) > 1
        ],
        physical_run_grouping_verified=False,
        pitch_training_images=0,
        local_human_review_receipts=0,
    )


def import_published_masks(archive, metadata, protocol, output):
    validate_protocol(protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema=SCHEMA,
        status="running",
        protocol=protocol,
        archive_path=str(Path(archive).resolve()),
        metadata_path=str(Path(metadata).resolve()),
        importer_sha256=file_sha256(__file__),
        scope=SCOPE,
    )
    save_json(output / "manifest.json", manifest)
    try:
        rows = scan(archive, metadata, protocol)
        (output / "images.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        )
        shutil.copyfile(__file__, output / "importer_snapshot.py")
        manifest.update(
            status="complete",
            index_sha256=file_sha256(output / "images.jsonl"),
            summary=summarize(rows),
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise


def audit_published_masks(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest["status"] != "complete":
        raise ValueError("Completed published-mask index required")
    if (
        file_sha256(root / "images.jsonl") != manifest["index_sha256"]
        or file_sha256(root / "importer_snapshot.py") != manifest["importer_sha256"]
    ):
        raise ValueError("Published-mask index/source snapshot changed")
    rows = [
        json.loads(line) for line in (root / "images.jsonl").read_text().splitlines()
    ]
    expected = scan(
        manifest["archive_path"], manifest["metadata_path"], manifest["protocol"]
    )
    if rows != expected or manifest["summary"] != summarize(expected):
        raise ValueError("Index/summary differs from original source reconstruction")
    return dict(status="passed", **manifest["summary"])


def preview_published_masks(root, output, *, maximum_width=900):
    """Show development-only source labels; never create a human review receipt."""
    root, output = Path(root), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        or manifest["status"] != "complete"
        or file_sha256(root / "images.jsonl") != manifest["index_sha256"]
    ):
        raise ValueError("Completed unchanged source index required")
    protocol = manifest["protocol"]
    if type(maximum_width) is not int or maximum_width < 200:
        raise ValueError("Preview width must be at least 200")
    if file_sha256(manifest["archive_path"]) != protocol["archive_sha256"]:
        raise ValueError("Source archive changed")
    rows = [json.loads(v) for v in (root / "images.jsonl").read_text().splitlines()]
    selected = []
    for dataset in sorted({r["source_dataset"] for r in rows}):
        training = [
            r
            for r in rows
            if r["source_dataset"] == dataset and r["source_split"] == "train"
        ]
        for condition in (lambda n: n == 0, lambda n: 0 < n <= 5, lambda n: n > 5):
            candidates = [
                r
                for r in training
                if condition(
                    sum(o["source_class"] == "crystal" for o in r["source_objects"])
                )
            ]
            if candidates:
                selected.append(candidates[0])
    output.mkdir(parents=True, exist_ok=False)
    previews = []
    with zipfile.ZipFile(manifest["archive_path"]) as archive:
        for row in selected:
            raw, image = image_bytes(archive, row["image_member"], protocol)
            payload = archive.read(row["mask_member"])
            if (
                hashlib.sha256(raw).hexdigest() != row["image_sha256"]
                or hashlib.sha256(payload).hexdigest() != row["mask_archive_sha256"]
            ):
                raise ValueError("Preview source bytes differ")
            overlay = image.copy()
            for label, mask in published_masks(payload, image.shape[:2], protocol):
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    (60, 200, 60) if label == "crystal" else (0, 180, 250),
                    max(1, image.shape[1] // 600),
                )
            width = min(image.shape[1], maximum_width // 2)
            height = round(image.shape[0] * width / image.shape[1])
            panel = np.concatenate(
                [
                    cv2.resize(v, (width, height), interpolation=cv2.INTER_AREA)
                    for v in (image, overlay)
                ],
                axis=1,
            )
            footer = np.full((65, panel.shape[1], 3), 245, np.uint8)
            count = sum(o["source_class"] == "crystal" for o in row["source_objects"])
            for i, text in enumerate(
                [
                    f"{row['source_dataset']} {Path(row['image_member']).name} | {count} selected crystal masks",
                    "Original / source outlines: crystals green; drops amber",
                    "Protein crystals; not KDP or exhaustive generic-crystal truth",
                ]
            ):
                cv2.putText(
                    footer,
                    text,
                    (8, 16 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (35, 35, 35),
                    1,
                    cv2.LINE_AA,
                )
            name = f"{row['source_dataset']}_{Path(row['image_member']).stem}.png"
            if not cv2.imwrite(str(output / name), np.concatenate((panel, footer))):
                raise OSError("Cannot save source preview")
            previews.append(
                dict(
                    sample_id=row["sample_id"],
                    source_split="train",
                    image=name,
                    image_sha256=file_sha256(output / name),
                )
            )
    result = dict(
        schema="crystalab_published_mask_preview_v1",
        source_index_sha256=manifest["index_sha256"],
        previews=previews,
        scope=SCOPE,
        local_human_reviews_created=0,
        visual_acceptance=None,
    )
    save_json(output / "manifest.json", result)
    return result
