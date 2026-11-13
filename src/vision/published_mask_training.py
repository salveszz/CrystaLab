# FILE PURPOSE: Export a separate author-selected protein crystal/drop pilot.
# Original expert masks remain intact; unsupported polygon images are excluded,
# never turned into negatives or pitch classes. No review receipt is created.
"""Source-bound auxiliary training data for published selected-instance masks."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from vision.published_masks import CLASSES, published_masks
from vision.training_data import mask_polygon, polygon_line, safe_path, save_json
from vision.video import file_sha256

SCHEMA = "crystalab_published_selected_mask_training_v1"


def source_rows(index, protocol):
    if (
        protocol["schema"] != "crystalab_chimp_selected_mask_pilot_v1"
        or protocol["task"] != "published_selected_protein_instances"
        or protocol["classes"] != list(CLASSES)
        or protocol["pitch_class_mapping"] is not None
    ):
        raise ValueError("Preserve the distinct published selected-object task")
    manifest = json.loads((index / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or file_sha256(index / "manifest.json") != protocol["source_manifest_sha256"]
        or file_sha256(index / "images.jsonl") != protocol["source_index_sha256"]
        or file_sha256(manifest["archive_path"])
        != manifest["protocol"]["archive_sha256"]
    ):
        raise ValueError("Published source identities changed")
    return manifest, [
        json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()
    ]


def select(rows, settings):
    selected = []
    for split, count in [
        ("train", settings["train_per_dataset"]),
        ("validation", settings["validation_per_dataset"]),
    ]:
        if type(count) is not int or count < 1:
            raise ValueError("Require positive source pilot counts")
        for dataset in settings["datasets"]:
            cohort = [
                r
                for r in rows
                if r["source_split"] == split and r["source_dataset"] == dataset
            ]
            cohort.sort(
                key=lambda r: hashlib.sha256(
                    (settings["hash_prefix"] + r["sample_id"]).encode()
                ).hexdigest()
            )
            if len(cohort) < count:
                raise ValueError("Source cohort too small for frozen pilot")
            selected.extend(cohort[:count])
    if len({r["sample_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate source image in pilot")
    return selected


def prepare(index, protocol_path, output):
    index, output = Path(index).resolve(), Path(output).resolve()
    protocol = json.loads(Path(protocol_path).read_text())
    manifest, rows = source_rows(index, protocol)
    selected = select(rows, protocol["selection"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output / "protocol.json")
    records = []
    with zipfile.ZipFile(manifest["archive_path"]) as archive:
        for ordinal, row in enumerate(selected):
            original = archive.read(row["image_member"])
            if hashlib.sha256(original).hexdigest() != row["image_sha256"]:
                raise ValueError("Original image changed")
            image_path = f"source_images/{ordinal:05d}.jpg"
            (output / image_path).parent.mkdir(exist_ok=True)
            (output / image_path).write_bytes(original)
            payload = archive.read(row["mask_member"])
            if hashlib.sha256(payload).hexdigest() != row["mask_archive_sha256"]:
                raise ValueError("Original source mask stack changed")
            objects, lines, failures = [], [], []
            for object_index, (label, plane) in enumerate(
                published_masks(
                    payload,
                    (row["image_height"], row["image_width"]),
                    manifest["protocol"],
                )
            ):
                original_object = row["source_objects"][object_index]
                if (
                    hashlib.sha256(plane.tobytes()).hexdigest()
                    != original_object["bool_plane_sha256"]
                    or label != original_object["source_class"]
                ):
                    raise ValueError("Original source mask plane changed")
                mask = plane.astype(np.uint8) * 255
                mask_path = f"ground_truth/{ordinal:05d}/{object_index:04d}.png"
                (output / mask_path).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output / mask_path), mask):
                    raise OSError("Cannot save original mask")
                conversion, failure = None, None
                try:
                    points, conversion = mask_polygon(
                        mask, protocol["polygon_export"]["minimum_native_iou"]
                    )
                    lines.append(
                        polygon_line(
                            CLASSES.index(label),
                            points,
                            row["image_width"],
                            row["image_height"],
                        )
                    )
                except ValueError as error:
                    failure = str(error)
                    failures.append(dict(object_index=object_index, reason=failure))
                objects.append(
                    dict(
                        object_index=object_index,
                        source_class=label,
                        mask_path=mask_path,
                        bool_plane_sha256=original_object["bool_plane_sha256"],
                        conversion=conversion,
                        representation_error=failure,
                    )
                )
            if len(objects) != len(row["source_objects"]):
                raise ValueError("Source object count changed")
            label_path = train_image_path = None
            if not failures:
                train_image_path = f"images/{row['source_split']}/{ordinal:05d}.jpg"
                label_path = f"labels/{row['source_split']}/{ordinal:05d}.txt"
                for relative in [train_image_path, label_path]:
                    (output / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output / image_path, output / train_image_path)
                (output / label_path).write_text("".join(line + "\n" for line in lines))
            records.append(
                dict(
                    sample_id=row["sample_id"],
                    split=row["source_split"],
                    source_dataset=row["source_dataset"],
                    image_path=image_path,
                    image_sha256=row["image_sha256"],
                    image_width=row["image_width"],
                    image_height=row["image_height"],
                    training_image_path=train_image_path,
                    label_path=label_path,
                    objects=objects,
                    eligible=not failures,
                    rejection_details=failures,
                    source_annotation_scope=row["annotation_scope"],
                    physical_run_id=None,
                    pitch_class_mapping=None,
                )
            )
    save_json(output / "records.json", records)
    save_json(
        output / "dataset.yaml",
        dict(
            path=str(output),
            train="images/train",
            val="images/validation",
            names=list(CLASSES),
        ),
    )
    save_json(
        output / "manifest.json",
        dict(
            schema=SCHEMA,
            status="complete",
            task=protocol["task"],
            classes=list(CLASSES),
            source_index=str(index),
            protocol_sha256=file_sha256(protocol_path),
            selected_images=len(records),
            eligible_images=sum(r["eligible"] for r in records),
            rejected_images=sum(not r["eligible"] for r in records),
            actual_human_reviews_created=0,
            scope=protocol["class_semantics"],
            pitch_class_mapping=None,
            independent_KDP_evaluation_groups=0,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        ),
    )
    return audit_training(output)


def audit_training(output):
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        manifest["schema"] != SCHEMA
        or manifest["task"] != "published_selected_protein_instances"
        or manifest["classes"] != list(CLASSES)
        or manifest["pitch_class_mapping"] is not None
        or manifest["actual_human_reviews_created"] != 0
        or file_sha256(root / "protocol.json") != manifest["protocol_sha256"]
    ):
        raise ValueError("Distinct published selected-mask task required")
    for name, digest in manifest["files"].items():
        if file_sha256(safe_path(root, name)) != digest:
            raise ValueError("Published-mask training artifact changed")
    _, rows = source_rows(Path(manifest["source_index"]), protocol)
    expected = select(rows, protocol["selection"])
    records = json.loads((root / "records.json").read_text())
    if [r["sample_id"] for r in records] != [r["sample_id"] for r in expected]:
        raise ValueError("Source pilot membership changed")
    for directory, expected_paths in [
        ("images", {r["training_image_path"] for r in records if r["eligible"]}),
        ("labels", {r["label_path"] for r in records if r["eligible"]}),
    ]:
        actual = {
            str(p.relative_to(root))
            for p in (root / directory).rglob("*")
            if p.is_file() and p.suffix in {".jpg", ".jpeg", ".png", ".txt"}
        }
        if actual != expected_paths:
            raise ValueError("Unindexed training image or label")
    for row, original in zip(records, expected):
        if (
            row["split"] != original["source_split"]
            or row["source_dataset"] != original["source_dataset"]
            or file_sha256(root / row["image_path"]) != original["image_sha256"]
            or len(row["objects"]) != len(original["source_objects"])
        ):
            raise ValueError("Source image membership/bytes changed")
        lines, failures = [], []
        for obj, source in zip(row["objects"], original["source_objects"]):
            mask = cv2.imread(str(safe_path(root, obj["mask_path"])), 0)
            if (
                mask.shape != (original["image_height"], original["image_width"])
                or not set(np.unique(mask)) <= {0, 255}
                or hashlib.sha256((mask > 0).tobytes()).hexdigest()
                != source["bool_plane_sha256"]
                or obj["source_class"] != source["source_class"]
            ):
                raise ValueError("Original expert mask/class changed")
            conversion, failure = None, None
            try:
                points, conversion = mask_polygon(
                    mask, protocol["polygon_export"]["minimum_native_iou"]
                )
                lines.append(
                    polygon_line(
                        CLASSES.index(source["source_class"]),
                        points,
                        original["image_width"],
                        original["image_height"],
                    )
                )
            except ValueError as error:
                failure = str(error)
                failures.append(dict(object_index=source["index"], reason=failure))
            if (
                obj["conversion"] != conversion
                or obj["representation_error"] != failure
            ):
                raise ValueError("Polygon representation record changed")
        if row["eligible"] != (not failures) or row["rejection_details"] != failures:
            raise ValueError("Unsupported mask cannot become a negative")
        if row["eligible"]:
            if file_sha256(safe_path(root, row["training_image_path"])) != original[
                "image_sha256"
            ] or safe_path(root, row["label_path"]).read_text() != "".join(
                line + "\n" for line in lines
            ):
                raise ValueError("Training pixels/polygons differ from source")
        elif row["label_path"] is not None or row["training_image_path"] is not None:
            raise ValueError("Rejected source image leaked into training")
    if json.loads((root / "dataset.yaml").read_text()) != dict(
        path=str(root),
        train="images/train",
        val="images/validation",
        names=list(CLASSES),
    ):
        raise ValueError("Source class map or paths changed")
    return dict(
        status="passed",
        task=manifest["task"],
        classes=list(CLASSES),
        selected_images=len(records),
        eligible_by_split=dict(Counter(r["split"] for r in records if r["eligible"])),
        rejected_images=sum(not r["eligible"] for r in records),
        eligible_masks_by_class=dict(
            Counter(
                o["source_class"]
                for r in records
                if r["eligible"]
                for o in r["objects"]
            )
        ),
        source_manifest_sha256=protocol["source_manifest_sha256"],
        independent_KDP_evaluation_groups=0,
        actual_human_reviews_created=0,
    )
