# FILE PURPOSE: Export audited optical proxies and explicitly reviewed real polygons for YOLO.
# Original family splits, source/receipt identity and polygon-conversion loss are
# retained. Unreviewed real proposals cannot enter this image-training adapter.
"""Provenance-preserving segmentation datasets with independent split auditing."""

from collections import Counter
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from rendering.optical import CLASSES, audit_optical
from vision.video import file_sha256


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe dataset artifact path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Dataset artifact escapes its root")
    return resolved


def polygon_line(class_id, points, width, height):
    points = np.asarray(points, dtype=float)
    if (
        type(class_id) is not int
        or not 0 <= class_id < len(CLASSES)
        or points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
        or np.any(points < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] >= height)
        or abs(cv2.contourArea(points.astype(np.float32))) < 1
    ):
        raise ValueError("Invalid segmentation polygon")
    normalized = points / [width, height]
    return str(class_id) + " " + " ".join(f"{p:.9f}" for p in normalized.ravel())


def polygon_masks(path, shape):
    """Rasterize audited YOLO polygons with nearest-pixel coordinates for diagnostics."""
    height, width = shape
    truth = []
    for line in Path(path).read_text().splitlines():
        values = line.split()
        points = np.asarray([float(v) for v in values[1:]]).reshape(-1, 2)
        points = np.rint(points * [width, height]).astype(np.int32)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [points], 255)
        truth.append(dict(class_id=int(values[0]), mask=mask))
    return truth


def mask_polygon(mask, minimum_iou=0.95):
    """YOLO external polygons fill holes; measure and bound that representation loss."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1 or len(contours[0]) < 3:
        raise ValueError("Expected one nondegenerate proxy component")
    reconstructed = np.zeros_like(mask)
    cv2.drawContours(reconstructed, contours, -1, 255, cv2.FILLED)
    intersection = np.count_nonzero((mask > 0) & (reconstructed > 0))
    union = np.count_nonzero((mask > 0) | (reconstructed > 0))
    iou = float(intersection / union) if union else 0.0
    if iou < minimum_iou:
        raise ValueError("External polygon loses too much mask detail; inspect source")
    return contours[0][:, 0, :], dict(
        original_mask_pixels=int(np.count_nonzero(mask)),
        polygon_mask_iou=iou,
        filled_hole_pixels=int(np.count_nonzero((reconstructed > 0) & (mask == 0))),
        minimum_iou=minimum_iou,
        policy="external_polygon_fills_small_holes",
    )


def _optical_records(root, manifest):
    audit_optical(root)
    qa_path = root / "visual_review.json"
    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    if qa.get("decision") not in {
        "software_smoke_accepted",
        "synthetic_development_QA_accepted",
    } or not qa.get("inspected_training_samples"):
        raise ValueError(
            "Synthetic source needs recorded development RGB/mask inspection"
        )
    records = []
    for line in (root / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        image = cv2.imread(str(safe_path(root, row["image_path"])))
        height, width = image.shape[:2]
        labels, conversions = [], []
        for obj in row["objects"]:
            if not obj["label_eligible"]:
                continue
            mask = cv2.imread(str(safe_path(root, obj["mask_path"])), 0)
            points, conversion = mask_polygon(mask)
            labels.append(polygon_line(obj["class_id"], points, width, height))
            conversions.append(dict(object_id=obj["id"], **conversion))
        records.append(
            dict(
                original=row,
                image_path=row["image_path"],
                sample_id=row["sample_id"],
                family_key="synthetic:" + row["family_id"],
                split=row["split"],
                evidence_kind="synthetic",
                material="illustrative_optical_geometry",
                label_lines=labels,
                conversions=conversions,
            )
        )
    return records, qa


def _reviewed_records(root, manifest):
    if manifest["classes"] != CLASSES:
        raise ValueError("Reviewed class map differs")
    for path, digest in manifest["artifact_sha256"].items():
        if file_sha256(safe_path(root, path)) != digest:
            raise ValueError("Reviewed source artifact identity changed")
    records = []
    for record in manifest["records"]:
        source, receipt = record["source_record"], record["review"]
        image_path = safe_path(root, record["image_path"])
        annotation_path = safe_path(root, record["annotation_path"])
        annotation = json.loads(annotation_path.read_text())
        if (
            source.get("evidence_kind") != "real"
            or receipt["sample_id"] != source["sample_id"]
            or not receipt.get("reviewer")
            or not receipt.get("acknowledgment")
            or receipt["image_sha256"] != file_sha256(image_path)
            or receipt["annotation_sha256"] != file_sha256(annotation_path)
            or (not annotation["shapes"] and not receipt.get("empty_reviewed"))
        ):
            raise ValueError(
                "Reviewed source lacks an exact matching real-image receipt"
            )
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (
            annotation["imageHeight"],
            annotation["imageWidth"],
        ):
            raise ValueError("Reviewed image dimensions changed")
        lines = []
        for shape in annotation["shapes"]:
            if (
                shape.get("flags", {}).get("uncertain")
                or shape["shape_type"] != "polygon"
            ):
                raise ValueError("Uncertain/nonpolygon real annotation")
            lines.append(
                polygon_line(
                    CLASSES.index(shape["label"]),
                    shape["points"],
                    annotation["imageWidth"],
                    annotation["imageHeight"],
                )
            )
        expected = "\n".join(lines) + ("\n" if lines else "")
        if safe_path(root, record["label_path"]).read_text() != expected:
            raise ValueError("Reviewed YOLO labels differ from receipted polygons")
        records.append(
            dict(
                original=record,
                image_path=record["image_path"],
                sample_id=source["sample_id"],
                family_key="real:" + source["source_id"] + ":" + source["family_id"],
                split=source["split"],
                evidence_kind="real",
                material=source["material"],
                label_lines=lines,
                conversions=[],
            )
        )
    return records, None


def prepare_segmentation(sources, output):
    """Combine source exports while preserving whole-family assignments; never overwrite."""
    if not sources:
        raise ValueError("At least one audited source is required")
    loaded, provenance = [], []
    for source in sources:
        root = Path(source).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] == "crystalab_optical_dataset_v1":
            records, qa = _optical_records(root, manifest)
        elif manifest["schema"] == "crystalab_reviewed_segmentation_v1":
            records, qa = _reviewed_records(root, manifest)
        else:
            raise ValueError(
                "Only audited optical or reviewed real exports are accepted"
            )
        provenance.append(
            dict(
                path=str(root),
                manifest_sha256=file_sha256(root / "manifest.json"),
                manifest=manifest,
                visual_review=qa,
            )
        )
        loaded.extend((root, len(provenance) - 1, r) for r in records)
    families, hashes, ids = {}, {}, set()
    for root, source_index, record in loaded:
        split = record["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown source split")
        if families.setdefault(record["family_key"], split) != split:
            raise ValueError("Related source families cross splits")
        digest = file_sha256(safe_path(root, record["image_path"]))
        if hashes.setdefault(digest, split) != split:
            raise ValueError("Duplicate source images cross splits")
        key = (source_index, record["sample_id"])
        if key in ids:
            raise ValueError("Duplicate source sample ID")
        ids.add(key)
    if not {"train", "validation"} <= set(families.values()):
        raise ValueError("Training and validation need independent source families")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (root, source_index, record) in enumerate(loaded):
        split = record["split"]
        image = cv2.imread(str(safe_path(root, record["image_path"])))
        name = f"sample_{index:06d}"
        image_path = output / "images" / split / (name + ".png")
        label_path = output / "labels" / split / (name + ".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image):
            raise OSError("Cannot export training image")
        lines = record.pop("label_lines")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            dict(
                record,
                source_index=source_index,
                image_path=str(image_path.relative_to(output)),
                label_path=str(label_path.relative_to(output)),
                objects=len(lines),
            )
        )
    data = dict(
        path=str(output), train="images/train", val="images/validation", names=CLASSES
    )
    if "test" in families.values():
        data["test"] = "images/test"
    save_json(output / "dataset.yaml", data)  # JSON is also valid YAML.
    manifest = dict(
        schema="crystalab_yolo_segmentation_v1",
        classes=CLASSES,
        export_environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            opencv=cv2.__version__,
        ),
        sources=provenance,
        records=rows,
        scope="Visible polygons; synthetic optical proxies and reviewed real evidence remain distinct",
        artifact_sha256={
            str(p.relative_to(output)): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    )
    save_json(output / "manifest.json", manifest)
    return audit_segmentation(output)


def audit_segmentation(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["schema"] != "crystalab_yolo_segmentation_v1"
        or manifest["classes"] != CLASSES
    ):
        raise ValueError("Unsupported segmentation dataset")
    expected = {}
    for index, source in enumerate(manifest["sources"]):
        root = Path(source["path"])
        if file_sha256(root / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("Original source manifest changed")
        original = json.loads((root / "manifest.json").read_text())
        if original != source["manifest"]:
            raise ValueError("Saved source manifest differs from its original")
        if original["schema"] == "crystalab_optical_dataset_v1":
            records, qa = _optical_records(root, original)
        elif original["schema"] == "crystalab_reviewed_segmentation_v1":
            records, qa = _reviewed_records(root, original)
        else:
            raise ValueError("Unsupported original source")
        if qa != source["visual_review"]:
            raise ValueError("Original visual review changed")
        expected.update({(index, r["sample_id"]): (root, r) for r in records})
    for path, digest in manifest["artifact_sha256"].items():
        if file_sha256(safe_path(output, path)) != digest:
            raise ValueError("Training dataset artifact identity changed")
    data = json.loads((output / "dataset.yaml").read_text())
    if data != dict(
        path=str(output),
        train="images/train",
        val="images/validation",
        names=CLASSES,
        **(
            {"test": "images/test"}
            if any(r["split"] == "test" for r in manifest["records"])
            else {}
        ),
    ):
        raise ValueError("Training YAML differs from audited paths/classes")
    families, hashes, paths, classes = {}, {}, set(), Counter()
    for row in manifest["records"]:
        root, original = expected.pop((row["source_index"], row["sample_id"]))
        for field in (
            "original",
            "family_key",
            "split",
            "evidence_kind",
            "material",
            "conversions",
        ):
            if row[field] != original[field]:
                raise ValueError("Export record differs from original source evidence")
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown segmentation split")
        if families.setdefault(row["family_key"], split) != split:
            raise ValueError("Training family crosses splits")
        path = safe_path(output, row["image_path"])
        if path.parent != output / "images" / split or str(path) in paths:
            raise ValueError("Duplicate/misplaced training image")
        paths.add(str(path))
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unreadable training image")
        source_image = cv2.imread(str(safe_path(root, original["image_path"])))
        if source_image is None or not np.array_equal(image, source_image):
            raise ValueError("Exported image pixels differ from original source")
        digest = file_sha256(path)
        if hashes.setdefault(digest, split) != split:
            raise ValueError("Identical decoded training images cross splits")
        labels = safe_path(output, row["label_path"])
        if labels != output / "labels" / split / (path.stem + ".txt"):
            raise ValueError("Image/label pairing differs")
        lines = labels.read_text().splitlines()
        if lines != original["label_lines"]:
            raise ValueError(
                "Training labels differ from original reviewed/proxy geometry"
            )
        if len(lines) != row["objects"]:
            raise ValueError("Object count differs")
        for line in lines:
            values = line.split()
            class_id = int(values[0])
            points = np.asarray([float(v) for v in values[1:]]).reshape(-1, 2)
            if (
                str(class_id) != values[0]
                or not 0 <= class_id < len(CLASSES)
                or len(points) < 3
                or not np.isfinite(points).all()
                or np.any(points < 0)
                or np.any(points > 1)
            ):
                raise ValueError("Invalid normalized YOLO polygon")
            classes[CLASSES[class_id]] += 1
    if (
        expected
        or set(str(p.resolve()) for p in (output / "images").rglob("*.png")) != paths
    ):
        raise ValueError("Unindexed training images")
    return dict(
        status="pass",
        images=len(paths),
        families=len(families),
        images_by_split=dict(Counter(r["split"] for r in manifest["records"])),
        images_by_evidence=dict(
            Counter(r["evidence_kind"] for r in manifest["records"])
        ),
        objects_by_class=dict(classes),
        manifest_sha256=file_sha256(output / "manifest.json"),
    )
