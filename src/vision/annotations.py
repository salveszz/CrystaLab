# FILE PURPOSE: Prepare editable real-image proposals and export only explicitly reviewed labels.
# Source run/family splits and original pixels are retained. Review receipts bind
# the exact image/annotation hashes; changing an accepted polygon invalidates its
# receipt. Empty proposals never silently become reviewed negative examples.
"""LabelMe-compatible polygon proposals, review receipts and segmentation export."""

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from .baseline import SeedTracker
from .data import SCHEMA, load_index_image, read_json
from .video import file_sha256

CLASSES = ["seed", "secondary_crystal", "inclusion_candidate"]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_bundle(
    index, run_id, output, *, frame_indices, baseline=None, proposal_width=960
):
    index, output = Path(index), Path(output)
    manifest = read_json(index / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
        raise ValueError("Incomplete real-image index")
    if file_sha256(index / "images.jsonl") != manifest["index_sha256"]:
        raise ValueError("Source index changed")
    if type(proposal_width) is not int or proposal_width < 2:
        raise ValueError("Invalid proposal image width")
    if (
        not frame_indices
        or any(type(v) is not int or v < 0 for v in frame_indices)
        or len(set(frame_indices)) != len(frame_indices)
    ):
        raise ValueError("Declare distinct nonnegative source frame indices")
    rows = [json.loads(v) for v in (index / "images.jsonl").read_text().splitlines()]
    rows = {r["frame_index"]: r for r in rows if r["run_id"] == run_id}
    if not rows or any(r["split"] != "train" for r in rows.values()):
        raise ValueError(
            "Pilot annotation proposals use training runs; final holdout annotation needs its own frozen protocol"
        )
    if not set(frame_indices) <= set(rows):
        raise ValueError("Requested annotation frames are unavailable")
    archive_path = manifest["archive_path"]
    if file_sha256(archive_path) != manifest["protocol"]["archive_sha256"]:
        raise ValueError("Source archive changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "annotations").mkdir()
    (output / "reviews").mkdir()
    bundle = dict(
        schema="crystalab_annotation_bundle_v1",
        status="preparing",
        run_id=run_id,
        classes=CLASSES,
        source_index_sha256=manifest["index_sha256"],
        source_protocol=manifest["protocol"],
        baseline_config=None if baseline is None else baseline.__dict__,
        samples=[],
    )
    save(output / "manifest.json", bundle)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for frame_index in sorted(frame_indices):
                row = rows[frame_index]
                pixels = load_index_image(archive, row)
                height, width = pixels.shape[:2]
                name = f"{run_id}_{frame_index:04d}"
                image_path = f"images/{name}.png"
                annotation_path = f"annotations/{name}.json"
                if not cv2.imwrite(str(output / image_path), pixels):
                    raise OSError("Cannot export annotation image")
                shapes = []
                proposal_reason = "no_proposal_model"
                if baseline is not None:
                    new_w = min(width, proposal_width)
                    new_h = round(height * new_w / width)
                    small = cv2.resize(
                        pixels, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    record, mask = SeedTracker(baseline).measure(small, 0)
                    proposal_reason = record["reason"]
                    if mask is not None:
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        contour = max(contours, key=cv2.contourArea)
                        contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
                        points = [
                            [float(x * width / new_w), float(y * height / new_h)]
                            for x, y in contour
                        ]
                        shapes = [
                            dict(
                                label="seed",
                                points=points,
                                group_id=None,
                                shape_type="polygon",
                                flags=dict(proposed=True, uncertain=False),
                                description="OpenCV proposal; inspect and edit before review",
                            )
                        ]
                annotation = dict(
                    version="5.0.0",
                    flags=dict(reviewed=False),
                    shapes=shapes,
                    imagePath=f"../{image_path}",
                    imageData=None,
                    imageHeight=height,
                    imageWidth=width,
                    description="Proposed labels only. Source manual 3D fits are not pixel-mask truth.",
                )
                save(output / annotation_path, annotation)
                bundle["samples"].append(
                    dict(
                        source_record=row,
                        image_path=image_path,
                        image_sha256=file_sha256(output / image_path),
                        annotation_path=annotation_path,
                        initial_proposal_sha256=file_sha256(output / annotation_path),
                        proposal_reason=proposal_reason,
                        review_path=f"reviews/{name}.json",
                    )
                )
        bundle["status"] = "ready_for_review"
        save(output / "manifest.json", bundle)
        return dict(status=bundle["status"], samples=len(bundle["samples"]), reviewed=0)
    except Exception as error:
        bundle.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(output / "manifest.json", bundle)
        raise


def _sample(bundle, annotation_name):
    matches = [
        r
        for r in bundle["samples"]
        if Path(r["annotation_path"]).name == annotation_name
    ]
    if len(matches) != 1:
        raise ValueError("Annotation is not a unique bundle member")
    return matches[0]


def _validated_annotation(root, sample):
    if file_sha256(root / sample["image_path"]) != sample["image_sha256"]:
        raise ValueError("Annotation image changed")
    annotation = read_json(root / sample["annotation_path"])
    original = sample["source_record"]
    width, height = annotation["imageWidth"], annotation["imageHeight"]
    if (width, height) != (original["image_width"], original["image_height"]):
        raise ValueError("Annotation dimensions differ from source")
    if (root / "annotations" / annotation["imagePath"]).resolve() != (
        root / sample["image_path"]
    ).resolve():
        raise ValueError("Annotation image reference changed")
    shapes = annotation["shapes"]
    for shape in shapes:
        if shape["label"] not in CLASSES or shape["shape_type"] != "polygon":
            raise ValueError("Unsupported polygon class/type")
        points = np.asarray(shape["points"], float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("Invalid polygon coordinates")
        if (
            np.any(points < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError("Polygon lies outside original pixels")
        if (
            len(np.unique(points, axis=0)) != len(points)
            or cv2.contourArea(points.astype(np.float32)) < 1
        ):
            raise ValueError("Degenerate polygon")
        if shape.get("flags", {}).get("uncertain"):
            raise ValueError("Uncertain polygons cannot become reviewed training truth")

        # Simple polygons may be concave but must not cross themselves.
        def cross(a, b, c):
            u, v = b - a, c - a
            return float(u[0] * v[1] - u[1] * v[0])

        for i in range(len(points)):
            a, b = points[i], points[(i + 1) % len(points)]
            for j in range(i + 2, len(points)):
                if (j + 1) % len(points) == i:
                    continue
                c, d = points[j], points[(j + 1) % len(points)]
                if (
                    cross(a, b, c) * cross(a, b, d) < 0
                    and cross(c, d, a) * cross(c, d, b) < 0
                ):
                    raise ValueError("Self-intersecting polygon")
    return annotation


def record_review(
    bundle_path,
    annotation_name,
    *,
    reviewer,
    acknowledge_image_reviewed=False,
    empty_reviewed=False,
    note="",
):
    """Explicit operator attestation; never called automatically during proposal export."""
    if not acknowledge_image_reviewed or not reviewer.strip():
        raise ValueError(
            "An explicit named reviewer and image-review acknowledgment are required"
        )
    root = Path(bundle_path)
    bundle = read_json(root / "manifest.json")
    if (
        bundle["schema"] != "crystalab_annotation_bundle_v1"
        or bundle["status"] != "ready_for_review"
    ):
        raise ValueError("Bundle is not ready for review")
    sample = _sample(bundle, annotation_name)
    annotation = _validated_annotation(root, sample)
    if not annotation["shapes"] and not empty_reviewed:
        raise ValueError(
            "Empty proposals need an explicit reviewed-negative acknowledgment"
        )
    receipt = dict(
        schema="crystalab_annotation_review_v1",
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        reviewer=reviewer,
        note=note,
        image_sha256=sample["image_sha256"],
        annotation_sha256=file_sha256(root / sample["annotation_path"]),
        sample_id=sample["source_record"]["sample_id"],
        empty_reviewed=empty_reviewed,
        acknowledgment="Reviewer inspected the original image, all visible target instances and current polygons",
    )
    path = root / sample["review_path"]
    if path.exists():
        raise FileExistsError(
            "Retain the previous review; use a new bundle for revisions"
        )
    save(path, receipt)
    return receipt


def export_reviewed(bundle_path, output):
    root, output = Path(bundle_path), Path(output)
    bundle = read_json(root / "manifest.json")
    if (
        bundle["schema"] != "crystalab_annotation_bundle_v1"
        or bundle["status"] != "ready_for_review"
    ):
        raise ValueError("Unsupported/incomplete annotation bundle")
    accepted = []
    for sample in bundle["samples"]:
        path = root / sample["review_path"]
        if not path.exists():
            continue
        receipt = read_json(path)
        annotation = _validated_annotation(root, sample)
        if (
            receipt["sample_id"] != sample["source_record"]["sample_id"]
            or receipt["image_sha256"] != sample["image_sha256"]
            or receipt["annotation_sha256"]
            != file_sha256(root / sample["annotation_path"])
        ):
            raise ValueError(
                "Review receipt no longer matches exact image/annotation bytes"
            )
        if (
            not receipt.get("reviewer")
            or not receipt.get("acknowledgment")
            or (not annotation["shapes"] and not receipt["empty_reviewed"])
        ):
            raise ValueError("Incomplete review receipt")
        accepted.append((sample, annotation, receipt))
    if not accepted:
        raise ValueError(
            "No explicitly reviewed annotations; proposals cannot be exported as truth"
        )
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for sample, annotation, receipt in accepted:
        original = sample["source_record"]
        split = original["split"]
        name = Path(sample["image_path"]).stem
        folder = output / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        labels = output / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / sample["image_path"], folder / f"{name}.png")
        annotations = output / "annotations" / split
        annotations.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / sample["annotation_path"], annotations / f"{name}.json")
        lines = []
        for shape in annotation["shapes"]:
            points = np.asarray(shape["points"]) / [
                annotation["imageWidth"],
                annotation["imageHeight"],
            ]
            lines.append(
                str(CLASSES.index(shape["label"]))
                + " "
                + " ".join(f"{x:.9f}" for x in points.ravel())
            )
        (labels / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        records.append(
            dict(
                source_record=original,
                review=receipt,
                image_path=str((folder / f"{name}.png").relative_to(output)),
                label_path=str((labels / f"{name}.txt").relative_to(output)),
                annotation_path=str((annotations / f"{name}.json").relative_to(output)),
            )
        )
    report = dict(
        schema="crystalab_reviewed_segmentation_v1",
        classes=CLASSES,
        records=records,
        reviewed_images=len(records),
        unreviewed_images=len(bundle["samples"]) - len(records),
        source_protocol=bundle["source_protocol"],
        scope="Reviewed visible polygons; source material and original run/family splits retained",
    )
    report["artifact_sha256"] = {
        str(p.relative_to(output)): file_sha256(p)
        for p in output.rglob("*")
        if p.is_file()
    }
    save(output / "manifest.json", report)
    return dict(
        reviewed_images=len(records), unreviewed_images=report["unreviewed_images"]
    )


def preview_bundle(bundle_path, output, *, max_width=960):
    """Create inspection overlays without creating or changing review receipts."""
    if type(max_width) is not int or max_width < 100:
        raise ValueError("Preview width must be an integer of at least 100 pixels")
    root, output = Path(bundle_path), Path(output)
    bundle = read_json(root / "manifest.json")
    if (
        bundle["schema"] != "crystalab_annotation_bundle_v1"
        or bundle["status"] != "ready_for_review"
    ):
        raise ValueError("Incomplete/unsupported annotation bundle")
    output.mkdir(parents=True, exist_ok=False)
    previews = []
    for sample in bundle["samples"]:
        annotation = _validated_annotation(root, sample)
        image = cv2.imread(str(root / sample["image_path"]))
        h, w = image.shape[:2]
        width = min(w, max_width)
        height = round(h * width / w)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for shape in annotation["shapes"]:
            points = np.rint(
                np.asarray(shape["points"]) * [width / w, height / h]
            ).astype(np.int32)
            cv2.polylines(image, [points], True, (0, 220, 255), 2)
        text = "PROPOSED / NOT REVIEWED / " + sample["source_record"]["sample_id"]
        cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(
            image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1
        )
        name = Path(sample["image_path"]).name
        if not cv2.imwrite(str(output / name), image):
            raise OSError("Cannot save annotation preview")
        previews.append(
            dict(
                image=name,
                source_sample_id=sample["source_record"]["sample_id"],
                annotation_sha256=file_sha256(root / sample["annotation_path"]),
            )
        )
    save(
        output / "manifest.json",
        dict(
            schema="crystalab_annotation_previews_v1",
            reviewed=False,
            previews=previews,
            note="Overlays aid review; no review receipt is created",
        ),
    )
    return dict(previews=len(previews), reviewed=0)
