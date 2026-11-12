# FILE PURPOSE: Preserve public KDP figure panels as source-bound, unreviewed candidates.
"""Publication JPEG crops are kept separate from raw camera/video observations."""

import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .annotations import CLASSES
from .training_data import safe_path, save_json
from .video import file_sha256

SCHEMA = "crystalab_publication_panels_index_v1"


def scan(cache, protocol):
    cache = Path(cache)
    if protocol.get("schema") != "crystalab_publication_panels_protocol_v1" or protocol.get("split") != "train":
        raise ValueError("Publication panels must use a development-only protocol")
    for key in ("source_id", "source_url", "material", "attribution", "license", "license_url",
                "dependency_family_id", "run_id", "split_semantics", "label_scope", "clock_semantics", "derivation"):
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            raise ValueError(f"Missing publication provenance: {key}")
    figures = {}
    for asset in protocol["metadata"] + protocol["figures"]:
        if Path(asset["file"]).name != asset["file"]:
            raise ValueError("Source assets must use plain filenames")
        path = safe_path(cache, asset["file"])
        if file_sha256(path) != asset["sha256"]:
            raise ValueError("Published source bytes changed")
        if "figure_id" in asset:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (asset["height"], asset["width"], 3):
                raise ValueError("Published figure dimensions changed")
            if asset["figure_id"] in figures:
                raise ValueError("Duplicate publication figure")
            figures[asset["figure_id"]] = (asset, image)
    records, images, seen = [], {}, set()
    if not protocol["panels"]:
        raise ValueError("Declare publication panels")
    for index, panel in enumerate(protocol["panels"]):
        name = panel["panel_id"]
        if not isinstance(name, str) or not name.replace("_", "").isalnum() or name in seen:
            raise ValueError("Unsafe or duplicate publication panel ID")
        seen.add(name)
        asset, image = figures[panel["figure_id"]]
        roi = panel["crop_xyxy"]
        if (not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi) or
            not 0 <= roi[0] < roi[2] <= image.shape[1] or not 0 <= roi[1] < roi[3] <= image.shape[0]):
            raise ValueError("Invalid integer publication crop")
        elapsed = panel["published_relative_time_s"]
        if elapsed is not None and (type(elapsed) is not int or elapsed < 0):
            raise ValueError("Published relative time must be a nonnegative integer or null")
        crop = image[roi[1]:roi[3], roi[0]:roi[2]].copy()
        pixel_hash = hashlib.sha256(crop.tobytes()).hexdigest()
        if pixel_hash in images:
            raise ValueError("Duplicate pixels cannot become new publication observations")
        images[pixel_hash] = crop
        record = dict(sample_id=f"{protocol['source_id']}:{name}", source_id=protocol["source_id"],
            run_id=protocol["run_id"], frame_index=index, dependency_family_id=protocol["dependency_family_id"],
            split="train", material=protocol["material"], data_kind="publication_panel",
            panel_id=name, original_figure_file=asset["file"], original_figure_sha256=asset["sha256"],
            crop_xyxy=roi, image_path=f"images/{name}.png", image_width=crop.shape[1], image_height=crop.shape[0],
            native_BGR_sha256=pixel_hash, domain=panel["domain"], phase_context=panel["context"],
            caption_id=name, source_url=protocol["source_url"], license=protocol["license"],
            attribution=protocol["attribution"], derivation=protocol["derivation"],
            published_relative_time_s=elapsed, publication_time_origin="first published Figure 2a panel" if elapsed is not None else None,
            acquisition_timestamp=None, physical_run_id=None, growth_run_start=None, first_hour_membership=None,
            pixel_scale_m=None, segmentation_labels=None, event_labels=None, sensor_values=None, applied_commands=None,
            quality_labels=None)
        records.append(record)
    return records, images


def import_panels(cache, protocol_path, output):
    cache, protocol_path, output = Path(cache).resolve(), Path(protocol_path).resolve(), Path(output).resolve()
    protocol = json.loads(protocol_path.read_text())
    records, images = scan(cache, protocol)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "source_snapshot").mkdir()
    shutil.copyfile(__file__, output / "source_snapshot/publication_panels.py")
    shutil.copyfile(protocol_path, output / "protocol.json")
    manifest = dict(schema=SCHEMA, status="writing", protocol=protocol, protocol_sha256=file_sha256(protocol_path),
                    cache_path=str(cache), source_sha256=file_sha256(__file__), opencv_version=cv2.__version__)
    save_json(output / "manifest.json", manifest)
    try:
        for row in records:
            if not cv2.imwrite(str(output / row["image_path"]), images[row["native_BGR_sha256"]]):
                raise OSError("Cannot preserve publication pixels")
        (output / "images.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
        manifest.update(status="complete", panels=len(records), dependency_families=1, reviewed_labels=0,
                        index_sha256=file_sha256(output / "images.jsonl"))
        save_json(output / "manifest.json", manifest)
        return dict(status="complete", panels=len(records), reviewed_labels=0)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise


def audit_index(index, protocol_path):
    index = Path(index).resolve()
    manifest = json.loads((index / "manifest.json").read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "complete" or
        manifest["protocol"] != protocol or manifest["protocol_sha256"] != file_sha256(protocol_path)):
        raise ValueError("Publication index differs from external frozen protocol")
    if (file_sha256(index / "protocol.json") != manifest["protocol_sha256"] or
        file_sha256(index / "source_snapshot/publication_panels.py") != manifest["source_sha256"] or
        file_sha256(index / "images.jsonl") != manifest["index_sha256"]):
        raise ValueError("Publication index or source snapshot changed")
    expected, images = scan(Path(manifest["cache_path"]), protocol)
    stored = [json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()]
    if stored != expected or len(stored) != manifest["panels"]:
        raise ValueError("Publication records differ from source pixels or metadata")
    for row in stored:
        actual = cv2.imread(str(safe_path(index, row["image_path"])), cv2.IMREAD_COLOR)
        if actual is None or not np.array_equal(actual, images[row["native_BGR_sha256"]]):
            raise ValueError("Publication crop pixels changed")
    return manifest, stored


def prepare_bundle(index, protocol_path, output):
    index, output = Path(index).resolve(), Path(output)
    manifest, records = audit_index(index, protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "annotations", "reviews"):
        (output / name).mkdir()
    bundle = dict(schema="crystalab_annotation_bundle_v1", status="preparing", classes=CLASSES,
                  run_id=manifest["protocol"]["run_id"], source_protocol=manifest["protocol"],
                  source_index_sha256=manifest["index_sha256"], samples=[])
    save_json(output / "manifest.json", bundle)
    for row in records:
        name = row["panel_id"]
        image_path, annotation_path = row["image_path"], f"annotations/{name}.json"
        shutil.copyfile(index / image_path, output / image_path)
        annotation = dict(version="5.0.0", flags=dict(reviewed=False, uncertain=True), shapes=[],
            imagePath=f"../{image_path}", imageData=None, imageWidth=row["image_width"], imageHeight=row["image_height"],
            description="Unreviewed publication crop. Source marks retained. Resolve visible boundaries and instance roles; an empty sheet is unknown, not a negative label.")
        save_json(output / annotation_path, annotation)
        bundle["samples"].append(dict(source_record=row, image_path=image_path,
            image_sha256=file_sha256(output / image_path), annotation_path=annotation_path,
            initial_proposal_sha256=file_sha256(output / annotation_path), proposal_reason="manual_publication_panel_review_needed",
            review_path=f"reviews/{name}.json"))
    bundle["status"] = "ready_for_review"
    save_json(output / "manifest.json", bundle)
    return dict(status=bundle["status"], samples=len(records), reviewed=0)
