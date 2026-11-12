# FILE PURPOSE: Build a separate one-target silhouette experiment from synthetic and author-shape proxies.
# The pitch-class exporter still refuses unreviewed real AI proposals. No secondary/inclusion truth is inferred.
"""Explicit weak-label task, common development evaluation and preserved source-family splits."""

from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from vision.source_shapes import audit_source_shapes
from vision.training_data import (
    audit_segmentation,
    mask_polygon,
    polygon_line,
    safe_path,
    save_json,
)
from vision.video import file_sha256

SCHEMA = "crystalab_source_fit_silhouette_training_v1"
CLASSES = ["main_crystal_silhouette"]
MODES = {"synthetic_only", "source_projection_only", "mixed"}


def source_records(projection, synthetic):
    projection, synthetic = Path(projection).resolve(), Path(synthetic).resolve()
    audit_source_shapes(projection)
    audit_segmentation(synthetic)
    sources = [
        dict(
            path=str(projection),
            manifest_sha256=file_sha256(projection / "manifest.json"),
        ),
        dict(
            path=str(synthetic),
            manifest_sha256=file_sha256(synthetic / "manifest.json"),
        ),
    ]
    records = []
    for r in json.loads((projection / "records.json").read_text()):
        records.append(
            dict(
                sample_id=r["sample_id"],
                split=r["split"],
                family_key="leeds:" + r["family_id"],
                evidence_kind="real",
                label_kind="source_author_3d_fit_projected_silhouette_proxy",
                pixel_reviewed=False,
                source_image=str(projection / r["image_path"]),
                source_mask=str(projection / r["mask_path"]),
            )
        )
    manifest = json.loads((synthetic / "manifest.json").read_text())
    for r in manifest["records"]:
        if r["split"] == "test":
            continue
        seed = [
            o
            for o in r["original"]["objects"]
            if o["class_id"] == 0 and o["label_eligible"]
        ]
        if len(seed) != 1:
            raise ValueError("One eligible synthetic main crystal is required")
        original = Path(manifest["sources"][r["source_index"]]["path"])
        records.append(
            dict(
                sample_id=r["sample_id"],
                split=r["split"],
                family_key=r["family_key"],
                evidence_kind="synthetic",
                label_kind="synthetic_optical_silhouette_proxy",
                pixel_reviewed=False,
                source_image=str(synthetic / r["image_path"]),
                source_mask=str(safe_path(original, seed[0]["mask_path"])),
            )
        )
    return sources, records


def chosen(records, mode):
    if mode not in MODES:
        raise ValueError(
            "Declare synthetic-only, source-projection-only or mixed training"
        )
    return [
        r
        for r in records
        if r["split"] == "validation"
        or mode == "mixed"
        or (r["evidence_kind"] == "real") == (mode == "source_projection_only")
    ]


def prepare_silhouette(projection, synthetic, mode, output):
    sources, records = source_records(projection, synthetic)
    records = chosen(records, mode)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    for i, r in enumerate(records):
        image = cv2.imread(r["source_image"])
        mask = cv2.imread(r["source_mask"], 0)
        if image.shape[:2] != mask.shape or set(np.unique(mask)) != {0, 255}:
            raise ValueError("Original silhouette image/mask must match")
        height, width = mask.shape
        polygon, conversion = mask_polygon(mask)
        image_path = f"images/{r['split']}/{i:05d}.png"
        mask_path = f"masks/{r['split']}/{i:05d}.png"
        label_path = f"labels/{r['split']}/{i:05d}.txt"
        for relative in [image_path, mask_path, label_path]:
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r["source_image"], output / image_path)
        shutil.copyfile(r["source_mask"], output / mask_path)
        (output / label_path).write_text(polygon_line(0, polygon, width, height) + "\n")
        saved.append(
            dict(
                r,
                image_path=image_path,
                mask_path=mask_path,
                label_path=label_path,
                conversion=conversion,
                source_image_sha256=file_sha256(r["source_image"]),
                source_mask_sha256=file_sha256(r["source_mask"]),
            )
        )
    save_json(output / "records.json", saved)
    save_json(
        output / "dataset.yaml",
        dict(
            path=str(output),
            train="images/train",
            val="images/validation",
            names=CLASSES,
        ),
    )
    save_json(
        output / "manifest.json",
        dict(
            schema=SCHEMA,
            status="complete",
            task="source_fitted_main_crystal_silhouette",
            mode=mode,
            classes=CLASSES,
            sources=sources,
            projection_path=str(Path(projection).resolve()),
            synthetic_path=str(Path(synthetic).resolve()),
            scope="One fitted main-crystal silhouette only. Real targets are author-3D-fit projections, not reviewed pixel/event labels. No KDP, secondary/inclusion or physical control claim.",
            final_test_exported=False,
            actual_human_reviews_created=0,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    return audit_silhouette(output)


def audit_silhouette(output):
    root = Path(output).resolve()
    m = json.loads((root / "manifest.json").read_text())
    if (
        m["schema"] != SCHEMA
        or m["status"] != "complete"
        or m["classes"] != CLASSES
        or m["final_test_exported"] is not False
        or m["actual_human_reviews_created"] != 0
    ):
        raise ValueError("Separate source-fit silhouette dataset required")
    for path, value in m["files"].items():
        if file_sha256(safe_path(root, path)) != value:
            raise ValueError("Silhouette dataset artifact changed")
    sources, expected = source_records(m["projection_path"], m["synthetic_path"])
    expected = chosen(expected, m["mode"])
    if sources != m["sources"]:
        raise ValueError("Silhouette sources changed")
    records = json.loads((root / "records.json").read_text())
    families = {}
    pixels = {}
    if len(expected) != len(records):
        raise ValueError("Silhouette source coverage changed")
    for folder, key in [("images", "image_path"), ("masks", "mask_path")]:
        if {
            str(p.relative_to(root)) for p in (root / folder).rglob("*") if p.is_file()
        } != {r[key] for r in records}:
            raise ValueError("Unindexed silhouette pixels")
    for r, e in zip(records, expected):
        if any(r[k] != v for k, v in e.items()):
            raise ValueError("Silhouette record differs from original source")
        if families.setdefault(r["family_key"], r["split"]) != r["split"]:
            raise ValueError("Silhouette family crosses splits")
        for kind in ["image", "mask"]:
            source = Path(r["source_" + kind])
            dest = safe_path(root, r[kind + "_path"])
            if file_sha256(source) != r["source_" + kind + "_sha256"] or file_sha256(
                source
            ) != file_sha256(dest):
                raise ValueError("Silhouette pixels differ from source")
        digest = r["source_image_sha256"]
        if pixels.setdefault(digest, r["split"]) != r["split"]:
            raise ValueError("Duplicate silhouette pixels cross splits")
        mask = cv2.imread(str(root / r["mask_path"]), 0)
        polygon, conversion = mask_polygon(mask)
        height, width = mask.shape
        if (
            r["conversion"] != conversion
            or (root / r["label_path"]).read_text()
            != polygon_line(0, polygon, width, height) + "\n"
        ):
            raise ValueError("Silhouette training polygon differs from source target")
    if json.loads((root / "dataset.yaml").read_text()) != dict(
        path=str(root), train="images/train", val="images/validation", names=CLASSES
    ):
        raise ValueError("Silhouette class or dataset path changed")
    return dict(
        status="pass",
        images=len(records),
        families=len(families),
        images_by_split=dict(Counter(r["split"] for r in records)),
        training_by_evidence=dict(
            Counter(r["evidence_kind"] for r in records if r["split"] == "train")
        ),
        validation_by_evidence=dict(
            Counter(r["evidence_kind"] for r in records if r["split"] == "validation")
        ),
        task=m["task"],
        pixel_reviewed=False,
        manifest_sha256=file_sha256(root / "manifest.json"),
    )
