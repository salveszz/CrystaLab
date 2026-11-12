# FILE PURPOSE: Derive explicitly weak silhouette targets from published manual 3D fits.
# No local AI proposals or human review receipts enter this separate source-fit task.
"""Leeds author-shape projection with preserved source bytes, splits and assumptions."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from simulation.crystal import Crystal
from vision.data import load_index_image
from vision.training_data import safe_path, save_json
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "crystalab_leeds_shape_projection_v1"


def fitted_vertices(fit):
    """Interpret only explicit, complete orthorhombic plane sets in the source data."""
    if (
        not isinstance(fit, dict)
        or fit.get("point_group_symbol") != "1"
        or fit.get("rotation_mode") != "axisangle"
    ):
        raise ValueError("Explicit full plane set and axis-angle pose required")
    lattice = np.asarray(fit["lattice_unit_cell"], dtype=float)
    angles = np.asarray(fit["lattice_angles"], dtype=float)
    hkl = np.asarray(fit["miller_indices"], dtype=float)
    distances = np.asarray(fit["distances"], dtype=float)
    origin, rotation = [np.asarray(fit[k], dtype=float) for k in ("origin", "rotation")]
    scale = fit["scale"]
    if (
        lattice.shape != (3,)
        or np.any(lattice <= 0)
        or angles.shape != (3,)
        or not np.allclose(angles, np.pi / 2, rtol=0, atol=1e-10)
        or hkl.ndim != 2
        or hkl.shape[1] != 3
        or np.any(hkl != np.rint(hkl))
        or distances.shape != (len(hkl),)
        or np.any(distances <= 0)
        or origin.shape != (3,)
        or rotation.shape != (3,)
        or isinstance(scale, bool)
        or not np.isfinite(scale)
        or scale <= 0
        or not all(
            np.isfinite(v).all()
            for v in (lattice, angles, hkl, distances, origin, rotation)
        )
    ):
        raise ValueError("Invalid source crystal lattice, planes, scale or pose")
    normals = hkl / lattice
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("A source plane must have a nonzero Miller normal")
    solid = Crystal(normals / lengths[:, None], distances)
    vertices = (
        solid.vertices @ Rotation.from_rotvec(rotation).as_matrix().T * scale + origin
    )
    return vertices


def camera_fields(payload):
    """Read the four admitted top-level source YAML scalars, without a general YAML loader."""
    fields = {}
    for line in payload.decode().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {
            "camera_type",
            "camera_distance",
            "camera_fov",
            "focal_length",
        }:
            if key in fields:
                raise ValueError("Duplicate source camera field")
            fields[key] = (
                value.strip() if key == "camera_type" else json.loads(value.strip())
            )
    if set(fields) != {"camera_type", "camera_distance", "camera_fov", "focal_length"}:
        raise ValueError("Source camera scale is incomplete")
    return fields


def projected_polygon(vertices, scene, size):
    if (
        scene["camera_type"] != "perspective"
        or scene.get("focal_length") is not None
        or not 0 < scene["camera_fov"] < 90
        or type(size) is not int
        or size < 64
    ):
        raise ValueError(
            "Declared square perspective-camera scale and output size required"
        )
    depth = float(np.mean(vertices[:, 2]))
    distance = float(scene["camera_distance"]) - depth
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Source crystal must be in front of its camera")
    zoom = 1 / (distance * np.tan(np.deg2rad(scene["camera_fov"]) / 2))
    coordinates = (0.5 + vertices[:, :2] * [zoom / 2, -zoom / 2]) * size
    # The supervised object is a raster mask: quantize vertices before taking its hull.
    # This removes sensitivity to almost collinear floating-point corner lists.
    polygon = cv2.convexHull(np.rint(coordinates).astype(np.int32))[:, 0, :].astype(
        float
    )
    if np.any(polygon < 0) or np.any(polygon >= size):
        raise ValueError(
            "Source silhouette leaves the declared crop; preserve for inspection"
        )
    return polygon, dict(depth_source_units=depth, orthographic_zoom=zoom)


def selected_sources(index, policy):
    index = Path(index).resolve()
    if file_sha256(index / "manifest.json") != policy["source_index_manifest_sha256"]:
        raise ValueError("Source-fit index manifest differs from frozen policy")
    manifest = json.loads((index / "manifest.json").read_text())
    if (
        manifest["schema"] != "crystalab_real_vision_index_v1"
        or manifest["status"] != "complete"
        or manifest["protocol"]["source_id"] != policy["source_id"]
        or file_sha256(index / "images.jsonl") != manifest["index_sha256"]
        or file_sha256(manifest["archive_path"]) != policy["source_archive_sha256"]
    ):
        raise ValueError("Source-fit index/archive identity changed")
    if (
        policy["schema"] != "crystalab_leeds_shape_projection_policy_v1"
        or policy["include_splits"] != ["train", "validation"]
        or policy["reserved_test_excluded"] is not True
        or policy["pixel_reviewed"] is not False
        or policy["label_kind"] != "source_author_3d_fit_projected_silhouette_proxy"
    ):
        raise ValueError(
            "Declare the separate weak source-fit task and preserve final test"
        )
    records = [json.loads(v) for v in (index / "images.jsonl").read_text().splitlines()]
    for record in records:
        settings = manifest["protocol"]["experiments"][record["run_id"]]
        if (
            record["split"] != settings["split"]
            or record["family_id"] != settings["family_id"]
        ):
            raise ValueError(
                "Source-fit records must preserve the original run/condition split"
            )
        member = record["labels"]["manual_crystal_fit"]
        if (
            member is not None
            and member
            != f"experiments/manual/{settings['annotation_id']}/{record['frame_index']:04d}.json"
        ):
            raise ValueError("Source fit must belong to the original frame")
    selected = [
        r
        for r in records
        if r["labels"]["manual_crystal_fit"] and r["split"] in policy["include_splits"]
    ]
    return manifest, selected


def source_target(archive, record, protocol, policy):
    member = record["labels"]["manual_crystal_fit"]
    payload = archive.read(member)
    if hashlib.sha256(payload).hexdigest() != record["labels"]["manual_fit_sha256"]:
        raise ValueError("Published source fit changed")
    fit = json.loads(payload)
    alias = protocol["experiments"][record["run_id"]]["annotation_id"]
    scene_member = f"experiments/scene0/scene0_{alias}.yml"
    scene_payload = archive.read(scene_member)
    scene = camera_fields(scene_payload)
    vertices = fitted_vertices(fit)
    size = policy["output_size"]
    polygon, geometry = projected_polygon(vertices, scene, size)
    original = load_index_image(archive, record)
    height, width = original.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    image = cv2.resize(
        original[top : top + side, left : left + side],
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    return (
        image,
        mask,
        dict(
            original=record,
            vertices_source_units=vertices.tolist(),
            polygon=polygon.tolist(),
            scene_member=scene_member,
            scene_sha256=hashlib.sha256(scene_payload).hexdigest(),
            scene=scene,
            geometry=geometry,
            transform=dict(
                crop_xywh=[left, top, side, side],
                output_size=size,
                interpolation="INTER_AREA",
            ),
            mask_area_px2=int(np.count_nonzero(mask)),
            label_kind=policy["label_kind"],
            pixel_reviewed=False,
        ),
    )


def export_source_shapes(index, policy_path, output):
    policy = json.loads(Path(policy_path).read_text())
    manifest, selected = selected_sources(index, policy)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "policy.json", policy)
    records = []
    with ZipFile(manifest["archive_path"]) as archive:
        for source in selected:
            image, mask, record = source_target(
                archive, source, manifest["protocol"], policy
            )
            stem = f"{source['run_id']}_{source['frame_index']:04d}"
            record.update(
                sample_id=source["sample_id"],
                family_id=source["family_id"],
                split=source["split"],
                image_path=f"images/{source['split']}/{stem}.png",
                mask_path=f"masks/{source['split']}/{stem}.png",
            )
            for relative, pixels in [
                (record["image_path"], image),
                (record["mask_path"], mask),
            ]:
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), pixels):
                    raise OSError("Cannot retain projected source target")
            records.append(record)
    sources = {}
    for path in [
        Path(__file__),
        ROOT / "src/simulation/crystal.py",
        ROOT / "scripts/project_source_shapes.py",
    ]:
        relative = str(path.relative_to(ROOT))
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[relative] = file_sha256(destination)
    save_json(output / "records.json", records)
    save_json(
        output / "manifest.json",
        dict(
            schema=SCHEMA,
            status="complete",
            source_index=str(Path(index).resolve()),
            source_index_manifest_sha256=file_sha256(Path(index) / "manifest.json"),
            source_sha256=sources,
            policy_sha256=file_sha256(output / "policy.json"),
            records=len(records),
            counts_by_split=dict(Counter(r["split"] for r in records)),
            scope=policy["scope"],
            actual_human_reviews_created=0,
            final_test_exported=False,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    return audit_source_shapes(output)


def audit_source_shapes(output):
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["schema"] != SCHEMA
        or manifest["status"] != "complete"
        or manifest["actual_human_reviews_created"] != 0
        or manifest["final_test_exported"] is not False
    ):
        raise ValueError("Completed weak source-shape projection required")
    for path, value in manifest["files"].items():
        if file_sha256(safe_path(root, path)) != value:
            raise ValueError("Projected source artifact changed")
    policy = json.loads((root / "policy.json").read_text())
    source, selected = selected_sources(manifest["source_index"], policy)
    if (
        file_sha256(root / "policy.json") != manifest["policy_sha256"]
        or file_sha256(Path(manifest["source_index"]) / "manifest.json")
        != manifest["source_index_manifest_sha256"]
    ):
        raise ValueError("Projection protocol/source identity changed")
    records = json.loads((root / "records.json").read_text())
    if len(records) != len(selected) or len(records) != manifest["records"]:
        raise ValueError("Source-fit coverage changed")
    with ZipFile(source["archive_path"]) as archive:
        for record, original in zip(records, selected):
            image, mask, expected = source_target(
                archive, original, source["protocol"], policy
            )
            if any(record[k] != v for k, v in expected.items()):
                raise ValueError("Projection differs from original author fit")
            for relative, pixels, mode in [
                (record["image_path"], image, cv2.IMREAD_COLOR),
                (record["mask_path"], mask, cv2.IMREAD_GRAYSCALE),
            ]:
                if not np.array_equal(
                    cv2.imread(str(safe_path(root, relative)), mode), pixels
                ):
                    raise ValueError(
                        "Projected image/mask differs from source reconstruction"
                    )
            if (
                record["sample_id"] != original["sample_id"]
                or record["split"] != original["split"]
                or record["family_id"] != original["family_id"]
            ):
                raise ValueError("Source grouping changed")
    return dict(
        status="pass",
        records=len(records),
        counts_by_split=dict(Counter(r["split"] for r in records)),
        label_kind=policy["label_kind"],
        pixel_reviewed=False,
        final_test_exported=False,
        manifest_sha256=file_sha256(root / "manifest.json"),
    )
