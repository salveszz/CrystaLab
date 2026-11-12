# FILE PURPOSE: Independently reconstruct projected real targets from author planes and pose.
"""Triple-plane intersections and Rodrigues rotation; no projection/export implementation called."""

from itertools import combinations
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from vision.video import file_sha256


def audit_projected_shapes(output):
    root = Path(output).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    policy = json.loads((root / "policy.json").read_text())
    index = Path(manifest["source_index"])
    source = json.loads((index / "manifest.json").read_text())
    assert (
        file_sha256(index / "manifest.json") == policy["source_index_manifest_sha256"]
    )
    assert file_sha256(source["archive_path"]) == policy["source_archive_sha256"]
    assert file_sha256(index / "images.jsonl") == source["index_sha256"]
    original = [
        json.loads(v) for v in (index / "images.jsonl").read_text().splitlines()
    ]
    selected = [
        r
        for r in original
        if r["labels"]["manual_crystal_fit"] and r["split"] in ["train", "validation"]
    ]
    records = json.loads((root / "records.json").read_text())
    assert len(records) == len(selected)
    maximum_vertex_error = maximum_projection_error = 0.0
    with ZipFile(source["archive_path"]) as archive:
        for row, record in zip(selected, records):
            assert (
                row == record["original"]
                and row["split"] != "test"
                and record["pixel_reviewed"] is False
            )
            data = archive.read(row["labels"]["manual_crystal_fit"])
            assert (
                hashlib.sha256(data).hexdigest() == row["labels"]["manual_fit_sha256"]
            )
            fit = json.loads(data)
            hkl = np.array(fit["miller_indices"])
            normals = hkl / fit["lattice_unit_cell"]
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            offsets = np.array(fit["distances"])
            triples = np.array(list(combinations(range(len(normals)), 3)))
            matrices = normals[triples]
            valid = np.abs(np.linalg.det(matrices)) > 1e-10
            points = np.linalg.solve(
                matrices[valid], offsets[triples[valid]][..., None]
            )[..., 0]
            points = points[np.all(points @ normals.T <= offsets + 1e-9, axis=1)]
            # Different triples can describe the same geometric vertex.
            points = np.unique(np.round(points, 10), axis=0)
            vector = np.array(fit["rotation"])
            angle = np.linalg.norm(vector)
            rotation = np.eye(3)
            if angle:
                x, y, z = vector / angle
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotation += np.sin(angle) * cross + (1 - np.cos(angle)) * (
                    cross @ cross
                )
            world = points @ rotation.T * fit["scale"] + fit["origin"]
            saved = np.array(record["vertices_source_units"])
            errors = cdist(world, saved)
            error = max(errors.min(axis=0).max(), errors.min(axis=1).max())
            assert error < 1e-7, (
                row["sample_id"],
                "vertices",
                error,
                len(world),
                len(saved),
            )
            maximum_vertex_error = max(maximum_vertex_error, float(error))
            scene = record["scene"]
            raw = archive.read(record["scene_member"])
            assert hashlib.sha256(raw).hexdigest() == record["scene_sha256"]
            for key, value in scene.items():
                line = next(
                    line
                    for line in raw.decode().splitlines()
                    if line.startswith(key + ":")
                )
                parsed = line.split(":", 1)[1].strip()
                assert (parsed if key == "camera_type" else json.loads(parsed)) == value
            factor = policy["output_size"] / (
                2
                * (scene["camera_distance"] - world[:, 2].mean())
                * np.tan(scene["camera_fov"] * np.pi / 360)
            )
            xy = policy["output_size"] / 2 + world[:, :2] * [factor, -factor]
            raster = np.unique(np.rint(xy).astype(np.int32), axis=0)
            polygon = raster[ConvexHull(raster).vertices]
            saved = np.array(record["polygon"])
            errors = cdist(polygon, saved)
            error = max(errors.min(axis=0).max(), errors.min(axis=1).max())
            assert error < 1e-7, (
                row["sample_id"],
                "projected_polygon",
                error,
                len(polygon),
                len(saved),
            )
            maximum_projection_error = max(maximum_projection_error, float(error))
            mask = np.zeros((policy["output_size"],) * 2, np.uint8)
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
            assert np.array_equal(mask, cv2.imread(str(root / record["mask_path"]), 0))
    return dict(
        status="pass",
        source_fits_reconstructed=len(records),
        maximum_vertex_error_source_units=maximum_vertex_error,
        maximum_projection_error_px=maximum_projection_error,
        pixel_reviewed=False,
        final_test_scored=False,
        scope="Independent geometric target reconstruction; not agreement with independently reviewed image boundaries",
    )
