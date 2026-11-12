# FILE PURPOSE: Check analytic source geometry, projection coordinates and held-out source integrity.
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import cv2
import numpy as np

from vision.source_shapes import (
    fitted_vertices,
    projected_polygon,
    camera_fields,
    export_source_shapes,
    audit_source_shapes,
)
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def cube_fit():
    return dict(
        point_group_symbol="1",
        rotation_mode="axisangle",
        lattice_unit_cell=[1, 1, 1],
        lattice_angles=[np.pi / 2] * 3,
        miller_indices=[
            *np.eye(3, dtype=int).tolist(),
            *(-np.eye(3, dtype=int)).tolist(),
        ],
        distances=[1] * 6,
        origin=[0, 0, 0],
        rotation=[0, 0, 0],
        scale=1,
    )


def fixture_source(root):
    index = root / "index"
    index.mkdir()
    archive = root / "source.zip"
    records = []
    with ZipFile(archive, "w") as z:
        for i, (run, split) in enumerate(
            [("a", "train"), ("b", "validation"), ("c", "test")]
        ):
            fit = json.dumps(cube_fit()).encode()
            im = np.full((48, 64, 3), 150 + i, np.uint8)
            ok, encoded = cv2.imencode(".jpg", im)
            assert ok
            payload = (
                encoded.tobytes()
                if split != "test"
                else b"reserved pixels must not be decoded"
            )
            member = f"experiments/img/{run}/0000.jpg"
            label = f"experiments/manual/{run}/0000.json"
            z.writestr(member, payload)
            z.writestr(label, fit)
            z.writestr(
                f"experiments/scene0/scene0_{run}.yml",
                "camera_type: perspective\ncamera_fov: 60.0\ncamera_distance: 10.0\nfocal_length: null\n",
            )
            records.append(
                dict(
                    source_id="fixture",
                    sample_id=run,
                    run_id=run,
                    frame_index=0,
                    split=split,
                    family_id=run,
                    archive_member=member,
                    image_sha256=hashlib.sha256(payload).hexdigest(),
                    image_height=48,
                    image_width=64,
                    labels=dict(
                        manual_crystal_fit=label,
                        manual_fit_sha256=hashlib.sha256(fit).hexdigest(),
                    ),
                )
            )
    (index / "images.jsonl").write_text("".join(json.dumps(v) + "\n" for v in records))
    protocol = dict(
        source_id="fixture",
        experiments={
            r["run_id"]: dict(
                annotation_id=r["run_id"], family_id=r["family_id"], split=r["split"]
            )
            for r in records
        },
    )
    (index / "manifest.json").write_text(
        json.dumps(
            dict(
                schema="crystalab_real_vision_index_v1",
                status="complete",
                protocol=protocol,
                archive_path=str(archive),
                index_sha256=file_sha256(index / "images.jsonl"),
            )
        )
    )
    policy = json.loads((ROOT / "configs/leeds_shape_projection.json").read_text())
    policy.update(
        source_id="fixture",
        source_index_manifest_sha256=file_sha256(index / "manifest.json"),
        source_archive_sha256=file_sha256(archive),
        output_size=128,
    )
    path = root / "policy.json"
    path.write_text(json.dumps(policy))
    return index, path


class SourceShapeTests(unittest.TestCase):
    def test_cube_pose_and_pixel_projection_have_closed_form_coordinates(self):
        fit = cube_fit()
        fit.update(rotation=[0, 0, np.pi / 2], origin=[0.5, -0.25, 0], scale=2)
        vertices = fitted_vertices(fit)
        np.testing.assert_allclose(vertices.min(axis=0), [-1.5, -2.25, -2], atol=1e-12)
        np.testing.assert_allclose(vertices.max(axis=0), [2.5, 1.75, 2], atol=1e-12)
        scene = dict(
            camera_type="perspective",
            focal_length=None,
            camera_distance=10,
            camera_fov=60,
        )
        polygon, info = projected_polygon(fitted_vertices(cube_fit()), scene, 128)
        delta = 128 / (20 * np.tan(np.pi / 6))
        np.testing.assert_allclose(
            polygon.min(axis=0), [round(64 - delta)] * 2, atol=1e-10
        )
        np.testing.assert_allclose(
            polygon.max(axis=0), [round(64 + delta)] * 2, atol=1e-10
        )
        self.assertEqual(info["depth_source_units"], 0)

    def test_unsupported_lattice_pose_camera_and_duplicate_scalar_are_refused(self):
        for change in [
            dict(point_group_symbol="222"),
            dict(rotation_mode="quaternion"),
            dict(lattice_angles=[1, 1, 1]),
            dict(scale=0),
            dict(distances=[float("nan")] * 6),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                fitted_vertices(dict(cube_fit(), **change))
        with self.assertRaises(ValueError):
            camera_fields(b"camera_distance: 10\ncamera_distance: 20\n")
        with self.assertRaises(ValueError):
            projected_polygon(
                fitted_vertices(cube_fit()),
                dict(
                    camera_type="thinlens",
                    focal_length=None,
                    camera_fov=60,
                    camera_distance=10,
                ),
                128,
            )

    def test_source_fit_export_never_decodes_reserved_test_or_invents_pixel_review(
        self,
    ):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            index, policy = fixture_source(root)
            out = root / "derived"
            result = export_source_shapes(index, policy, out)
            self.assertEqual(result["counts_by_split"], dict(train=1, validation=1))
            self.assertFalse(result["pixel_reviewed"])
            self.assertFalse(result["final_test_exported"])
            records = json.loads((out / "records.json").read_text())
            self.assertEqual({r["sample_id"] for r in records}, {"a", "b"})
            path = out / records[0]["mask_path"]
            cv2.imwrite(str(path), np.zeros((128, 128), np.uint8))
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["files"][records[0]["mask_path"]] = file_sha256(path)
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source reconstruction"):
                audit_source_shapes(out)

    def test_resealed_index_cannot_move_a_source_run_across_its_original_split(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            index, path = fixture_source(root)
            records = [
                json.loads(v) for v in (index / "images.jsonl").read_text().splitlines()
            ]
            records[-1]["split"] = "train"
            (index / "images.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            m = json.loads((index / "manifest.json").read_text())
            m["index_sha256"] = file_sha256(index / "images.jsonl")
            (index / "manifest.json").write_text(json.dumps(m))
            policy = json.loads(path.read_text())
            policy["source_index_manifest_sha256"] = file_sha256(
                index / "manifest.json"
            )
            path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, "original run/condition split"):
                export_source_shapes(index, path, root / "invalid")
