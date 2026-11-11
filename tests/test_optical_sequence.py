# FILE PURPOSE: Verify optical replay export clocks and reserved-scene exclusion.
# Tiny artificial RGB frames isolate encoding/mapping from the independently
# audited Blender dataset; no physical timing or model accuracy is implied.

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.optical_sequence import export_optical_sequence
from vision.training_data import save_json
from vision.video import VideoReader, file_sha256


class OpticalSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.rows = []
        for i in range(3):
            name = f"frame_{i}.png"
            cv2.imwrite(
                str(self.source / name), np.full((32, 32, 3), 50 + i * 40, np.uint8)
            )
            self.rows.append(
                dict(
                    scene_id="fixture",
                    family_id="family",
                    split="train",
                    frame_index=i,
                    sample_id=f"fixture:{i}",
                    image_path=name,
                    synthetic_time=i,
                )
            )
        (self.source / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        save_json(
            self.source / "manifest.json",
            dict(
                artifact_sha256={
                    r["image_path"]: file_sha256(self.source / r["image_path"])
                    for r in self.rows
                }
            ),
        )

    def test_encoded_sequence_preserves_source_steps_without_physical_clock(self):
        output = self.root / "export"
        # Pixel/mask source auditing is separately exercised on actual optical data.
        with patch(
            "vision.optical_sequence.audit_optical", return_value=dict(status="pass")
        ):
            manifest = export_optical_sequence(self.source, "fixture", output)
        self.assertEqual(manifest["frames"], 3)
        with VideoReader(output / "sequence.avi") as reader:
            frames = list(reader)
        np.testing.assert_allclose([f.video_time_s for f in frames], [0, 0.25, 0.5])
        mapping = [
            json.loads(line)
            for line in (output / "frame_map.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [r["source_sample_id"] for r in mapping],
            ["fixture:0", "fixture:1", "fixture:2"],
        )
        self.assertEqual([r["synthetic_time"] for r in mapping], [0, 1, 2])
        self.assertTrue(all(r["acquisition_time_s"] is None for r in mapping))
        self.assertTrue(
            all(
                r["time_semantics"] == "illustrative_geometry_step_not_physical_time"
                for r in mapping
            )
        )
        info = json.loads((output / "source_info.json").read_text())
        self.assertEqual(
            info["frame_map_sha256"], file_sha256(output / "frame_map.jsonl")
        )

    def test_reserved_scene_rejected_before_output_creation(self):
        for row in self.rows:
            row["split"] = "test"
        (self.source / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        with patch(
            "vision.optical_sequence.audit_optical", return_value=dict(status="pass")
        ):
            with self.assertRaisesRegex(ValueError, "test excluded"):
                export_optical_sequence(self.source, "fixture", self.root / "blocked")
        self.assertFalse((self.root / "blocked").exists())
