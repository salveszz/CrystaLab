# FILE PURPOSE: Verify real-image indexing, frozen run/family splits and honest labels.
# Small JPEG archives exercise source identity, filename variants, manual-fit
# sidecars, reconstruction audits and cross-split duplicate rejection offline.

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.data import audit_image_index, import_leeds, load_index_image
from vision.sequence import export_pilot
from vision.video import VideoReader


class RealVisionImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "real.zip"
        self.design = dict(
            schema="crystalab_leeds_import_v1",
            source_id="SOURCE",
            material="alpha_LGA",
            license="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            nominal_frame_interval_s=300,
            clock_provenance="test only nominal cadence",
            source_url="https://example.invalid/source",
            citation="Source fixture, not real measured data",
            experiments={
                run: dict(
                    annotation_id=run, family_id=run, split=split, expected_frames=2
                )
                for run, split in (("a", "train"), ("b", "validation"), ("c", "test"))
            },
        )
        self.make_archive()

    def tearDown(self):
        self.temporary.cleanup()

    def make_archive(self, *, duplicate=False, unsafe=False):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for run_i, run in enumerate(self.design["experiments"]):
                for index in range(2):
                    pixels = np.full(
                        (40, 60, 3),
                        80 if duplicate else 40 + run_i * 30 + index * 5,
                        np.uint8,
                    )
                    ok, jpg = cv2.imencode(".jpg", pixels)
                    self.assertTrue(ok)
                    prefix = "LGA_alpha_" if run == "b" else ""
                    archive.writestr(
                        f"experiments/img/{run}/{prefix}{index:04d}.jpg", jpg.tobytes()
                    )
                archive.writestr(
                    f"experiments/manual/{run}/0000.json",
                    json.dumps(
                        dict(
                            distances=[1, 2],
                            miller_indices=[[1, 0, 0], [-1, 0, 0]],
                            rotation=[0, 0, 0],
                            origin=[0, 0, 0],
                            scale=1,
                        )
                    ),
                )
            if unsafe:
                archive.writestr("../escape", b"no")
        self.design["archive_sha256"] = hashlib.sha256(
            self.archive.read_bytes()
        ).hexdigest()

    def test_real_images_remain_distinct_from_manual_fits_and_missing_mask_labels(self):
        output = self.root / "index"
        manifest = import_leeds(self.archive, self.design, output)
        self.assertEqual(manifest["summary"]["images"], 6)
        self.assertEqual(manifest["summary"]["images_with_manual_shape_fit"], 3)
        self.assertEqual(audit_image_index(output)["status"], "pass")
        rows = [
            json.loads(v) for v in (output / "images.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(r["acquisition_time_s"] is None for r in rows))
        self.assertTrue(all(not r["segmentation_training_ready"] for r in rows))
        self.assertTrue(all(r["labels"]["segmentation_mask"] is None for r in rows))
        self.assertEqual(rows[1]["nominal_elapsed_s"], 300)
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(load_index_image(archive, rows[0]).shape, (40, 60, 3))
        with self.assertRaises(FileExistsError):
            import_leeds(self.archive, self.design, output)

    def test_source_hash_and_corrupted_index_fail(self):
        output = self.root / "index"
        import_leeds(self.archive, self.design, output)
        with (output / "images.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            audit_image_index(output)
        self.design["archive_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "archive hash"):
            import_leeds(self.archive, self.design, self.root / "bad")
        self.assertEqual(
            json.loads((self.root / "bad/manifest.json").read_text())["status"],
            "failed",
        )

    def test_development_video_preserves_frame_map_and_rejects_holdout_tuning(self):
        index = self.root / "index"
        import_leeds(self.archive, self.design, index)
        output = self.root / "pilot"
        report = export_pilot(index, "a", output, display_fps=4, max_width=30)
        self.assertEqual(report["status"], "complete")
        mapping = [
            json.loads(v) for v in (output / "frame_map.jsonl").read_text().splitlines()
        ]
        self.assertEqual(mapping[1]["display_time_s"], 0.25)
        self.assertEqual(mapping[1]["nominal_elapsed_s"], 300)
        self.assertIsNone(mapping[1]["acquisition_time_s"])
        with VideoReader(output / "sequence.avi") as reader:
            self.assertEqual(len(list(reader)), 2)
        with self.assertRaisesRegex(ValueError, "training run"):
            export_pilot(index, "c", self.root / "test_pilot")

    def test_related_families_and_duplicate_images_cannot_cross_splits(self):
        self.design["experiments"]["b"]["family_id"] = "a"
        with self.assertRaisesRegex(ValueError, "family"):
            import_leeds(self.archive, self.design, self.root / "related")
        self.design["experiments"]["b"]["family_id"] = "b"
        self.make_archive(duplicate=True)
        with self.assertRaisesRegex(ValueError, "Identical real images"):
            import_leeds(self.archive, self.design, self.root / "duplicates")

    def test_missing_frames_and_unsafe_members_are_rejected(self):
        self.design["experiments"]["a"]["expected_frames"] = 3
        with self.assertRaisesRegex(ValueError, "source frames"):
            import_leeds(self.archive, self.design, self.root / "missing")
        self.design["experiments"]["a"]["expected_frames"] = 2
        self.make_archive(unsafe=True)
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            import_leeds(self.archive, self.design, self.root / "unsafe")


if __name__ == "__main__":
    unittest.main()
