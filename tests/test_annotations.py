# FILE PURPOSE: Verify that proposals cannot become real training truth without explicit review.
# Fixture polygons exercise source/split preservation, reviewed negatives,
# receipt invalidation after edits and rejection of ambiguous/invalid geometry.

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.annotations import (
    prepare_bundle,
    record_review,
    export_reviewed,
    preview_bundle,
)
from vision.baseline import BaselineConfig
from vision.data import SCHEMA
from vision.video import file_sha256


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = self.root / "index"
        self.index.mkdir()
        pixels = np.full((80, 100, 3), 220, np.uint8)
        pixels[25:55, 30:60] = 30
        ok, data = cv2.imencode(".jpg", pixels)
        self.assertTrue(ok)
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("image.jpg", data.tobytes())
        import hashlib

        row = dict(
            sample_id="fixture:run:0",
            run_id="run",
            family_id="family",
            split="train",
            frame_index=0,
            evidence_kind="fixture",
            archive_member="image.jpg",
            image_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            image_width=100,
            image_height=80,
        )
        (self.index / "images.jsonl").write_text(json.dumps(row) + "\n")
        m = dict(
            schema=SCHEMA,
            status="complete",
            archive_path=str(archive),
            index_sha256=file_sha256(self.index / "images.jsonl"),
            protocol=dict(
                archive_sha256=file_sha256(archive),
                material="fixture",
                license="fixture only",
            ),
        )
        (self.index / "manifest.json").write_text(json.dumps(m))
        self.bundle = self.root / "bundle"
        prepare_bundle(
            self.index,
            "run",
            self.bundle,
            frame_indices=[0],
            baseline=BaselineConfig((45, 40), threshold=110, morphology_size=1),
            proposal_width=100,
        )
        self.annotation = self.bundle / "annotations/run_0000.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_proposals_are_not_exported_without_explicit_exact_image_review(self):
        with self.assertRaisesRegex(ValueError, "No explicitly reviewed"):
            export_reviewed(self.bundle, self.root / "early")
        with self.assertRaisesRegex(ValueError, "acknowledgment"):
            record_review(
                self.bundle, self.annotation.name, reviewer="fixture reviewer"
            )
        receipt = record_review(
            self.bundle,
            self.annotation.name,
            reviewer="fixture reviewer",
            acknowledge_image_reviewed=True,
        )
        self.assertEqual(receipt["annotation_sha256"], file_sha256(self.annotation))
        result = export_reviewed(self.bundle, self.root / "accepted")
        self.assertEqual(result["reviewed_images"], 1)
        m = json.loads((self.root / "accepted/manifest.json").read_text())
        self.assertEqual(m["records"][0]["source_record"]["family_id"], "family")
        self.assertEqual(m["records"][0]["source_record"]["split"], "train")
        self.assertTrue(
            (self.root / "accepted/labels/train/run_0000.txt")
            .read_text()
            .startswith("0 ")
        )
        with self.assertRaises(FileExistsError):
            export_reviewed(self.bundle, self.root / "accepted")

    def test_an_edit_after_review_invalidates_the_receipt(self):
        record_review(
            self.bundle,
            self.annotation.name,
            reviewer="fixture reviewer",
            acknowledge_image_reviewed=True,
        )
        d = json.loads(self.annotation.read_text())
        d["shapes"][0]["points"][0][0] += 1
        self.annotation.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "receipt no longer matches"):
            export_reviewed(self.bundle, self.root / "edited")

    def test_previews_do_not_create_review_receipts(self):
        result = preview_bundle(self.bundle, self.root / "preview", max_width=100)
        self.assertEqual(result, dict(previews=1, reviewed=0))
        self.assertEqual(list((self.bundle / "reviews").iterdir()), [])
        image = cv2.imread(str(self.root / "preview/run_0000.png"))
        self.assertEqual(image.shape[:2], (80, 100))
        with self.assertRaisesRegex(ValueError, "No explicitly reviewed"):
            export_reviewed(self.bundle, self.root / "still_unreviewed")

    def test_empty_proposals_require_reviewed_negative_acknowledgment(self):
        d = json.loads(self.annotation.read_text())
        d["shapes"] = []
        self.annotation.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "reviewed-negative"):
            record_review(
                self.bundle,
                self.annotation.name,
                reviewer="fixture reviewer",
                acknowledge_image_reviewed=True,
            )
        record_review(
            self.bundle,
            self.annotation.name,
            reviewer="fixture reviewer",
            acknowledge_image_reviewed=True,
            empty_reviewed=True,
        )
        export_reviewed(self.bundle, self.root / "negative")
        self.assertEqual(
            (self.root / "negative/labels/train/run_0000.txt").read_text(), ""
        )

    def test_uncertain_out_of_bounds_and_self_crossing_polygons_fail(self):
        original = json.loads(self.annotation.read_text())
        for points, uncertain in [
            ([[10, 10], [30, 10], [30, 30]], True),
            ([[-1, 10], [30, 10], [30, 30]], False),
            ([[10, 10], [30, 30], [30, 10], [10, 30]], False),
        ]:
            d = json.loads(json.dumps(original))
            d["shapes"][0]["points"] = points
            d["shapes"][0]["flags"]["uncertain"] = uncertain
            self.annotation.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                record_review(
                    self.bundle,
                    self.annotation.name,
                    reviewer="fixture reviewer",
                    acknowledge_image_reviewed=True,
                )

    def test_changed_original_pixels_and_holdout_proposals_are_rejected(self):
        image = self.bundle / "images/run_0000.png"
        image.write_bytes(image.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "image changed"):
            record_review(
                self.bundle,
                self.annotation.name,
                reviewer="fixture reviewer",
                acknowledge_image_reviewed=True,
            )
        row = json.loads((self.index / "images.jsonl").read_text())
        row["split"] = "test"
        (self.index / "images.jsonl").write_text(json.dumps(row) + "\n")
        m = json.loads((self.index / "manifest.json").read_text())
        m["index_sha256"] = file_sha256(self.index / "images.jsonl")
        (self.index / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "training runs"):
            prepare_bundle(self.index, "run", self.root / "holdout", frame_indices=[0])


if __name__ == "__main__":
    unittest.main()
