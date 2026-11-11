# FILE PURPOSE: Verify real-review provenance, leakage barriers and polygon-conversion limits.
# All review receipts here are artificial test fixtures in temporary directories.
# They never review or approve actual project images.

import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from models.segmentation import training_options
from vision.training_data import (
    CLASSES,
    audit_segmentation,
    mask_polygon,
    prepare_segmentation,
    save_json,
)
from vision.video import file_sha256


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, name, split, family, value, *, empty=False):
        root = self.root / name
        root.mkdir()
        image = root / "image.png"
        cv2.imwrite(str(image), np.full((16, 16, 3), value, np.uint8))
        annotation = dict(
            imageWidth=16,
            imageHeight=16,
            shapes=[]
            if empty
            else [
                dict(
                    label="seed",
                    shape_type="polygon",
                    flags={},
                    points=[[2, 2], [12, 2], [12, 12], [2, 12]],
                )
            ],
        )
        save_json(root / "annotation.json", annotation)
        (root / "label.txt").write_text(
            ""
            if empty
            else "0 0.125000000 0.125000000 0.750000000 0.125000000 0.750000000 0.750000000 0.125000000 0.750000000\n"
        )
        receipt = dict(
            sample_id=name,
            reviewer="TEST FIXTURE ONLY",
            acknowledgment=True,
            empty_reviewed=empty,
            image_sha256=file_sha256(image),
            annotation_sha256=file_sha256(root / "annotation.json"),
        )
        record = dict(
            source_record=dict(
                evidence_kind="real",
                source_id="TEST_FIXTURE",
                material="fixture",
                sample_id=name,
                family_id=family,
                split=split,
            ),
            review=receipt,
            image_path="image.png",
            annotation_path="annotation.json",
            label_path="label.txt",
        )
        save_json(
            root / "manifest.json",
            dict(
                schema="crystalab_reviewed_segmentation_v1",
                classes=CLASSES,
                records=[record],
                artifact_sha256={p.name: file_sha256(p) for p in root.iterdir()},
            ),
        )
        return root

    def test_reviewed_sources_keep_splits_and_exact_polygons(self):
        sources = [
            self.source("a", "train", "a", 20),
            self.source("b", "validation", "b", 40),
            self.source("c", "test", "c", 60),
        ]
        out = self.root / "dataset"
        report = prepare_segmentation(sources, out)
        self.assertEqual(
            report["images_by_split"], {"train": 1, "validation": 1, "test": 1}
        )
        self.assertEqual(report["images_by_evidence"], {"real": 3})
        self.assertEqual(report["objects_by_class"], {"seed": 3})
        self.assertEqual(audit_segmentation(out), report)
        with self.assertRaises(FileExistsError):
            prepare_segmentation(sources, out)

    def test_reject_proposals_and_invalidated_receipts(self):
        proposal = self.root / "proposed"
        proposal.mkdir()
        save_json(
            proposal / "manifest.json", dict(schema="crystalab_annotation_bundle_v1")
        )
        with self.assertRaisesRegex(ValueError, "Only audited optical or reviewed"):
            prepare_segmentation([proposal], self.root / "out")
        a = self.source("a", "train", "a", 20)
        b = self.source("b", "validation", "b", 40)
        m = json.loads((a / "manifest.json").read_text())
        m["records"][0]["review"]["annotation_sha256"] = "invalid"
        save_json(a / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "receipt"):
            prepare_segmentation([a, b], self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_family_and_duplicate_image_leakage_are_rejected(self):
        a = self.source("a", "train", "same", 20)
        b = self.source("b", "validation", "same", 40)
        with self.assertRaisesRegex(ValueError, "families cross splits"):
            prepare_segmentation([a, b], self.root / "out")
        c = self.source("c", "validation", "different", 20)
        with self.assertRaisesRegex(ValueError, "images cross splits"):
            prepare_segmentation([a, c], self.root / "out")

    def test_labels_reconstructed_from_source_even_if_export_hashes_are_rewritten(self):
        a = self.source("a", "train", "a", 20)
        b = self.source("b", "validation", "b", 40)
        out = self.root / "out"
        prepare_segmentation([a, b], out)
        label = out / "labels/train/sample_000000.txt"
        label.write_text(label.read_text().replace("0.125000000", "0.250000000"))
        m = json.loads((out / "manifest.json").read_text())
        m["artifact_sha256"][str(label.relative_to(out))] = file_sha256(label)
        save_json(out / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "original reviewed/proxy geometry"):
            audit_segmentation(out)

    def test_empty_images_require_explicit_negative_review(self):
        a = self.source("a", "train", "a", 20, empty=True)
        b = self.source("b", "validation", "b", 40)
        m = json.loads((a / "manifest.json").read_text())
        m["records"][0]["review"]["empty_reviewed"] = False
        save_json(a / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "receipt"):
            prepare_segmentation([a, b], self.root / "out")
        m["records"][0]["review"]["empty_reviewed"] = True
        save_json(a / "manifest.json", m)
        report = prepare_segmentation([a, b], self.root / "out")
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["objects_by_class"], {"seed": 1})

    def test_polygon_hole_loss_is_bounded_and_disconnected_objects_rejected(self):
        mask = np.zeros((32, 32), np.uint8)
        mask[2:30, 2:30] = 255
        mask[10, 10] = 0
        _, conversion = mask_polygon(mask)
        self.assertEqual(conversion["filled_hole_pixels"], 1)
        self.assertGreater(conversion["polygon_mask_iou"], 0.99)
        mask[8:24, 8:24] = 0
        with self.assertRaisesRegex(ValueError, "loses too much"):
            mask_polygon(mask)
        mask[:] = 0
        mask[2:8, 2:8] = 255
        mask[20:26, 20:26] = 255
        with self.assertRaisesRegex(ValueError, "one nondegenerate"):
            mask_polygon(mask)

    def test_training_contract_requires_declared_cpu_experiment(self):
        config = dict(
            schema="crystalab_yolov8_seg_train_v1",
            purpose="plumbing_check",
            epochs=3,
            imgsz=256,
            batch=4,
            threads=2,
            seed=11,
            device="cpu",
            learning_rate=0.001,
        )
        options = training_options(config)
        self.assertFalse(options["amp"])
        self.assertEqual(options["workers"], 0)
        for change in [
            dict(purpose="production_accuracy"),
            dict(epochs=0),
            dict(imgsz=255),
            dict(device="cuda"),
            dict(learning_rate=float("nan")),
            dict(purpose="overfit_check"),
        ]:
            invalid = copy.deepcopy(config)
            invalid.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                training_options(invalid)
