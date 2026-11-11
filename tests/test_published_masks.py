# FILE PURPOSE: Verify streamed source masks, source-split integrity and honest training eligibility.
# All pixels and source receipts here are artificial temporary test data.
"""The importer must reject corrupt/unsafe labels without inventing pitch classes."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.published_masks import (
    CLASSES,
    audit_published_masks,
    import_published_masks,
    preview_published_masks,
    published_masks,
    scan,
)
from vision.video import file_sha256


def npz_bytes(labels, masks):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, class_labels=np.asarray(labels), masks=masks)
    return buffer.getvalue()


class PublishedMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / "source.zip"
        self.metadata = self.root / "metadata.json"
        self.output = self.root / "index"
        self.entries = {}
        self.protocol = dict(
            schema="crystalab_chimp_import_v1",
            source_id="FIXTURE-ONLY",
            citation="Generated test pixels, not a published dataset",
            license_id="cc-by-4.0",
            source_classes=list(CLASSES),
            pitch_class_mapping=None,
            zenodo_record_id=1,
            archive_name="source.zip",
            maximum_instances=16,
            maximum_pixels=2000,
            maximum_stack_bytes=100000,
            datasets={},
        )
        for dataset in ("a", "b"):
            self.protocol["datasets"][dataset] = dict(
                directory=dataset,
                training_csv="train.csv",
                validation_csv="validation.csv",
                expected_images=4,
            )
            self.entries[f"{dataset}/train.csv"] = b"filename\n0.jpg\n1.jpg\n2.jpg\n"
            self.entries[f"{dataset}/validation.csv"] = b"filename\n3.jpg\n"
            for number, count in enumerate((0, 1, 6, 1)):
                pixels = np.full((30, 40, 3), 180 + number + (dataset == "b"), np.uint8)
                ok, encoded = cv2.imencode(".jpg", pixels)
                self.assertTrue(ok)
                self.entries[f"{dataset}/images/{number}.jpg"] = encoded.tobytes()
                labels = ["drop"] + ["crystal"] * count
                masks = np.zeros((len(labels), 30, 40), bool)
                masks[0, 1:29, 1:39] = True
                for i in range(count):
                    masks[i + 1, 3 + i : 10 + i, 5 + i : 13 + i] = True
                if count:
                    masks[1, 6, 8] = False  # Keep a source hole; never polygon-fill it.
                self.entries[f"{dataset}/masks/{number}.npz"] = npz_bytes(labels, masks)
        self.repack()

    def tearDown(self):
        self.temp.cleanup()

    def repack(self):
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in self.entries.items():
                archive.writestr(name, data)
        self.protocol["archive_sha256"] = file_sha256(self.archive)
        self.protocol["publisher_md5"] = hashlib.md5(
            self.archive.read_bytes()
        ).hexdigest()
        metadata = dict(
            id=1,
            metadata=dict(license=dict(id="cc-by-4.0")),
            files=[
                dict(
                    key="source.zip",
                    checksum="md5:" + self.protocol["publisher_md5"],
                    size=self.archive.stat().st_size,
                )
            ],
        )
        self.metadata.write_text(json.dumps(metadata))
        self.protocol["metadata_sha256"] = file_sha256(self.metadata)

    def import_index(self):
        return import_published_masks(
            self.archive, self.metadata, self.protocol, self.output
        )

    def test_original_masks_and_author_membership_are_retained_without_pitch_relabeling(
        self,
    ):
        manifest = self.import_index()
        self.assertEqual(manifest["summary"]["source_instance_masks"], 24)
        self.assertEqual(
            manifest["summary"]["masks_by_source_class"], dict(drop=8, crystal=16)
        )
        self.assertEqual(
            manifest["summary"]["images_by_source_split"], dict(train=6, validation=2)
        )
        self.assertEqual(manifest["summary"]["pitch_training_images"], 0)
        rows = [
            json.loads(v)
            for v in (self.output / "images.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            all(
                r["physical_run_id"] is None
                and r["pitch_class_mapping"] is None
                and not r["pitch_training_ready"]
                and r["local_human_review_receipt"] is None
                for r in rows
            )
        )
        with zipfile.ZipFile(self.archive) as archive:
            row = rows[1]
            raw = archive.read(row["mask_member"])
            source = list(published_masks(raw, (30, 40), self.protocol))
        self.assertFalse(source[1][1][6, 8])
        self.assertEqual(
            row["source_objects"][1]["bool_plane_sha256"],
            hashlib.sha256(source[1][1].tobytes()).hexdigest(),
        )
        self.assertEqual(audit_published_masks(self.output)["status"], "passed")

    def test_mask_stack_is_streamed_and_only_small_class_array_uses_numpy_load(self):
        payload = self.entries["a/masks/2.npz"]
        calls = []
        original = np.load

        def small_load(source, **kwargs):
            self.assertIs(kwargs.get("allow_pickle"), False)
            calls.append(len(source.getbuffer()))
            return original(source, **kwargs)

        with patch("vision.published_masks.np.load", side_effect=small_load):
            masks = published_masks(payload, (30, 40), self.protocol)
            first = next(masks)
            self.assertEqual(first[1].nbytes, 1200)
            self.assertEqual(len(list(masks)), 6)
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0], 1200)

    def test_empty_publisher_stack_is_preserved_without_invented_negative_truth(self):
        payload = npz_bytes([], np.array([], dtype=bool))
        self.assertEqual(list(published_masks(payload, (30, 40), self.protocol)), [])
        self.entries["a/masks/0.npz"] = payload
        self.repack()
        manifest = self.import_index()
        self.assertEqual(manifest["summary"]["empty_source_annotation_images"], 1)
        self.assertEqual(manifest["summary"]["images_without_selected_crystal"], 2)
        self.assertEqual(manifest["summary"]["local_human_review_receipts"], 0)

    def test_bad_mask_axes_dtype_class_pickle_and_empty_planes_are_rejected(self):
        for labels, masks in (
            (["crystal"], np.ones((1, 40, 30), bool)),
            (["crystal"], np.ones((1, 30, 40), np.uint8)),
            (["seed"], np.ones((1, 30, 40), bool)),
            (np.array(["crystal"], dtype=object), np.ones((1, 30, 40), bool)),
            (["crystal"], np.zeros((1, 30, 40), bool)),
            (["crystal", "drop"], np.ones((1, 30, 40), bool)),
            (["crystal"], np.asfortranarray(np.ones((1, 30, 40), bool))),
        ):
            with (
                self.subTest(dtype=str(np.asarray(masks).dtype), labels=str(labels)),
                self.assertRaises(ValueError),
            ):
                list(published_masks(npz_bytes(labels, masks), (30, 40), self.protocol))

    def test_declared_instance_and_expanded_stack_bounds_are_enforced(self):
        payload = self.entries["a/masks/2.npz"]
        for change in (dict(maximum_instances=2), dict(maximum_stack_bytes=200)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                list(published_masks(payload, (30, 40), dict(self.protocol, **change)))

    def test_source_membership_duplicate_and_missing_label_files_fail(self):
        for field in ("duplicate", "missing", "unsafe", "unaccounted"):
            saved = copy.deepcopy(self.entries)
            if field == "duplicate":
                self.entries["a/validation.csv"] = b"filename\n0.jpg\n3.jpg\n"
            elif field == "missing":
                del self.entries["a/masks/1.npz"]
            elif field == "unsafe":
                self.entries["../escape.jpg"] = b"not extracted"
            else:
                self.entries["extra.npz"] = self.entries["a/masks/1.npz"]
            self.repack()
            with self.subTest(field=field), self.assertRaises(ValueError):
                scan(self.archive, self.metadata, self.protocol)
            self.entries = saved
        self.repack()

    def test_changed_archive_or_license_and_pitch_mapping_are_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.import_index()
        self.assertEqual(
            json.loads((self.output / "manifest.json").read_text())["status"], "failed"
        )
        self.repack()
        metadata = json.loads(self.metadata.read_text())
        metadata["metadata"]["license"]["id"] = "unknown"
        self.metadata.write_text(json.dumps(metadata))
        self.protocol["metadata_sha256"] = file_sha256(self.metadata)
        with self.assertRaisesRegex(ValueError, "license"):
            scan(self.archive, self.metadata, self.protocol)
        self.repack()
        self.protocol["pitch_class_mapping"] = {"crystal": "seed"}
        with self.assertRaisesRegex(ValueError, "unmapped"):
            scan(self.archive, self.metadata, self.protocol)

    def test_rehashed_index_and_summary_tampering_fails_source_reconstruction(self):
        self.import_index()
        index = self.output / "images.jsonl"
        manifest_path = self.output / "manifest.json"
        rows = [json.loads(v) for v in index.read_text().splitlines()]
        rows[1]["source_objects"][1]["pixels"] += 1
        index.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = json.loads(manifest_path.read_text())
        manifest["index_sha256"] = file_sha256(index)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source reconstruction"):
            audit_published_masks(self.output)

    def test_duplicate_pixels_across_author_splits_are_retained_and_flagged(self):
        self.entries["a/images/3.jpg"] = self.entries["a/images/1.jpg"]
        self.repack()
        manifest = self.import_index()
        self.assertEqual(len(manifest["summary"]["cross_split_duplicate_groups"]), 1)
        self.assertEqual(manifest["summary"]["pitch_training_images"], 0)

    def test_preview_uses_training_only_and_never_creates_review_receipts(self):
        self.import_index()
        before = {p.name: file_sha256(p) for p in self.output.iterdir()}
        report = preview_published_masks(self.output, self.root / "preview")
        self.assertEqual(len(report["previews"]), 6)
        self.assertTrue(all(p["source_split"] == "train" for p in report["previews"]))
        self.assertEqual(report["local_human_reviews_created"], 0)
        self.assertIsNone(report["visual_acceptance"])
        self.assertEqual(
            before, {p.name: file_sha256(p) for p in self.output.iterdir()}
        )
        with self.assertRaises(FileExistsError):
            preview_published_masks(self.output, self.root / "preview")
        with self.assertRaises(FileExistsError):
            self.import_index()


if __name__ == "__main__":
    unittest.main()
