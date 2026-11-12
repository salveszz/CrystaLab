# FILE PURPOSE: Guard crop provenance, unknown labels and development grouping.
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from dashboard.review import ReviewStore
from vision.annotations import export_reviewed
from vision.publication_panels import audit_index, import_panels, prepare_bundle, scan

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_asset_audit", ROOT / "scripts/audit_data_sources.py")
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublicationPanelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.protocol = json.loads((ROOT / "configs/kdp_gel_panels.json").read_text())
        self.protocol.update(source_id="PUBLICATION-TEST", dependency_family_id="whole-study-test", run_id="test_panels")
        self.protocol["metadata"] = []
        yy, xx = np.mgrid[:24, :32]
        image = np.stack([xx * 7, yy * 9, (xx + yy) * 4], axis=2).astype(np.uint8)
        cv2.imwrite(str(self.cache / "figure.jpg"), image)
        self.protocol["figures"] = [dict(figure_id="one", file="figure.jpg", width=32, height=24,
                                         sha256=digest(self.cache / "figure.jpg"))]
        self.protocol["panels"] = [dict(panel_id="first", figure_id="one", crop_xyxy=[1, 2, 17, 18],
            domain="test_publication", context="Test source, not real evidence", published_relative_time_s=600)]
        self.path = self.root / "protocol.json"
        self.write_protocol()

    def write_protocol(self):
        self.path.write_text(json.dumps(self.protocol))

    def test_exact_crop_and_group_survive_review_preview_without_truth(self):
        index, bundle = self.root / "index", self.root / "bundle"
        import_panels(self.cache, self.path, index)
        prepare_bundle(index, self.path, bundle)
        manifest, rows = audit_index(index, self.path)
        row = rows[0]
        expected = cv2.imread(str(self.cache / "figure.jpg"))[2:18, 1:17]
        np.testing.assert_array_equal(cv2.imread(str(index / row["image_path"])), expected)
        self.assertEqual(row["dependency_family_id"], "whole-study-test")
        self.assertEqual(row["published_relative_time_s"], 600)
        self.assertIsNone(row["first_hour_membership"])
        self.assertIsNone(row["event_labels"])
        before = {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()}
        store = ReviewStore(bundle, read_only=True)
        self.assertEqual(store.summary()["reviewed"], 0)
        self.assertEqual(store.get(0)["status"], "proposed")
        self.assertTrue(store.get(0)["annotation"]["flags"]["uncertain"])
        self.assertEqual(store.get(0)["source"]["data_kind"], "publication_panel")
        store.image(0)
        self.assertEqual(before, {str(p.relative_to(bundle)): digest(p) for p in bundle.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "No explicitly reviewed"):
            export_reviewed(bundle, self.root / "export")
        self.assertFalse((self.root / "export").exists())

    def test_fake_record_with_recomputed_hash_still_fails_source_reconstruction(self):
        index = self.root / "index"
        import_panels(self.cache, self.path, index)
        row = json.loads((index / "images.jsonl").read_text())
        row["published_relative_time_s"] = 1
        (index / "images.jsonl").write_text(json.dumps(row) + "\n")
        manifest = json.loads((index / "manifest.json").read_text())
        manifest["index_sha256"] = digest(index / "images.jsonl")
        (index / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "records differ"):
            audit_index(index, self.path)

    def test_external_protocol_and_source_pixels_cannot_silently_change(self):
        index = self.root / "index"
        import_panels(self.cache, self.path, index)
        self.protocol["license"] = "Different terms"
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, "external frozen"):
            audit_index(index, self.path)
        with (self.cache / "figure.jpg").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            scan(self.cache, self.protocol)

    def test_noninteger_outside_duplicate_and_holdout_crops_are_refused(self):
        variants = []
        for roi in ([0.5, 0, 10, 10], [0, 0, 100, 20], [0, 0, 0, 5]):
            protocol = copy.deepcopy(self.protocol)
            protocol["panels"][0]["crop_xyxy"] = roi
            variants.append(protocol)
        protocol = copy.deepcopy(self.protocol)
        panel = copy.deepcopy(protocol["panels"][0]);panel["panel_id"] = "duplicate"
        protocol["panels"].append(panel);variants.append(protocol)
        protocol = copy.deepcopy(self.protocol);protocol["split"] = "test";variants.append(protocol)
        protocol = copy.deepcopy(self.protocol);protocol["panels"][0]["published_relative_time_s"] = True;variants.append(protocol)
        for protocol in variants:
            with self.subTest(protocol=protocol):
                with self.assertRaises(ValueError):
                    scan(self.cache, protocol)

    def test_source_auditor_decodes_images_and_rejects_wrong_format(self):
        payload = (self.cache / "figure.jpg").read_bytes()
        self.assertEqual(ASSETS._describe_payload(payload, "jpg"), dict(width=32, height=24, mode="RGB"))
        stream = io.BytesIO()
        Image.new("RGB", (4, 5)).save(stream, format="PNG")
        with self.assertRaisesRegex(ValueError, "format differs"):
            ASSETS._describe_payload(stream.getvalue(), "jpg")
