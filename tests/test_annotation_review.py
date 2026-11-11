# FILE PURPOSE: Verify explicit review, draft/version/export integrity and local HTTP boundaries.
"""All review receipts in these tests attest only to artificial fixture images."""

import copy
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotation_review_fixture import prepare_review_fixture
from dashboard.review import ReviewConflict, ReviewServer, ReviewStore
from vision.video import file_sha256


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = prepare_review_fixture(self.root)
        self.store = ReviewStore(self.bundle)

    def tearDown(self):
        self.temp.cleanup()

    def draft(self, index=0, **changes):
        current = self.store.get(index)
        return dict(
            bundle_id=current["bundle_id"],
            revision=current["revision"],
            shapes=copy.deepcopy(current["annotation"]["shapes"]),
            image_uncertain=False,
            note="Fixture correction",
            **changes,
        )

    def review(self, index=0, **changes):
        current = self.store.get(index)
        payload = dict(
            bundle_id=current["bundle_id"],
            revision=current["revision"],
            reviewer="AUTOMATED FIXTURE ONLY",
            ack_image_reviewed=True,
            empty_reviewed=False,
        )
        payload.update(changes)
        return self.store.review(index, payload)


class ReviewTests(ReviewFixture):
    def test_holdout_and_escaping_artifact_paths_cannot_be_opened(self):
        path = self.bundle / "manifest.json"
        original = json.loads(path.read_text())
        for field, value in (
            ("split", "test"),
            ("image_path", "../outside.png"),
            ("annotation_path", "/tmp/outside.json"),
        ):
            manifest = copy.deepcopy(original)
            sample = manifest["samples"][0]
            if field == "split":
                sample["source_record"][field] = value
            else:
                sample[field] = value
            path.write_text(json.dumps(manifest))
            with self.subTest(field=field), self.assertRaises(ValueError):
                ReviewStore(self.bundle)
        path.write_text(json.dumps(original))

    def test_read_and_draft_do_not_create_receipts_or_accepted_labels(self):
        self.assertEqual(self.store.summary()["reviewed"], 0)
        before = self.store.get(0)
        payload = self.draft()
        payload["shapes"][0]["points"][0][0] += 2
        after = self.store.save_draft(0, payload)
        self.assertEqual(after["status"], "draft")
        self.assertNotEqual(after["revision"], before["revision"])
        self.assertIsNone(after["review"])
        self.assertEqual(list((self.bundle / "reviews").iterdir()), [])
        with self.assertRaisesRegex(ValueError, "No explicitly reviewed"):
            self.store.export(dict(bundle_id=before["bundle_id"]))

    def test_uncertainty_can_be_saved_but_not_reviewed(self):
        for level in ("image", "shape"):
            with self.subTest(level=level):
                payload = self.draft()
                if level == "image":
                    payload["image_uncertain"] = True
                else:
                    payload["shapes"][0]["flags"]["uncertain"] = True
                self.store.save_draft(0, payload)
                with self.assertRaisesRegex(ValueError, "Uncertain"):
                    self.review()
                self.assertEqual(self.store.summary()["reviewed"], 0)

    def test_review_requires_named_explicit_boolean_acknowledgments(self):
        for changes in (
            {"reviewer": " "},
            {"reviewer": True},
            {"ack_image_reviewed": 1},
            {"ack_image_reviewed": "yes"},
            {"empty_reviewed": "false"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.review(**changes)
        payload = self.draft()
        payload["shapes"] = []
        self.store.save_draft(0, payload)
        with self.assertRaisesRegex(ValueError, "reviewed-negative"):
            self.review()
        self.assertEqual(self.review(empty_reviewed=True)["status"], "reviewed")

    def test_invalid_geometry_is_rejected_before_draft_bytes_change(self):
        before = self.store.get(0)
        for points in (
            [[[-1, 10], [50, 10], [50, 50]]],
            [[[10, 10], [60, 10], [60, 60], [35, 10], [10, 60]]],
            [[[10, 10], [60, 60], [60, 10], [10, 60]]],
        ):
            payload = self.draft()
            payload["shapes"][0]["points"] = points[0]
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.store.save_draft(0, payload)
            self.assertEqual(self.store.get(0)["revision"], before["revision"])

    def test_optimistic_revision_and_manifest_conflicts_prevent_lost_edits(self):
        old_payload = self.draft()
        self.store.save_draft(0, old_payload)
        with self.assertRaisesRegex(ReviewConflict, "another window"):
            self.store.save_draft(0, old_payload)
        payload = self.draft()
        payload["bundle_id"] = "another bundle"
        with self.assertRaises(ReviewConflict):
            self.store.save_draft(0, payload)
        manifest = self.bundle / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaisesRegex(ReviewConflict, "changed on disk"):
            self.store.get(0)

    def test_revision_preserves_original_and_other_valid_receipts(self):
        for index in (0, 1):
            self.review(index)
        original = {
            str(p.relative_to(self.bundle)): file_sha256(p)
            for p in self.bundle.rglob("*")
            if p.is_file()
        }
        old = self.store.get(0)
        with self.assertRaisesRegex(ReviewConflict, "already has a review"):
            self.store.save_draft(0, self.draft())
        revised = self.store.revise(
            0, dict(bundle_id=old["bundle_id"], revision=old["revision"])
        )
        self.assertIsNone(revised["sample"]["review"])
        self.assertEqual(self.store.get(1)["status"], "reviewed")
        self.assertEqual(self.store.summary()["reviewed"], 1)
        self.assertNotEqual(self.store.bundle_id, old["bundle_id"])
        self.assertEqual(
            original,
            {
                str(p.relative_to(self.bundle)): file_sha256(p)
                for p in self.bundle.rglob("*")
                if p.is_file()
            },
        )
        self.store.save_draft(0, self.draft())
        self.assertEqual(self.review()["status"], "reviewed")

    def test_stale_receipt_requires_revision_and_blocks_export(self):
        self.review()
        path = self.bundle / "annotations/run_0000.json"
        annotation = json.loads(path.read_text())
        annotation["review_note"] = "External edit"
        path.write_text(json.dumps(annotation))
        current = self.store.get(0)
        self.assertEqual(current["status"], "stale_review")
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.store.export(dict(bundle_id=current["bundle_id"]))
        revised = self.store.revise(
            0, dict(bundle_id=current["bundle_id"], revision=current["revision"])
        )
        self.assertEqual(revised["summary"]["reviewed"], 0)
        self.assertIsNone(revised["sample"]["review"])

    def test_export_is_reviewed_subset_with_exact_source_and_distinct_outputs(self):
        self.review()
        first = self.store.export(dict(bundle_id=self.store.bundle_id))
        second = self.store.export(dict(bundle_id=self.store.bundle_id))
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(first["reviewed_images"], 1)
        self.assertEqual(first["unreviewed_images"], 2)
        exported = json.loads((Path(first["output"]) / "manifest.json").read_text())
        source = exported["records"][0]["source_record"]
        self.assertEqual(source["family_id"], "fixture_family")
        self.assertEqual(source["split"], "train")
        self.assertEqual(
            exported["source_protocol"]["license"], "Generated test pixels only"
        )

    def test_read_only_store_rejects_all_write_actions(self):
        self.store = ReviewStore(self.bundle, read_only=True)
        current = self.store.get(0)
        for action, payload in (
            (self.store.save_draft, self.draft()),
            (self.store.review, dict(bundle_id=current["bundle_id"])),
            (self.store.revise, dict(bundle_id=current["bundle_id"])),
        ):
            with (
                self.subTest(action=action.__name__),
                self.assertRaises(PermissionError),
            ):
                action(0, payload)
        with self.assertRaises(PermissionError):
            self.store.export(dict(bundle_id=current["bundle_id"]))


class ReviewHTTPTests(ReviewFixture):
    # Only HTTP-specific cases are collected here; the shared setup creates fixture pixels.
    def request(self, method, path, payload=None, headers=None):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        values = {"Content-Type": "application/json"}
        if headers:
            values.update(headers)
        try:
            conn.request(
                method,
                path,
                body=None if payload is None else json.dumps(payload),
                headers=values,
            )
            response = conn.getresponse()
            return response.status, response.read(), response.getheaders()
        finally:
            conn.close()

    def setUp(self):
        super().setUp()
        self.server = ReviewServer(self.bundle, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        super().tearDown()

    def test_local_page_and_api_require_current_session_for_writes(self):
        for path, content in (
            ("/", b"Check the crystal"),
            ("/assets/review.js", b"saveDraft"),
            ("/assets/review.css", b"viewport"),
            ("/api/image/0", b"PNG"),
        ):
            with self.subTest(path=path):
                status, body, headers = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(content, body)
                self.assertIn(
                    "frame-ancestors 'none'", dict(headers)["Content-Security-Policy"]
                )
        _, body, _ = self.request("GET", "/api/session")
        token = json.loads(body)["session_token"]
        payload = self.draft()
        status, _, _ = self.request("POST", "/api/sample/0/draft", payload)
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            "POST", "/api/sample/0/draft", payload, {"X-CrystaLab-Session": token}
        )
        self.assertEqual(status, 200)

    def test_foreign_origin_host_and_path_are_rejected_and_read_only_is_enforced(self):
        _, body, _ = self.request("GET", "/api/session")
        token = json.loads(body)["session_token"]
        for headers in (
            {"Host": "untrusted.invalid"},
            {"Origin": "https://foreign.invalid"},
        ):
            with self.subTest(headers=headers):
                self.assertEqual(
                    self.request("GET", "/api/session", headers=headers)[0], 403
                )
        self.assertEqual(self.request("GET", "/assets/../manifest.json")[0], 404)
        self.server.store.read_only = True
        self.assertEqual(
            self.request(
                "POST",
                "/api/export",
                dict(bundle_id=self.store.bundle_id),
                {"X-CrystaLab-Session": token},
            )[0],
            403,
        )


if __name__ == "__main__":
    unittest.main()
