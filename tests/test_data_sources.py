# FILE PURPOSE: Verify offline source auditing, honest download failures and
# cached-file identity checks. No live network, model fitting or experimental
# measurements are needed by these tests.

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_data_sources import _describe_payload, audit_sources, read_registry


class DataSourceTests(unittest.TestCase):
    def test_publisher_md5_is_checked_in_addition_to_saved_sha256(self):
        data = b"time_s,value\n0,2\n"
        asset = dict(
            name="values.csv",
            url="https://example.invalid/data",
            kind="csv",
            max_bytes=100,
            md5="wrong",
        )
        registry = dict(
            sources=[
                dict(
                    source_id="FIXTURE",
                    data_status="reference_only",
                    reason="Publisher digest check",
                    assets=[asset],
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "FIXTURE"
            folder.mkdir()
            (folder / "values.csv").write_bytes(data)
            result = audit_sources(registry, directory)["sources"][0]["assets"][0]
            self.assertEqual(result["status"], "unavailable_or_invalid")
            asset["md5"] = hashlib.md5(data).hexdigest()
            result = audit_sources(registry, directory)["sources"][0]["assets"][0]
            self.assertEqual(result["publisher_md5_verified"], asset["md5"])
            self.assertEqual(result["sha256"], hashlib.sha256(data).hexdigest())

    def test_archives_are_complete_crc_checked_and_paths_are_bounded(self):
        for name in ("images/frame.png", "../escape", "/absolute", "dir\\escape"):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr(name, b"data")
            if name == "images/frame.png":
                self.assertEqual(
                    _describe_payload(stream.getvalue(), "zip")["archive_entries"], 1
                )
            else:
                with self.assertRaises(ValueError):
                    _describe_payload(stream.getvalue(), "zip")
        with self.assertRaises(ValueError):
            _describe_payload(b"PK truncated", "zip")

    def setUp(self):
        self.registry = read_registry(ROOT / "configs/data_sources.json")

    def test_offline_audit_does_not_use_network(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("urllib.request.urlopen") as opener,
        ):
            report = audit_sources(self.registry, directory)
            opener.assert_not_called()
        self.assertEqual(report["verified_kdp_trajectory_sources"], [])
        self.assertTrue(
            all(
                a["status"] == "not_cached"
                for s in report["sources"]
                for a in s["assets"]
            )
        )

    def test_html_and_invalid_file_formats_rejected(self):
        for kind in ("pdf", "mov", "csv", "text"):
            with self.assertRaises(ValueError):
                _describe_payload(b"<!DOCTYPE html><html>challenge</html>", kind)
        with self.assertRaises(ValueError):
            _describe_payload(b"not a pdf", "pdf")
        self.assertEqual(
            _describe_payload(b"time_s,value\n0,2\n", "csv")["data_rows"], 1
        )

    def test_download_failure_is_retained_without_creating_data(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("urllib.request.urlopen", side_effect=OSError("unavailable")),
        ):
            report = audit_sources(self.registry, directory, fetch=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue(
            all(
                a["status"] == "unavailable_or_invalid"
                for s in report["sources"]
                for a in s["assets"]
            )
        )

    def test_cached_identity_is_verified_without_overwriting(self):
        data = b"time_s,value\n0,2\n"
        registry = {
            "sources": [
                {
                    "source_id": "FIXTURE",
                    "data_status": "reference_only",
                    "reason": "File-check fixture",
                    "assets": [
                        {
                            "name": "values.csv",
                            "url": "https://example.invalid/values.csv",
                            "kind": "csv",
                            "max_bytes": 100,
                            "sha256": "wrong",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FIXTURE/values.csv"
            path.parent.mkdir()
            path.write_bytes(data)
            report = audit_sources(registry, directory)
            self.assertEqual(
                report["sources"][0]["assets"][0]["status"], "unavailable_or_invalid"
            )
            registry["sources"][0]["assets"][0]["sha256"] = hashlib.sha256(
                data
            ).hexdigest()
            report = audit_sources(registry, directory)
            self.assertEqual(report["sources"][0]["assets"][0]["data_rows"], 1)
            self.assertEqual(path.read_bytes(), data)

    def test_duplicate_registry_identifiers_rejected(self):
        registry = json.loads(json.dumps(self.registry))
        registry["sources"].append(registry["sources"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_registry(path)


if __name__ == "__main__":
    unittest.main()
