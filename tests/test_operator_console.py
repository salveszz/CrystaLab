# FILE PURPOSE: Verify operator replay actions, provenance, rollback and local HTTP behavior.
# All writable reviews use generated fixtures and are labeled automated QA.
import json
from http.client import HTTPConnection
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_candidate_fixture import generate_candidate_fixture
from controllers.audit import audit_control_benchmark
from controllers.benchmark import sha256
from dashboard.operator import OperatorServer, OperatorStore, prepare_operator_session
from dashboard.review import ReviewConflict
from vision.candidates import CandidateConfig

ROOT = Path(__file__).resolve().parents[1]


def prepare_operator_fixture(root, *, adaptive_reference=None):
    generate_candidate_fixture(root / "fixture", candidate_config=CandidateConfig())
    analysis = root / "fixture_candidates"
    config = json.loads((ROOT / "configs/supervised_control_fixture.json").read_text())
    config["clock_mapping"]["expected_source_video_sha256"] = json.loads(
        (analysis / "source_replay_manifest.json").read_text()
    )["input"]["sha256"]
    path = root / "simulation.json"
    path.write_text(json.dumps(config, indent=2))
    session = root / "operator"
    prepare_operator_session(
        analysis,
        ROOT / "configs/event_rule_policy.json",
        session,
        config=path,
        qa_mode=True,
        adaptive_reference=adaptive_reference,
    )
    return session


def review_payload(store, frame=13, action="approve_pause"):
    s = store.summary()
    entry = next(e for e in s["proposals"] if e["proposal"]["kind"] == "ramp_review")
    return dict(
        revision=s["revision"],
        frame_ordinal=frame,
        proposal_id=entry["proposal"]["proposal_id"],
        proposal_sha256=entry["proposal_sha256"],
        action=action,
        reviewer="Automated fixture check",
        comment="Generated fixture only",
        acknowledged=True,
        pause_duration_model_s=60,
    )


class OperatorConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.prepared = prepare_operator_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.session = self.case / "session"
        shutil.copytree(self.prepared, self.session)
        self.store = OperatorStore(self.session)

    def test_pause_manual_resume_and_pid_response_are_independently_audited(self):
        source_hashes = {
            str(p): sha256(p)
            for p in (self.root / "fixture_candidates").rglob("*")
            if p.is_file()
        }
        saved = self.store.review(review_payload(self.store))
        self.assertEqual(len(saved["reviews"]), 1)
        history, _ = self.store.history()
        path = self.session / history["latest_run"]
        first = json.loads((path / "manifest.json").read_text())
        self.assertEqual(first["operator_reviews_created"], 0)
        self.assertFalse(first["executions"][0]["operator_reviewed"])
        self.assertEqual(first["executions"][0]["basis"], "automated_fixture_qa")
        self.assertEqual(first["runs"][2]["recipe_phase_lag_s"], 60)
        self.assertEqual(audit_control_benchmark(path)["status"], "pass")
        state = self.store.frame(15)
        self.assertTrue(state["can_resume"])
        payload = review_payload(self.store, frame=15, action="resume_recipe")
        payload["pause_review_id"] = saved["reviews"][0]["review_id"]
        saved = self.store.review(payload)
        history, _ = self.store.history()
        path = self.session / history["latest_run"]
        final = json.loads((path / "manifest.json").read_text())
        self.assertEqual(final["runs"][2]["recipe_phase_lag_s"], 20)
        self.assertEqual(final["executions"][0]["pause_until_model_s"], 150)
        self.assertEqual(len(final["manual_resumptions"]), 1)
        self.assertEqual(audit_control_benchmark(path)["status"], "pass")
        self.assertTrue(all(sha256(p) == d for p, d in source_hashes.items()))
        self.assertFalse(self.store.frame(16)["can_resume"])

    def test_missing_review_expiry_unknown_view_and_bad_duration_do_not_create_receipts(
        self,
    ):
        for change in (
            dict(acknowledged=False),
            dict(reviewer=""),
            dict(frame_ordinal=0),
            dict(frame_ordinal=33),
            dict(frame_ordinal=23),
            dict(pause_duration_model_s=61),
        ):
            with self.subTest(change=change):
                p = review_payload(self.store)
                p.update(change)
                with self.assertRaises((ValueError, ReviewConflict)):
                    self.store.review(p)
                self.assertEqual(self.store.summary()["reviews"], [])

    def test_concurrent_stale_review_and_duplicate_disposition_are_refused(self):
        payload = review_payload(self.store)
        self.store.review(payload)
        with self.assertRaises(ReviewConflict):
            self.store.review(payload)
        with self.assertRaises(ReviewConflict):
            self.store.review(review_payload(self.store, frame=14))

    def test_decline_keeps_recipe_running_and_orphan_receipt_is_not_overwritten(self):
        saved = self.store.review(review_payload(self.store, action="decline"))
        self.assertEqual(saved["runs"][2]["recipe_phase_lag_s"], 0)
        history, _ = self.store.history()
        self.assertIsNotNone(history["entries"][0]["result_path"])
        (self.session / "reviews/000002.json").write_text("{}")
        with self.assertRaisesRegex(ReviewConflict, "incomplete review"):
            self.store.summary()

    def test_tampering_with_manual_end_or_resealed_review_time_fails_semantic_audit(
        self,
    ):
        self.store.review(review_payload(self.store))
        history, _ = self.store.history()
        source = self.session / history["latest_run"]
        edited = self.case / "edited"
        shutil.copytree(source, edited)
        manifest = json.loads((edited / "manifest.json").read_text())
        manifest["executions"][0]["pause_until_model_s"] -= 1
        (edited / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "actual pause end"):
            audit_control_benchmark(edited)
        edited = self.case / "edited_review"
        shutil.copytree(source, edited)
        p = edited / "reviews.json"
        reviews = json.loads(p.read_text())
        reviews[0]["source_time_s"] = 3
        p.write_text(json.dumps(reviews))
        m = json.loads((edited / "manifest.json").read_text())
        m["reviews_sha256"] = m["files"]["reviews.json"] = sha256(p)
        (edited / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "review frame clock"):
            audit_control_benchmark(edited)
        edited = self.case / "edited_source"
        shutil.copytree(source, edited)
        p = edited / "reviews.json"
        reviews = json.loads(p.read_text())
        reviews[0]["source_manifest_sha256"] = "0" * 64
        p.write_text(json.dumps(reviews))
        m = json.loads((edited / "manifest.json").read_text())
        m["reviews_sha256"] = m["files"]["reviews.json"] = sha256(p)
        (edited / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "review source/scope"):
            audit_control_benchmark(edited)

    def test_read_only_and_changed_configuration_cannot_save(self):
        readonly = OperatorStore(self.session, read_only=True)
        with self.assertRaises(PermissionError):
            readonly.review(review_payload(self.store))
        p = self.session / "simulation_config.json"
        p.write_text(p.read_text() + "\n")
        with self.assertRaises(ReviewConflict):
            self.store.review(review_payload(self.store))

    def test_http_assets_local_request_guards_and_invalid_frame_are_enforced(self):
        server = OperatorServer(self.session, port=0, read_only=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(path, method="GET", headers=None, body=None):
            c = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                c.request(method, path, headers=headers or {}, body=body)
                r = c.getresponse()
                return r.status, r.read()
            finally:
                c.close()

        try:
            for path in (
                "/",
                "/assets/operator.js",
                "/assets/operator.css",
                "/api/session",
                "/api/image/0",
            ):
                self.assertEqual(request(path)[0], 200)
            self.assertEqual(request("/api/image/-1")[0], 422)
            self.assertEqual(request("/api/evidence/-1")[0], 422)
            self.assertEqual(
                request("/", headers={"Host": "untrusted.example"})[0], 403
            )
            self.assertEqual(
                request(
                    "/api/review",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body="{}",
                )[0],
                403,
            )
            self.assertEqual(
                request(
                    "/api/review",
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-CrystaLab-Session": server.session_token,
                        "Origin": "http://untrusted.example",
                    },
                    body="{}",
                )[0],
                403,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
