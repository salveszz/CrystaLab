# FILE PURPOSE: Verify end-to-end candidate events against independent fixture labels.
# Real-review receipt tests use artificial temporary files only and never review
# actual project footage. Coverage, censoring and label/source binding are explicit.

import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_candidate_fixture import generate_candidate_fixture
from vision.candidates import CandidateConfig
from vision.candidate_replay import analyze_candidates
from vision.event_evaluation import (
    load_annotations,
    review_annotations,
    score_candidate_events,
)
from vision.training_data import save_json
from vision.video import file_sha256


class EventEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root / "fixture"
        generate_candidate_fixture(cls.fixture)
        cls.analysis = cls.root / "analysis"
        analyze_candidates(
            cls.fixture / "observations", CandidateConfig(), cls.analysis
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        self.annotations = self.case / "annotations.json"
        shutil.copyfile(self.fixture / "annotations.json", self.annotations)
        shutil.copytree(self.fixture / "truth_masks", self.case / "truth_masks")

    def test_fixture_pipeline_distinguishes_startup_new_and_unknown_without_hiding_false_alert(
        self,
    ):
        result = score_candidate_events(
            self.analysis, self.annotations, self.case / "score"
        )
        self.assertEqual(result["events"], 4)
        self.assertEqual(result["truth_events"], 3)
        self.assertEqual(result["per_kind"]["crystal_candidate"]["tp"], 2)
        self.assertEqual(result["per_kind"]["crystal_candidate"]["fp"], 1)
        self.assertEqual(result["per_kind"]["inclusion_candidate"]["tp"], 1)
        self.assertEqual(result["new_visibility_claims"], dict(tp=2, fp=0, fn=0))
        self.assertTrue(all(m["origin_correct"] for m in result["matches"]))
        self.assertAlmostEqual(
            result["negative_coverage"]["crystal_candidate"][
                "usable_annotated_negative_source_seconds"
            ],
            0.9,
        )
        self.assertAlmostEqual(
            result["negative_coverage"]["inclusion_candidate"][
                "usable_annotated_negative_source_seconds"
            ],
            2.2,
        )
        self.assertEqual(
            result["negative_coverage"]["crystal_candidate"][
                "false_events_in_negative_intervals"
            ],
            1,
        )
        self.assertIsNone(
            result["negative_coverage"]["crystal_candidate"][
                "false_alerts_per_observed_negative_hour"
            ]
        )
        self.assertFalse(result["first_hour_claim_evaluated"])
        self.assertEqual(result["duplicate_alert_ids"], [])
        self.assertEqual(
            result["complete_followup"]["crystal_candidate"]["eligible"], 2
        )

    def test_real_labels_need_named_review_and_exact_receipt_not_an_automatic_import(
        self,
    ):
        truth = json.loads(self.annotations.read_text())
        truth["evidence_kind"] = "real"
        save_json(self.annotations, truth)
        digest = truth["source_video_sha256"]
        with self.assertRaisesRegex(ValueError, "explicit source"):
            load_annotations(self.annotations, digest)
        with self.assertRaisesRegex(ValueError, "explicitly acknowledge"):
            review_annotations(
                self.annotations, self.fixture / "sequence.avi", "TEST FIXTURE"
            )
        receipt = review_annotations(
            self.annotations,
            self.fixture / "sequence.avi",
            "TEST FIXTURE ONLY",
            ack_event_source_reviewed=True,
        )
        self.assertEqual(load_annotations(self.annotations, digest)[2], receipt)
        with self.assertRaises(FileExistsError):
            review_annotations(
                self.annotations,
                self.fixture / "sequence.avi",
                "TEST FIXTURE",
                ack_event_source_reviewed=True,
            )
        truth["label_semantics"] = "edited after review"
        save_json(self.annotations, truth)
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            load_annotations(self.annotations, digest)

    def test_uncertain_negatives_and_changed_masks_cannot_be_scored_as_truth(self):
        truth = json.loads(self.annotations.read_text())
        truth["negative_intervals"][0]["uncertain"] = True
        save_json(self.annotations, truth)
        with self.assertRaisesRegex(ValueError, "Uncertain interval"):
            load_annotations(self.annotations, truth["source_video_sha256"])
        truth["negative_intervals"][0].pop("uncertain")
        save_json(self.annotations, truth)
        mask = self.case / truth["events"][0]["first_visible_mask_path"]
        mask.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mask identity"):
            load_annotations(self.annotations, truth["source_video_sha256"])

    def test_label_source_identity_and_constructed_real_evidence_are_bound(self):
        truth = json.loads(self.annotations.read_text())
        truth["source_video_sha256"] = "wrong"
        save_json(self.annotations, truth)
        with self.assertRaisesRegex(ValueError, "clock identity"):
            score_candidate_events(self.analysis, self.annotations, self.case / "wrong")
        truth["source_video_sha256"] = file_sha256(self.fixture / "sequence.avi")
        truth["evidence_kind"] = "real"
        save_json(self.annotations, truth)
        review_annotations(
            self.annotations,
            self.fixture / "sequence.avi",
            "TEST FIXTURE",
            ack_event_source_reviewed=True,
        )
        with self.assertRaisesRegex(ValueError, "masquerade"):
            score_candidate_events(self.analysis, self.annotations, self.case / "mixed")

    def test_short_followup_is_censored_and_outside_window_targets_are_reported(self):
        analysis = self.case / "short"
        shutil.copytree(self.analysis, analysis)
        rows = [
            json.loads(line)
            for line in (analysis / "candidate_observations.jsonl")
            .read_text()
            .splitlines()
        ]
        (analysis / "candidate_observations.jsonl").write_text(
            "".join(
                json.dumps(row) + "\n" for row in rows if row["video_time_s"] <= 2.9
            )
        )
        manifest = json.loads((analysis / "manifest.json").read_text())
        manifest["files"]["candidate_observations.jsonl"] = file_sha256(
            analysis / "candidate_observations.jsonl"
        )
        save_json(analysis / "manifest.json", manifest)
        truth = json.loads(self.annotations.read_text())
        late = copy.deepcopy(truth["events"][0])
        late.update(event_id="outside-window", onset_interval_source_s=[4.0, 4.1])
        truth["events"].append(late)
        save_json(self.annotations, truth)
        result = score_candidate_events(analysis, self.annotations, self.case / "score")
        self.assertEqual(
            result["excluded_outside_window_event_ids"], ["outside-window"]
        )
        self.assertEqual(
            result["complete_followup"]["inclusion_candidate"]["eligible"], 0
        )
        self.assertEqual(
            result["complete_followup"]["inclusion_candidate"]["censored_event_ids"],
            ["new_inclusion_B"],
        )

    def test_elapsed_clip_does_not_count_unobservable_class_deadline_as_complete(self):
        baseline = score_candidate_events(
            self.analysis, self.annotations, self.case / "baseline"
        )
        self.assertEqual(
            baseline["observable_deadline_followup"]["inclusion_candidate"]["eligible"],
            1,
        )
        analysis = self.case / "gap"
        shutil.copytree(self.analysis, analysis)
        path = analysis / "candidate_observations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        index = next(
            i for i, row in enumerate(rows) if math.isclose(row["video_time_s"], 3.0)
        )
        rows[index]["observable"]["inclusion_candidate"] = False
        for i in (index, index + 1):
            rows[i]["usable_intervals"] = [
                v
                for v in rows[i]["usable_intervals"]
                if v["candidate_type"] != "inclusion_candidate"
            ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = json.loads((analysis / "manifest.json").read_text())
        manifest["files"][path.name] = file_sha256(path)
        save_json(analysis / "manifest.json", manifest)
        result = score_candidate_events(analysis, self.annotations, self.case / "score")
        self.assertEqual(
            result["complete_followup"]["inclusion_candidate"]["eligible"], 1
        )
        strict = result["observable_deadline_followup"]["inclusion_candidate"]
        self.assertEqual(strict["eligible"], 0)
        self.assertEqual(
            strict["excluded"][0]["reason"], "unobservable_or_gap_within_deadline"
        )
        self.assertEqual(
            result["observable_deadline_followup"]["crystal_candidate"],
            baseline["observable_deadline_followup"]["crystal_candidate"],
        )
        self.assertEqual(result["per_kind"]["inclusion_candidate"]["tp"], 1)

    def test_rehashed_usable_interval_cannot_bridge_unknown_observations(self):
        analysis = self.case / "forged"
        shutil.copytree(self.analysis, analysis)
        path = analysis / "candidate_observations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        index = next(
            i for i, row in enumerate(rows) if math.isclose(row["video_time_s"], 3.0)
        )
        rows[index]["observable"]["inclusion_candidate"] = False
        # Leave the old interval claiming that this unknown endpoint was usable.
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = json.loads((analysis / "manifest.json").read_text())
        manifest["files"][path.name] = file_sha256(path)
        save_json(analysis / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "adjacent class-specific"):
            score_candidate_events(analysis, self.annotations, self.case / "score")

    def test_physical_rate_needs_real_review_and_supported_clock_mapping(self):
        # Explicitly artificial metadata mutation to test unit conversion, never real evidence.
        analysis = self.case / "clock"
        shutil.copytree(self.analysis, analysis)
        source = json.loads((analysis / "source_replay_manifest.json").read_text())
        source["source_info"].update(
            evidence_kind="real", experiment_clock_provenance="TEST FIXTURE ONLY"
        )
        source["experiment_clock"] = dict(origin_s=0, seconds_per_video_second=2)
        save_json(analysis / "source_replay_manifest.json", source)
        manifest = json.loads((analysis / "manifest.json").read_text())
        digest = file_sha256(analysis / "source_replay_manifest.json")
        manifest["source_replay_manifest_sha256"] = digest
        manifest["files"]["source_replay_manifest.json"] = digest
        save_json(analysis / "manifest.json", manifest)
        truth = json.loads(self.annotations.read_text())
        truth["evidence_kind"] = "real"
        save_json(self.annotations, truth)
        review_annotations(
            self.annotations,
            self.fixture / "sequence.avi",
            "TEST FIXTURE ONLY",
            ack_event_source_reviewed=True,
            ack_physical_clock_reviewed=True,
        )
        result = score_candidate_events(analysis, self.annotations, self.case / "score")
        self.assertTrue(result["physical_clock_scored"])
        rate = result["negative_coverage"]["crystal_candidate"]
        self.assertAlmostEqual(rate["physical_negative_seconds"], 1.8)
        self.assertAlmostEqual(rate["false_alerts_per_observed_negative_hour"], 2000)
