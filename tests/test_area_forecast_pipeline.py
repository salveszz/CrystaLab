# FILE PURPOSE: Verify forecast input/label provenance with independent artificial mask sequences.
# Supplied predictions deliberately differ from truth. Actual Blender/YOLO execution
# is audited separately; these fixtures test clocks, grouped fitting and tamper guards.

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from models.area_forecast import AreaForecastConfig
from models.area_forecast_run import (
    forecast_replay,
    load_forecast_model,
    train_forecast,
)
from vision.area_forecast_data import audit_cohort, make_samples
from vision.learned import LearnedObservationConfig, LearnedSeedTracker
from vision.optical_sequence import export_optical_sequence
from vision.replay import replay
from vision.training_data import save_json
from vision.video import file_sha256


def fixture_cohort(root):
    original = root / "original"
    cohort = root / "cohort"
    original.mkdir()
    cohort.mkdir()
    rows = []
    images = []
    all_masks = {}
    for group, split in enumerate(["train", "train", "validation", "test"]):
        family = f"fixture_{group}"
        masks = []
        for index in range(3):
            half = 10 + group + index
            mask = np.zeros((64, 64), np.uint8)
            mask[32 - half : 32 + half, 32 - half : 32 + half] = 255
            predicted = np.zeros_like(mask)
            predicted[31 - half : 33 + half, 31 - half : 33 + half] = 255
            masks.append(predicted)
            image = np.full((64, 64, 3), 230, np.uint8)
            image[mask > 0] = 40
            mask_path = f"{family}_{index}_mask.png"
            image_path = f"{family}_{index}.png"
            cv2.imwrite(str(original / mask_path), mask)
            cv2.imwrite(str(original / image_path), image)
            images.extend([mask_path, image_path])
            rows.append(
                dict(
                    scene_id=family,
                    family_id=family,
                    split=split,
                    frame_index=index,
                    sample_id=f"{family}:{index}",
                    image_path=image_path,
                    synthetic_time=index,
                    objects=[
                        dict(
                            class_id=0,
                            id="seed",
                            label_eligible=True,
                            mask_path=mask_path,
                        )
                    ],
                )
            )
        all_masks[family] = masks
    (original / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    save_json(
        original / "manifest.json",
        dict(
            artifact_sha256={
                name: file_sha256(original / name)
                for name in [*images, "samples.jsonl"]
            }
        ),
    )
    for name in ["manifest.json", "samples.jsonl"]:
        shutil.copyfile(
            original / name,
            cohort
            / (
                "optical_manifest.json"
                if name == "manifest.json"
                else "optical_samples.jsonl"
            ),
        )
    config = AreaForecastConfig()
    save_json(cohort / "forecast_config.json", asdict(config))
    samples = []
    cases = []
    observation_config = LearnedObservationConfig((32, 32))
    for group, split in enumerate(["train", "train", "validation"]):
        family = f"fixture_{group}"
        sequence = cohort / "sequences" / family
        with patch(
            "vision.optical_sequence.audit_optical", return_value=dict(status="pass")
        ):
            export_optical_sequence(original, family, sequence)

        def factory(output):
            current = 0

            def predict(image):
                nonlocal current
                mask = all_masks[family][current]
                current += 1
                return [dict(class_id=0, mask=mask, score=0.9)]

            observer = LearnedSeedTracker(observation_config, predict)
            observer.info = dict(
                method="fixture_supplied_masks_not_neural_model",
                model_accuracy_claim=False,
            )
            return observer

        folder = cohort / "observations" / family
        replay(
            sequence / "sequence.avi",
            observation_config,
            folder,
            run_id=family,
            observer_factory=factory,
            source_info=json.loads((sequence / "source_info.json").read_text()),
            frame_map=sequence / "frame_map.jsonl",
        )
        batch = make_samples(
            folder, config, original, {row["sample_id"]: row for row in rows}, cohort
        )
        samples.extend(batch)
        cases.append(
            dict(
                scene_id=family,
                family_id=family,
                split=split,
                replay_path=str(folder.relative_to(cohort)),
                replay_manifest_sha256=file_sha256(folder / "manifest.json"),
            )
        )
    (cohort / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    save_json(
        cohort / "manifest.json",
        dict(
            schema="crystalab_area_forecast_cohort_v1",
            status="complete",
            source_manifest_sha256=file_sha256(original / "manifest.json"),
            samples=len(samples),
            cases=cases,
            files={
                str(p.relative_to(cohort)): file_sha256(p)
                for p in cohort.rglob("*")
                if p.is_file()
            },
        ),
    )
    return cohort


class AreaForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.cohort = fixture_cohort(cls.root)
        cls.fit = cls.root / "fit"
        cls.report = train_forecast(cls.cohort, cls.fit)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_inputs_proxy_labels_warmup_and_frozen_groups_audit(self):
        audit = audit_cohort(self.cohort)
        self.assertEqual(audit["samples"], 6)
        self.assertEqual(audit["families_by_split"], dict(train=2, validation=1))
        self.assertEqual(audit["rate_ready_by_split"], dict(train=2, validation=1))
        self.assertEqual(
            self.report["calibration"]["training_family_ids"],
            ["fixture_0", "fixture_1"],
        )
        model = load_forecast_model(self.fit / "model.json")
        self.assertFalse(model["can_authorize_actuation"])
        self.assertFalse(model["action_conditioned"])
        val = self.report["metrics"]["validation"]
        self.assertEqual(val["persistence"]["prediction_coverage"], 1)
        self.assertEqual(val["recent_rate"]["prediction_coverage"], 0.5)
        self.assertEqual(val["recent_rate"]["common_history_support"]["samples"], 1)

    def test_replay_preserves_source_clock_and_does_not_score_unreviewed_data(self):
        output = self.root / "forecast"
        result = forecast_replay(
            self.cohort / "observations/fixture_2", self.fit / "model.json", output
        )
        self.assertEqual(result["predictions_by_method"]["recent_rate"], 2)
        self.assertEqual(result["unavailable_reasons"], dict(insufficient_history=1))
        rows = [
            json.loads(s) for s in (output / "forecasts.jsonl").read_text().splitlines()
        ]
        self.assertEqual(rows[1]["target_time_source_s"], 0.5)
        self.assertEqual(rows[1]["source_frame"]["source_sample_id"], "fixture_2:1")
        self.assertTrue(
            all(
                r["requires_operator_review"]
                and not r["can_authorize_actuation"]
                and r["physical_horizon_s"] is None
                for r in rows
            )
        )
        self.assertIsNone(result["real_accuracy"])
        self.assertEqual(result["numerical_actuator_commands"], 0)

    def test_rehashed_wrong_future_mask_cannot_become_an_accepted_label(self):
        edited = self.root / "edited"
        shutil.copytree(self.cohort, edited)
        path = edited / "samples.jsonl"
        samples = [json.loads(s) for s in path.read_text().splitlines()]
        samples[1]["label_refs"]["future"] = samples[1]["label_refs"]["current"].copy()
        path.write_text("".join(json.dumps(s) + "\n" for s in samples))
        m = json.loads((edited / "manifest.json").read_text())
        m["files"]["samples.jsonl"] = file_sha256(path)
        save_json(edited / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "original source sample"):
            audit_cohort(edited)

    def test_model_tampering_and_existing_output_are_rejected(self):
        edited = self.root / "edited_fit"
        shutil.copytree(self.fit, edited)
        model = json.loads((edited / "model.json").read_text())
        model["calibration"]["coefficient"] = 999
        save_json(edited / "model.json", model)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            load_forecast_model(edited / "model.json")
        with self.assertRaises(FileExistsError):
            train_forecast(self.cohort, self.fit)

    def test_rehashed_input_horizon_and_removed_warmup_are_rejected(self):
        for name, mutate in [
            (
                "horizon",
                lambda rows: rows[1]["trend"]["features"].update(horizon_source_s=2.0),
            ),
            ("missing_warmup", lambda rows: rows.pop(0)),
        ]:
            edited = self.root / name
            shutil.copytree(self.cohort, edited)
            path = edited / "samples.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            mutate(rows)
            path.write_text("".join(json.dumps(s) + "\n" for s in rows))
            m = json.loads((edited / "manifest.json").read_text())
            m["files"]["samples.jsonl"] = file_sha256(path)
            m["samples"] = len(rows)
            save_json(edited / "manifest.json", m)
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit_cohort(edited)
