# FILE PURPOSE: Check weak-task separation, common evaluation membership and explicit training budgets.
# The synthetic audit is stubbed only for tiny fixtures; production reconstruction is exercised on real exports.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from test_source_shapes import fixture_source
from models.segmentation import training_options
from models.observation import YOLOObserver
from vision.source_shapes import export_source_shapes
from vision.source_shape_audit import audit_projected_shapes
from vision.silhouette_training import prepare_silhouette, audit_silhouette
from vision.training_data import audit_segmentation


def synthetic_fixture(root):
    source = root / "optical"
    source.mkdir()
    dataset = root / "synthetic"
    dataset.mkdir()
    records = []
    for i, split in enumerate(["train", "validation", "test"]):
        im = np.full((64, 64, 3), 50 + i, np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[15:45, 15:45] = 255
        cv2.imwrite(str(dataset / f"{i}.png"), im)
        cv2.imwrite(str(source / f"{i}.png"), mask)
        records.append(
            dict(
                sample_id=f"s{i}",
                split=split,
                family_key=f"synthetic:{i}",
                image_path=f"{i}.png",
                source_index=0,
                original=dict(
                    objects=[
                        dict(class_id=0, label_eligible=True, mask_path=f"{i}.png")
                    ]
                ),
            )
        )
    (dataset / "manifest.json").write_text(
        json.dumps(dict(sources=[dict(path=str(source))], records=records))
    )
    return dataset


class SilhouetteTaskTests(unittest.TestCase):
    def test_comparisons_share_validation_and_keep_source_roles_and_final_test(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            index, policy = fixture_source(root)
            projection = root / "projected"
            export_source_shapes(index, policy, projection)
            self.assertEqual(
                audit_projected_shapes(projection)["source_fits_reconstructed"], 2
            )
            synthetic = synthetic_fixture(root)
            members = []
            with patch("vision.silhouette_training.audit_segmentation"):
                for mode, training in [
                    ("synthetic_only", {"synthetic": 1}),
                    ("source_projection_only", {"real": 1}),
                    ("mixed", {"real": 1, "synthetic": 1}),
                ]:
                    output = root / mode
                    result = prepare_silhouette(projection, synthetic, mode, output)
                    self.assertEqual(result["training_by_evidence"], training)
                    self.assertEqual(
                        result["validation_by_evidence"], {"real": 1, "synthetic": 1}
                    )
                    self.assertFalse(result["pixel_reviewed"])
                    rows = json.loads((output / "records.json").read_text())
                    self.assertTrue(all(r["split"] != "test" for r in rows))
                    members.append(
                        {r["sample_id"] for r in rows if r["split"] == "validation"}
                    )
                    with self.assertRaisesRegex(ValueError, "Unsupported segmentation"):
                        audit_segmentation(output)
                self.assertTrue(all(v == members[0] for v in members))
                path = root / "mixed/images/train/extra.png"
                cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
                with self.assertRaisesRegex(ValueError, "Unindexed"):
                    audit_silhouette(root / "mixed")

    def test_auxiliary_task_is_rejected_before_loading_a_pitch_observer(self):
        with (
            patch(
                "models.observation.load_run_metadata",
                return_value=(
                    {"dataset_task": "source_fit_silhouette"},
                    Path("weights"),
                ),
            ),
            patch("models.observation.local_yolo") as loader,
        ):
            with self.assertRaisesRegex(ValueError, "pitch instance/event classes"):
                YOLOObserver(None, "run", "settings")
            loader.assert_not_called()

    def test_equal_update_schedule_requires_explicit_finite_learning_parameters(self):
        config = dict(
            schema="crystalab_yolov8_seg_train_v1",
            purpose="grouped_experiment",
            epochs=6,
            imgsz=256,
            batch=4,
            threads=2,
            seed=123,
            device="cpu",
            learning_rate=0.001,
            warmup_epochs=0,
            lrf=1,
        )
        options = training_options(config)
        self.assertEqual(options["warmup_epochs"], 0)
        self.assertEqual(options["lrf"], 1)
        for change in [
            dict(warmup_epochs=-1),
            dict(warmup_epochs=True),
            dict(lrf=0),
            dict(lrf=float("nan")),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                training_options(dict(config, **change))
