# FILE PURPOSE: Adapt a verified local YOLOv8-seg model to the timestamped observation interface.
# Raw per-instance masks and uncalibrated scores are retained separately from the
# associated seed component. This adapter never issues controller commands.
"""Local CPU inference with pinned model identity and explicit real-domain limits."""

from pathlib import Path

import numpy as np

from models.segmentation import load_run_metadata, local_yolo
from vision.learned import LearnedSeedTracker
from vision.training_data import CLASSES
from vision.video import file_sha256


class YOLOObserver(LearnedSeedTracker):
    def __init__(self, config, run, settings_dir):
        self.run = Path(run).resolve()
        manifest, self.weights = load_run_metadata(self.run)
        self.manifest_sha256 = file_sha256(self.run / "manifest.json")
        YOLO, versions = local_yolo(settings_dir, manifest["runtime"])
        import torch

        torch.set_num_threads(config.threads)
        self.model = YOLO(str(self.weights), task="segment")
        if [self.model.names[i] for i in range(len(CLASSES))] != CLASSES:
            raise ValueError(
                "Trained model class map differs from observation contract"
            )
        self.image_size = manifest["config"]["imgsz"]
        self.info = dict(
            method="yolov8n_seg_seed_component_overlap_v1",
            run_path=str(self.run),
            training_manifest_sha256=self.manifest_sha256,
            weights_sha256=manifest["best_weights_sha256"],
            training_manifest=manifest,
            inference=dict(
                device="cpu",
                imgsz=self.image_size,
                threads=config.threads,
                confidence=config.confidence,
                mask_threshold=0.5,
                retina_masks=True,
            ),
            packages=versions,
            classes=CLASSES,
            score_semantics="Uncalibrated detector score; not event probability or physical confidence",
            role_semantics="Explicit seed point and overlap establish seed role across configured crystal classes; raw per-frame labels are not object identity",
            scope="Experimental learned observation; real domain/event performance unvalidated",
            quality_or_healthy_state_available=False,
        )
        root = Path(__file__).resolve().parents[2]
        self.source_files = [
            Path(__file__),
            root / "src/models/segmentation.py",
            root / "src/vision/learned.py",
            root / "src/vision/training_data.py",
        ]
        super().__init__(config, self.predict_instances)

    def predict_instances(self, image):
        result = self.model.predict(
            image,
            device="cpu",
            imgsz=self.image_size,
            conf=self.config.confidence,
            retina_masks=True,
            verbose=False,
            save=False,
        )[0]
        if result.masks is None:
            if result.boxes is not None and len(result.boxes):
                raise ValueError("Segmentation predictions omitted their masks")
            return []
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        if (
            len(masks) != len(classes)
            or len(masks) != len(scores)
            or masks.shape[1:] != image.shape[:2]
            or not np.isfinite(masks).all()
        ):
            raise ValueError("Prediction dimensions/values differ from original frame")
        return [
            dict(
                class_id=int(class_id),
                class_name=CLASSES[class_id],
                score=float(score),
                mask=(mask > 0.5).astype(np.uint8) * 255,
                score_is_calibrated=False,
            )
            for mask, class_id, score in zip(masks, classes, scores)
        ]

    def verify(self):
        manifest, weights = load_run_metadata(self.run)
        if (
            file_sha256(self.run / "manifest.json") != self.manifest_sha256
            or file_sha256(weights) != self.info["weights_sha256"]
        ):
            raise ValueError("Model identity changed during observation")
