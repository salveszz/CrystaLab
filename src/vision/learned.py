# FILE PURPOSE: Associate learned seed masks over time without treating scores as validated confidence.
# Initialization uses an explicit seed point. Ambiguous, clipped, missing and
# long-gap observations stay invalid; raw candidate masks remain available.
"""Model-independent learned-mask tracking and projected pixel measurements."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from vision.baseline import BaselineConfig


@dataclass(frozen=True)
class LearnedObservationConfig:
    seed_point: tuple[int, int]
    schema: str = "crystalab_learned_observation_v1"
    roi: tuple[int, int, int, int] | None = None
    confidence: float = 0.25
    seed_candidate_class_ids: tuple[int, ...] = (0, 1)
    compatible_seed_iou: float = 0.9
    threads: int = 2
    minimum_area_px: int = 32
    maximum_area_fraction: float = 0.8
    minimum_contrast: float = 5.0
    minimum_focus: float = 0.0
    minimum_iou: float = 0.1
    ambiguity_margin: float = 0.05
    maximum_gap_s: float = 2.0

    def __post_init__(self):
        if self.schema != "crystalab_learned_observation_v1":
            raise ValueError("Unsupported learned observation contract")
        if (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0 < self.confidence < 1
        ):
            raise ValueError(
                "Detection threshold must be finite and between zero and one"
            )
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("CPU thread count must be a positive integer")
        if (
            isinstance(self.compatible_seed_iou, bool)
            or not math.isfinite(self.compatible_seed_iou)
            or not 0 < self.compatible_seed_iou <= 1
        ):
            raise ValueError("Compatible seed-mask IoU must be finite and in (0,1]")
        if (
            not self.seed_candidate_class_ids
            or len(set(self.seed_candidate_class_ids))
            != len(self.seed_candidate_class_ids)
            or any(
                type(value) is not int or value not in {0, 1}
                for value in self.seed_candidate_class_ids
            )
        ):
            raise ValueError(
                "Seed association accepts distinct crystal class IDs zero/one only"
            )
        BaselineConfig(
            **{
                name: getattr(self, name)
                for name in (
                    "seed_point",
                    "roi",
                    "minimum_area_px",
                    "maximum_area_fraction",
                    "minimum_contrast",
                    "minimum_focus",
                    "minimum_iou",
                    "ambiguity_margin",
                    "maximum_gap_s",
                )
            }
        )


class LearnedSeedTracker:
    """Associate crystal components by explicit seed role, never largest-object fallback.

    Predictor returns class_id, class_name, score and an original-size binary mask
    for each raw instance. `instances` is replaced on every processed frame.
    """

    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        self.previous_mask = self.last_valid_time = self.last_time = self.shape = None
        self.initialized = False
        self.previous_raw_class = None
        self.instances = []

    def measure(self, image, timestamp):
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("Measurement timestamps must strictly increase")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an original-size uint8 BGR image")
        if self.shape is not None and self.shape != image.shape[:2]:
            raise ValueError("Frame dimensions changed during a tracked run")
        self.shape = image.shape[:2]
        self.last_time = timestamp
        self.instances = []
        c = self.config
        height, width = self.shape
        x, y, w, h = c.roi or (0, 0, width, height)
        sx, sy = c.seed_point
        if x + w > width or y + h > height or not (x <= sx < x + w and y <= sy < y + h):
            raise ValueError("ROI/seed point lies outside frame or observation ROI")
        gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        contrast = float(gray.max()) - float(gray.min())
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.instances = list(self.predictor(image))
        for instance in self.instances:
            mask = instance["mask"]
            if (
                type(instance["class_id"]) is not int
                or not 0 <= instance["class_id"] <= 2
                or not math.isfinite(instance["score"])
                or not 0 <= instance["score"] <= 1
                or mask.dtype != np.uint8
                or mask.shape != self.shape
                or not set(np.unique(mask)) <= {0, 255}
            ):
                raise ValueError(
                    "Invalid learned instance class, score or original-size binary mask"
                )
        record = dict(
            valid=False,
            status="unobserved",
            reason=None,
            projected_area_px=None,
            width_px=None,
            height_px=None,
            centroid_px=None,
            bbox_px=None,
            overlap_iou=None,
            contrast=contrast,
            focus=focus,
            quality_target_available=False,
            optical_quality=None,
            method="yolov8n_seg_seed_component_overlap_v1",
            seed_score=None,
            confidence_is_calibrated=False,
            seed_instance_index=None,
            seed_raw_class_id=None,
            seed_role_basis="explicit_initial_point_then_mask_overlap; raw visual label is retained",
            seed_alternatives=[],
            projected_area_range_px=None,
            area_range_semantics="range of compatible prediction hypotheses; not a calibrated uncertainty interval",
            requires_inspection=True,
            inspection_reason=None,
            inspection_reasons=[],
            seed_component_id=None,
            excluded_seed_instance_pixels=None,
            mask_semantics="one connected component associated by explicit point or prior mask",
            target_absence_is_healthy=False,
            tracking_time_basis="source_video_time_s",
        )

        def invalid(reason):
            record.update(status="invalid", reason=reason)
            record.update(inspection_reason=reason, inspection_reasons=[reason])
            return record, None

        if self.initialized and timestamp - self.last_valid_time > c.maximum_gap_s:
            return invalid("identity_requires_reinitialization")
        if contrast < c.minimum_contrast:
            return invalid("low_contrast")
        if focus < c.minimum_focus:
            return invalid("out_of_focus")
        candidates = []
        for instance_index, instance in enumerate(self.instances):
            if (
                instance["class_id"] not in c.seed_candidate_class_ids
                or instance["score"] < c.confidence
            ):
                continue
            count, components, stats, centroids = cv2.connectedComponentsWithStats(
                instance["mask"]
            )
            for component in range(1, count):
                bx, by, bw, bh, area = (int(v) for v in stats[component])
                if (
                    area < c.minimum_area_px
                    or area > c.maximum_area_fraction * w * h
                    or bx <= x
                    or by <= y
                    or bx + bw >= x + w
                    or by + bh >= y + h
                ):
                    continue
                mask = (components == component).astype(np.uint8) * 255
                if not self.initialized:
                    if mask[sy, sx] == 0:
                        continue
                    overlap = None
                else:
                    intersection = np.count_nonzero(
                        (mask > 0) & (self.previous_mask > 0)
                    )
                    union = np.count_nonzero((mask > 0) | (self.previous_mask > 0))
                    overlap = float(intersection / union)
                    if overlap < c.minimum_iou:
                        continue
                candidates.append(
                    dict(
                        mask=mask,
                        overlap=overlap,
                        area=area,
                        bounds=[bx, by, bw, bh],
                        centroid=centroids[component].tolist(),
                        instance_index=instance_index,
                        component_id=component,
                        score=float(instance["score"]),
                        raw_class_id=instance["class_id"],
                        excluded_pixels=int(np.count_nonzero(instance["mask"])) - area,
                    )
                )
        if not candidates:
            return invalid("seed_not_identified_or_clipped")
        candidates.sort(
            key=lambda row: row["overlap"] if self.initialized else row["score"],
            reverse=True,
        )
        selected = candidates[0]
        alternatives = []
        distinct = []
        for candidate in candidates[1:]:
            a, b = selected["mask"] > 0, candidate["mask"] > 0
            overlap = float(np.count_nonzero(a & b) / np.count_nonzero(a | b))
            if (
                candidate["raw_class_id"] != selected["raw_class_id"]
                and overlap >= c.compatible_seed_iou
            ):
                alternatives.append(
                    dict(
                        instance_index=candidate["instance_index"],
                        component_id=candidate["component_id"],
                        raw_class_id=candidate["raw_class_id"],
                        score=candidate["score"],
                        projected_area_px=candidate["area"],
                        mask_iou_to_selected=overlap,
                    )
                )
            else:
                distinct.append(candidate)
        if distinct and (
            not self.initialized
            or selected["overlap"] - distinct[0]["overlap"] <= c.ambiguity_margin
        ):
            return invalid("ambiguous_seed_identity")
        inspection_reasons = []
        if alternatives:
            inspection_reasons.append("compatible_cross_class_seed_hypotheses")
        if (
            self.previous_raw_class is not None
            and self.previous_raw_class != selected["raw_class_id"]
        ):
            inspection_reasons.append("raw_seed_class_transition")
        record.update(
            valid=True,
            status="tracked_with_alternatives" if alternatives else "tracked",
            projected_area_px=selected["area"],
            width_px=selected["bounds"][2],
            height_px=selected["bounds"][3],
            bbox_px=selected["bounds"],
            centroid_px=selected["centroid"],
            overlap_iou=selected["overlap"],
            seed_score=selected["score"],
            seed_instance_index=selected["instance_index"],
            seed_raw_class_id=selected["raw_class_id"],
            seed_component_id=selected["component_id"],
            excluded_seed_instance_pixels=selected["excluded_pixels"],
            seed_alternatives=alternatives,
            projected_area_range_px=[
                min(
                    [selected["area"]] + [a["projected_area_px"] for a in alternatives]
                ),
                max(
                    [selected["area"]] + [a["projected_area_px"] for a in alternatives]
                ),
            ],
            requires_inspection=bool(inspection_reasons),
            inspection_reasons=inspection_reasons,
            inspection_reason=";".join(inspection_reasons)
            if inspection_reasons
            else None,
        )
        self.previous_mask = selected["mask"]
        self.last_valid_time = timestamp
        self.initialized = True
        self.previous_raw_class = selected["raw_class_id"]
        return record, selected["mask"]
