# FILE PURPOSE: Measure and track a seeded projected silhouette using OpenCV.
# Seed initialization is explicit; ambiguous, clipped and lost observations stay
# invalid. Scores describe geometric overlap, never calibrated model confidence.
"""Inspectably limited threshold/contour baseline for a fixed camera view."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineConfig:
    seed_point: tuple[int, int]
    roi: tuple[int, int, int, int] | None = None
    polarity: str = "dark"
    threshold: int | None = None
    minimum_area_px: int = 32
    maximum_area_fraction: float = 0.8
    morphology_size: int = 3
    minimum_contrast: float = 5.0
    minimum_focus: float = 0.0
    minimum_iou: float = 0.1
    ambiguity_margin: float = 0.05
    maximum_gap_s: float = 2.0

    def __post_init__(self):
        if len(self.seed_point) != 2 or any(
            type(v) is not int or v < 0 for v in self.seed_point
        ):
            raise ValueError("Seed point needs two nonnegative integer coordinates")
        if self.roi is not None and (
            len(self.roi) != 4
            or any(type(v) is not int or v < 0 for v in self.roi)
            or min(self.roi[2:]) <= 0
        ):
            raise ValueError("ROI needs nonnegative x/y and positive width/height")
        if self.polarity not in {"dark", "bright"}:
            raise ValueError("Polarity must be dark or bright")
        if self.threshold is not None and (
            type(self.threshold) is not int or not 0 <= self.threshold <= 255
        ):
            raise ValueError("Threshold must be an integer in [0,255] or null for Otsu")
        for key in ("minimum_area_px", "morphology_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if self.morphology_size % 2 == 0:
            raise ValueError("Morphology size must be odd")
        for key in (
            "maximum_area_fraction",
            "minimum_contrast",
            "minimum_focus",
            "minimum_iou",
            "ambiguity_margin",
            "maximum_gap_s",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if not 0 < self.maximum_area_fraction < 1:
            raise ValueError("Maximum area fraction must be in (0,1)")
        if not 0 < self.minimum_iou <= 1 or self.ambiguity_margin > 1:
            raise ValueError("Invalid overlap/ambiguity threshold")


class SeedTracker:
    """Initialize at the configured seed point and follow mask overlap.

    After a long observation gap identity requires manual reinitialization.
    A missing seed never falls back to the largest unrelated object.
    """

    def __init__(self, config):
        self.config = config
        self.previous_mask = self.last_valid_time = self.last_time = None
        self.initialized = False
        self.shape = None

    def measure(self, image, timestamp):
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or (self.last_time is not None and timestamp <= self.last_time)
        ):
            raise ValueError("Measurement timestamps must strictly increase")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a uint8 BGR image")
        if self.shape is not None and self.shape != image.shape[:2]:
            raise ValueError("Frame dimensions changed during a tracked run")
        self.shape = image.shape[:2]
        self.last_time = timestamp
        c = self.config
        height, width = self.shape
        x, y, w, h = c.roi or (0, 0, width, height)
        sx, sy = c.seed_point
        if x + w > width or y + h > height or not (x <= sx < x + w and y <= sy < y + h):
            raise ValueError("ROI/seed point lies outside the frame or observation ROI")
        gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        contrast = float(gray.max()) - float(gray.min())
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
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
            method="opencv_threshold_seed_overlap_v1",
        )

        def invalid(reason):
            record.update(status="invalid", reason=reason)
            return record, None

        if self.initialized and timestamp - self.last_valid_time > c.maximum_gap_s:
            return invalid("identity_requires_reinitialization")
        if contrast < c.minimum_contrast:
            return invalid("low_contrast")
        if focus < c.minimum_focus:
            return invalid("out_of_focus")
        mode = cv2.THRESH_BINARY_INV if c.polarity == "dark" else cv2.THRESH_BINARY
        if c.threshold is None:
            mode |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(gray, c.threshold or 0, 255, mode)
        if c.morphology_size > 1:
            kernel = np.ones((c.morphology_size, c.morphology_size), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            local = np.zeros_like(gray)
            cv2.drawContours(local, [contour], -1, 255, cv2.FILLED)
            area = int(np.count_nonzero(local))
            bx, by, bw, bh = cv2.boundingRect(contour)
            if (
                area < c.minimum_area_px
                or area > c.maximum_area_fraction * w * h
                or bx == 0
                or by == 0
                or bx + bw == w
                or by + bh == h
            ):
                continue
            mask = np.zeros((height, width), np.uint8)
            mask[y : y + h, x : x + w] = local
            if not self.initialized:
                if mask[sy, sx] == 0:
                    continue
                score = None
            else:
                intersection = np.count_nonzero((mask > 0) & (self.previous_mask > 0))
                union = np.count_nonzero((mask > 0) | (self.previous_mask > 0))
                score = float(intersection / union)
                if score < c.minimum_iou:
                    continue
            candidates.append((score, mask, (x + bx, y + by, bw, bh), area))
        if not candidates:
            return invalid("seed_not_identified_or_clipped")
        candidates.sort(key=lambda row: row[0] or 0.0, reverse=True)
        if len(candidates) > 1 and (
            not self.initialized
            or candidates[0][0] - candidates[1][0] <= c.ambiguity_margin
        ):
            return invalid("ambiguous_seed_identity")
        score, mask, bounds, area = candidates[0]
        ys, xs = np.nonzero(mask)
        record.update(
            valid=True,
            status="tracked",
            projected_area_px=area,
            width_px=bounds[2],
            height_px=bounds[3],
            bbox_px=list(bounds),
            centroid_px=[float(xs.mean()), float(ys.mean())],
            overlap_iou=score,
        )
        self.previous_mask, self.last_valid_time = mask, timestamp
        self.initialized = True
        return record, mask
