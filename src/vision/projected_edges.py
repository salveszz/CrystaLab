# FILE PURPOSE: Measure anchored projected edge advance only with explicit view/reference support.
# A planar similarity fit does not recover crystal pose or certify attachment.
# Directional mask supports are not physical facets; proposals contain no commands.
"""Causal image-reference diagnostic with explicit identifiability preconditions."""

from collections import deque
import hashlib
import json
import math

import cv2
import numpy as np

# Numerical equality only, far below raster uncertainty; not a real-data budget.
LENGTH_ROUNDOFF_PX = 1e-9
FRACTION_ROUNDOFF = 1e-12


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def similarity_registration(current, reference):
    """Least-squares orientation-preserving 2D similarity, current -> reference."""
    current, reference = np.asarray(current, float), np.asarray(reference, float)
    if current.shape != (3, 2) or reference.shape != (3, 2) or not np.isfinite([current, reference]).all():
        raise ValueError("Three finite corresponding image-reference points are required")
    x, y = current.T
    system = np.zeros((6, 4))
    system[0::2] = np.column_stack([x, -y, np.ones(3), np.zeros(3)])
    system[1::2] = np.column_stack([y, x, np.zeros(3), np.ones(3)])
    parameters, _, rank, _ = np.linalg.lstsq(system, reference.ravel(), rcond=None)
    if rank != 4:
        raise ValueError("Reference points do not determine a similarity")
    a, b, tx, ty = parameters
    matrix = np.array([[a, -b, tx], [b, a, ty]])
    residual = current @ matrix[:, :2].T + matrix[:, 2] - reference
    return matrix, float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def marker_centroids(image, colors, config):
    result = []
    for color in colors:
        mask = (np.max(abs(image.astype(np.int16) - np.asarray(color)), axis=2) <= config["marker_color_tolerance"]).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= config["marker_minimum_pixels"]]
        if len(candidates) != 1 or stats[candidates[0], cv2.CC_STAT_AREA] > config["marker_maximum_pixels"]:
            return None
        result.append(centers[candidates[0]])
    return np.asarray(result)


class ProjectedEdgeTracker:
    def __init__(self, config, marker_colors, *, run_id, clock_id, evidence_kind):
        self.config = dict(config)
        if evidence_kind not in {"fixture", "synthetic", "real"} or not run_id or not clock_id:
            raise ValueError("Declare run, clock and evidence kind")
        self.run_id, self.clock_id, self.evidence_kind = run_id, clock_id, evidence_kind
        self.colors = np.asarray(marker_colors)
        if (self.colors.shape != (3, 3) or not np.isfinite(self.colors).all() or
            np.any(self.colors != np.rint(self.colors)) or np.any(self.colors < 0) or np.any(self.colors > 255)):
            raise ValueError("Declare three distinct image marker colors")
        if len({tuple(v) for v in self.colors}) != 3:
            raise ValueError("Reference marker colors must be distinct")
        weights = np.asarray(config["anchor_barycentric_weights"], float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or abs(weights.sum() - 1) > 1e-12:
            raise ValueError("Reference anchor weights must sum to one")
        for key in ("maximum_registration_RMS_px", "minimum_reference_triangle_area_px2", "horizon_source_s",
                    "maximum_gap_source_s", "minimum_total_advance_px", "maximum_negative_advance_px", "imbalance_difference_px"):
            if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"Invalid projected-edge setting: {key}")
        if not 0 < config["imbalance_fraction"] <= 1 or type(config["persistence_windows"]) is not int or config["persistence_windows"] < 1:
            raise ValueError("Invalid projected-edge decision settings")
        for key in ("marker_minimum_pixels", "marker_maximum_pixels", "minimum_mask_pixels"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError("Marker/mask pixel budgets must be positive integers")
        limits = config["registration_scale_range"]
        if (len(limits) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in limits) or
            not 0 < limits[0] <= limits[1] or config["marker_maximum_pixels"] < config["marker_minimum_pixels"] or
            type(config["marker_color_tolerance"]) is not int or not 0 <= config["marker_color_tolerance"] <= 255):
            raise ValueError("Invalid registration/marker limits")
        self.weights = weights
        self.last_time = self.last_frame = self.shape = self.signature = self.reference = None
        self.history = deque()
        self.streak = [0, 0]
        self.active = [False, False]
        self.proposals = []

    def _clear_history(self):
        self.history.clear()
        self.streak, self.active = [0, 0], [False, False]

    def measure(self, image, mask, *, timestamp, frame_index, context=None):
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0 or
            (self.last_time is not None and timestamp <= self.last_time) or type(frame_index) is not int or frame_index < 0 or
            (self.last_frame is not None and frame_index <= self.last_frame)):
            raise ValueError("Source timestamps and original frame indices must increase")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected native uint8 BGR image")
        if self.shape is not None and self.shape != image.shape[:2]:
            raise ValueError("Image dimensions changed")
        self.shape = image.shape[:2]
        gap = self.last_time is not None and timestamp - self.last_time > self.config["maximum_gap_source_s"]
        self.last_time, self.last_frame = timestamp, frame_index
        row = dict(run_id=self.run_id, clock_id=self.clock_id, evidence_kind=self.evidence_kind,
            frame_index=frame_index, source_time_s=timestamp, image_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            context_sha256=None if context is None else value_hash(context), context=context,
            valid=False, reason=None, projected_supports_px=None, anchor_reference_px=None,
            registration=None, comparison=None, proposals=[], physical_face_rate_available=False,
            physical_commands=[])

        def unavailable(reason):
            self._clear_history()
            row["reason"] = reason
            return row

        if context is None:
            return unavailable("missing_external_view_reference")
        if (context.get("evidence_kind") != self.evidence_kind or context.get("run_id") != self.run_id or
            context.get("frame_index") != frame_index or not context.get("provenance")):
            raise ValueError("View assertions must bind the current source frame and evidence kind")
        observed, available = context.get("observed_at_source_s"), context.get("available_at_source_s")
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (observed, available)) or
            not 0 <= observed <= available <= timestamp):
            raise ValueError("View assertions must be available on the current source clock")
        for key, reason in [("attachment_supported", "attachment_not_established"),
                            ("pose_supported", "object_pose_not_comparable"),
                            ("full_visibility_supported", "incomplete_silhouette_view")]:
            if context.get(key) is not True:
                return unavailable(reason)
        if not context.get("reference_id") or not context.get("pose_id"):
            return unavailable("missing_reference_or_pose_identity")
        if gap:
            self._clear_history()
        signature = (context["reference_id"], context["pose_id"])
        if signature != self.signature:
            self._clear_history()
            self.reference = None
            self.signature = signature
        markers = marker_centroids(image, self.colors, self.config)
        if markers is None:
            return unavailable("reference_markers_missing_or_ambiguous")
        triangle = np.linalg.det(np.stack([markers[1] - markers[0], markers[2] - markers[0]])) / 2
        if abs(triangle) < self.config["minimum_reference_triangle_area_px2"]:
            return unavailable("reference_marker_geometry_degenerate")
        if self.reference is None:
            self.reference = markers.copy()
        matrix, rms = similarity_registration(markers, self.reference)
        scale = float(np.linalg.norm(matrix[0, :2]))
        row["registration"] = dict(current_to_reference=matrix.tolist(), RMS_px=rms, scale=scale,
                                   marker_centroids_px=markers.tolist(), reference_points_px=self.reference.tolist())
        if rms > self.config["maximum_registration_RMS_px"]:
            return unavailable("reference_is_not_a_supported_similarity")
        if not self.config["registration_scale_range"][0] <= scale <= self.config["registration_scale_range"][1]:
            return unavailable("reference_scale_outside_declared_range")
        if mask is None:
            return unavailable("seed_mask_unavailable")
        if mask.dtype != np.uint8 or mask.shape != self.shape or not set(np.unique(mask)) <= {0, 255}:
            raise ValueError("Expected native binary seed mask")
        row["mask_pixels_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
        foreground = mask > 0
        if np.count_nonzero(foreground) < self.config["minimum_mask_pixels"]:
            return unavailable("seed_mask_too_small")
        if np.any(foreground[[0, -1], :]) or np.any(foreground[:, [0, -1]]):
            return unavailable("seed_silhouette_clipped")
        if cv2.connectedComponents(foreground.astype(np.uint8))[0] != 2:
            return unavailable("seed_mask_identity_not_unique")
        ys, xs = np.nonzero(foreground)
        points = np.column_stack([xs, ys]) @ matrix[:, :2].T + matrix[:, 2]
        anchor = self.weights @ self.reference
        current_anchor = self.weights @ markers
        ax, ay = np.rint(current_anchor).astype(int)
        if not (0 <= ay < mask.shape[0] and 0 <= ax < mask.shape[1] and foreground[ay, ax]):
            return unavailable("independent_anchor_outside_seed")
        relative = points - anchor
        supports = np.array([relative[:, 0].max(), -relative[:, 0].min(),
                             relative[:, 1].max(), -relative[:, 1].min()])
        row.update(valid=True, reason="insufficient_comparable_history", projected_supports_px=supports.tolist(),
                   anchor_reference_px=anchor.tolist())
        cutoff = timestamp - self.config["horizon_source_s"]
        candidates = [entry for entry in self.history if entry["source_time_s"] <= cutoff + 1e-10]
        if candidates:
            previous = candidates[-1]
            delta = supports - np.asarray(previous["projected_supports_px"])
            checks = []
            for axis in range(2):
                positive, negative = delta[2 * axis:2 * axis + 2]
                total, difference = float(positive + negative), float(abs(positive - negative))
                available = (min(positive, negative) + LENGTH_ROUNDOFF_PX >= -self.config["maximum_negative_advance_px"] and
                             total + LENGTH_ROUNDOFF_PX >= self.config["minimum_total_advance_px"])
                fraction = difference / total if available else None
                triggered = bool(available and difference + LENGTH_ROUNDOFF_PX >= self.config["imbalance_difference_px"] and
                                 fraction + FRACTION_ROUNDOFF >= self.config["imbalance_fraction"])
                self.streak[axis] = self.streak[axis] + 1 if triggered else 0
                if not triggered:
                    self.active[axis] = False
                check = dict(axis=("image_reference_x", "image_reference_y")[axis], positive_advance_px=float(positive),
                    negative_advance_px=float(negative), difference_px=difference, imbalance_fraction=fraction,
                    comparable_growth=bool(available), threshold_exceeded=triggered, persistent_windows=self.streak[axis])
                checks.append(check)
                if self.streak[axis] >= self.config["persistence_windows"] and not self.active[axis]:
                    self.active[axis] = True
                    proposal = dict(proposal_id=f"{self.run_id}:edge-review:{len(self.proposals) + 1:04d}",
                        kind="projected_edge_advance_review", run_id=self.run_id, clock_id=self.clock_id,
                        source_frame_index=frame_index, proposed_at_source_s=timestamp,
                        baseline_frame_index=previous["frame_index"], context_sha256=row["context_sha256"],
                        image_BGR_sha256=row["image_BGR_sha256"], mask_pixels_sha256=row["mask_pixels_sha256"],
                        baseline_image_BGR_sha256=previous["image_BGR_sha256"],
                        baseline_mask_pixels_sha256=previous["mask_pixels_sha256"], axis=check["axis"],
                        recommendation="review_reference_pose_and_rotation_or_circulation", requires_operator_review=True,
                        physical_defect_confirmed=False, physical_command=None)
                    self.proposals.append(proposal)
                    row["proposals"].append(proposal)
            row.update(reason="compared_projected_supports", comparison=dict(baseline_frame_index=previous["frame_index"],
                baseline_source_time_s=previous["source_time_s"], horizon_source_s=timestamp - previous["source_time_s"],
                baseline_context_sha256=previous["context_sha256"],
                baseline_image_BGR_sha256=previous["image_BGR_sha256"],
                baseline_mask_pixels_sha256=previous["mask_pixels_sha256"], axes=checks))
        self.history.append({key:row[key] for key in ("frame_index", "source_time_s", "projected_supports_px", "context_sha256", "image_BGR_sha256", "mask_pixels_sha256")})
        while len(self.history) > 1 and self.history[1]["source_time_s"] <= cutoff:
            self.history.popleft()
        return row
