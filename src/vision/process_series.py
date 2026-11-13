# FILE PURPOSE: Build past-only development inputs from source images and reported process logs.
# Source clock quantization is explicit. Threshold contours are observation proxies,
# not accepted masks, physical calibration, defect labels or causal action effects.
"""Source-time alignment and candidate observable forecasting examples."""

import math

import cv2
import numpy as np

INPUT_FIELDS = (
    "current_contour_area_px2",
    "current_sqrt_area_px",
    "trailing_sqrt_area_rate_px_per_h",
    "reported_infusion_rate_ml_per_h",
    "trailing_reported_infusion_mean_ml_per_h",
)


def integer_log_seconds(hours, tolerance_s):
    """Invert this source's integer-second -> six-decimal-hour representation."""
    if not math.isfinite(tolerance_s) or not 0 <= tolerance_s < 0.5:
        raise ValueError("Require a finite sub-half-second representation tolerance")
    values = np.asarray(hours)
    if values.ndim != 1 or values.dtype.kind not in "iuf" or not len(values):
        raise ValueError("Require a nonempty real source-clock vector")
    seconds = values.astype(float) * 3600
    if not np.isfinite(seconds).all() or np.any(seconds < 0):
        raise ValueError("Source elapsed time must be finite and nonnegative")
    rounded = np.rint(seconds)
    error = abs(seconds - rounded)
    if np.max(error) > tolerance_s or np.any(np.diff(rounded) <= 0):
        raise ValueError(
            "Source clock is not strictly increasing integer seconds within its declared precision"
        )
    return rounded.astype(np.int64), float(error.max())


def asof_index(seconds, cutoff_s, maximum_age_s):
    if not math.isfinite(maximum_age_s) or maximum_age_s < 0:
        raise ValueError("Require a finite nonnegative maximum source-log age")
    index = int(np.searchsorted(seconds, cutoff_s, side="right") - 1)
    if index < 0 or cutoff_s - seconds[index] > maximum_age_s:
        return None
    return index


def trailing_reported_mean(seconds, rates, cutoff_s, history_s):
    """Time-weighted, zero-order-held reported state, using no row after cutoff."""
    start = cutoff_s - history_s
    first = int(np.searchsorted(seconds, start, side="right") - 1)
    last = int(np.searchsorted(seconds, cutoff_s, side="right") - 1)
    if history_s <= 0 or first < 0 or last < first:
        return None
    integral = 0.0
    for index in range(first, last + 1):
        end = min(cutoff_s, seconds[index + 1]) if index < last else cutoff_s
        integral += max(0.0, end - max(start, seconds[index])) * rates[index]
    return float(integral / history_s), first, last


def primary_contour(image_bgr, settings, previous_centroid=None):
    """Source RGB thresholds on native pixels, with an explicit analysis ROI."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("Require native uint8 BGR pixels")
    roi = np.zeros(image_bgr.shape[:2], np.uint8)
    cv2.circle(
        roi,
        tuple(settings["analysis_circle_xy"]),
        settings["analysis_radius_px"],
        255,
        -1,
    )
    selected = cv2.inRange(
        image_bgr,
        np.array(settings["lower_RGB"][::-1], np.uint8),
        np.array(settings["upper_RGB"][::-1], np.uint8),
    )
    selected &= roi
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        moment = cv2.moments(contour)
        if moment["m00"] > 0:
            center = (moment["m10"] / moment["m00"], moment["m01"] / moment["m00"])
            candidates.append((contour, float(cv2.contourArea(contour)), center))
    if not candidates:
        return dict(
            integrity_valid=False,
            reason="no_positive_contour",
            contour_area_px2=None,
            centroid_xy=None,
            contour_count=len(contours),
        ), np.zeros(roi.shape, np.uint8)
    if previous_centroid is None:
        contour, area, center = max(candidates, key=lambda item: item[1])
    else:
        contour, area, center = min(
            candidates, key=lambda item: math.dist(item[2], previous_centroid)
        )
    mask = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    jump = None if previous_centroid is None else math.dist(center, previous_centroid)
    touches_roi = bool(
        np.any((mask > 0) & (cv2.erode(roi, np.ones((3, 3), np.uint8)) == 0))
    )
    valid = not touches_roi and (
        jump is None or jump <= settings["maximum_centroid_step_px"]
    )
    return dict(
        integrity_valid=valid,
        reason="threshold_observation" if valid else "roi_contact_or_centroid_jump",
        contour_area_px2=area,
        centroid_xy=list(center),
        centroid_step_px=jump,
        contour_count=len(contours),
        positive_contours=len(candidates),
        largest_and_tracked_agree=area == max(item[1] for item in candidates),
        contour_xy=contour[:, 0, :].tolist(),
        filled_mask_pixels=int(np.count_nonzero(mask)),
        physical_calibration_verified=False,
        accepted_mask_ground_truth=False,
    ), mask


def candidate_samples(
    observations, seconds, reported_rates, settings, maximum_asof_age_s
):
    """Future values enter labels only; every feature uses a recorded prefix.

    A shared source-clock origin and held reported infusion state are declared
    assumptions. They do not establish actual actuator response or causal effects.
    """
    times = np.array([r["filename_elapsed_s"] for r in observations], dtype=float)
    seconds, reported_rates = (
        np.asarray(seconds),
        np.asarray(reported_rates, dtype=float),
    )
    if (
        len(observations) < 2
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(times).all()
        or np.any(times < 0)
        or seconds.ndim != 1
        or reported_rates.shape != seconds.shape
        or not np.isfinite(reported_rates).all()
        or len(seconds) < 2
        or not np.isfinite(seconds).all()
        or np.any(np.diff(seconds) <= 0)
    ):
        raise ValueError(
            "Require ordered observations and finite, ordered process rows"
        )
    if settings["reported_rate_availability"] != "next_source_row":
        raise ValueError(
            "Require next-source-row availability for post-image command values"
        )
    availability_seconds = seconds[1:]
    available_rates = reported_rates[:-1]
    history, horizon = settings["history_s"], settings["horizon_s"]
    if (
        history <= 0
        or horizon <= 0
        or settings["target_lateness_s"] < 0
        or settings["minimum_history_images"] < 2
    ):
        raise ValueError(
            "Require positive history/horizon and nonnegative target lateness"
        )
    rows, rejected = [], []
    for index, observation in enumerate(observations):
        cutoff = times[index]
        reason = None
        last_log = asof_index(availability_seconds, cutoff, maximum_asof_age_s)
        weighted = trailing_reported_mean(
            availability_seconds, available_rates, cutoff, history
        )
        history_indexes = [
            i
            for i in range(index + 1)
            if cutoff - history <= times[i] <= cutoff
            and observations[i]["integrity_valid"]
        ]
        future = int(np.searchsorted(times, cutoff + horizon, side="left"))
        if not observation["integrity_valid"]:
            reason = "invalid_current_observation"
        elif (
            cutoff - times[0] < history
            or len(history_indexes) < settings["minimum_history_images"]
        ):
            reason = "insufficient_image_history"
        elif last_log is None or weighted is None:
            reason = "missing_or_stale_past_process_state"
        elif (
            future >= len(times)
            or times[future] > cutoff + horizon + settings["target_lateness_s"]
            or not observations[future]["integrity_valid"]
        ):
            reason = "missing_comparable_future_proxy"
        if reason:
            rejected.append(
                dict(
                    sample_id=observation["sample_id"],
                    cutoff_s=float(cutoff),
                    reason=reason,
                )
            )
            continue
        areas = np.array([observations[i]["contour_area_px2"] for i in history_indexes])
        if not np.isfinite(areas).all() or np.any(areas <= 0):
            raise ValueError("Valid observation proxies must have positive finite area")
        if (
            not math.isfinite(observations[future]["contour_area_px2"])
            or observations[future]["contour_area_px2"] <= 0
        ):
            raise ValueError("Future proxy area must be positive and finite")
        t = times[history_indexes] - cutoff
        values = np.sqrt(areas)
        centered = t - t.mean()
        slope = float(
            np.dot(centered, values - values.mean()) / np.dot(centered, centered) * 3600
        )
        inputs = dict(
            current_contour_area_px2=float(observation["contour_area_px2"]),
            current_sqrt_area_px=math.sqrt(observation["contour_area_px2"]),
            trailing_sqrt_area_rate_px_per_h=slope,
            reported_infusion_rate_ml_per_h=float(available_rates[last_log]),
            trailing_reported_infusion_mean_ml_per_h=weighted[0],
        )
        assert tuple(inputs) == INPUT_FIELDS
        rows.append(
            dict(
                sample_id=observation["sample_id"],
                cutoff_s=float(cutoff),
                inputs=inputs,
                input_image_sample_ids=[
                    observations[i]["sample_id"] for i in history_indexes
                ],
                input_log_row_range=[weighted[1], last_log],
                last_input_log_time_s=int(seconds[last_log]),
                last_input_availability_bound_s=int(availability_seconds[last_log]),
                source_log_age_s=float(cutoff - seconds[last_log]),
                maximum_input_time_s=float(max(cutoff, availability_seconds[last_log])),
                target_sample_id=observations[future]["sample_id"],
                target_time_s=float(times[future]),
                actual_horizon_s=float(times[future] - cutoff),
                target_sqrt_area_px=math.sqrt(observations[future]["contour_area_px2"]),
                target_kind="future_unreviewed_threshold_contour_proxy",
                split="development",
                independent_ground_truth=False,
                physical_command=None,
            )
        )
    return rows, rejected
