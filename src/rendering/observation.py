# FILE PURPOSE: Render timestamped projected observations and explicit invalid states.
# Overlays distinguish baseline/learned measurements and unverified candidate objects.
"""BGR overlays for seed-tracking evidence and replay videos."""

import cv2
import numpy as np


def observation_overlay(image, record, mask=None, *, instances=()):
    output = image.copy()
    color = (60, 220, 60) if record["valid"] else (40, 60, 240)
    if record["valid"] and record.get("requires_inspection"):
        color = (0, 200, 255)
    for instance in instances:
        contours, _ = cv2.findContours(
            instance["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidate_color = [(190, 190, 190), (0, 200, 255), (255, 180, 0)][
            instance["class_id"]
        ]
        cv2.drawContours(output, contours, -1, candidate_color, 1)
        if contours:
            bx, by, _, _ = cv2.boundingRect(np.concatenate(contours))
            text = ["crystal?", "crystal?", "inclusion?"][
                instance["class_id"]
            ] + f" {instance['score']:.2f}"
            cv2.putText(
                output,
                text,
                (max(0, bx), max(10, by - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                candidate_color,
                1,
            )
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
    lines = [
        f"frame={record['frame_index']} video={record['video_time_s']:.3f}s "
        f"({record['timestamp_basis']})",
        "YOLOv8-seg / projected seed / unvalidated"
        if record["method"].startswith("yolov8")
        else "BASELINE / projected silhouette / unvalidated",
    ]
    if record["valid"]:
        lines.append(
            f"area={record['projected_area_px']} px2 "
            f"size={record['width_px']}x{record['height_px']} px"
        )
    else:
        lines.append("INVALID: " + record["reason"])
    if record["valid"] and record.get("requires_inspection"):
        labels = {
            "compatible_cross_class_seed_hypotheses": "alternative seed outlines",
            "raw_seed_class_transition": "model label changed",
        }
        lines.append(
            "INSPECT: "
            + "; ".join(
                labels.get(reason, reason) for reason in record["inspection_reasons"]
            )
        )
        low, high = record["projected_area_range_px"]
        if low != high:
            lines.append(f"Area hypotheses: {low}..{high} px2")
    if instances:
        lines.append(f"Raw candidates: {len(instances)} / scores uncalibrated")
    largest = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0] for line in lines
    )
    scale = max(0.18, min(0.43, 0.43 * (image.shape[1] - 16) / max(largest, 1)))
    for i, line in enumerate(lines):
        origin = (8, 18 + i * 18)
        cv2.putText(output, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(output, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    return np.ascontiguousarray(output)
