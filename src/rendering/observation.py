# FILE PURPOSE: Render timestamped projected observations and explicit invalid states.
# Overlays label baseline measurements and never present them as optical quality.
"""BGR overlays for seed-tracking evidence and replay videos."""

import cv2
import numpy as np


def observation_overlay(image, record, mask=None):
    output = image.copy()
    color = (60, 220, 60) if record["valid"] else (40, 60, 240)
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)
    lines = [
        f"frame={record['frame_index']} video={record['video_time_s']:.3f}s "
        f"({record['timestamp_basis']})",
        "BASELINE / projected silhouette / unvalidated",
    ]
    if record["valid"]:
        lines.append(
            f"area={record['projected_area_px']} px2 "
            f"size={record['width_px']}x{record['height_px']} px"
        )
    else:
        lines.append("INVALID: " + record["reason"])
    for i, line in enumerate(lines):
        origin = (8, 18 + i * 18)
        cv2.putText(output, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3)
        cv2.putText(output, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
    return np.ascontiguousarray(output)
