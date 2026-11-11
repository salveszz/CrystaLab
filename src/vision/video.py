# FILE PURPOSE: Decode file/camera frames with explicit clocks and source identity.
# Container timestamps, nominal-FPS estimates and camera arrival times retain
# distinct meanings. No video clock is silently relabeled as experiment time.
"""Timestamped video input for reproducible observation and camera adapters."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import cv2
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Frame:
    index: int
    video_time_s: float
    timestamp_basis: str
    image: np.ndarray


class VideoReader:
    """Read every decoded frame; invalid/nonmonotonic clocks raise explicitly.

    OpenCV properties depend on the backend. Files default to its container
    timestamp; use clock='nominal_fps' explicitly for files lacking usable PTS.
    Camera times are monotonic arrival times, not sensor exposure timestamps.
    """

    def __init__(self, source, *, camera=False, clock="container", fps=None):
        if camera:
            if isinstance(source, bool) or not isinstance(source, int) or source < 0:
                raise ValueError("Camera source must be a nonnegative device index")
            if clock != "arrival" or fps is not None:
                raise ValueError(
                    "Camera input requires arrival clock and no FPS override"
                )
        else:
            source = Path(source)
            if not source.is_file():
                raise FileNotFoundError(source)
            if clock not in {"container", "nominal_fps"}:
                raise ValueError("File clock must be container or nominal_fps")
            if fps is not None and clock != "nominal_fps":
                raise ValueError("FPS override requires an explicit nominal_fps clock")
        if fps is not None and (
            isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0
        ):
            raise ValueError("FPS must be positive and finite")
        self.source, self.camera, self.clock = source, camera, clock
        self.capture = cv2.VideoCapture(source if camera else str(source))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video input: {source}")
        reported_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps is not None else reported_fps
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = None
        if clock == "nominal_fps" and self.fps is None:
            self.close()
            raise ValueError("No usable nominal FPS; supply an explicit FPS")
        count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reported_count = int(count) if math.isfinite(count) and count > 0 else None
        self.metadata = {
            "kind": "camera" if camera else "file",
            "source": source if camera else str(source.resolve()),
            "sha256": None if camera else file_sha256(source),
            "clock": clock,
            "reported_fps": reported_fps if math.isfinite(reported_fps) else None,
            "nominal_fps": self.fps,
            "reported_frame_count": None if camera else self.reported_count,
            "backend": self.capture.getBackendName(),
            "opencv_version": cv2.__version__,
        }
        self.index, self.previous_time, self.started = 0, None, None

    def __iter__(self):
        return self

    def __next__(self):
        ok, image = self.capture.read()
        if not ok:
            if self.camera:
                raise OSError("Camera stopped delivering frames")
            if self.index == 0:
                raise OSError("Video has no decodable frames")
            if self.reported_count is not None and self.index < self.reported_count - 1:
                raise OSError(
                    f"Decode ended at {self.index}; container advertised "
                    f"{self.reported_count} frames"
                )
            raise StopIteration
        if image is None or image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError("Decoder did not return a uint8 BGR frame")
        if self.clock == "arrival":
            now = time.monotonic()
            if self.started is None:
                self.started = now
            timestamp = now - self.started
        elif self.clock == "nominal_fps":
            timestamp = self.index / self.fps
        else:
            timestamp = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or self.previous_time is not None
            and timestamp <= self.previous_time
        ):
            raise ValueError(
                "Video timestamps are unavailable/nonmonotonic; inspect the source "
                "or explicitly select nominal_fps. No clock fallback was applied."
            )
        frame = Frame(self.index, timestamp, self.clock, image)
        self.index += 1
        self.previous_time = timestamp
        return frame

    def close(self):
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
