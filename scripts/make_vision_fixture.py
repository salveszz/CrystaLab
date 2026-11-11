# FILE PURPOSE: Generate a labeled non-optical video fixture for pipeline checks.
# Analytic pixel silhouettes, distractors and an observation gap exercise tracking.
# This is not Blender imagery, measured crystal data or a physical defect model.
"""Make a small reproducible seed-observation integration fixture."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def generate_fixture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "truth_masks").mkdir()
    width, height, fps, frames = 320, 240, 10, 24
    writer = cv2.VideoWriter(
        str(output / "fixture.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError("MJPG fixture writer is unavailable")
    truth = []
    try:
        for i in range(frames):
            image = np.full((height, width, 3), 220, np.uint8)
            mask = np.zeros((height, width), np.uint8)
            if not 10 <= i < 14:
                mask[90:130, 65 : 105 + i] = 255
                image[mask > 0] = 40
            # Larger than the seed, but never a valid replacement for its identity.
            cv2.circle(image, (250, 145), 32, (55, 55, 55), -1)
            writer.write(image)
            path = f"truth_masks/frame_{i:08d}.png"
            if not cv2.imwrite(str(output / path), mask):
                raise OSError("Cannot write fixture mask")
            truth.append(
                dict(
                    frame_index=i,
                    video_time_s=i / fps,
                    visible_seed=bool(mask.any()),
                    projected_area_px=int(np.count_nonzero(mask)),
                    mask_path=path,
                )
            )
    finally:
        writer.release()

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    save(
        "baseline.json",
        dict(
            seed_point=[85, 110],
            polarity="dark",
            threshold=110,
            morphology_size=1,
            minimum_area_px=32,
            maximum_gap_s=2,
        ),
    )
    save(
        "source_info.json",
        dict(
            evidence_kind="fixture",
            material="none",
            source_id="analytic_seed_fixture_v1",
            split="development",
            limitations="Pixel geometry only; no optics, growth physics or defect labels",
        ),
    )
    save("truth.json", dict(schema="crystalab_seed_fixture_v1", frames=truth))
    save(
        "manifest.json",
        dict(
            status="complete",
            evidence_kind="fixture",
            fps=fps,
            frames=frames,
            scene_id="analytic_seed_fixture_v1",
            video_sha256=hashlib.sha256(
                (output / "fixture.avi").read_bytes()
            ).hexdigest(),
        ),
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_fixture(args.output).resolve())


if __name__ == "__main__":
    main()
