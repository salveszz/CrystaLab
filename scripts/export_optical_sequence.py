# FILE PURPOSE: Export one audited optical development scene as a timestamp-mapped convenience video.
# Original family/split identity and illustrative geometry time remain explicit.
"""Create a synthetic replay clip without asserting physical growth timing."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.optical_sequence import export_optical_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--display-fps", type=float, default=4.0)
    p.add_argument("--codec", choices=("MJPG", "FFV1"), default="MJPG")
    a = p.parse_args()
    print(
        json.dumps(
            export_optical_sequence(
                a.source, a.scene, a.output, display_fps=a.display_fps, codec=a.codec
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
