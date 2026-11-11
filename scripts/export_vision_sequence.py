# FILE PURPOSE: Create a traceable development replay from indexed real microscope JPEGs.
# Preserve a per-frame map and attribution; playback FPS never becomes experiment
# timing. Holdout experiments are excluded from this visual-tuning command.
"""Export a training-run pilot video for seed-baseline inspection."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.sequence import export_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-fps", type=float, default=4)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=960)
    args = parser.parse_args()
    result = export_pilot(
        args.index,
        args.run_id,
        args.output,
        display_fps=args.display_fps,
        max_frames=args.max_frames,
        max_width=args.max_width,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "frames", "output_width", "output_height")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
