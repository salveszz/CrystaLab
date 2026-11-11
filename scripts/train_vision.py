# FILE PURPOSE: Run a declared local YOLOv8-seg experiment with audited image labels.
# Pretrained weights are explicitly pinned; output is never overwritten and failed
# manifests remain inspectable. This CLI does not issue hardware commands.
"""Train the pitch's vision model locally without cloud integrations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.segmentation import train_segmentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--runtime", type=Path, default=Path("configs/yolov8n_seg_runtime.json")
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = train_segmentation(
        args.dataset,
        json.loads(args.config.read_text()),
        json.loads(args.runtime.read_text()),
        args.output,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "elapsed_seconds",
                    "evaluation_scope",
                    "metrics",
                    "best_weights_sha256",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
