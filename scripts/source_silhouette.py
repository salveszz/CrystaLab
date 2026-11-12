# FILE PURPOSE: Prepare and train an explicit weak-label main-crystal silhouette experiment.
"""Synthetic, source-author geometry and mixed YOLO fits; no pitch-class relabeling."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.silhouette_training import prepare_silhouette, audit_silhouette
from models.segmentation import train_segmentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--projection", type=Path, required=True)
    prepare.add_argument("--synthetic", type=Path, required=True)
    prepare.add_argument(
        "--mode",
        choices=["synthetic_only", "source_projection_only", "mixed"],
        required=True,
    )
    prepare.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--runtime", type=Path, default=Path("configs/yolov8n_seg_runtime.json")
    )
    train.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "prepare":
        result = prepare_silhouette(a.projection, a.synthetic, a.mode, a.output)
    elif a.action == "audit":
        result = audit_silhouette(a.dataset)
    else:
        result = train_segmentation(
            a.dataset,
            json.loads(a.config.read_text()),
            json.loads(a.runtime.read_text()),
            a.output,
            dataset_task="source_fit_silhouette",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
