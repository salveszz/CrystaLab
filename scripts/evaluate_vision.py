# FILE PURPOSE: Evaluate saved segmentation weights on audited development images only.
# This preserves model/data identity and exports masks, overlays and instance errors.
"""Inspect a first local vision fit without scoring the final test split."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.segmentation_eval import evaluate_development


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--iou-gate", type=float, default=0.5)
    a = p.parse_args()
    result = evaluate_development(
        a.run,
        a.dataset,
        a.output,
        split=a.split,
        confidence=a.confidence,
        iou_gate=a.iou_gate,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "images",
                    "families",
                    "images_by_evidence",
                    "per_class",
                    "latency",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
