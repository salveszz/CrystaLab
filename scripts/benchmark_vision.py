# FILE PURPOSE: Compare the OpenCV seed baseline with saved development model predictions.
# Both methods use identical audited images and polygons; final test is excluded.
"""Measure projected seed errors without dropping missed/invalid observations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.vision_benchmark import benchmark_seed, calibrate_seed_baseline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--learned", type=Path)
    p.add_argument(
        "--calibration",
        type=Path,
        help="Select baseline settings using training images only",
    )
    p.add_argument("--split", choices=["train", "validation"], default="validation")
    a = p.parse_args()
    if a.calibration:
        if a.learned:
            p.error("Calibration does not consume learned/validation predictions")
        result = calibrate_seed_baseline(
            a.dataset,
            json.loads(a.config.read_text()),
            json.loads(a.calibration.read_text()),
            a.output,
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ["status", "selected_index", "training_sample_ids"]
                },
                indent=2,
            )
        )
        return
    result = benchmark_seed(
        a.dataset,
        json.loads(a.config.read_text()),
        a.output,
        split=a.split,
        learned=a.learned,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "images",
                    "families",
                    "report",
                    "baseline_invalid_reasons",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
