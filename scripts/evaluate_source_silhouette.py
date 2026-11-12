# FILE PURPOSE: Evaluate the frozen single-main-crystal proxy experiment on common development targets.
"""Retain negative comparisons and misses; exclude final test and physical claims."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.silhouette_eval import evaluate_silhouette


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study", type=Path, default=Path("configs/source_silhouette_experiment.json")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate_silhouette(a.study, a.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                models=result["models"],
                metrics=result["metrics"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
