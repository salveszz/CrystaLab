# FILE PURPOSE: Generate or resume fully recorded synthetic growth episodes; no ML.
# The JSON protocol fixes scenarios, whole-family splits, actions, clocks and seeds.
# Output is created fresh unless an identity-checked resume is explicitly requested.
"""Command line entry point for the existing 3D simulator's data pipeline."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from environment.dataset import canonical, generate, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_growth.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = generate(
        read_json(args.config), args.output, jobs=args.jobs, resume=args.resume
    )
    print(canonical(report))


if __name__ == "__main__":
    main()
