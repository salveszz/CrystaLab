# FILE PURPOSE: Produce source-clock review proposals from a verified candidate analysis.
# Existing outputs are refused and no actuator or physical sensor is contacted.
"""Run the event-rule supervisory baseline in proposal-only replay mode."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.shadow import run_shadow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_shadow(args.analysis, args.policy, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "processed_frames",
                    "proposals_by_kind",
                    "numerical_actuator_commands",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
