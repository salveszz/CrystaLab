# FILE PURPOSE: Replay causal projected-area forecasts without action dispatch or real-accuracy claims.
# Invalid/inspection frames reset history and retain explicit unavailable predictions.
"""Apply a verified local projected-area forecast model to observation records."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.area_forecast_run import forecast_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = forecast_replay(args.replay, args.model, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "processed_frames",
                    "predictions_by_method",
                    "unavailable_reasons",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
