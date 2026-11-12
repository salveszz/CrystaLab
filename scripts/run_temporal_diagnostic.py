# FILE PURPOSE: Run a declared frozen-model event/area experiment after optical rendering completes.
"""No fitting, reserved-scene replay, human attestation or physical action is performed."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.temporal_diagnostic import run_temporal_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_temporal_diagnostic(args.source, args.experiment, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "development_scenes",
                    "event_metrics",
                    "forecast_metrics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
