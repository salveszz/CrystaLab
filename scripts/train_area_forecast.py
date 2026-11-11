# FILE PURPOSE: Fit training-only area-gain calibration and report grouped development baselines.
# Original proxy labels remain separate from causal learned observation features.
"""Train and evaluate a transparent one-step projected-area forecast diagnostic."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.area_forecast_run import train_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train_forecast(args.cohort, args.output)
    print(
        json.dumps(
            {key: result[key] for key in ["status", "calibration", "metrics"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
