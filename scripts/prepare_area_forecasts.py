# FILE PURPOSE: Prepare or audit grouped optical projected-area forecasting examples.
# Original mask proxies stay separate from learned causal inputs; final test is excluded.
"""Create a bounded forecast cohort from an existing optical dataset and local model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.area_forecast_data import audit_cohort, prepare_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-run", type=Path)
    parser.add_argument("--observer-config", type=Path)
    parser.add_argument("--forecast-config", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit_cohort(args.output)
    else:
        if any(
            v is None
            for v in [
                args.source,
                args.model_run,
                args.observer_config,
                args.forecast_config,
            ]
        ):
            parser.error("Preparation needs source, model run and both configurations")
        result = prepare_cohort(
            args.source,
            args.model_run,
            args.observer_config,
            args.forecast_config,
            args.output,
        )
        result = {
            key: result[key]
            for key in ["status", "samples", "cases", "elapsed_seconds"]
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
