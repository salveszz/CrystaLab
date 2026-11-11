# FILE PURPOSE: Run a declared fixture/synthetic event-rule and PID software comparison.
# Explicit source-bound clock mapping and scripted decisions never authorize hardware.
"""Compare fixed recipe, PID and rule-supervised PID on a thermal software fixture."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.supervised_demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_demo(args.analysis, args.policy, args.config, args.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                scripted_executions=result["scripted_executions"],
                runs=[
                    {
                        key: r[key]
                        for key in [
                            "method",
                            "nominal_recipe_sample_mae_K",
                            "recipe_phase_lag_s",
                            "metrics",
                        ]
                    }
                    for r in result["runs"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
