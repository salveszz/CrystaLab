# FILE PURPOSE: Run an auditable fixed-recipe/PID comparison on a simulated thermal plant.
# The experiment uses declared software fixtures and cannot command physical equipment.
"""Compare causal feedback with a fixed recipe under declared thermal scenarios."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(args.config, args.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                runs=[
                    {key: r[key] for key in ["scenario", "method", "metrics"]}
                    for r in result["runs"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
