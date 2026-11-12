# FILE PURPOSE: Compare fixed recipe, PID, rules and forecast-informed adaptive supervision.
"""Run a source-bound thermal software comparison without hardware or human review claims."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.adaptive_demo import run_adaptive_demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "policy", "config", "reference", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    r = run_adaptive_demo(a.analysis, a.policy, a.config, a.reference, a.output)
    print(
        json.dumps(
            {
                k: r[k]
                for k in ("status", "adaptive_executions", "final_adaptive_parameters")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
