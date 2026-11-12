# FILE PURPOSE: Fit training normalization, replay causal adaptive forecasts or score development cohorts.
"""Run a bounded observable-area adaptation experiment without physical actuation."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.adaptive_area_run import fit_reference, replay_adaptive, evaluate_adaptive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit-reference", "replay", "evaluate"))
    p.add_argument("--cohort", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--replay", type=Path)
    p.add_argument("--baseline-model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.action == "fit-reference":
        if a.cohort is None or a.config is None:
            p.error("Reference fitting requires cohort and policy configuration")
        r = fit_reference(a.cohort, a.config, a.output)
    elif a.action == "replay":
        if a.replay is None or a.reference is None:
            p.error("Replay requires observations and a reference")
        r = replay_adaptive(a.replay, a.reference, a.output)
    else:
        if any(v is None for v in (a.cohort, a.reference, a.baseline_model)):
            p.error("Evaluation requires cohort, reference and baseline model")
        r = evaluate_adaptive(a.cohort, a.reference, a.baseline_model, a.output)
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "status",
                    "positive_relative_gain_reference",
                    "initial_error_ratio",
                    "metrics",
                    "frames",
                )
                if k in r
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
