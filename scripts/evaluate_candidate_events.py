# FILE PURPOSE: Score persistent candidate events against source-bound visible-event annotations.
# Startup/unknown origins, misses, duplicates and usable negative coverage remain explicit.
"""Evaluate inspection events without inferring physical time or defect benefit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.event_evaluation import score_candidate_events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--maximum-delay-s", type=float, default=1.0)
    a = p.parse_args()
    result = score_candidate_events(
        a.analysis,
        a.annotations,
        a.output,
        match_iou=a.match_iou,
        maximum_delay_s=a.maximum_delay_s,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "events",
                    "truth_events",
                    "per_kind",
                    "new_visibility_claims",
                    "negative_coverage",
                    "physical_clock_scored",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
