# FILE PURPOSE: Record an explicit operator review receipt for exact real-event labels.
# This tool cannot infer a review; a named reviewer must inspect source events,
# masks and negative periods and explicitly acknowledge that work.
"""Bind a real visible-event review to source and annotation hashes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.event_evaluation import review_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--source-video", type=Path, required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--ack-event-source-reviewed", action="store_true")
    p.add_argument("--ack-physical-clock-reviewed", action="store_true")
    a = p.parse_args()
    result = review_annotations(
        a.annotations,
        a.source_video,
        a.reviewer,
        ack_event_source_reviewed=a.ack_event_source_reviewed,
        ack_physical_clock_reviewed=a.ack_physical_clock_reviewed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
