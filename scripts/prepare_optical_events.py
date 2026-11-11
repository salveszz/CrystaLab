# FILE PURPOSE: Export source-construction event annotations independently of detector predictions.
# Optical proxy eligibility is distinct from real visible-event review.
"""Prepare bounded optical event labels for a development replay."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.event_evaluation import build_optical_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = build_optical_annotations(a.replay, a.output)
    print(
        json.dumps(
            dict(
                events=len(result["events"]),
                evidence_kind=result["evidence_kind"],
                label_semantics=result["label_semantics"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
