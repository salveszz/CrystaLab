# FILE PURPOSE: Prepare editable LabelMe proposals, record explicit reviews and export accepted polygons.
# Proposed masks never automatically become real ground truth; review receipts
# bind source images and edited labels while preserving experiment-family splits.
"""Manage an auditable real-image segmentation annotation batch."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.annotations import (
    prepare_bundle,
    record_review,
    export_reviewed,
    preview_bundle,
)
from vision.baseline import BaselineConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    a = s.add_parser("prepare")
    a.add_argument("--index", type=Path, required=True)
    a.add_argument("--run-id", required=True)
    a.add_argument("--frames", type=int, nargs="+", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--baseline", type=Path)
    a.add_argument("--proposal-width", type=int, default=960)
    a = s.add_parser("review")
    a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--annotation", required=True)
    a.add_argument("--reviewer", required=True)
    a.add_argument("--ack-image-reviewed", action="store_true")
    a.add_argument("--empty-reviewed", action="store_true")
    a.add_argument("--note", default="")
    a = s.add_parser("export")
    a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = s.add_parser("preview")
    a.add_argument("--bundle", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--max-width", type=int, default=960)
    a = p.parse_args()
    if a.command == "prepare":
        baseline = (
            None
            if a.baseline is None
            else BaselineConfig(**json.loads(a.baseline.read_text()))
        )
        result = prepare_bundle(
            a.index,
            a.run_id,
            a.output,
            frame_indices=a.frames,
            baseline=baseline,
            proposal_width=a.proposal_width,
        )
    elif a.command == "review":
        result = record_review(
            a.bundle,
            a.annotation,
            reviewer=a.reviewer,
            acknowledge_image_reviewed=a.ack_image_reviewed,
            empty_reviewed=a.empty_reviewed,
            note=a.note,
        )
    elif a.command == "preview":
        result = preview_bundle(a.bundle, a.output, max_width=a.max_width)
    else:
        result = export_reviewed(a.bundle, a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
