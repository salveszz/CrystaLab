# FILE PURPOSE: Prepare or audit grouped YOLO image datasets without accepting real proposals.
# Only audited optical image sources and explicitly reviewed real exports are accepted.
"""Export source-preserving segmentation images and normalized polygon labels."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.training_data import prepare_segmentation, audit_segmentation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()
    result = (
        audit_segmentation(args.output)
        if args.audit
        else prepare_segmentation(args.source, args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
