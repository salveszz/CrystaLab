# FILE PURPOSE: Import/audit the reviewed Leeds real-microscopy archive locally.
# Retain source hashes, run/family splits and original manual-fit semantics;
# unavailable segmentation/event labels stay null rather than becoming pseudo-truth.
"""Index the real microscope image archive, or reconstruct its audit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.data import audit_image_index, import_leeds, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        report = audit_image_index(args.output, archive_path=args.archive)
    else:
        if args.archive is None or args.protocol is None:
            parser.error("Import requires --archive and --protocol")
        report = import_leeds(args.archive, read_json(args.protocol), args.output)
        report = dict(status=report["status"], **report["summary"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
