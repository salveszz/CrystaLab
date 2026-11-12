# FILE PURPOSE: Import/audit public figure pixels and prepare explicitly unreviewed sheets.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.publication_panels import audit_index, import_panels, prepare_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["import", "audit", "bundle"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kdp_gel_panels.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external_vision/KDP-GEL-2026")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "import":
        if args.output is None:
            parser.error("Import requires --output")
        result = import_panels(args.cache, args.protocol, args.output)
    else:
        if args.index is None or (args.stage == "bundle" and args.output is None):
            parser.error("Audit/bundle requires --index; bundle also requires --output")
        if args.stage == "bundle":
            result = prepare_bundle(args.index, args.protocol, args.output)
        else:
            manifest, records = audit_index(args.index, args.protocol)
            result = dict(status="passed", panels=len(records), source_id=manifest["protocol"]["source_id"])
    print(json.dumps(result, indent=2, allow_nan=False))
