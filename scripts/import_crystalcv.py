# FILE PURPOSE: Import public CrystalCV observations and source manual measurements.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.crystalcv_data import import_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = import_data(args.cache, args.protocol, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "counts",
                    "formats",
                    "roles",
                    "duplicate_pixel_groups",
                )
            },
            indent=2,
        )
    )
