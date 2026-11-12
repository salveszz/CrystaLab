# FILE PURPOSE: Export/audit source-author 3D-fit silhouette proxies without accepting AI proposals.
"""Prepare separate weak real-image targets; published shape fits are not reviewed pixel masks."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.source_shapes import export_source_shapes, audit_source_shapes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("data/real_vision_leeds_v1"))
    p.add_argument(
        "--policy", type=Path, default=Path("configs/leeds_shape_projection.json")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    a = p.parse_args()
    result = (
        audit_source_shapes(a.output)
        if a.audit
        else export_source_shapes(a.index, a.policy, a.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
