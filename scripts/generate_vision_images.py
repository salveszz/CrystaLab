# FILE PURPOSE: Generate/audit Blender optical scenes with explicit mask semantics.
# Frozen scene families stay in one split; geometric and emission-proxy masks are
# separate, and all optics/events remain illustrative rather than calibrated KDP.
"""Render the pitch's optical synthetic image path using Blender/Cycles."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rendering.optical import generate_optical, audit_optical


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--blender", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--timeout", type=float, default=1800)
    a = p.parse_args()
    if a.audit:
        result = audit_optical(a.output)
    else:
        if a.config is None or a.blender is None:
            p.error("Generation needs --config and --blender")
        result = generate_optical(
            json.loads(a.config.read_text()), a.output, a.blender, timeout_s=a.timeout
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
