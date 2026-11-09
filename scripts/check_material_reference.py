# FILE PURPOSE: Audit the hash-pinned COD structural reference, print its SI
# density and crystal-frame normals, and keep the illustrative simulator intact.
# The archive is read directly, so a separately extracted data/COD is optional.

"""Check the selected KDP lattice reference and its explicit limits."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.material import load_kdp_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "configs/kdp_material_reference.json",
    )
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    structure = load_kdp_structure(ROOT / reference["archive"], reference)
    print(
        json.dumps(
            {
                "reference": asdict(structure),
                "normal_100": structure.face_normal([1, 0, 0]).tolist(),
                "normal_101": structure.face_normal([1, 0, 1]).tolist(),
                "mass_of_one_physical_mm3_kg": structure.crystal_mass_kg(1e-9),
                "limits": reference["limits"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
