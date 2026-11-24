# FILE PURPOSE: Independently re-read synthetic episodes, fields, causal samples and
# exact labels, verifying fingerprints, conservation and whole-scenario splits.
# This command fits no model and does not trust an earlier saved audit result.
"""Audit a completed synthetic growth dataset without rerunning the solver."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from environment.dataset import audit_dataset, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(canonical(audit_dataset(args.dataset)))


if __name__ == "__main__":
    main()
