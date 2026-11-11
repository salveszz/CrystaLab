# FILE PURPOSE: Independently audit a saved simulated temperature-control comparison.
# The optional report is written outside the immutable run directory.
"""Reconstruct saved software thermal trajectories and command provenance."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.audit import audit_control_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        if args.output.resolve().is_relative_to(args.run.resolve()):
            parser.error("Write the audit report outside the immutable run")
        if args.output.exists():
            raise FileExistsError("Retain prior audit reports; choose a new output")
    result = audit_control_benchmark(args.run)
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
