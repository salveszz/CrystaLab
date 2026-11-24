# FILE PURPOSE: Compare fixed-recipe and PID control on the complete declared
# software thermal benchmark, reusing its existing provenance and native audit.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controllers.benchmark import run_benchmark
from controllers.audit import audit_control_benchmark

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/temperature_control_baseline.json",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run_benchmark(a.config, a.output)
    print(json.dumps(audit_control_benchmark(a.output), indent=2))
