# FILE PURPOSE: Train the complete synthetic volume-gain XGBoost comparison with
# frozen group splits, native missing values and validation-only model selection.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.growth_xgboost import train

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/xgboost_growth.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = train(a.dataset, a.config, a.output)
    print(json.dumps(result, indent=2))
