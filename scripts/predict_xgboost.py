# FILE PURPOSE: Reload a verified synthetic-growth model and predict volume gain
# from the 14 causal features/missing indicators, never from labels or metadata.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from environment.dataset import read_rows
from models.growth_xgboost import predict_saved

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows = read_rows(a.input)
    predictions = predict_saved(a.run, rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        for i, (row, value) in enumerate(zip(rows, predictions)):
            stream.write(
                json.dumps(
                    dict(
                        sample_id=row.get("sample_id", str(i)),
                        predicted_volume_gain=float(value),
                        units="model_length^3",
                        physical_model_promoted=False,
                    )
                )
                + "\n"
            )
