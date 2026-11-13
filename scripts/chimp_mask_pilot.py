# FILE PURPOSE: Prepare/audit/train the declared selected protein-instance pilot,
# retaining author labels and source membership without a pitch-class mapping.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.published_mask_training import prepare, audit_training
from models.segmentation import train_segmentation

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit", "train"])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        result = prepare(
            ROOT / "data/published_masks_chimp_v1",
            ROOT / "configs/chimp_mask_pilot.json",
            args.dataset,
        )
    elif args.action == "audit":
        result = audit_training(args.dataset)
    else:
        result = train_segmentation(
            args.dataset,
            json.loads((ROOT / "configs/chimp_mask_pilot_train.json").read_text()),
            json.loads((ROOT / "configs/yolov8n_seg_runtime.json").read_text()),
            args.output,
            dataset_task="published_selected_protein_instances",
        )
    print(json.dumps(result, indent=2))
