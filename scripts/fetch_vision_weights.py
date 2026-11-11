# FILE PURPOSE: Acquire the explicitly pinned official YOLOv8n-seg initialization locally.
# Cached weights must match SHA256. Partial downloads remain separate and cannot
# be mistaken for accepted model initialization; no training data is uploaded.
"""Download or verify pretrained vision weights from the pinned official URL."""

import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.video import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime", type=Path, default=Path("configs/yolov8n_seg_runtime.json")
    )
    args = parser.parse_args()
    config = json.loads(args.runtime.read_text())
    path = Path(config["local_path"])
    if config["architecture"] != "YOLOv8n-seg" or not config["url"].startswith(
        "https://github.com/ultralytics/assets/releases/download/"
    ):
        raise ValueError("Only the pinned official YOLOv8 initialization is supported")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with (
            urllib.request.urlopen(config["url"], timeout=120) as source,
            partial.open("xb") as target,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
        if file_sha256(partial) != config["sha256"]:
            raise ValueError(
                "Downloaded weight identity mismatch; partial file retained"
            )
        partial.rename(path)
    if file_sha256(path) != config["sha256"]:
        raise ValueError("Cached pretrained weight identity mismatch")
    print(
        json.dumps(
            dict(status="verified", path=str(path), sha256=config["sha256"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
