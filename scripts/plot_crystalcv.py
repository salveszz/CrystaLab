# FILE PURPOSE: Inspect source thermal-cycle views and published rectangle measurements.
# The figure shows perovskite development data, not KDP masks or control accuracy.
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.crystalcv_data import load_image


def plot(index, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((index / "manifest.json").read_text())
    records = [
        json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()
    ]
    manual = [
        json.loads(line)
        for line in (index / "manual_measurements.jsonl").read_text().splitlines()
    ]
    protocol = manifest["protocol"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.6), layout="constrained")
    shown = []
    for axis, name in zip(axes[0], ("25h 40m 01s.jpg", "27h 00m 01s.jpg")):
        row = next(
            r
            for r in records
            if r["archive_id"] == protocol["manual_archive_id"]
            and r["source_folder"] == protocol["thermal_image_directory"]
            and Path(r["member"]).name == name
        )
        image, _ = load_image(index, row["sample_id"])
        axis.imshow(image[:, :, ::-1])
        axis.set_title(f"Native source view: {name.removesuffix('.jpg')}")
        axis.axis("off")
        shown.append(
            {key: row[key] for key in ("sample_id", "sha256", "native_BGR_sha256")}
        )
    t = np.array([r["filename_elapsed_s"] for r in manual]) / 3600
    truth = np.array([r["manual_rectangle_area_px2"] for r in manual])
    automatic = np.array([r["source_reported_rectangle_area_px2"] for r in manual])
    axes[1, 0].plot(
        t, truth, "o-", markersize=3, label="Published manual FIJI rectangle"
    )
    axes[1, 0].plot(t, automatic, ".-", label="Published automatic rectangle")
    axes[1, 0].set(
        xlabel="Source filename elapsed time [h]",
        ylabel="Rectangle area [pixel²]",
        title="One source trajectory, 54 published measurements",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].scatter(truth, 100 * (automatic / truth - 1), s=20)
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set(
        xlabel="Published manual rectangle area [pixel²]",
        ylabel="Automatic rectangle error [%]",
        title="Source baseline disagreement with manual boxes",
    )
    for axis in axes[1]:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Public MAPbBr₃ thermal-cycle study — source images and rectangle measurements\nNo silhouette masks, KDP accuracy or counterfactual control benefit established",
        fontsize=12,
    )
    fig.savefig(output / "crystalcv_source.png", dpi=160)
    fig.savefig(output / "crystalcv_source.svg")
    plt.close(fig)
    sources = [
        index / "manifest.json",
        index / "images.jsonl",
        index / "manual_measurements.jsonl",
        Path(__file__),
    ]
    record = dict(
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        shown_images=shown,
    )
    (output / "source.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.index, args.output)
