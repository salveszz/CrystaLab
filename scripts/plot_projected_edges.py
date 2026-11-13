# FILE PURPOSE: Show saved fixture images and measured/baseline edge imbalances.
"""Standalone scientific figure; no new measurements, labels or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((run / "protocol.json").read_text())
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    cases = [("unequal_anchored", 3, "Unequal advance; attachment asserted"),
             ("symmetric_camera_motion", 9, "Symmetric expansion; camera moves"),
             ("symmetric_free_translation", 3, "Same raw RGB; attachment unsupported")]
    for column, (case, selected, title) in enumerate(cases):
        data = rows(run / case / "measurements.jsonl")
        frame = data[selected]
        image = np.asarray(Image.open(run / case / frame["source_image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(run / case / frame["observed_mask_path"]))
        axes[0, column].imshow(image)
        axes[0, column].contour(mask, levels=[127], colors=["#02bfc0"], linewidths=.8)
        registration = frame["registration"]
        if registration is not None:
            markers = np.asarray(registration["marker_centroids_px"])
            axes[0, column].scatter(*markers.T, s=100, facecolors="none", edgecolors="#cc8f00", linewidth=1.1)
            anchor = np.asarray(protocol["measurement"]["anchor_barycentric_weights"]) @ markers
            axes[0, column].scatter(*anchor, marker="+", s=80, c="#d54a12")
        axes[0, column].set_title(title + f"\nFrame {selected}; artificial RGB and observed contour", fontsize=10)
        axes[0, column].set_axis_off()
        timestamps, raw, centered, qualified = [], [], [], []
        for step, row in enumerate(data):
            timestamps.append(row["source_time_s"])
            baseline = [p for p in data[:step] if p["source_time_s"] <= row["source_time_s"] - protocol["measurement"]["horizon_source_s"]]
            for key, destination in [("unregistered_supports_px", raw), ("centroid_centered_supports_px", centered)]:
                if not baseline or row[key] is None or baseline[-1][key] is None:
                    destination.append(float("nan"))
                else:
                    delta = np.asarray(row[key]) - baseline[-1][key]
                    destination.append(float(abs(delta[0] - delta[1])))
            qualified.append(float("nan") if row["comparison"] is None else row["comparison"]["axes"][0]["difference_px"])
        axis = axes[1, column]
        axis.plot(timestamps, raw, color="#d47915", linestyle="--", label="Unregistered fixed origin")
        axis.plot(timestamps, centered, color="#666a70", linestyle=":", label="Recentered on mask")
        axis.plot(timestamps, qualified, color="#166dab", marker=".", label="Qualified reference" if any(np.isfinite(qualified)) else "Qualified: unavailable")
        axis.axhline(protocol["measurement"]["imbalance_difference_px"], color="#9d3935", alpha=.8, linewidth=1, label="4 px difference gate")
        for row in data:
            for proposal in row["proposals"]:
                axis.axvline(proposal["proposed_at_source_s"], color="#137d64", linewidth=1)
        if case == "symmetric_free_translation":
            axis.text(.5, .65, "No qualified comparison:\nattachment is not established", transform=axis.transAxes,
                      ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        axis.set(xlabel="Fixture source time (s)", ylabel="Opposed x-advance difference (px)", ylim=(-.3, 10))
        axis.grid(alpha=.2)
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle("Anchored projected-edge diagnostic — constructed images and assertions\n"
                   "Planar registration does not establish crystal pose, attachment or physical face rates", fontsize=13)
    figure.supxlabel("Review proposals also require a growth/fraction gate and persistence. No hardware command or real-data accuracy is shown.", fontsize=9)
    figure.savefig(output / "projected_edges.png", dpi=180)
    figure.savefig(output / "projected_edges.svg")
    plt.close(figure)
    (output / "source.json").write_text(json.dumps(dict(run=str(run),
        run_manifest_sha256=hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run, args.output)
