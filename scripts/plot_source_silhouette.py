# FILE PURPOSE: Export scientific proxy-comparison plots and the declared training-only outline preview.
"""Static source-fit diagnostics; no model selection, image-label review or data mutation."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--projection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    a.output.mkdir(parents=True, exist_ok=False)
    metrics = read(a.evaluation / "metrics.json")
    methods = [
        "real_training_mean_template",
        "synthetic_only",
        "source_projection_only",
        "mixed",
    ]
    labels = [
        "Static real-training\ntemplate",
        "Synthetic\nonly",
        "Real projection\nonly",
        "Mixed",
    ]
    colors = ["#8996a2", "#b98c4d", "#658e7e", "#387db0"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for axis, kind, title in zip(
        axes,
        ["real", "synthetic"],
        [
            "12 real source-fit proxy images / 2 families",
            "44 synthetic proxy images / 6 families",
        ],
    ):
        values = [
            metrics[m]["by_evidence"][kind]["proxy_mask_iou_mean"] for m in methods
        ]
        axis.bar(labels, values, color=colors, width=0.65)
        axis.scatter(
            range(len(methods)),
            [
                metrics[m]["by_evidence"][kind]["proxy_mask_iou_family_macro"]
                for m in methods
            ],
            marker="D",
            color="#253344",
            s=22,
            label="Family macro mean",
            zorder=3,
        )
        for i, value in enumerate(values):
            axis.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Proxy-mask IoU")
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Fixed-budget main-crystal silhouettes / author-projected real targets / no pixel-gold or KDP validation",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.output / "proxy_comparison.png", dpi=160)
    plt.close(fig)
    policy = read(a.projection / "policy.json")
    records = read(a.projection / "records.json")
    tiles = []
    for sample in policy["inspection_training_sample_ids"]:
        row = next(r for r in records if r["sample_id"] == sample)
        assert row["split"] == "train"
        image = cv2.imread(str(a.projection / row["image_path"]))
        cv2.polylines(
            image, [np.rint(row["polygon"]).astype(np.int32)], True, (0, 235, 255), 1
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (25, 25, 25), -1)
        cv2.putText(
            image, sample, (7, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1
        )
        cv2.putText(
            image,
            "AUTHOR SHAPE PROXY / PIXEL EDGE NOT REVIEWED",
            (7, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (0, 235, 255),
            1,
        )
        tiles.append(image)
    assert len(tiles) == 6
    cv2.imwrite(
        str(a.output / "training_projection_preview.png"),
        np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]),
    )
    metadata = dict(
        evaluation_manifest_sha256=sha(a.evaluation / "manifest.json"),
        projection_manifest_sha256=sha(a.projection / "manifest.json"),
        script_sha256=sha(__file__),
        preview_training_ids=policy["inspection_training_sample_ids"],
        files={f.name: sha(f) for f in a.output.glob("*.png")},
    )
    (a.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            dict(output=str(a.output), figures=list(metadata["files"])), indent=2
        )
    )


if __name__ == "__main__":
    main()
