# FILE PURPOSE: Plot the frozen public observable comparison, including negative
# learned-model results and the distinction between one run and independent data.
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(result, output):
    output.mkdir(parents=True, exist_ok=False)
    for name, digest in json.loads((result / "manifest.json").read_text()).items():
        assert sha(result / name) == digest
    report = json.loads((result / "report.json").read_text())
    parts = json.loads((result / "partitions.json").read_text())
    rows = [
        json.loads(line)
        for line in (result / "predictions.jsonl").read_text().splitlines()
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cohort"], row["method"]].append(row)
    labels = dict(
        persistence="No change",
        recent_rate="Recent rate",
        train_mean_gain="Training mean",
        ridge_vision="Ridge: vision",
        ridge_with_process="Ridge: + process",
    )
    colors = dict(
        persistence="#888888",
        recent_rate="#166f9c",
        train_mean_gain="#aa334d",
        ridge_vision="#716296",
        ridge_with_process="#d07820",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    ax = axes[0, 0]
    for name, color in zip(parts["cohorts"], ["#c2d7ed", "#cce8d6", "#f3d5af"]):
        selected = grouped[name, "persistence"]
        ax.plot(
            [r["cutoff_s"] / 3600 for r in selected],
            [r["current_sqrt_area_px"] for r in selected],
            color="#333333",
            linewidth=1.5,
        )
        bounds = parts["cohorts"][name]["support_range_s"]
        ax.axvspan(
            bounds[0] / 3600,
            bounds[1] / 3600,
            color=color,
            alpha=0.65,
            label=name.replace("_", " "),
        )
    ax.set(
        title="Disjoint cohort supports; one source run",
        xlabel="Recorded elapsed time (h)",
        ylabel="Current √ contour area (px)",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    methods = list(labels)
    x = np.arange(len(methods))
    for shift, name, color in [
        (-0.2, "later_block_1", "#347f6b"),
        (0.2, "later_block_2", "#c47c33"),
    ]:
        values = [
            report["metrics"][name][method]["all_windows"]["MAE_px"]
            for method in methods
        ]
        ax.bar(x + shift, values, width=0.38, color=color, label=name.replace("_", " "))
    ax.set_xticks(
        x,
        [labels[k].replace(": ", ":\n").replace(" ", "\n", 1) for k in methods],
        fontsize=8,
    )
    ax.set(
        ylabel="Future √ area MAE (px)", title="Recent rate beats both fitted models"
    )
    ax.legend(fontsize=8)
    for ax, name in zip(axes[1], ["later_block_1", "later_block_2"]):
        selected = grouped[name, "recent_rate"]
        times = [r["cutoff_s"] / 3600 for r in selected]
        observed_gain = [
            r["observed_future_sqrt_area_px"] - r["current_sqrt_area_px"]
            for r in selected
        ]
        ax.plot(
            times,
            observed_gain,
            color="#222222",
            linewidth=1.6,
            label="Observed proxy gain",
        )
        for method in ["recent_rate", "ridge_vision", "ridge_with_process"]:
            ax.plot(
                times,
                [r["predicted_gain_px"] for r in grouped[name, method]],
                color=colors[method],
                linewidth=1.2,
                label=labels[method],
            )
        ax.set(
            title=f"{name.replace('_', ' ')} · {len(selected)} overlapping windows",
            xlabel="Prediction cutoff time (h)",
            ylabel="Future √ area − current √ area (px)",
        )
        ax.legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Public MAPbBr₃ forecast diagnostic · 30-minute nominal horizon", fontsize=15
    )
    fig.supxlabel(
        "Automatic pixel proxies; 275 training windows, 69/66 later windows. Only 4 disjoint-support windows per later block. No KDP or causal-control validation.",
        fontsize=9,
    )
    fig.savefig(output / "forecast_comparison.png", dpi=160)
    fig.savefig(output / "forecast_comparison.svg")
    plt.close(fig)
    shutil.copyfile(Path(__file__), output / "plotter_snapshot.py")
    (output / "source.json").write_text(
        json.dumps(
            dict(
                result_manifest_sha256=sha(result / "manifest.json"),
                plotter_sha256=sha(Path(__file__)),
                independent_evaluation_groups=0,
                physical_command=None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(output / "forecast_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.result, args.output)
