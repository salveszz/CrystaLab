# FILE PURPOSE: Export scientific forecast/state and four-way thermal diagnostic figures.
"""Plot the frozen synthetic adaptive experiment without fitting or selecting a model."""

import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    ROOT = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study", type=Path, default=Path("configs/adaptive_experiment.json")
    )
    p.add_argument("--output", type=Path, required=True)
    plot_args = p.parse_args()
    output = plot_args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text())
    rows = lambda p: [json.loads(v) for v in Path(p).read_text().splitlines()]
    study = read(ROOT / plot_args.study)
    evaluation = ROOT / study["forecast_evaluation_output"]
    metrics = read(evaluation / "metrics.json")["validation"]
    methods = [
        "persistence",
        "recent_rate",
        "calibrated_recent_rate",
        "adaptive_recent_rate",
        "train_mean_fraction",
    ]
    labels = [
        "Persistence",
        "Recent\nrate",
        "Frozen\ncalibration",
        "Online\nadaptation",
        "Training\nmean",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [
        metrics[m]["common_history_support"]["future_area_mae_px2"] for m in methods
    ]
    colors = ["#8799a9", "#367c6b", "#9974ad", "#3476ab", "#bd8a45"]
    axes[0].bar(labels, values, color=colors, width=0.68)
    for i, v in enumerate(values):
        axes[0].text(i, v + 3, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(values) * 1.18)
    axes[0].set_ylabel("Future proxy-area MAE (px²)")
    axes[0].set_title("28 shared windows / two validation families")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    for i, color in enumerate(["#8d7488", "#8a9b68", "#b68b57", "#23699c", "#319178"]):
        scene = f"optical_temporal_{i:03d}"
        s = rows(evaluation / f"replays/{scene}/states.jsonl")
        axes[1].plot(
            [r["estimate"]["observed_at_s"] for r in s],
            [r["parameter_after"]["coefficient"] for r in s],
            color=color,
            linestyle="--" if i < 3 else "-",
            linewidth=1.5 if i < 3 else 2,
            label=f"{'Train' if i < 3 else 'Validation'} {i:03d}",
        )
    axes[1].axhline(1, color="#777", linewidth=0.7, alpha=0.5)
    axes[1].set_xlabel("Source-video time (s; not physical growth time)")
    axes[1].set_ylabel("Online gain coefficient (bounded to [0, 3])")
    axes[1].set_title("Parameters update only after observed outcomes arrive")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Synthetic development diagnostic / adaptation loses to plain recent rate / no model promotion",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "forecast_and_state.png", dpi=160)
    plt.close(fig)
    case = next(c for c in study["control_cases"] if c["split"] == "validation")
    root = ROOT / case["output"]
    m = read(root / "manifest.json")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for r, color in zip(m["runs"], ["#bd8a45", "#3b71a5", "#458b69", "#8b65a4"]):
        rr = rows(root / r["records_path"])
        t = [v["time_s"] for v in rr]
        axes[0].plot(
            t,
            [v["plant_truth"]["temperature_K"] for v in rr],
            color=color,
            label=r["method"].replace("_", " "),
        )
        axes[1].plot(
            t,
            [v.get("supervision", {}).get("recipe_phase_s", v["time_s"]) for v in rr],
            color=color,
            label=r["method"].replace("_", " "),
        )
    axes[0].plot(
        t,
        [v["nominal_recipe_setpoint_K"] for v in rr],
        color="#333",
        linestyle="--",
        label="original recipe target",
    )
    for a in m["scripted_executions"]:
        axes[1].axvspan(
            a["model_decision_time_s"],
            a["pause_until_model_s"],
            color="#458b69",
            alpha=0.12,
        )
    for a in m["adaptive_executions"]:
        axes[1].axvspan(
            a["model_decision_time_s"],
            a["pause_until_model_s"],
            color="#8b65a4",
            alpha=0.16,
        )
    s = rows(root / "adaptive_supervised_area_states.jsonl")
    axes[2].plot(
        [v["computed_at_source_s"] * 100 for v in s],
        [v["parameter_after"]["coefficient"] for v in s],
        color="#8b65a4",
        marker="o",
        markersize=3,
        label="observed-area coefficient",
    )
    for ax in axes:
        ax.axvline(375, color="#666", linestyle=":", linewidth=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Simulated temperature (K)")
    axes[1].set_ylabel("Recipe phase (model s)")
    axes[2].set_ylabel("Adaptive coefficient")
    axes[2].set_xlabel(
        "Thermal fixture time (model s); source video ends at 375 model s"
    )
    fig.suptitle(
        "Validation scene 003 / fixed video, thermal software fixture / no crystal action-response benefit",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "control_validation.png", dpi=150)
    plt.close(fig)
    metadata = dict(
        study_sha256=hashlib.sha256((ROOT / plot_args.study).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in output.glob("*.png")
        },
        scope="Static synthetic development plots; no physical acceptance or action-response claim",
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(dict(output=str(output), figures=list(metadata["files"])), indent=2)
    )


if __name__ == "__main__":
    main()
