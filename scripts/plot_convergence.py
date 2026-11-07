# HOW THIS FILE WORKS
# Read a completed convergence report without rerunning or changing the solver.
# Plot the isolated grid/time histories, the finest-pair numerical differences
# and each run's conservation diagnostics against the predeclared tolerances.
# Zero conservation error is drawn at a labeled display floor on the log axis;
# the underlying report values are unchanged. Export one standalone PNG for review.

"""Render the saved coupled-convergence study as a static scientific figure."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def plot_report(report_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(Path(report_path).read_text())
    protocol = report["protocol"]
    cases = report["results"]
    spatial = sorted(
        [c for c in cases if c["config"]["dt"] == protocol["timesteps"][0]],
        key=lambda c: c["config"]["nx"],
    )
    temporal = sorted(
        [
            c
            for c in cases
            if [c["config"][n] for n in ("nx", "ny", "nz")] == protocol["grids"][-1]
        ],
        key=lambda c: -c["config"]["dt"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fig.suptitle(
        "Coupled 3D growth and displacement: numerical refinement (uncalibrated model)"
    )
    for axis, group, title in [
        (axes[0, 0], spatial, "Spatial refinement at fixed dt"),
        (axes[0, 1], temporal, "Timestep refinement on fixed finest grid"),
    ]:
        for case in group:
            c = case["config"]
            label = (
                f"{c['nx']}×{c['ny']}×{c['nz']}"
                if group is spatial
                else f"dt={c['dt']:g}"
            )
            axis.plot(
                [r["time"] for r in case["history"]],
                [r["volume"] for r in case["history"]],
                label=label,
            )
        axis.set(title=title, xlabel="Simulation time", ylabel="Crystal volume")
        axis.legend()
        axis.grid(alpha=0.2)

    metrics = [
        ("volume_gain_relative_change", "Net growth"),
        ("volume_trajectory_relative_change", "Growth trajectory"),
        ("face_advance_relative_change", "Face advance"),
        ("level_rise_relative_change", "Level rise"),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    for position, name in enumerate(("spatial", "temporal")):
        row = report["comparisons"][name][-1]
        axes[1, 0].bar(
            x + (position - 0.5) * width,
            [100 * row[k] for k, _ in metrics],
            width,
            label=name.capitalize(),
        )
    axes[1, 0].axhline(
        100 * protocol["relative_change_tolerance"],
        color="#bc3434",
        linestyle="--",
        label="Declared tolerance",
    )
    axes[1, 0].set(
        xticks=x,
        xticklabels=[label for _, label in metrics],
        ylabel="Difference (%)",
        title="Finest adjacent pairs: selected refinement metrics",
    )
    axes[1, 0].tick_params(axis="x", rotation=12)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    display_floor = 1e-17
    x = np.arange(len(cases))
    for position, (key, label) in enumerate(
        [
            ("max_mass_relative_error", "Combined mass"),
            ("max_carrier_relative_error", "Carrier volume"),
        ]
    ):
        values = [case[key] for case in cases]
        bars = axes[1, 1].bar(
            x + (position - 0.5) * width,
            [max(v, display_floor) for v in values],
            width,
            label=label,
        )
        for bar, value in zip(bars, values):
            if value == 0.0:
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2,
                    display_floor * 1.2,
                    "0",
                    ha="center",
                    fontsize=7,
                )
    axes[1, 1].axhline(
        protocol["mass_relative_tolerance"],
        color="#bc3434",
        linestyle="--",
        label="Mass tolerance",
    )
    axes[1, 1].axhline(
        protocol["carrier_relative_tolerance"],
        color="#8b5e29",
        linestyle=":",
        label="Carrier tolerance",
    )
    labels = [
        f"{c['config']['nx']}×{c['config']['ny']}×{c['config']['nz']}\ndt={c['config']['dt']:g}"
        for c in cases
    ]
    axes[1, 1].set(
        yscale="log",
        xticks=x,
        xticklabels=labels,
        ylabel="Maximum relative error",
        title="Conservation diagnostics (zero shown at display floor)",
    )
    axes[1, 1].set_ylim(
        display_floor,
        max(protocol["mass_relative_tolerance"], protocol["carrier_relative_tolerance"])
        * 10,
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "results/convergence/report.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(plot_report(args.report, args.output or args.report.with_name("summary.png")))


if __name__ == "__main__":
    main()
