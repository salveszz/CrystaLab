# FILE PURPOSE: Plot saved thermal verification results with declared numerical scope.
# This figure contains no measured KDP performance or inferred crystal risk.
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


def plot(run, output):
    output.mkdir(parents=True, exist_ok=False)
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), layout="constrained")
    dt = np.array([r["dt_s"] for r in report["shear"]])
    error = np.array([r["relative_velocity_error"] for r in report["shear"]])
    axes[0].loglog(dt, error, "o-", label="Computed error")
    axes[0].loglog(dt, error[-1] * dt / dt[-1], "--", label="First-order guide")
    axes[0].set(
        xlabel="Time step [s]",
        ylabel="Relative velocity RMS error",
        title="Buoyant shear: time refinement",
    )
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(sorted(dt), [f"{x:g}" for x in sorted(dt)])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    n = np.array([r["counts_x"] for r in report["wall_conduction"]])
    error = np.array([r["rms_error_K"] for r in report["wall_conduction"]])
    axes[1].loglog(n, error, "o-", label="Computed error")
    axes[1].loglog(n, error[-1] * (n[-1] / n) ** 2, "--", label="Second-order guide")
    axes[1].set(
        xlabel="Cells in x",
        ylabel="Temperature RMS error [K]",
        title="Wall conduction: grid refinement",
    )
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(n, [str(x) for x in n])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    paths = [run / row["artifact"] for row in report["mixing"]]
    with np.load(paths[0]) as off, np.load(paths[1]) as on:
        difference = (on["w_m_per_s"] - off["w_m_per_s"])[4] * 1000
    limit = float(abs(difference).max())
    im = axes[2].imshow(
        difference,
        origin="lower",
        extent=[0, 10, 0, 10],
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
    )
    axes[2].set(
        xlabel="x [mm]",
        ylabel="y [mm]",
        title="3D flow: buoyancy on minus off\nVertical speed, z = 5 mm face, t = 0.25 s",
    )
    fig.colorbar(im, ax=axes[2], label="Vertical speed difference [mm/s]")
    for axis in axes[:2]:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        "Fixed-domain heat / solute / buoyancy verification — declared test properties, not KDP calibration",
        fontsize=12,
    )
    fig.savefig(output / "thermal_verification.png", dpi=170)
    fig.savefig(output / "thermal_verification.svg")
    plt.close(fig)
    sources = [report_path, run / "manifest.json", *paths, Path(__file__)]
    record = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / "source.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)
