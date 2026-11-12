"""Plot recorded CPU/GPU times and an explicitly uncalibrated scalar indicator."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(study, output):
    output.mkdir(exist_ok=False)
    summary_path = study / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = summary["timings"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for axis, key, title in zip(axes, ("cold_total_s", "stepping_s"),
                               ("Construction + simulation steps", "Simulation steps")):
        for offset, backend, label, color in ((-.18,"cpu","CPU","#285b90"),(.18,"gpu","Hybrid GPU","#bd5b29")):
            values = np.array([r[backend][key] for r in rows])
            median = np.median(values, axis=1)
            x = np.arange(len(rows))+offset
            axis.bar(x, median, width=.32, label=label, color=color, alpha=.85)
            for i, times in enumerate(values):
                axis.scatter(np.full(3,x[i]), times, s=14, color="black", zorder=3)
        axis.set(xticks=range(len(rows)), xticklabels=[f"{r['n']}³" for r in rows],
                 xlabel="Grid cells", ylabel="Wall time (seconds)", title=title)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Recorded local CPU / hybrid Vulkan comparison", fontsize=13)
    fig.savefig(output/"timings.png", dpi=170)
    fig.savefig(output/"timings.pdf")
    plt.close(fig)

    n = max(r["n"] for r in rows)
    paths = [study/f"n{n}_measured_0_{backend}"/"fields.npz" for backend in ("cpu","gpu")]
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        cpu = a["positive_concentration_excess_indicator"][n//2]
        gpu = b["positive_concentration_excess_indicator"][n//2]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout="constrained")
    vmax = max(cpu.max(),gpu.max())
    for axis, field, title in zip(axes[:2], (cpu,gpu), ("CPU", "Hybrid GPU")):
        image = axis.imshow(field, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        axis.set(title=title, xlabel="x cell", ylabel="y cell")
    fig.colorbar(image, ax=axes[:2], label="Positive concentration excess / initial mean", shrink=.8)
    delta = abs(gpu-cpu)
    image = axes[2].imshow(delta, origin="lower", cmap="magma", vmin=0)
    axes[2].set(title="Absolute GPU − CPU difference", xlabel="x cell", ylabel="y cell")
    fig.colorbar(image, ax=axes[2], shrink=.8)
    fig.suptitle(f"{n}³ midplane at 0.125 s · numerical indicator, uncalibrated physical risk", fontsize=12)
    fig.savefig(output/"concentration_indicator.png", dpi=170)
    fig.savefig(output/"concentration_indicator.pdf")
    plt.close(fig)
    manifest = dict(study=str(study),plotter_sha256=sha(__file__),
                    input_sha256={str(p):sha(p) for p in [summary_path,*paths]},
                    output_sha256={p.name:sha(p) for p in sorted(output.iterdir())},
                    interpretation="Recorded timings; dots are all three measured runs. Map is a positive concentration-excess indicator, not a physical nucleation or defect prediction.")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plot(args.study,args.output)
