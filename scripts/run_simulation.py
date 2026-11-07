# HOW TO RUN AND READ THIS DEMO
# Run: python scripts/run_simulation.py (or add --headless without a display).
# Load settings from configs/default.json, then apply command-line overrides.
# Run the simulation to completion and retain snapshots of the tank and crystal.
# Plot temperature and concentration with crystal overlays, alongside the
# history of crystal width and height. Animation plays the saved snapshots;
# the window is a replay, not an interactive live controller.
# Export growth.gif, final_tank.png, measurements.csv, and the resolved config
# into the output directory. Show the animation window if a display is available.
# The numerical model lives in src/simulation; this script handles presentation
# and export. Displayed time and lengths are illustrative, not real-world units.

"""python scripts/run_simulation.py --help."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig")
)

from simulation.engine import Simulation
from simulation.tank import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/default.json",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Save animation without opening a window",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--concentration", type=float)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for name in ("duration", "dt", "seed", "temperature", "concentration"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    sim = Simulation(SimulationConfig(**settings))
    headless = args.headless or not (
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or sys.platform in ("win32", "darwin")
    )
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    from dataclasses import asdict

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    # Retain at most ~121 frames while advancing all configured timesteps.
    snapshots = []
    frame_interval = max(sim.config.duration / 120, sim.config.dt)
    next_frame = 0.0

    def capture():
        k = sim.crystal
        snapshots.append(
            (
                sim.time,
                sim.tank.temperature.copy(),
                sim.tank.concentration.copy(),
                (k.left, k.bottom, k.width, k.height),
                k.defects,
            )
        )

    capture()
    next_frame += frame_interval
    while sim.time < sim.config.duration - 1e-12:
        sim.step(min(sim.config.dt, sim.config.duration - sim.time))
        if sim.time >= next_frame - 1e-12:
            capture()
            next_frame += frame_interval
    if snapshots[-1][0] != sim.time:
        capture()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sim.history[0].keys())
        writer.writeheader()
        writer.writerows(sim.history)
    (args.output / "config.json").write_text(
        json.dumps(asdict(sim.config), indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    fig.suptitle(
        "CrystaLab • illustrative 2D crystallization (uncalibrated model units)"
    )
    extent = (0, sim.config.width, 0, sim.config.height)
    images, patches = [], []
    for index, (label, cmap) in enumerate(
        (("Temperature", "coolwarm"), ("Solute concentration", "viridis"))
    ):
        vmin = min(float(s[index + 1].min()) for s in snapshots)
        vmax = max(float(s[index + 1].max()) for s in snapshots)
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 0.1, vmax + 0.1
        images.append(
            axes[index].imshow(
                snapshots[0][index + 1],
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
        )
        fig.colorbar(images[-1], ax=axes[index], shrink=0.8)
        patch = Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#bdefff",
            edgecolor="#123b5d",
            linewidth=2,
            alpha=0.9,
        )
        axes[index].add_patch(patch)
        patches.append(patch)
        axes[index].set(title=label, xlabel="Tank x", ylabel="Tank y")
    times = [row["time"] for row in sim.history]
    axes[2].plot(times, [row["width"] for row in sim.history], label="Width")
    axes[2].plot(times, [row["height"] for row in sim.history], label="Height")
    cursor = axes[2].axvline(0, color="gray", linestyle="--")
    axes[2].set(title="Crystal geometry", xlabel="Simulated time", ylabel="Length")
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    status = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

    def update(index):
        time, temperature, concentration, bounds, defects = snapshots[index]
        for img, field in zip(images, (temperature, concentration)):
            img.set_data(field)
        for patch in patches:
            patch.set_bounds(*bounds)
        cursor.set_xdata([time, time])
        status.set_text(
            f"t = {time:.1f}  |  dissolved concentration = {np.mean(concentration):.4f}  |  synthetic defects = {defects}"
        )
        return images + patches + [cursor, status]

    animation = FuncAnimation(
        fig, update, frames=len(snapshots), interval=60, repeat=True
    )
    animation.save(args.output / "growth.gif", writer=PillowWriter(fps=16))
    update(len(snapshots) - 1)
    fig.savefig(args.output / "final_tank.png", dpi=150)
    print(
        f"Crystal: {sim.history[0]['width']:.3f} × {sim.history[0]['height']:.3f} -> {sim.crystal.width:.3f} × {sim.crystal.height:.3f}"
    )
    print(f"Mass drift: {sim.total_mass - sim.history[0]['total_mass']:.3e}")
    print(
        f"Results: {args.output.resolve()} (growth.gif, final_tank.png, measurements.csv, config.json)"
    )
    if not headless:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
