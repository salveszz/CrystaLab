# HOW TO RUN AND READ THIS DEMO
# Run python scripts/run_simulation.py, or add --headless to export without a window.
# Load the 3D JSON configuration, apply command-line overrides and run until duration.
# Capture volumetric fields and actual face polygons, then render a full 3D tank,
# a crystal close-up, rising liquid level, displacement arrows and field slices.
# The slices are views of volumetric data, not separate planar simulations.
# Export a single-pass GIF, final image, volume/mass CSV, resolved settings and
# fields.npz containing actual 3D scalar fields, liquid volumes and motion vectors.
# The optional window replays the same frames once and holds the final state.
# Geometry, fields and mass accounting live in src/simulation; no AI is used.

"""Run the illustrative 3D crystal-growth simulator and export its results."""

import argparse
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import sys

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
        "--headless", action="store_true", help="Export without opening a window"
    )
    for name in ("duration", "dt", "temperature", "concentration"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    for name in ("duration", "dt", "temperature", "concentration", "seed"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    sim = Simulation(SimulationConfig(**settings))
    c = sim.config
    iz, iy = c.nz // 2, c.ny // 2
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
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from PIL import Image
    from matplotlib.patches import Polygon, Rectangle

    # Capture at most about 61 volumetric snapshots while advancing every step.
    snapshots = []
    frame_interval = max(c.duration / 60, c.dt)
    next_frame = frame_interval

    def capture():
        k = sim.crystal
        visible = [i for i, face in enumerate(k.faces) if len(face) >= 3]
        snapshots.append(
            dict(
                time=sim.time,
                temperature=sim.tank.temperature.copy(),
                concentration=sim.tank.concentration.copy(),
                volume=k.volume,
                defects=k.defects,
                faces=[k.vertices[k.faces[i]].copy() for i in visible],
                normals=k.normals[visible].copy(),
                rates=sim.face_rates[visible].copy(),
                liquid_level=sim.tank.liquid_level,
                liquid_fraction=sim.tank.liquid_fraction.copy(),
                solid_fraction=sim.tank.geometry.solid_fraction.copy(),
                velocity=sim.tank.displacement_velocity.copy(),
                sections=[
                    k.cross_section(1, sim.tank.y[iy])[:, [0, 2]],
                    k.cross_section(2, sim.tank.z[iz])[:, [0, 1]],
                    k.cross_section(1, sim.tank.y[iy])[:, [0, 2]],
                ],
            )
        )

    capture()
    while sim.time < c.duration:
        sim.step()
        if sim.time >= next_frame:
            capture()
            next_frame += frame_interval
    if snapshots[-1]["time"] != sim.time:
        capture()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sim.history[0].keys())
        writer.writeheader()
        writer.writerows(sim.history)
    (args.output / "config.json").write_text(json.dumps(asdict(c), indent=2) + "\n")
    # Preserve the actual volumetric state, not just its projected pictures.
    np.savez_compressed(
        args.output / "fields.npz",
        temperature=sim.tank.temperature,
        concentration=sim.tank.concentration,
        liquid_volumes=sim.tank.liquid_volumes,
        liquid_fraction=sim.tank.liquid_fraction,
        solid_fraction=sim.tank.geometry.solid_fraction,
        velocity=sim.tank.displacement_velocity,
        vertices=sim.crystal.vertices,
        normals=sim.crystal.normals,
        offsets=sim.crystal.offsets,
        liquid_level=sim.tank.liquid_level,
    )

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    fig.suptitle("CrystaLab: 3D crystal growth (uncalibrated model units)")
    fig.get_layout_engine().set(rect=(0.0, 0.04, 1.0, 0.96))
    tank_ax = fig.add_subplot(2, 3, 1, projection="3d")
    zoom_ax = fig.add_subplot(2, 3, 2, projection="3d")
    history_ax = fig.add_subplot(2, 3, 3)
    slice_axes = [fig.add_subplot(2, 3, index) for index in (4, 5, 6)]
    # Face colors encode normal growth rate; lighting reveals the true facet
    # orientation. Display arrows are scaled, not a change to physical motion.
    rate_max = max(
        1e-10, max(float(frame["rates"].max(initial=0.0)) for frame in snapshots)
    )
    speed_max = max(
        1e-12,
        max(
            float(np.linalg.norm(frame["velocity"], axis=-1).max())
            for frame in snapshots
        ),
    )
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)

    def face_colors(frame):
        colors = plt.colormaps["viridis"](frame["rates"] / rate_max)
        colors[:, :3] *= (0.65 + 0.35 * np.maximum(0.0, frame["normals"] @ light))[
            :, None
        ]
        return colors

    collections = []
    for ax in (tank_ax, zoom_ax):
        crystal = Poly3DCollection(
            snapshots[0]["faces"],
            facecolors=face_colors(snapshots[0]),
            edgecolor="#163b51",
            linewidths=0.8,
            alpha=1.0,
        )
        ax.add_collection3d(crystal)
        ax.set(xlabel="x / width", ylabel="y / depth", zlabel="z / height")
        ax.view_init(elev=22, azim=-55)
        collections.append(crystal)

    # Draw all twelve tank edges using 3D coordinates. Axis proportions are
    # physical lengths, preventing a rectangular tank from looking like a cube.
    corners = np.array(
        [
            [x, y, z]
            for x in (0.0, c.width)
            for y in (0.0, c.depth)
            for z in (0.0, c.height)
        ]
    )
    edges = [
        [a, b]
        for i, a in enumerate(corners)
        for b in corners[i + 1 :]
        if np.count_nonzero(a != b) == 1
    ]
    tank_ax.add_collection3d(Line3DCollection(edges, colors="#657786", linewidths=0.8))
    tank_ax.set(
        xlim=(0, c.width),
        ylim=(0, c.depth),
        zlim=(0, c.height),
        title="Crystal inside 3D tank",
    )
    tank_ax.set_box_aspect((c.width, c.depth, c.height))
    bounds = sim.crystal.bounds
    margin = max(0.1, float(sim.crystal.dimensions.max()) * 0.3)
    for axis, limits in zip(
        (zoom_ax.set_xlim, zoom_ax.set_ylim, zoom_ax.set_zlim), bounds.T
    ):
        axis(limits[0] - margin, limits[1] + margin)
    zoom_ax.set_box_aspect(sim.crystal.dimensions + 2 * margin)
    zoom_ax.set_title("Facets and liquid displacement\n(arrows scaled)", fontsize=11)
    # The overview labels z; omit its repeated close-up label beside the colorbar.
    zoom_ax.set_zlabel("")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig.colorbar(
        ScalarMappable(norm=Normalize(0.0, rate_max), cmap="viridis"),
        ax=zoom_ax,
        shrink=0.6,
        pad=0.02,
        label="Face normal growth rate",
    )

    def surface(level):
        return [
            [0, 0, level],
            [c.width, 0, level],
            [c.width, c.depth, level],
            [0, c.depth, level],
        ]

    water = Poly3DCollection(
        [surface(snapshots[0]["liquid_level"])],
        facecolor="#3098c5",
        edgecolor="#2389b0",
        linewidths=0.8,
        alpha=0.15,
    )
    tank_ax.add_collection3d(water)
    arrows = {"flow": None, "normal": None}

    times = [row["time"] for row in sim.history]
    history_ax.plot(
        times, [row["volume"] for row in sim.history], label="Actual crystal volume"
    )
    history_ax.set(title="Volumetric growth", xlabel="Simulation time", ylabel="Volume")
    history_ax.grid(alpha=0.2)
    level_ax = history_ax.twinx()
    levels = [row["liquid_level"] for row in sim.history]
    level_ax.plot(times, levels, color="#c85437", linestyle="--", label="Liquid level")
    level_ax.set_ylabel("Liquid level", color="#c85437")
    history_ax.set_ylim(0.0, max(row["volume"] for row in sim.history) * 1.1)
    level_ax.set_ylim(
        levels[0], levels[0] + max(0.001, (max(levels) - levels[0]) * 1.25)
    )
    level_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    lines = history_ax.lines[:1] + level_ax.lines[:1]
    history_ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
    cursor = history_ax.axvline(0.0, color="gray", linestyle="--")

    # Use fixed central planes and fixed color scales through the replay. XY is
    # field[z,:,:], while XZ is field[:,y,:]; z is always the vertical direction.
    iz, iy = c.nz // 2, c.ny // 2
    specifications = [
        (
            "temperature",
            lambda field: field[:, iy, :],
            (0, c.width, 0, c.height),
            f"Temperature, XZ at y={sim.tank.y[iy]:.2f}",
            "z / height",
            "coolwarm",
        ),
        (
            "concentration",
            lambda field: field[iz, :, :],
            (0, c.width, 0, c.depth),
            f"Concentration, XY at z={sim.tank.z[iz]:.2f}",
            "y / depth",
            "viridis",
        ),
        (
            "liquid_fraction",
            lambda field: field[:, iy, :],
            (0, c.width, 0, c.height),
            f"Liquid occupancy, XZ at y={sim.tank.y[iy]:.2f}",
            "z / height",
            "Blues",
        ),
    ]
    images, solids = [], []
    for panel, (ax, (key, section, extent, title, ylabel, cmap)) in enumerate(
        zip(slice_axes, specifications)
    ):
        if key == "liquid_fraction":
            lo, hi = 0.0, 1.0
        else:
            samples = [
                section(frame[key])[section(frame["liquid_fraction"]) > 0.0]
                for frame in snapshots
            ]
            lo = min(float(values.min()) for values in samples if values.size)
            hi = max(float(values.max()) for values in samples if values.size)
            if hi - lo < 1e-9:
                lo, hi = lo - 0.1, hi + 0.1
        field = section(snapshots[0][key])
        displayed = (
            field
            if key == "liquid_fraction"
            else np.ma.masked_where(
                section(snapshots[0]["liquid_fraction"]) <= 0.0, field
            )
        )
        image = ax.imshow(
            displayed,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        images.append(image)
        # Gray polygons are exact solid cross-sections, rather than voxel-center
        # masks. Air and solid are not displayed as zero-concentration liquid.
        solid = Polygon(
            snapshots[0]["sections"][panel],
            closed=True,
            facecolor="#343c44",
            edgecolor="#202a35",
            linewidth=0.6,
        )
        ax.add_patch(solid)
        solids.append(solid)
        fig.colorbar(image, ax=ax, shrink=0.8)
        ax.set(title=title, xlabel="x / width", ylabel=ylabel)
    status = fig.text(0.5, 0.005, "", ha="center", fontsize=9)

    def update(index):
        frame = snapshots[index]
        for collection in collections:
            collection.set_verts(frame["faces"])
            collection.set_facecolor(face_colors(frame))
        water.set_verts([surface(frame["liquid_level"])])
        for panel, (image, solid, (key, section, *_rest)) in enumerate(
            zip(images, solids, specifications)
        ):
            field = section(frame[key])
            image.set_data(
                field
                if key == "liquid_fraction"
                else np.ma.masked_where(section(frame["liquid_fraction"]) <= 0.0, field)
            )
            solid.set_xy(frame["sections"][panel])
            if panel == 0:
                # The last wet cell may be a thin cut cell. Clip its displayed
                # temperature to the actual surface, not the top of the whole voxel.
                image.set_clip_path(
                    Rectangle(
                        (0.0, 0.0),
                        c.width,
                        frame["liquid_level"],
                        transform=slice_axes[panel].transData,
                    )
                )
        if arrows["flow"] is not None:
            arrows["flow"].remove()
        if arrows["normal"] is not None:
            arrows["normal"].remove()
        centers = np.asarray([face.mean(axis=0) for face in frame["faces"]])
        normal_length = 0.13 * float(sim.crystal.dimensions.max())
        arrows["normal"] = zoom_ax.quiver(
            *centers.T,
            *(frame["normals"] * normal_length).T,
            color="#c55e25",
            linewidth=0.8,
            arrow_length_ratio=0.25,
        )
        positions = sim.tank.positions
        velocity = frame["velocity"].reshape(-1, 3)
        speed = np.linalg.norm(velocity, axis=1)
        inside = np.all(
            (positions >= bounds[0] - margin) & (positions <= bounds[1] + margin),
            axis=1,
        )
        selected = np.flatnonzero(
            inside
            & (frame["liquid_fraction"].ravel() > 0.95)
            & (speed > speed_max * 0.025)
        )
        if len(selected) > 48:
            selected = selected[np.linspace(0, len(selected) - 1, 48, dtype=int)]
        if len(selected):
            scaled = velocity[selected] * (
                0.2 * float(sim.crystal.dimensions.max()) / speed_max
            )
            endpoints = positions[selected] + scaled
            visible = np.all(
                (endpoints >= bounds[0] - margin) & (endpoints <= bounds[1] + margin),
                axis=1,
            )
            selected, scaled = selected[visible], scaled[visible]
            arrows["flow"] = zoom_ax.quiver(
                *positions[selected].T,
                *scaled.T,
                color="#256da8",
                linewidth=0.7,
                arrow_length_ratio=0.25,
            )
        else:
            arrows["flow"] = None
        cursor.set_xdata([frame["time"], frame["time"]])
        status.set_text(
            f"t={frame['time']:.2f}/{c.duration:g}   volume={frame['volume']:.5f}   "
            f"liquid level={frame['liquid_level']:.5f}   (blue arrows: quasistatic displacement; orange: face normals)"
        )
        return collections + images + solids + [water, cursor, status]

    # Export with Pillow directly: omitting its loop option makes the GIF play
    # once. The window's repeat=False separately prevents interactive replay loops.
    gif_frames = []
    for index in range(len(snapshots)):
        update(index)
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=90)
            buffer.seek(0)
            with Image.open(buffer) as rendered:
                gif_frames.append(rendered.convert("RGB"))
    gif_frames[0].save(
        args.output / "growth.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=80,
        disposal=2,
    )
    del gif_frames
    update(len(snapshots) - 1)
    fig.savefig(args.output / "final_tank.png", dpi=150)
    print(f"Volume: {sim.history[0]['volume']:.6f} -> {sim.crystal.volume:.6f}")
    print(
        f"Final time: {sim.time:g}; mass drift: {sim.total_mass - sim.history[0]['total_mass']:.3e}"
    )
    print(
        f"Liquid level: {snapshots[0]['liquid_level']:.6f} -> {sim.tank.liquid_level:.6f}"
    )
    print(
        f"Carrier-volume drift: {sim.tank.liquid_volume - sim.tank.carrier_volume:.3e}"
    )
    print(f"Results: {args.output.resolve()}")
    if not headless:
        # Keep a reference until the window closes; the final frame stays visible.
        animation = FuncAnimation(
            fig, update, frames=len(snapshots), interval=80, repeat=False
        )
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
