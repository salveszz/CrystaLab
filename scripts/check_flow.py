# HOW THIS FILE WORKS
# Verify the new SI momentum component independently of the growth engine.
# Compare periodic 3D channel flow with its quadratic steady solution and a
# viscous eigenmode decay with its exact time dependence. All three axes remain
# present. An ABC/Beltrami vortex exercises nonlinear advection and pressure in
# all three velocity components; its analytic solution is derived below.
# Run rest and manufactured vortical forcing around the physical fixed cube.
# Save declared settings/source fingerprints before measurements. Export raw
# fields, metrics and plots to a fresh directory, with all model limits explicit.
# No experimental calibration, quality labels or growing-boundary claim occurs.

"""Verify isothermal three-dimensional Navier-Stokes against analytic flows."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.flow import IncompressibleFlow
from simulation.units import FluidPropertiesSI
from describe_physical_setup import describe_setup


def abc_velocity(position, t, length, speed, nu):
    """curl(u)=k*u: nonlinear transport is grad(|u|2/2), absorbed by pressure.

    Laplacian(u)=-k2*u, so u(t)=exp(-nu*k2*t)*u(0),
    p(t)=-rho*|u(t)|2/2 plus a spatially uniform gauge.
    """
    x, y, z = np.moveaxis(position * (2 * np.pi / length), -1, 0)
    amplitude = speed * np.exp(-nu * (2 * np.pi / length) ** 2 * t)
    return amplitude * np.stack(
        (np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)), axis=-1
    )


def abc_pressure(flow, t, length, speed):
    z, y, x = np.meshgrid(
        *[(np.arange(n) + 0.5) * h for n, h in zip(flow.shape, flow.spacing)],
        indexing="ij",
    )
    velocity = abc_velocity(
        np.stack((x, y, z), axis=-1),
        t,
        length,
        speed,
        flow.properties.kinematic_viscosity_m2_per_s,
    )
    pressure = -0.5 * flow.properties.density_kg_per_m3 * np.sum(velocity**2, axis=-1)
    return pressure - pressure.mean()


def channel_case(properties, settings, nz):
    lengths = settings["lengths_m"]
    flow = IncompressibleFlow(
        lengths,
        [*settings["horizontal_counts"], nz],
        properties,
        periodic_xyz=(True, True, False),
    )
    nu = properties.kinematic_viscosity_m2_per_s
    H = lengths[2]
    force = 2 * nu * settings["target_steady_max_speed_m_per_s"] / H**2
    flow.run_until(settings["duration_s"], settings["max_dt_s"], (force, 0, 0))
    z = flow.face_positions(0)[..., 2]
    exact = force * (2 * H * z - z**2) / (2 * nu)
    error = np.sqrt(np.mean((flow.velocity[0] - exact) ** 2)) / np.sqrt(
        np.mean(exact**2)
    )
    return flow, dict(
        relative_rms_error=float(error), nz=nz, acceleration_m_per_s2=force
    )


def decay_case(properties, settings, dt):
    flow = IncompressibleFlow(
        settings["lengths_m"],
        settings["counts_xyz"],
        properties,
        periodic_xyz=(True, True, False),
    )
    k = np.pi / (2 * settings["lengths_m"][2])
    initial = settings["initial_speed_m_per_s"] * np.sin(
        k * flow.face_positions(0)[..., 2]
    )
    flow.set_velocity(
        [initial, np.zeros_like(flow.velocity[1]), np.zeros_like(flow.velocity[2])]
    )
    flow.run_until(settings["duration_s"], dt)
    exact = initial * np.exp(
        -properties.kinematic_viscosity_m2_per_s * k * k * settings["duration_s"]
    )
    error = np.sqrt(np.mean((flow.velocity[0] - exact) ** 2)) / np.sqrt(
        np.mean(exact**2)
    )
    return flow, dict(relative_rms_error=float(error), dt_s=dt)


def abc_case(properties, settings, n):
    length, speed = settings["length_m"], settings["coefficient_speed_m_per_s"]
    flow = IncompressibleFlow(
        [length] * 3, [n] * 3, properties, periodic_xyz=(True, True, True)
    )
    nu = properties.kinematic_viscosity_m2_per_s
    flow.set_velocity(
        [
            abc_velocity(flow.face_positions(i), 0, length, speed, nu)[..., i]
            for i in range(3)
        ]
    )
    flow.set_pressure(abc_pressure(flow, 0, length, speed))
    flow.run_until(settings["duration_s"], settings["max_dt_s"])
    error, scale = 0.0, 0.0
    for i, component in enumerate(flow.components):
        exact = abc_velocity(flow.face_positions(i), flow.time_s, length, speed, nu)[
            ..., i
        ]
        index = tuple(component.cells.T)
        error += float(np.sum((flow.velocity[i][index] - exact[index]) ** 2))
        scale += float(np.sum(exact[index] ** 2))
    exact_pressure = abc_pressure(flow, flow.time_s, length, speed)
    p_error = np.sqrt(np.mean((flow.pressure - exact_pressure) ** 2)) / np.sqrt(
        np.mean(exact_pressure**2)
    )
    return flow, dict(
        n=n,
        relative_velocity_error=float(np.sqrt(error / scale)),
        relative_pressure_error=float(p_error),
    )


def reference_flow(settings, counts):
    describe_setup(settings)
    domain, crystal = settings["domain"], settings["crystal"]
    center, size = np.asarray(crystal["center_m"]), np.asarray(crystal["size_m"])
    lengths = [*domain["size_m"][:2], domain["liquid_height_m"]]
    return IncompressibleFlow(
        lengths,
        counts,
        FluidPropertiesSI(**settings["fluid"]),
        solid_box_m=[center - size / 2, center + size / 2],
    )


def obstacle_case(reference, settings):
    flow = reference_flow(reference, settings["counts_xyz"])
    force = settings["vortical_acceleration_scale_m_per_s2"]

    def acceleration(position, _time):
        x, y, z = np.moveaxis(position / flow.lengths, -1, 0)
        # This manufactured, nonconservative force drives a 3D recirculation.
        # It is not interpreted as an experimentally calibrated pump or stirrer.
        envelope = np.sin(np.pi * z)
        return force * np.stack(
            (-(y - 0.5) * envelope, (x - 0.5) * envelope, np.zeros_like(z)), axis=-1
        )

    flow.run_until(settings["duration_s"], settings["max_dt_s"], acceleration)
    return flow, dict(
        acceleration_scale_m_per_s2=force,
        forcing="manufactured_vortical_body_acceleration",
    )


def _save_fields(path, flow):
    np.savez_compressed(
        path,
        u_x_m_per_s=flow.velocity[0],
        u_y_m_per_s=flow.velocity[1],
        u_z_m_per_s=flow.velocity[2],
        centered_velocity_m_per_s=flow.centered_velocity(),
        dynamic_pressure_Pa=flow.pressure,
        fluid=flow.fluid,
        lengths_m=flow.lengths,
        solid_vertices_m=flow.solid_vertices_m,
    )


def run_study(protocol, reference, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh flow-verification output directory")
    properties = FluidPropertiesSI(**reference["fluid"])
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/simulation/flow.py",
            "src/simulation/units.py",
            "src/simulation/crystal.py",
            "scripts/check_flow.py",
            "scripts/describe_physical_setup.py",
        )
    }
    manifest = dict(
        protocol=protocol,
        reference=reference,
        sources=sources,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=importlib.metadata.version("scipy"),
            threads={
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
        ),
        method="uniform MAC, limited linear conservative momentum advection, implicit viscosity, incremental pressure correction; first order in time",
    )
    output.mkdir(parents=True)
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for name, digest in sources.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Source changed before flow verification began")
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    results = []

    def record(name, flow, metrics, elapsed):
        row = dict(case=name, **metrics, **flow.measurements(), runtime_s=elapsed)
        row["constraints_pass"] = (
            row["scaled_divergence"] <= protocol["max_scaled_divergence"]
            and row["blocked_face_speed_m_per_s"]
            <= protocol["max_blocked_face_speed_m_per_s"]
        )
        results.append(row)
        _save_fields(output / f"{name}.npz", flow)
        print(name, json.dumps(row), flush=True)
        return row

    rest = reference_flow(reference, protocol["obstacle"]["counts_xyz"])
    start = time.perf_counter()
    rest.run_until(reference["duration_s"], 1.0)
    row = record("rest", rest, {}, time.perf_counter() - start)
    rest_pass = (
        row["max_speed_m_per_s"] == 0 and row["time_s"] == reference["duration_s"]
    )
    for family, values, builder in (
        ("channel", protocol["channel"]["vertical_counts"], channel_case),
        ("decay", protocol["decay"]["timesteps_s"], decay_case),
        ("abc", protocol["abc"]["counts"], abc_case),
    ):
        for value in values:
            start = time.perf_counter()
            flow, metrics = builder(properties, protocol[family], value)
            record(f"{family}_{value}", flow, metrics, time.perf_counter() - start)
    start = time.perf_counter()
    obstacle, metrics = obstacle_case(reference, protocol["obstacle"])
    record("obstacle", obstacle, metrics, time.perf_counter() - start)
    gates = dict(
        rest=rest_pass,
        incompressibility_and_walls=all(r["constraints_pass"] for r in results),
        nonzero_obstacle=results[-1]["max_speed_m_per_s"] > 1e-6,
    )
    for family, metric, tolerance in (
        (
            "channel",
            "relative_rms_error",
            protocol["channel"]["finest_relative_rms_tolerance"],
        ),
        (
            "decay",
            "relative_rms_error",
            protocol["decay"]["finest_relative_rms_tolerance"],
        ),
        (
            "abc",
            "relative_velocity_error",
            protocol["abc"]["finest_relative_velocity_tolerance"],
        ),
    ):
        errors = [r[metric] for r in results if r["case"].startswith(family)]
        gates[family] = (
            all(b < a for a, b in zip(errors, errors[1:])) and errors[-1] <= tolerance
        )
    pressure = [
        r["relative_pressure_error"] for r in results if r["case"].startswith("abc")
    ]
    gates["abc_pressure"] = (
        all(b < a for a, b in zip(pressure, pressure[1:]))
        and pressure[-1] <= protocol["abc"]["finest_relative_pressure_tolerance"]
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Source changed during flow verification; results cannot be accepted"
            )
    report = dict(
        results=results,
        gates=gates,
        overall_pass=all(gates.values()),
        protocol=protocol,
        limits=protocol["limits"],
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Fixed-geometry 3D flow verification",
        "",
        protocol["limits"],
        "",
        "| Case | Velocity error | Pressure error | Scaled divergence | Blocked-face speed (m/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        e = row.get("relative_rms_error", row.get("relative_velocity_error"))
        p = row.get("relative_pressure_error")
        lines.append(
            f"| {row['case']} | {e if e is not None else 'n/a'} | {p if p is not None else 'n/a'} | {row['scaled_divergence']:.3e} | {row['blocked_face_speed_m_per_s']:.3e} |"
        )
    lines += [
        "",
        f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}.",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in gates.items()
        ],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot_report(report, obstacle, output)
    return report


def plot_report(report, obstacle, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, family, x, error in zip(
        axes,
        ("channel", "decay", "abc"),
        ("nz", "dt_s", "n"),
        ("relative_rms_error", "relative_rms_error", "relative_velocity_error"),
    ):
        rows = [r for r in report["results"] if r["case"].startswith(family)]
        axis.loglog(
            [r[x] for r in rows], [r[error] for r in rows], "o-", label="velocity"
        )
        if family == "abc":
            axis.loglog(
                [r[x] for r in rows],
                [r["relative_pressure_error"] for r in rows],
                "s-",
                label="pressure",
            )
            axis.legend()
        axis.set(xlabel=x, ylabel="Relative RMS error", title=family)
        axis.set_xticks([r[x] for r in rows], labels=[f"{r[x]:g}" for r in rows])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=0.25)
    fig.suptitle("Analytic 3D momentum verification; not experimental KDP accuracy")
    fig.savefig(output / "summary.png", dpi=140)
    plt.close(fig)
    center = obstacle.centered_velocity()
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    z, y, x = np.meshgrid(
        *[
            (np.arange(n) + 0.5) * h * 1000
            for n, h in zip(obstacle.shape, obstacle.spacing)
        ],
        indexing="ij",
    )
    sample = np.zeros(obstacle.shape, bool)
    sample[::2, ::2, ::2] = True
    sample &= obstacle.fluid
    ax.quiver(
        x[sample],
        y[sample],
        z[sample],
        *np.moveaxis(center[sample], -1, 0),
        length=1e4,
        normalize=False,
        color="#2371a8",
        linewidth=0.5,
    )
    v = obstacle.solid_vertices_m * 1000
    if len(v):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        lo, hi = v.min(axis=0), v.max(axis=0)
        for dimension in range(3):
            other = [d for d in range(3) if d != dimension]
            for position in (lo[dimension], hi[dimension]):
                corners = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    point = lo.copy()
                    point[dimension] = position
                    point[other[0]] = (lo, hi)[a][other[0]]
                    point[other[1]] = (lo, hi)[b][other[1]]
                    corners.append(point)
                ax.add_collection3d(
                    Poly3DCollection(
                        [corners], facecolor="#535d67", edgecolor="#222222", alpha=0.9
                    )
                )
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel="z (mm)",
        xlim=(0, obstacle.lengths[0] * 1000),
        ylim=(0, obstacle.lengths[1] * 1000),
        zlim=(0, obstacle.lengths[2] * 1000),
        title="Manufactured forcing; arrows scaled",
    )
    # Matplotlib normalizes this array in place; keep solver dimensions intact.
    ax.set_box_aspect(obstacle.lengths.copy())
    panel = fig.add_subplot(1, 2, 2)
    iz = obstacle.shape[0] // 2
    image = panel.imshow(
        np.ma.masked_where(~obstacle.fluid[iz], obstacle.pressure[iz]),
        origin="lower",
        extent=(0, obstacle.lengths[0] * 1000, 0, obstacle.lengths[1] * 1000),
    )
    panel.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Dynamic pressure slice (Pa; mean-zero gauge)",
    )
    fig.colorbar(image, ax=panel)
    fig.suptitle("Fixed 3D crystal flow example; no growth or calibrated stirring")
    fig.savefig(output / "fixed_crystal_flow.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/flow_verification.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/flow_verification"
    )
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    reference = json.loads(
        (args.config.parent / protocol["fluid_reference"]).read_text()
    )
    report = run_study(protocol, reference, args.output)
    print("ASSESSMENT:", "PASS" if report["overall_pass"] else "FAIL", args.output)
    raise SystemExit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
