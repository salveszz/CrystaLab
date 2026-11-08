# HOW THIS FILE WORKS
# Verify actual metric-dependent SI flow and solute operators on unequal grids.
# Reuse the independent channel/decay/ABC references with volume-weighted errors.
# Run true cell-average scalar waves/diffusion and coupled uniform solute on the
# same focused mesh as nonlinear flow. An aligned cube has explicitly refined
# near-surface cells; nonzero manufactured forcing tests actual coupling there.
# Declare all criteria before measuring; archive exact sources, mesh coordinates,
# reference/final fields, metrics and figures in a fresh results folder.
# This is numerical verification, not oblique cut-cell or physical growth validation.

"""Verify rectilinear and focused-grid flow/solute with independent references."""

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
from simulation.mesh import focused_axis_edges
from simulation.solute import FlowSoluteSimulation, SoluteTransport
from simulation.units import FluidPropertiesSI, finite_number
from check_flow import abc_case, abc_pressure, abc_velocity, channel_case, decay_case
from check_solute import obstacle_case
from describe_physical_setup import describe_setup


def stretched_edges(lengths, counts, amplitudes):
    result = []
    for length, n, amplitude in zip(lengths, counts, amplitudes):
        amplitude = finite_number(amplitude, "stretch amplitude")
        if abs(amplitude) >= 1:
            raise ValueError("Smooth stretch amplitudes must have magnitude <1")
        coordinate = np.linspace(0, 1, n + 1)
        edge = length * (
            coordinate + amplitude * np.sin(2 * np.pi * coordinate) / (2 * np.pi)
        )
        edge[0], edge[-1] = 0, length
        result.append(edge)
    return result


def aligned_cube_edges(reference, settings):
    lengths = [
        *reference["domain"]["size_m"][:2],
        reference["domain"]["liquid_height_m"],
    ]
    center, size = reference["crystal"]["center_m"], reference["crystal"]["size_m"]
    inner = settings["cells_across_solid"]
    result = []
    for length, n, midpoint, width in zip(
        lengths, settings["counts_xyz"], center, size
    ):
        low, high = midpoint - width / 2, midpoint + width / 2
        left = round((n - inner) * low / (length - width))
        right = n - inner - left
        if min(left, inner, right) < 1:
            raise ValueError("Aligned-cube mesh needs cells on both sides of its solid")
        result.append(
            np.r_[
                np.linspace(0, low, left + 1)[:-1],
                np.linspace(low, high, inner + 1)[:-1],
                np.linspace(high, length, right + 1),
            ]
        )
    return result


def focused_abc_case(properties, settings, n):
    L, speed = settings["length_m"], settings["coefficient_speed_m_per_s"]
    edges = [focused_axis_edges(L, n, L / 2, settings["focus_width_m"])] * 3
    flow = IncompressibleFlow(
        [L] * 3, [n] * 3, properties, periodic_xyz=(True,) * 3, grid_edges_m=edges
    )
    nu = properties.kinematic_viscosity_m2_per_s
    flow.set_velocity(
        [
            abc_velocity(flow.face_positions(i), 0, L, speed, nu)[..., i]
            for i in range(3)
        ]
    )
    flow.set_pressure(abc_pressure(flow, 0, L, speed))
    model = FlowSoluteSimulation(flow, settings["concentration_kg_per_m3"])
    model.run_until(settings["duration_s"], settings["max_dt_s"])
    error, scale = 0.0, 0.0
    for i, component in enumerate(flow.components):
        exact = abc_velocity(flow.face_positions(i), flow.time_s, L, speed, nu)[..., i]
        ix = tuple(component.cells.T)
        error += float(
            np.sum(component.volumes * (flow.velocity[i][ix] - exact[ix]) ** 2)
        )
        scale += float(np.sum(component.volumes * exact[ix] ** 2))
    pressure = abc_pressure(flow, flow.time_s, L, speed)
    p_error = np.sqrt(
        np.sum(flow.cell_volumes * (flow.pressure - pressure) ** 2)
        / np.sum(flow.cell_volumes * pressure**2)
    )
    return (
        flow,
        model.solute,
        dict(
            n=n,
            relative_velocity_error=float(np.sqrt(error / scale)),
            relative_pressure_error=float(p_error),
            uniform_error_kg_per_m3=float(
                np.max(
                    abs(
                        model.solute.concentration - settings["concentration_kg_per_m3"]
                    )
                )
            ),
        ),
        None,
    )


def scalar_case(properties, settings, n, amplitudes, *, diffusion=False):
    lengths = np.array(settings["lengths_m"])
    edges = stretched_edges(lengths, [n] * 3, amplitudes)
    flow = IncompressibleFlow(
        lengths,
        [n] * 3,
        properties,
        periodic_xyz=(not diffusion,) * 3,
        grid_edges_m=edges,
    )
    speed = np.zeros(3) if diffusion else np.array(settings["velocity_m_per_s"])
    flow.set_velocity([np.full_like(v, u) for v, u in zip(flow.velocity, speed)])
    k = (np.pi if diffusion else 2 * np.pi) / lengths
    xyz = flow.cell_positions()
    average = flow.grid.product(
        [np.sinc(ka * w / (2 * np.pi)) for ka, w in zip(k[::-1], flow.grid.widths)]
    )
    mean, amplitude, duration = (
        settings["mean_kg_per_m3"],
        settings["amplitude_kg_per_m3"],
        settings["duration_s"],
    )
    mode = np.prod(np.cos(xyz * k), axis=-1) if diffusion else np.cos(xyz @ k)
    initial = mean + amplitude * average * mode
    scalar = SoluteTransport(flow, initial)
    scalar.advance(duration, max_substep_s=settings.get("max_substep_s"))
    final_mode = mode if diffusion else np.cos(xyz @ k - np.dot(k, speed) * duration)
    exact = (
        mean
        + amplitude
        * average
        * np.exp(-properties.solute_diffusivity_m2_per_s * np.dot(k, k) * duration)
        * final_mode
    )
    error = np.sqrt(
        np.sum(flow.cell_volumes * (scalar.concentration - exact) ** 2)
        / np.sum(flow.cell_volumes * (exact - mean) ** 2)
    )
    violation = max(
        float(initial.min() - scalar.concentration.min()),
        float(scalar.concentration.max() - initial.max()),
        0,
    )
    return (
        flow,
        scalar,
        dict(
            n=n,
            relative_rms_error=float(error),
            concentration_bound_violation_kg_per_m3=violation,
        ),
        exact,
    )


def _validate(protocol):
    for family, key in (
        ("channel", "vertical_counts"),
        ("decay", "timesteps_s"),
        ("abc", "counts"),
        ("focused_abc_solute", "counts"),
        ("wave", "counts"),
        ("diffusion", "counts"),
    ):
        values = protocol[family][key]
        if len(values) < 3:
            raise ValueError("Declare at least three refinement levels")
        for value in values:
            finite_number(value, key, positive=True)
            if key != "timesteps_s" and (not isinstance(value, int) or value < 3):
                raise ValueError("Grid counts must be integers >=3")
        if not all(
            b < a if key == "timesteps_s" else a < b for a, b in zip(values, values[1:])
        ):
            raise ValueError("Use increasing grid counts and decreasing timesteps")
    if len(protocol["stretch_amplitudes_xyz"]) != 3:
        raise ValueError("Declare three stretch amplitudes")
    for value in protocol["stretch_amplitudes_xyz"]:
        if abs(finite_number(value, "stretch amplitude")) >= 1:
            raise ValueError("Smooth stretch amplitudes must have magnitude <1")


def run_study(protocol, reference, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh rectilinear-verification directory")
    describe_setup(reference)
    _validate(protocol)
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/simulation/flow.py",
            "src/simulation/mesh.py",
            "src/simulation/solute.py",
            "src/simulation/units.py",
            "src/simulation/crystal.py",
            "scripts/check_flow.py",
            "scripts/check_solute.py",
            "scripts/check_rectilinear_flow.py",
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
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
        ),
        method="Rectilinear MAC; actual primal/dual volumes and face areas; compatible pressure projection; symmetric viscous stiffness; conservative limited momentum and scalar face transfers.",
    )
    output.mkdir(parents=True)
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for name, digest in sources.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Source changed before verification")
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    rows = []

    def record(name, builder):
        start = time.perf_counter()
        flow, scalar, metrics, exact = builder()
        row = dict(case=name, **flow.measurements())
        archive = dict(
            u_x_m_per_s=flow.velocity[0],
            u_y_m_per_s=flow.velocity[1],
            u_z_m_per_s=flow.velocity[2],
            dynamic_pressure_Pa=flow.pressure,
            fluid=flow.fluid,
            lengths_m=flow.lengths,
            x_edges_m=flow.grid.edges[2],
            y_edges_m=flow.grid.edges[1],
            z_edges_m=flow.grid.edges[0],
            cell_volumes_m3=flow.cell_volumes,
            solid_vertices_m=flow.solid_vertices_m,
        )
        if scalar is not None:
            row.update(scalar.measurements())
            row["relative_mass_drift"] = abs(
                scalar.mass_kg / scalar.initial_mass_kg - 1
            )
            archive["concentration_kg_per_m3"] = scalar.concentration
        if exact is not None:
            archive["reference_concentration_kg_per_m3"] = exact
        row.update(metrics, runtime_s=time.perf_counter() - start)
        row["constraints_pass"] = (
            row["scaled_divergence"] <= protocol["max_scaled_divergence"]
            and row["blocked_face_speed_m_per_s"]
            <= protocol["max_blocked_face_speed_m_per_s"]
            and row.get("relative_mass_drift", 0) <= protocol["max_relative_mass_drift"]
            and row.get("uniform_error_kg_per_m3", 0)
            <= protocol["max_uniform_error_kg_per_m3"]
            and row.get("concentration_bound_violation_kg_per_m3", 0)
            <= protocol["max_concentration_bound_violation_kg_per_m3"]
        )
        archive["time_s"] = row["time_s"]
        np.savez_compressed(output / f"{name}.npz", **archive)
        rows.append(row)
        print(name, json.dumps(row, allow_nan=False), flush=True)

    properties = FluidPropertiesSI(**reference["fluid"])
    amplitudes = protocol["stretch_amplitudes_xyz"]
    for n in protocol["channel"]["vertical_counts"]:
        counts = [*protocol["channel"]["horizontal_counts"], n]
        edges = stretched_edges(protocol["channel"]["lengths_m"], counts, amplitudes)

        def channel():
            flow, metrics = channel_case(
                properties, protocol["channel"], n, grid_edges_m=edges
            )
            return flow, None, dict(metrics, n=n), None

        record(f"channel_{n}", channel)
    for dt in protocol["decay"]["timesteps_s"]:
        edges = stretched_edges(
            protocol["decay"]["lengths_m"], protocol["decay"]["counts_xyz"], amplitudes
        )

        def decay():
            flow, metrics = decay_case(
                properties, protocol["decay"], dt, grid_edges_m=edges
            )
            return flow, None, metrics, None

        record(f"decay_{dt}", decay)
    for n in protocol["abc"]["counts"]:
        edges = stretched_edges([protocol["abc"]["length_m"]] * 3, [n] * 3, amplitudes)

        def abc():
            flow, metrics = abc_case(properties, protocol["abc"], n, grid_edges_m=edges)
            return flow, None, metrics, None

        record(f"abc_{n}", abc)
    for n in protocol["focused_abc_solute"]["counts"]:
        record(
            f"focused_abc_solute_{n}",
            lambda: focused_abc_case(properties, protocol["focused_abc_solute"], n),
        )
    for family in ("wave", "diffusion"):
        for n in protocol[family]["counts"]:
            record(
                f"{family}_{n}",
                lambda: scalar_case(
                    properties,
                    protocol[family],
                    n,
                    amplitudes,
                    diffusion=family == "diffusion",
                ),
            )
    edges = aligned_cube_edges(reference, protocol["obstacle"])
    for uniform in (True, False):

        def obstacle():
            scalar, _exact, metrics, initial = obstacle_case(
                reference, protocol["obstacle"], uniform=uniform, grid_edges_m=edges
            )
            metrics["concentration_bound_violation_kg_per_m3"] = max(
                float(
                    initial[scalar.flow.fluid].min()
                    - scalar.concentration[scalar.flow.fluid].min()
                ),
                float(
                    scalar.concentration[scalar.flow.fluid].max()
                    - initial[scalar.flow.fluid].max()
                ),
                0,
            )
            if uniform:
                metrics["uniform_error_kg_per_m3"] = metrics[
                    "uniform_max_error_kg_per_m3"
                ]
            return scalar.flow, scalar, metrics, None

        record(
            "aligned_cube_uniform" if uniform else "aligned_cube_nonuniform", obstacle
        )
    gates = dict(invariants=all(row["constraints_pass"] for row in rows))
    for family in (
        "channel",
        "decay",
        "abc",
        "focused_abc_solute",
        "wave",
        "diffusion",
    ):
        for metric, tolerance in (
            (
                ("relative_velocity_error", "finest_relative_velocity_tolerance"),
                ("relative_pressure_error", "finest_relative_pressure_tolerance"),
            )
            if family in ("abc", "focused_abc_solute")
            else (("relative_rms_error", "finest_relative_rms_tolerance"),)
        ):
            errors = [
                row[metric] for row in rows if row["case"].startswith(family + "_")
            ]
            gates[family + "_" + metric] = (
                all(b < a for a, b in zip(errors, errors[1:]))
                and errors[-1] <= protocol[family][tolerance]
            )
    gates["nonzero_aligned_cube"] = all(
        row["max_speed_m_per_s"] > 1e-6
        for row in rows
        if row["case"].startswith("aligned_cube_")
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Source changed during verification; results are not accepted"
            )
    report = dict(
        results=rows,
        gates=gates,
        overall_pass=all(gates.values()),
        protocol=protocol,
        limits=protocol["limits"],
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Rectilinear flow and solute verification",
        "",
        protocol["limits"],
        "",
        "| Case | Velocity/scalar error | Pressure error | Relative mass drift |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['case']} | {r.get('relative_velocity_error', r.get('relative_rms_error', 'n/a'))} | {r.get('relative_pressure_error', 'n/a')} | {r.get('relative_mass_drift', 'n/a')} |"
        for r in rows
    ]
    lines += [
        "",
        f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}.",
        "",
        *[f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot_report(report, output)
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for axis, family in zip(
        axes.flat,
        ("channel", "decay", "abc", "focused_abc_solute", "wave", "diffusion"),
    ):
        rows = [r for r in report["results"] if r["case"].startswith(family + "_")]
        x = [r["dt_s"] if family == "decay" else r["n"] for r in rows]
        key = (
            "relative_velocity_error"
            if family in ("abc", "focused_abc_solute")
            else "relative_rms_error"
        )
        axis.loglog(
            x,
            [r[key] for r in rows],
            "o-",
            label="Velocity" if "velocity" in key else "Field",
        )
        if family in ("abc", "focused_abc_solute"):
            axis.loglog(
                x, [r["relative_pressure_error"] for r in rows], "s-", label="Pressure"
            )
            axis.legend()
        axis.set(
            title=family.replace("_", " "),
            xlabel="Timestep (s)" if family == "decay" else "Cells per refined axis",
            ylabel="Relative volume-weighted RMS error",
        )
        axis.set_xticks(x, labels=[f"{value:g}" for value in x])
        axis.set_xticks([], minor=True)
        axis.grid(alpha=0.25)
    fig.savefig(output / "summary.png", dpi=150)
    plt.close(fig)
    with np.load(output / "aligned_cube_nonuniform.npz") as data:
        z = data["fluid"].shape[0] // 2
        concentration = np.ma.masked_where(
            ~data["fluid"][z], data["concentration_kg_per_m3"][z]
        )
        fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
        mesh = axis.pcolormesh(
            data["x_edges_m"] * 1000,
            data["y_edges_m"] * 1000,
            concentration,
            cmap="viridis",
            edgecolors=(0, 0, 0, 0.15),
            linewidth=0.25,
        )
        axis.set(
            xlabel="x (mm)",
            ylabel="y (mm)",
            title="Actual focused cells around the fixed cube",
            aspect="equal",
        )
        fig.colorbar(mesh, ax=axis, label="C (kg KDP / m³ solution)")
        fig.savefig(output / "focused_cube.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=ROOT / "configs/rectilinear_verification.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/rectilinear_verification"
    )
    args = parser.parse_args()
    protocol = json.loads(args.study_config.read_text())
    reference = json.loads(
        (args.study_config.parent / protocol["fluid_reference"]).read_text()
    )
    report = run_study(protocol, reference, args.output)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
