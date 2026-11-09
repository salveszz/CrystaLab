# HOW THIS FILE WORKS
# Verify monolithic fixed cut-cell pressure/viscosity coupling against independent
# manufactured velocity/pressure averages, gradient-only forcing, periodic decay
# and coupled time refinement. Audit volume conservation, gauges, wall velocity,
# pressure work and implicit kinetic-energy balance on real cut geometries.
# Archive the declared protocol, source copies/hashes, raw states and figures in
# a fresh output directory. Preserve failed cases and gates without overwriting.
# These checks establish linear Stokes only, not advection or growing-crystal flow.

"""Independent conservation, energy and accuracy checks for cut-cell Stokes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.crystal import Crystal
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid
from simulation.stokes import CutCellStokes
from simulation.units import finite_number
from check_cutcell import wall_geometry, cube_geometry
from check_viscosity import manufactured_means, rms_error


def operator_for(geometry, protocol):
    return CutCellStokes(
        geometry,
        kinematic_viscosity_m2_per_s=protocol["kinematic_viscosity_m2_per_s"],
        density_kg_per_m3=protocol["density_kg_per_m3"],
    )


def pressure_reference(operator, protocol):
    """Exact liquid-volume mean of affine p/rho, with analytic constant gradient."""
    length, speed, nu = (
        protocol[k]
        for k in ("length_m", "speed_m_per_s", "kinematic_viscosity_m2_per_s")
    )
    scale = nu * speed / length
    pi = (operator.geometry.centroids_m / length) @ np.array([1, 2, -3]) * scale
    pi = operator.projection.mean_zero(pi)
    gradient = scale / length * np.array([1, 2, -3])
    return pi * operator.density, gradient, scale


def normalized_metrics(
    operator, state, protocol, *, previous=None, dt=None, force=None
):
    metrics = operator.balances(state, previous=previous, dt_s=dt, acceleration=force)
    g, v = operator.geometry, operator.viscosity
    length, speed, nu = (
        protocol[k]
        for k in ("length_m", "speed_m_per_s", "kinematic_viscosity_m2_per_s")
    )
    power_scale = max(
        abs(metrics["body_power_m5_per_s3"]) + abs(metrics["boundary_power_m5_per_s3"]),
        nu * speed**2 * length,
    )
    if previous is not None:
        previous_energy = float(np.sum(g.volumes_m3[..., None] * previous**2) / 2)
        metrics["previous_energy_m5_per_s2"] = previous_energy
        metrics["relative_energy_increase"] = (
            metrics["energy_m5_per_s2"] / previous_energy - 1
        )
        power_scale = max(power_scale, previous_energy / dt)
    native = operator.normal_face_velocity(state)
    # This independent Cartesian face sum also permits manufactured nonzero
    # outer traces; PressureProjection.net_flux deliberately rejects those BCs.
    native_net = sum(
        np.diff(values * g.open_areas_m2[axis], axis=axis)
        for values, axis in zip(native, (2, 1, 0))
    )
    metrics.update(
        relative_integral_flux=metrics["max_integral_flux_m3_per_s"]
        / (speed * length**2),
        relative_pressure_work=abs(metrics["pressure_work_m5_per_s3"]) / power_scale,
        relative_balance_error=max(
            abs(metrics["work_residual_m5_per_s3"]) / power_scale,
            metrics["max_cell_momentum_residual_m4_per_s2"] / (nu * speed * length),
            metrics["max_face_momentum_residual_m4_per_s2"] / (nu * speed * length),
        ),
        relative_native_flux_disagreement=float(
            np.max(abs(native_net - operator.net_flux(state)))
        )
        / (speed * length**2),
        wall_trace_exactly_zero=not np.any(state.trace_velocity_m_per_s[v.wall_ids]),
        max_pressure_gauge_over_scale=max(
            abs(
                float(
                    np.average(
                        state.pressure_pa[g.fluid][nodes], weights=v.volumes[nodes]
                    )
                )
            )
            for nodes in operator._groups
        )
        / (operator.density * nu * speed / length),
        connected_liquid_regions=len(operator._groups),
    )
    return metrics


def shear_case(protocol, n, focused=False):
    g = wall_geometry(protocol["length_m"], protocol["sloped_wall"], n, focused=focused)
    operator = operator_for(g, protocol)
    exact, force, traces = manufactured_means(operator.viscosity, protocol)
    pressure, gradient, scale = pressure_reference(operator, protocol)
    force[g.fluid] += gradient
    state = operator.steady(
        force, boundary_velocity_m_per_s=traces[operator.viscosity.boundary]
    )
    metrics = normalized_metrics(operator, state, protocol, force=force)
    metrics.update(
        n=n,
        relative_velocity_rms_error=rms_error(
            state.cell_velocity_m_per_s, exact, g.volumes_m3
        ),
        pressure_rms_over_scale=float(
            np.sqrt(
                np.sum(g.volumes_m3 * (state.pressure_pa - pressure) ** 2)
                / g.volumes_m3.sum()
            )
        )
        / (operator.density * scale),
    )
    gradient_force = np.zeros_like(force)
    gradient_force[g.fluid] = gradient
    gradient_state = operator.steady(gradient_force)
    metrics["gradient_leakage_over_speed"] = (
        float(
            np.sqrt(
                np.sum(
                    g.volumes_m3[..., None] * gradient_state.cell_velocity_m_per_s**2
                )
                / g.volumes_m3.sum()
            )
        )
        / protocol["speed_m_per_s"]
    )
    metrics["gradient_pressure_rms_over_scale"] = float(
        np.sqrt(
            np.sum(g.volumes_m3 * (gradient_state.pressure_pa - pressure) ** 2)
            / g.volumes_m3.sum()
        )
    ) / (operator.density * scale)
    gradient_metrics = normalized_metrics(
        operator, gradient_state, protocol, force=gradient_force
    )
    for metric in (
        "relative_balance_error",
        "relative_integral_flux",
        "relative_pressure_work",
    ):
        metrics["gradient_" + metric] = gradient_metrics[metric]
    fields = dict(
        exact_velocity_m_per_s=exact,
        exact_pressure_pa=pressure,
        acceleration_m_per_s2=force,
        gradient_velocity_m_per_s=gradient_state.cell_velocity_m_per_s,
        gradient_pressure_pa=gradient_state.pressure_pa,
        gradient_trace_velocity_m_per_s=gradient_state.trace_velocity_m_per_s,
    )
    return operator, state, metrics, fields


def decay_case(protocol, geometry):
    operator = operator_for(geometry, protocol)
    rng = np.random.default_rng(protocol["seed"])
    old = rng.normal(size=(*geometry.grid.shape, 3)) * protocol["speed_m_per_s"]
    old[~geometry.fluid] = 0
    state = operator.step(old, protocol["dt_s"])
    return (
        operator,
        state,
        normalized_metrics(
            operator, state, protocol, previous=old, dt=protocol["dt_s"]
        ),
        dict(previous_velocity_m_per_s=old),
    )


def periodic_time_case(protocol, steps):
    settings = protocol["temporal"]
    n, length = settings["n"], protocol["length_m"]
    g = CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
    operator = operator_for(g, protocol)
    start = np.zeros((*g.grid.shape, 3))
    start[..., 1] = (
        protocol["speed_m_per_s"]
        * np.sin(2 * np.pi * g.grid.centers[2] / length)
        * np.sinc(1 / n)
    )
    rate = 4 * operator.nu * np.sin(np.pi / n) ** 2 / (length / n) ** 2
    reference = start * np.exp(-rate * settings["duration_s"])
    velocity = start.copy()
    dt = settings["duration_s"] / steps
    for _ in range(steps):
        old = velocity
        state = operator.step(old, dt)
        velocity = state.cell_velocity_m_per_s
    metrics = normalized_metrics(operator, state, protocol, previous=old, dt=dt)
    metrics.update(
        steps=steps,
        dt_s=dt,
        relative_temporal_rms_error=rms_error(velocity, reference, g.volumes_m3),
    )
    return (
        operator,
        state,
        metrics,
        dict(
            semidiscrete_reference_velocity_m_per_s=reference,
            initial_velocity_m_per_s=start,
        ),
    )


def coupled_trajectory(protocol, steps):
    """Analytic forcing and outer traces for e^(-t/tau)*(u_exact,p_exact).

    Gaussian liquid means and the analytic Laplacian/pressure gradient determine
    forcing independently. A separate fine numerical trajectory is the temporal
    reference only; it is not a new claim of spatial/experimental accuracy.
    """
    settings = protocol["temporal"]
    g = wall_geometry(protocol["length_m"], protocol["sloped_wall"], settings["n"])
    operator = operator_for(g, protocol)
    exact, force, traces = manufactured_means(operator.viscosity, protocol)
    pressure, gradient, _scale = pressure_reference(operator, protocol)
    force[g.fluid] += gradient
    force -= exact / settings["decay_time_s"]
    dt = settings["duration_s"] / steps
    velocity = exact.copy()
    for step in range(1, steps + 1):
        factor = np.exp(-step * dt / settings["decay_time_s"])
        old = velocity
        state = operator.step(
            old,
            dt,
            acceleration_m_per_s2=factor * force,
            boundary_velocity_m_per_s=factor * traces[operator.viscosity.boundary],
        )
        velocity = state.cell_velocity_m_per_s
    metrics = normalized_metrics(
        operator, state, protocol, previous=old, dt=dt, force=factor * force
    )
    metrics.update(steps=steps, dt_s=dt)
    return (
        operator,
        state,
        metrics,
        dict(
            analytic_velocity_m_per_s=factor * exact,
            analytic_pressure_pa=factor * pressure,
        ),
    )


def _validate(protocol):
    for name in (
        "length_m",
        "kinematic_viscosity_m2_per_s",
        "density_kg_per_m3",
        "speed_m_per_s",
        "dt_s",
    ):
        finite_number(protocol[name], name, positive=True)
    for values, minimum in (
        (protocol["sloped_wall"]["counts"], 3),
        (protocol["temporal"]["steps"], 1),
    ):
        if (
            len(values) < 3
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n < minimum
                for n in values
            )
            or any(a >= b for a, b in zip(values, values[1:]))
        ):
            raise ValueError("Declare three increasing integer refinement levels")
    for name, value in protocol["gates"].items():
        finite_number(value, name, positive=True)
    temporal = protocol["temporal"]
    finite_number(temporal["duration_s"], "duration", positive=True)
    finite_number(temporal["decay_time_s"], "decay time", positive=True)
    if (
        isinstance(temporal["reference_steps"], bool)
        or not isinstance(temporal["reference_steps"], int)
        or temporal["reference_steps"] < 4 * temporal["steps"][-1]
    ):
        raise ValueError(
            "Temporal reference needs at least four times the finest step count"
        )


def run_study(protocol, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh Stokes verification directory")
    _validate(protocol)
    names = [
        "src/simulation/" + name + ".py"
        for name in (
            "stokes",
            "viscosity",
            "pressure",
            "cutcell",
            "crystal",
            "mesh",
            "linear",
            "units",
        )
    ]
    names += [
        "scripts/check_stokes.py",
        "scripts/check_viscosity.py",
        "scripts/check_cutcell.py",
        "configs/stokes_verification.json",
        "tests/test_stokes.py",
    ]
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    output.mkdir(parents=True)
    manifest = dict(
        protocol=protocol,
        sources=sources,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=importlib.metadata.version("scipy"),
            threads={
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
        ),
    )
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for name in names:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    rows = []

    def record(name, action):
        try:
            operator, state, metrics, fields = action()
            g, v = operator.geometry, operator.viscosity
            fields.update(
                velocity_m_per_s=state.cell_velocity_m_per_s,
                trace_velocity_m_per_s=state.trace_velocity_m_per_s,
                pressure_pa=state.pressure_pa,
                liquid_volumes_m3=g.volumes_m3,
                liquid_centroids_m=g.centroids_m,
                wall_cells=g.wall_cells,
                wall_face_ids=g.wall_face_ids,
                wall_areas_m2=g.wall_areas_m2,
                wall_normals=g.wall_normals,
                wall_centroids_m=g.wall_centroids_m,
                axis_refinement_history=g.refinement_history,
                trace_centroids_m=v.face_centroids_m,
                boundary_ids=v.boundary_ids,
                wall_ids=v.wall_ids,
            )
            for axis, label in enumerate(("z", "y", "x")):
                fields[label + "_edges_m"] = g.grid.edges[axis]
                fields[label + "_open_areas_m2"] = g.open_areas_m2[axis]
                fields[label + "_face_ids"] = v.face_ids[axis]
            if not all(np.isfinite(value).all() for value in fields.values()):
                raise ArithmeticError("Nonfinite archived Stokes field")
            np.savez_compressed(output / (name + ".npz"), **fields)
            row = dict(case=name, **metrics, **g.measurements())
        except (ValueError, ArithmeticError, RuntimeError) as error:
            row = dict(case=name, error=str(error))
        rows.append(row)
        (output / "cases.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row, allow_nan=False), flush=True)

    for focused in (False, True):
        for n in protocol["sloped_wall"]["counts"]:
            record(
                f"{'focused' if focused else 'uniform'}_shear_{n}",
                lambda n=n, focused=focused: shear_case(protocol, n, focused),
            )
    for name in ("rotated_cube", "subcell_cube"):
        settings = protocol[name]
        record(
            name,
            lambda settings=settings: decay_case(
                protocol,
                cube_geometry(protocol["length_m"], settings, settings["counts_xyz"]),
            ),
        )
    tiny = protocol["tiny_layer"]
    thin = dict(
        protocol["sloped_wall"],
        intercept_m=protocol["length_m"] * (1 - tiny["fraction"] / tiny["n"]),
        slope_dz_dy=0,
    )
    record(
        "tiny_layer",
        lambda: decay_case(
            protocol, wall_geometry(protocol["length_m"], thin, tiny["n"])
        ),
    )
    record(
        "periodic_sloped_wall",
        lambda: decay_case(
            protocol,
            wall_geometry(
                protocol["length_m"],
                protocol["sloped_wall"],
                protocol["sloped_wall"]["counts"][0],
                periodic=(True, False, False),
            ),
        ),
    )
    length = protocol["length_m"]
    slab = Crystal.box([length / 2] * 3, [0.2 * length, 1.2 * length, 1.2 * length])
    record(
        "disconnected_regions",
        lambda: decay_case(
            protocol,
            CutCellGeometry(RectilinearGrid([length] * 3, [10] * 3, [False] * 3), slab),
        ),
    )
    for steps in protocol["temporal"]["steps"]:
        record(
            f"periodic_time_{steps}",
            lambda steps=steps: periodic_time_case(protocol, steps),
        )
    try:
        _op, fine, _metrics, _fields = coupled_trajectory(
            protocol, protocol["temporal"]["reference_steps"]
        )

        def coupled_case(steps):
            op, state, metrics, fields = coupled_trajectory(protocol, steps)
            metrics["relative_temporal_rms_error"] = rms_error(
                state.cell_velocity_m_per_s,
                fine.cell_velocity_m_per_s,
                op.geometry.volumes_m3,
            )
            scale = op.density * op.nu * protocol["speed_m_per_s"] / length
            metrics["temporal_pressure_rms_over_scale"] = (
                float(
                    np.sqrt(
                        np.sum(
                            op.geometry.volumes_m3
                            * (state.pressure_pa - fine.pressure_pa) ** 2
                        )
                        / op.geometry.volumes_m3.sum()
                    )
                )
                / scale
            )
            fields.update(
                fine_time_reference_velocity_m_per_s=fine.cell_velocity_m_per_s,
                fine_time_reference_pressure_pa=fine.pressure_pa,
            )
            return op, state, metrics, fields

        for steps in protocol["temporal"]["steps"]:
            record(f"coupled_time_{steps}", lambda steps=steps: coupled_case(steps))
    except (ValueError, ArithmeticError, RuntimeError) as error:
        rows.append(dict(case="coupled_time_reference", error=str(error)))
    limits = protocol["gates"]
    gates = dict(
        all_cases_completed=all("error" not in r for r in rows),
        wall_traces_zero=all(r.get("wall_trace_exactly_zero", False) for r in rows),
    )
    gates["conservation_and_work"] = all(
        r.get("relative_balance_error", float("inf"))
        < limits["maximum_relative_balance_error"]
        and r.get("relative_integral_flux", float("inf"))
        < limits["maximum_relative_integral_flux"]
        and r.get("relative_pressure_work", float("inf"))
        < limits["maximum_relative_pressure_work"]
        and r.get("relative_native_flux_disagreement", float("inf"))
        < limits["maximum_relative_integral_flux"]
        and r.get("max_pressure_gauge_over_scale", float("inf"))
        < limits["maximum_relative_integral_flux"]
        for r in rows
    )
    decays = [
        r
        for r in rows
        if r["case"]
        in (
            "rotated_cube",
            "subcell_cube",
            "tiny_layer",
            "periodic_sloped_wall",
            "disconnected_regions",
        )
    ]
    gates["unforced_energy_decreases"] = all(
        r.get("relative_energy_increase", float("inf"))
        <= limits["maximum_relative_energy_increase"]
        and r.get("dissipation_m5_per_s3", -1) >= 0
        for r in decays
    )
    for family in ("uniform", "focused"):
        selected = [r for r in rows if r["case"].startswith(family + "_shear_")]
        for metric, tolerance in (
            ("relative_velocity_rms_error", "finest_relative_velocity_rms_error"),
            ("pressure_rms_over_scale", "finest_pressure_rms_over_scale"),
            ("gradient_leakage_over_speed", "finest_gradient_leakage_over_speed"),
            ("gradient_pressure_rms_over_scale", "finest_pressure_rms_over_scale"),
        ):
            values = [r.get(metric, float("inf")) for r in selected]
            counts = protocol["sloped_wall"]["counts"]
            rates = [
                np.log(a / b) / np.log(n2 / n1)
                if np.isfinite(a) and np.isfinite(b) and b > 0
                else -float("inf")
                for a, b, n1, n2 in zip(values, values[1:], counts, counts[1:])
            ]
            gates[family + "_" + metric] = bool(
                values[-1] < limits[tolerance]
                and all(rate >= limits["minimum_spatial_rate"] for rate in rates)
            )
        gates[family + "_gradient_conservation_and_work"] = all(
            r.get("gradient_relative_balance_error", float("inf"))
            < limits["maximum_relative_balance_error"]
            and r.get("gradient_relative_integral_flux", float("inf"))
            < limits["maximum_relative_integral_flux"]
            and r.get("gradient_relative_pressure_work", float("inf"))
            < limits["maximum_relative_pressure_work"]
            for r in selected
        )
    for family in ("periodic", "coupled"):
        selected = [
            r for r in rows if r["case"].startswith(family + "_time_") and "steps" in r
        ]
        rates = [
            np.log(a["relative_temporal_rms_error"] / b["relative_temporal_rms_error"])
            / np.log(b["steps"] / a["steps"])
            for a, b in zip(selected, selected[1:])
        ]
        gates[family + "_first_order_time"] = bool(
            len(selected) == len(protocol["temporal"]["steps"])
            and all(rate >= limits["minimum_temporal_rate"] for rate in rates)
        )
        if family == "coupled":
            rates = [
                np.log(
                    a["temporal_pressure_rms_over_scale"]
                    / b["temporal_pressure_rms_over_scale"]
                )
                / np.log(b["steps"] / a["steps"])
                for a, b in zip(selected, selected[1:])
            ]
            gates["coupled_pressure_first_order_time"] = bool(
                len(selected) == len(protocol["temporal"]["steps"])
                and all(rate >= limits["minimum_temporal_rate"] for rate in rates)
            )
    tiny_row = next(r for r in rows if r["case"] == "tiny_layer")
    gates["tiny_capacity_retained"] = (
        abs(tiny_row.get("min_liquid_fraction", 0) / tiny["fraction"] - 1) < 1e-6
    )
    gates["disconnected_gauges"] = (
        next(r for r in rows if r["case"] == "disconnected_regions").get(
            "connected_liquid_regions"
        )
        == 2
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during Stokes verification")
    report = dict(
        protocol=protocol, results=rows, gates=gates, overall_pass=all(gates.values())
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_report(report, output)
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for family in ("uniform", "focused"):
        rows = [
            r
            for r in report["results"]
            if r["case"].startswith(family + "_shear_") and "n" in r
        ]
        for axis, metric in zip(
            axes[:2], ("relative_velocity_rms_error", "pressure_rms_over_scale")
        ):
            axis.loglog(
                [r["n"] for r in rows], [r[metric] for r in rows], "o-", label=family
            )
    for family in ("periodic", "coupled"):
        rows = [
            r
            for r in report["results"]
            if r["case"].startswith(family + "_time_") and "dt_s" in r
        ]
        axes[2].loglog(
            [r["dt_s"] for r in rows],
            [r["relative_temporal_rms_error"] for r in rows],
            "o-",
            label=family,
        )
    for axis, label in zip(
        axes,
        (
            "Velocity RMS / reference RMS",
            "Pressure RMS / prescribed scale",
            "Velocity time error / reference RMS",
        ),
    ):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes[:2]:
        axis.set_xlabel("Cells per initial axis")
    axes[2].set_xlabel("Time step (s)")
    fig.savefig(output / "refinement.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/stokes_verification.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(json.loads(args.config.read_text()), args.output)
    print("PASS" if report["overall_pass"] else "FAIL")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
