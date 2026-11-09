# HOW THIS FILE WORKS
# Measure the standalone fixed cut-cell viscosity operator against affine shear,
# independently integrated 3D manufactured no-slip shear and a periodic decay
# eigenmode. Verify refinement and implicit energy/flux balances on sloped,
# rotated, subcell and tiny-layer geometry. Archive sources, protocol, native
# geometry and solutions before reporting success; keep failures and refuse
# existing output directories. These checks do not enable the coupled flow step.

"""Independent references for stationary embedded-wall viscous diffusion."""

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
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid
from simulation.units import finite_number
from simulation.viscosity import CutCellViscosity
from check_cutcell import wall_geometry, cube_geometry


def _average_f(ylo, yhi, zlo, zhi, a, b, order):
    """Exact Gaussian integral of d*(1-d) over a clipped y/z rectangle.

    Zero-width y or z bounds denote a face integral. Split at the wall crossing;
    no discrete stiffness is used to construct this reference or its forcing.
    """
    nodes, weights = leggauss(order)
    if ylo == yhi:
        lower = max(zlo, a + b * ylo)
        if lower >= zhi:
            return 0.0
        z = (zhi + lower) / 2 + (zhi - lower) / 2 * nodes
        d = z - a - b * ylo
        return float(weights @ (d * (1 - d)) / 2)
    if zlo == zhi:
        upper = min(yhi, (zlo - a) / b)
        if upper <= ylo:
            return 0.0
        y = (upper + ylo) / 2 + (upper - ylo) / 2 * nodes
        d = zlo - a - b * y
        return float(weights @ (d * (1 - d)) / 2)
    breaks = sorted(
        {ylo, yhi, *[v for v in ((zlo - a) / b, (zhi - a) / b) if ylo < v < yhi]}
    )
    integral, area = 0.0, 0.0
    for low, high in zip(breaks, breaks[1:]):
        if a + b * (low + high) / 2 >= zhi:
            continue
        y = (low + high) / 2 + (high - low) / 2 * nodes
        bottom = np.maximum(zlo, a + b * y)
        z = (bottom[:, None] + zhi) / 2 + (zhi - bottom[:, None]) / 2 * nodes
        d = z - a - b * y[:, None]
        factor = weights[:, None] * weights * (high - low) * (zhi - bottom[:, None]) / 4
        integral += np.sum(factor * d * (1 - d))
        area += np.sum(factor)
    return float(integral / area) if area else 0.0


def manufactured_means(operator, protocol):
    """Volume/face averages of U*[f(d), sin(pi*x)*f(d), b*sin(pi*x)*f(d)].

    Coordinates are divided by L; d=z-a-b*y. This field is divergence-free,
    vanishes on the embedded plane, and has analytically known -nu Laplacian.
    The quadrature integrates only the liquid, independently of geometry moments.
    """
    g = operator.geometry
    length, speed, nu = (
        protocol[k]
        for k in ("length_m", "speed_m_per_s", "kinematic_viscosity_m2_per_s")
    )
    settings = protocol["sloped_wall"]
    a, b = settings["intercept_m"] / length, settings["slope_dz_dy"]
    order = settings["quadrature_order"]
    edges = [e / length for e in g.grid.edges]
    cell = np.zeros((*g.grid.shape, 3))
    force = np.zeros_like(cell)
    for raw in operator.cells:
        iz, iy, ix = raw
        yz = _average_f(*edges[1][iy : iy + 2], *edges[0][iz : iz + 2], a, b, order)
        xlo, xhi = edges[2][ix : ix + 2]
        sx = (np.cos(np.pi * xlo) - np.cos(np.pi * xhi)) / (np.pi * (xhi - xlo))
        cell[tuple(raw)] = speed * np.array([yz, sx * yz, b * sx * yz])
        c = 2 * (1 + b * b)
        force[tuple(raw)] = (
            nu
            * speed
            / length**2
            * np.array([c, sx * (np.pi**2 * yz + c), b * sx * (np.pi**2 * yz + c)])
        )
    traces = np.zeros((len(operator.boundary), 3))
    for axis, ids in enumerate(operator.face_ids):
        for raw in np.argwhere(ids >= 0):
            iz, iy, ix = raw
            bounds = [edges[k][raw[k] : raw[k] + 2] for k in range(3)]
            coordinate = edges[axis][raw[axis]]
            bounds[axis] = [coordinate, coordinate]
            yz = _average_f(*bounds[1], *bounds[0], a, b, order)
            xlo, xhi = bounds[2]
            sx = (
                np.sin(np.pi * xlo)
                if axis == 2
                else (np.cos(np.pi * xlo) - np.cos(np.pi * xhi)) / (np.pi * (xhi - xlo))
            )
            traces[ids[tuple(raw)]] = speed * np.array([yz, sx * yz, b * sx * yz])
    return cell, force, traces


def rms_error(value, reference, volume):
    return float(
        np.sqrt(
            np.sum(volume[..., None] * (value - reference) ** 2)
            / np.sum(volume[..., None] * reference**2)
        )
    )


def shear_case(protocol, n, focused=False):
    g = wall_geometry(protocol["length_m"], protocol["sloped_wall"], n, focused=focused)
    operator = CutCellViscosity(g)
    exact, force, traces = manufactured_means(operator, protocol)
    nu = protocol["kinematic_viscosity_m2_per_s"]
    state = operator.steady(
        force, nu, boundary_velocity_m_per_s=traces[operator.boundary]
    )
    b = protocol["sloped_wall"]["slope_dz_dy"]
    a = protocol["sloped_wall"]["intercept_m"] / protocol["length_m"]
    position = operator.face_centroids_m[operator.boundary] / protocol["length_m"]
    affine_boundary = (
        protocol["speed_m_per_s"]
        * (position[:, 2] - a - b * position[:, 1])[:, None]
        * [1, 1, b]
    )
    affine_boundary[operator.embedded[operator.boundary]] = 0
    affine = operator.steady(
        np.zeros_like(exact), nu, boundary_velocity_m_per_s=affine_boundary
    )
    position = g.centroids_m / protocol["length_m"]
    exact_affine = (
        protocol["speed_m_per_s"]
        * (position[..., 2] - a - b * position[..., 1])[..., None]
        * [1, 1, b]
    )
    exact_affine[~g.fluid] = 0
    metrics = dict(
        n=n,
        relative_volume_rms_error=rms_error(
            state.cell_velocity_m_per_s, exact, g.volumes_m3
        ),
        affine_relative_max_error=float(
            np.max(abs(affine.cell_velocity_m_per_s - exact_affine))
        )
        / protocol["speed_m_per_s"],
        max_relative_moment_defect=operator.max_relative_moment_defect,
        **operator.balances(state, nu, acceleration=force),
    )
    fields = dict(
        exact_velocity_m_per_s=exact,
        acceleration_m_per_s2=force,
        affine_velocity_m_per_s=affine.cell_velocity_m_per_s,
        exact_affine_velocity_m_per_s=exact_affine,
    )
    return operator, state, metrics, fields


def decay_case(protocol, geometry):
    operator = CutCellViscosity(geometry)
    rng = np.random.default_rng(protocol["seed"])
    old = rng.normal(size=(*geometry.grid.shape, 3)) * protocol["speed_m_per_s"]
    old[~geometry.fluid] = 0
    nu, dt = protocol["kinematic_viscosity_m2_per_s"], protocol["dt_s"]
    state = operator.implicit(old, dt, nu)
    metrics = operator.balances(state, nu, previous=old, dt_s=dt)
    previous_energy = float(np.sum(geometry.volumes_m3[..., None] * old**2) / 2)
    power_scale = previous_energy / dt
    flux_scale = nu * protocol["speed_m_per_s"] * protocol["length_m"]
    metrics.update(
        previous_energy_m5_per_s2=previous_energy,
        relative_energy_increase=metrics["energy_m5_per_s2"] / previous_energy - 1,
        relative_balance_error=max(
            abs(metrics["work_residual_m5_per_s3"]) / power_scale,
            metrics["max_cell_momentum_residual_m4_per_s2"] / flux_scale,
            metrics["max_interior_flux_residual_m4_per_s2"] / flux_scale,
        ),
        max_relative_moment_defect=operator.max_relative_moment_defect,
    )
    return operator, state, metrics, dict(previous_velocity_m_per_s=old)


def temporal_case(protocol, steps):
    settings = protocol["temporal"]
    n, length = settings["n"], protocol["length_m"]
    geometry = CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
    operator = CutCellViscosity(geometry)
    h = length / n
    rate = 4 * protocol["kinematic_viscosity_m2_per_s"] * np.sin(np.pi / n) ** 2 / h**2
    start = np.zeros((*geometry.grid.shape, 3))
    # Exact volume mean of sin(2*pi*x/L); the uniform-grid discrete eigenvalue
    # isolates Euler time error from spatial truncation. It is analytic, not
    # measured by applying this implementation's stiffness to its solution.
    x = geometry.grid.centers[2]
    start[..., 1] = (
        protocol["speed_m_per_s"] * np.sin(2 * np.pi * x / length) * np.sinc(1 / n)
    )
    reference = start * np.exp(-rate * settings["duration_s"])
    velocity = start.copy()
    dt = settings["duration_s"] / steps
    for _ in range(steps):
        state = operator.implicit(
            velocity, dt, protocol["kinematic_viscosity_m2_per_s"]
        )
        velocity = state.cell_velocity_m_per_s
    return (
        operator,
        state,
        dict(
            steps=steps,
            dt_s=dt,
            relative_temporal_rms_error=rms_error(
                velocity, reference, geometry.volumes_m3
            ),
        ),
        dict(
            semidiscrete_reference_velocity_m_per_s=reference,
            initial_velocity_m_per_s=start,
        ),
    )


def _validate(protocol):
    for name in ("length_m", "kinematic_viscosity_m2_per_s", "speed_m_per_s", "dt_s"):
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
    settings = protocol["sloped_wall"]
    a = finite_number(settings["intercept_m"], "intercept", positive=True)
    b = finite_number(settings["slope_dz_dy"], "slope", positive=True)
    if a + b * protocol["length_m"] >= protocol["length_m"]:
        raise ValueError("Manufactured wall must remain below the top")
    order = settings["quadrature_order"]
    if isinstance(order, bool) or not isinstance(order, int) or order < 4:
        raise ValueError("Polynomial reference needs Gaussian order >=4")
    for name, value in protocol["gates"].items():
        finite_number(value, name, positive=True)


def run_study(protocol, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh viscosity verification directory")
    _validate(protocol)
    names = [
        "src/simulation/" + name + ".py"
        for name in ("viscosity", "cutcell", "crystal", "mesh", "linear", "units")
    ]
    names += [
        "scripts/check_viscosity.py",
        "scripts/check_cutcell.py",
        "tests/test_viscosity.py",
        "configs/viscosity_verification.json",
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
            g = operator.geometry
            fields.update(
                velocity_m_per_s=state.cell_velocity_m_per_s,
                trace_velocity_m_per_s=state.trace_velocity_m_per_s,
                liquid_volumes_m3=g.volumes_m3,
                liquid_centroids_m=g.centroids_m,
                wall_cells=g.wall_cells,
                wall_face_ids=g.wall_face_ids,
                wall_areas_m2=g.wall_areas_m2,
                wall_normals=g.wall_normals,
                wall_centroids_m=g.wall_centroids_m,
                axis_refinement_history=g.refinement_history,
                trace_centroids_m=operator.face_centroids_m,
                boundary_ids=operator.boundary_ids,
                wall_ids=operator.wall_ids,
            )
            for axis, label in enumerate(("z", "y", "x")):
                fields[label + "_edges_m"] = g.grid.edges[axis]
                fields[label + "_open_areas_m2"] = g.open_areas_m2[axis]
                fields[label + "_face_ids"] = operator.face_ids[axis]
            wall_zero = not np.any(state.trace_velocity_m_per_s[operator.wall_ids])
            metrics.update(g.measurements(), wall_trace_exactly_zero=wall_zero)
            if (
                "previous_energy_m5_per_s2" not in metrics
                and "dissipation_m5_per_s3" in metrics
            ):
                power_scale = max(
                    abs(metrics["body_power_m5_per_s3"]),
                    abs(metrics["boundary_power_m5_per_s3"]),
                    np.finfo(float).tiny,
                )
                flux_scale = (
                    protocol["kinematic_viscosity_m2_per_s"]
                    * protocol["speed_m_per_s"]
                    * protocol["length_m"]
                )
                metrics["relative_steady_balance_error"] = max(
                    abs(metrics["work_residual_m5_per_s3"]) / power_scale,
                    metrics["max_cell_momentum_residual_m4_per_s2"] / flux_scale,
                    metrics["max_interior_flux_residual_m4_per_s2"] / flux_scale,
                )
            if not all(np.isfinite(v).all() for v in fields.values()):
                raise ArithmeticError("Nonfinite archived field")
            np.savez_compressed(output / (name + ".npz"), **fields)
            row = dict(case=name, **metrics)
        except (ValueError, ArithmeticError) as error:
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
    # Construct cases inside record so a failed solve is retained in cases.json.
    for steps in protocol["temporal"]["steps"]:
        record(
            f"temporal_{steps}",
            lambda steps=steps: temporal_case(protocol, steps),
        )
    limits = protocol["gates"]
    gates = dict(
        all_cases_completed=all("error" not in r for r in rows),
        all_wall_traces_zero=all(r.get("wall_trace_exactly_zero", False) for r in rows),
    )
    for family in ("uniform", "focused"):
        selected = [r for r in rows if r["case"].startswith(family + "_shear_")]
        errors = [r.get("relative_volume_rms_error", float("inf")) for r in selected]
        rates = [
            np.log(a / b) / np.log(n2 / n1)
            for a, b, n1, n2 in zip(
                errors,
                errors[1:],
                protocol["sloped_wall"]["counts"],
                protocol["sloped_wall"]["counts"][1:],
            )
        ]
        gates[family + "_spatial_refinement"] = bool(
            errors[-1] < limits["finest_relative_volume_rms_error"]
            and all(rate >= limits["minimum_spatial_rate"] for rate in rates)
        )
        gates[family + "_affine_exactness"] = all(
            r.get("affine_relative_max_error", float("inf"))
            < limits["affine_relative_max_error"]
            for r in selected
        )
    decays = [r for r in rows if "previous_energy_m5_per_s2" in r]
    gates["energy_and_flux_balances"] = len(decays) == 4 and all(
        r["relative_balance_error"] < limits["maximum_relative_balance_error"]
        and r["relative_energy_increase"] <= limits["maximum_relative_energy_increase"]
        and r["dissipation_m5_per_s3"] >= 0
        for r in decays
    )
    shears = [r for r in rows if "_shear_" in r["case"]]
    gates["steady_work_and_flux_balances"] = all(
        r.get("relative_steady_balance_error", float("inf"))
        < limits["maximum_relative_balance_error"]
        for r in shears
    )
    tiny_row = next(r for r in rows if r["case"] == "tiny_layer")
    gates["tiny_capacity_retained"] = (
        abs(tiny_row.get("min_liquid_fraction", 0) / tiny["fraction"] - 1) < 1e-6
    )
    times = [r for r in rows if r["case"].startswith("temporal_")]
    rates = [
        np.log(
            a.get("relative_temporal_rms_error", float("inf"))
            / b.get("relative_temporal_rms_error", float("inf"))
        )
        / np.log(b["steps"] / a["steps"])
        for a, b in zip(times, times[1:])
        if "steps" in a and "steps" in b
    ]
    gates["first_order_implicit_time"] = bool(
        len(rates) == len(times) - 1
        and all(rate >= limits["minimum_temporal_rate"] for rate in rates)
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during viscosity verification")
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for family in ("uniform", "focused"):
        rows = [
            r
            for r in report["results"]
            if r["case"].startswith(family + "_shear_")
            and "relative_volume_rms_error" in r
        ]
        axes[0].loglog(
            [r["n"] for r in rows],
            [r["relative_volume_rms_error"] for r in rows],
            "o-",
            label=family,
        )
    axes[0].set(
        xlabel="Cells per initial axis",
        ylabel="Relative volume-weighted velocity RMS error",
        title="3D manufactured sloped-wall shear",
    )
    axes[0].legend()
    rows = [r for r in report["results"] if "relative_temporal_rms_error" in r]
    axes[1].loglog(
        [r["dt_s"] for r in rows],
        [r["relative_temporal_rms_error"] for r in rows],
        "o-",
    )
    axes[1].set(
        xlabel="Time step (s)",
        ylabel="Relative RMS error against discrete eigenmode",
        title="Implicit Euler time refinement",
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.savefig(output / "refinement.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/viscosity_verification.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(json.loads(args.config.read_text()), args.output)
    print("PASS" if report["overall_pass"] else "FAIL")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
