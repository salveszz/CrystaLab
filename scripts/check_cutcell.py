# HOW THIS FILE WORKS
# Verify SI solid/cell intersections and a compatible pressure projection around
# actual sloped/rotated walls. Compare manufactured Poisson potentials with a
# source integrated independently over the true liquid region by Gaussian rules.
# Check projection of an analytic gradient, random-flux conservation, weighted
# norm decrease, small capacities, periodic seams and disconnected-region gauges.
# Reject split liquid cells explicitly. Archive all sources, declared protocol,
# native geometry/fields and plots in a fresh results folder. This validates the
# geometry/pressure foundation, not a complete no-slip cut-cell flow time step.
# Record bounded axis refinement of disconnected pockets and retain failed cases.

"""Verify fixed cut-cell geometry and pressure with independent references."""

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
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.crystal import Crystal
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid, focused_axis_edges
from simulation.pressure import PressureProjection
from simulation.units import finite_number


def wall_geometry(
    length, settings, n, *, focused=False, periodic=(False, False, False)
):
    box = Crystal.box([length / 2] * 3, [length] * 3)
    solid = Crystal(
        np.vstack((box.normals, [0, -settings["slope_dz_dy"], 1])),
        np.r_[box.offsets, settings["intercept_m"]],
    )
    edges = (
        [focused_axis_edges(length, n, length / 2, settings["focus_width_m"])] * 3
        if focused
        else None
    )
    grid = RectilinearGrid([length] * 3, [n] * 3, periodic, edges)
    return CutCellGeometry(grid, solid)


def _polynomial(y, z, a, b):
    fy, fz = y * y * (1 - y) ** 2, z * z * (1 - z) ** 2
    dy, dz = 2 * y - 6 * y * y + 4 * y**3, 2 * z - 6 * z * z + 4 * z**3
    dyy, dzz = 2 - 12 * y + 12 * y * y, 2 - 12 * z + 12 * z * z
    d = z - a - b * y
    potential = fy * fz * d * d
    laplacian = (
        (dyy * fz + fy * dzz) * d * d
        - 4 * b * dy * fz * d
        + 4 * fy * dz * d
        + 2 * (1 + b * b) * fy * fz
    )
    gradient_y = dy * fz * d * d - 2 * b * fy * fz * d
    gradient_z = fy * dz * d * d + 2 * fy * fz * d
    return potential, laplacian, gradient_y, gradient_z


def exact_potential(position, length, settings):
    x, y, z = np.moveaxis(position / length, -1, 0)
    p, _lap, _gy, _gz = _polynomial(
        y, z, settings["intercept_m"] / length, settings["slope_dz_dy"]
    )
    return settings["potential_scale_m2_per_s"] * np.cos(np.pi * x) * p


def exact_gradient(position, length, settings):
    x, y, z = np.moveaxis(position / length, -1, 0)
    p, _lap, gy, gz = _polynomial(
        y, z, settings["intercept_m"] / length, settings["slope_dz_dy"]
    )
    scale = settings["potential_scale_m2_per_s"] / length
    return scale * np.stack(
        (
            -np.pi * np.sin(np.pi * x) * p,
            np.cos(np.pi * x) * gy,
            np.cos(np.pi * x) * gz,
        ),
        axis=-1,
    )


def integrated_source(geometry, length, settings):
    """True liquid-average -Laplacian(phi), independent of the discrete matrix.

    Split y integration where the oblique wall crosses a z cell edge. Within
    each segment the polynomial z limit is affine. Gaussian integration is
    exact to rounding at the default degree; the x cosine integral is analytic.
    """
    a, b = settings["intercept_m"] / length, settings["slope_dz_dy"]
    if b <= 0:
        raise ValueError("Manufactured wall benchmark requires positive dz/dy")
    edges = [e / length for e in geometry.grid.edges]
    source = np.zeros(geometry.grid.shape)
    nodes, weights = leggauss(settings["quadrature_order"])
    x = edges[2]
    x_average = (np.sin(np.pi * x[1:]) - np.sin(np.pi * x[:-1])) / (np.pi * np.diff(x))
    for iz, (zlo, zhi) in enumerate(zip(edges[0][:-1], edges[0][1:])):
        for iy, (ylo, yhi) in enumerate(zip(edges[1][:-1], edges[1][1:])):
            area = geometry.volumes_m3[iz, iy, 0] / (
                geometry.grid.widths[2][0] * length**2
            )
            if area <= 0:
                continue
            breaks = sorted(
                {
                    ylo,
                    yhi,
                    *[v for v in ((zlo - a) / b, (zhi - a) / b) if ylo < v < yhi],
                }
            )
            total = 0.0
            for low, high in zip(breaks, breaks[1:]):
                if a + b * 0.5 * (low + high) >= zhi:
                    continue
                y = 0.5 * (high + low) + 0.5 * (high - low) * nodes
                bottom = np.maximum(zlo, a + b * y)
                z = (
                    0.5 * (zhi + bottom[:, None])
                    + 0.5 * (zhi - bottom[:, None]) * nodes[None, :]
                )
                p, lap, _gy, _gz = _polynomial(y[:, None], z, a, b)
                integrand = np.pi**2 * p - lap
                inner = (
                    np.sum(integrand * weights[None, :], axis=1) * 0.5 * (zhi - bottom)
                )
                total += float(np.sum(inner * weights) * 0.5 * (high - low))
            source[iz, iy, :] = (
                settings["potential_scale_m2_per_s"]
                / length**2
                * x_average
                * total
                / area
            )
    return source


def projection_metrics(geometry, settings):
    operator = PressureProjection(
        geometry.grid, geometry.volumes_m3, geometry.open_areas_m2
    )
    rng = np.random.default_rng(settings["random_seed"])
    fields = []
    for face in operator.faces:
        value = np.zeros(face.mask.shape)
        value[tuple(face.cells.T)] = rng.normal(
            0, settings["speed_scale_m_per_s"], len(face.cells)
        )
        operator._periodic_alias(value, face.axis)
        fields.append(value)
    flux_scale = max(
        float(abs(q).max(initial=0)) for q in operator.volume_fluxes(fields)
    )
    before = operator.projection_norm_squared(fields)
    projected, phi = operator.project(fields)
    after = operator.projection_norm_squared(projected)
    residual = float(abs(operator.net_flux(projected)).max(initial=0)) / max(
        flux_scale, np.finfo(float).tiny
    )
    return (
        operator,
        projected,
        phi,
        dict(
            relative_integral_flux_residual=residual,
            relative_projection_norm_change=(after - before) / before if before else 0,
            max_divergence_per_s=float(
                abs(operator.divergence(projected)).max(initial=0)
            ),
            blocked_face_speed_m_per_s=max(
                float(abs(v[~f.mask]).max(initial=0))
                for v, f in zip(projected, operator.faces)
            ),
            connected_liquid_regions=len(operator.groups),
        ),
    )


def poisson_case(length, settings, n, projection_settings, *, focused=False):
    geometry = wall_geometry(length, settings, n, focused=focused)
    operator, projected, random_phi, metrics = projection_metrics(
        geometry, projection_settings
    )
    source = integrated_source(geometry, length, settings)
    phi = operator.solve_poisson_source(source)
    exact = operator.mean_zero(
        exact_potential(geometry.grid.positions(), length, settings)
    )
    error = np.sqrt(
        np.sum(geometry.volumes_m3 * (phi - exact) ** 2)
        / np.sum(geometry.volumes_m3 * exact**2)
    )
    gradient = []
    for component, face in enumerate(operator.faces):
        value = exact_gradient(geometry.face_centroids_m[face.axis], length, settings)[
            ..., component
        ]
        value[~face.mask] = 0
        operator._periodic_alias(value, face.axis)
        gradient.append(value)
    residual, _ = operator.project(gradient)
    gradient_error = np.sqrt(
        operator.projection_norm_squared(residual)
        / operator.projection_norm_squared(gradient)
    )
    metrics.update(
        n=n,
        relative_potential_rms_error=float(error),
        relative_gradient_projection_error=float(gradient_error),
        geometry_volume_relative_error=abs(
            geometry.volumes_m3.sum()
            / (
                length**2
                * (
                    length
                    - settings["intercept_m"]
                    - settings["slope_dz_dy"] * length / 2
                )
            )
            - 1
        ),
    )
    fields = dict(
        pressure_potential_m2_per_s=phi,
        reference_potential_m2_per_s=exact,
        poisson_source_per_s=source,
        random_projection_potential_m2_per_s=random_phi,
        projected_u_x_m_per_s=projected[0],
        projected_u_y_m_per_s=projected[1],
        projected_u_z_m_per_s=projected[2],
    )
    return geometry, metrics, fields


def cube_geometry(length, settings, counts):
    angle = settings.get("angles_rad_xyz", [0, 0, 0])
    x, y, z = angle
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    solid = (
        Crystal.box([0] * 3, [settings["width_m"]] * 3)
        .rotated(rz @ ry @ rx)
        .translated(settings["center_m"])
    )
    return CutCellGeometry.resolved(
        RectilinearGrid([length] * 3, counts, [False] * 3),
        solid,
        max_axis_refinements=settings.get("max_axis_refinements", 48),
    )


def _validate(protocol):
    finite_number(protocol["length_m"], "length", positive=True)
    for family in ("sloped_wall", "rotated_cube"):
        counts = protocol[family]["counts"]
        if (
            len(counts) < 3
            or not all(
                isinstance(n, int) and not isinstance(n, bool) and n >= 3
                for n in counts
            )
            or not all(a < b for a, b in zip(counts, counts[1:]))
        ):
            raise ValueError("Declare three increasing integer refinement levels")
    if protocol["sloped_wall"]["quadrature_order"] < 6:
        raise ValueError("Use Gaussian order >=6 for this polynomial source")
    for key in (
        "max_geometry_volume_relative_error",
        "max_normalized_area_closure",
        "max_normalized_moment_closure",
    ):
        finite_number(protocol[key], key, positive=True)


def run_study(protocol, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh cut-cell verification directory")
    _validate(protocol)
    names = (
        "src/simulation/crystal.py",
        "src/simulation/cutcell.py",
        "src/simulation/mesh.py",
        "src/simulation/pressure.py",
        "src/simulation/linear.py",
        "src/simulation/units.py",
        "scripts/check_cutcell.py",
    )
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
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
    length = protocol["length_m"]

    def record(name, builder):
        start = time.perf_counter()
        try:
            geometry, metrics, fields = builder()
        except (ValueError, ArithmeticError) as error:
            row = dict(
                case=name,
                constraints_pass=False,
                error_type=type(error).__name__,
                error=str(error),
                runtime_s=time.perf_counter() - start,
            )
            rows.append(row)
            (output / "case_failures.json").write_text(
                json.dumps([r for r in rows if "error" in r], indent=2, allow_nan=False)
                + "\n"
            )
            print(name, json.dumps(row), flush=True)
            return
        row = dict(
            case=name,
            **geometry.measurements(),
            **metrics,
            runtime_s=time.perf_counter() - start,
        )
        row["constraints_pass"] = (
            row["normalized_area_closure"] <= protocol["max_normalized_area_closure"]
            and row["normalized_moment_closure"]
            <= protocol["max_normalized_moment_closure"]
            and row.get("geometry_volume_relative_error", 0)
            <= row.get(
                "geometry_volume_tolerance",
                protocol["max_geometry_volume_relative_error"],
            )
            and row.get("relative_integral_flux_residual", 0)
            <= protocol["projection"]["max_relative_integral_flux_residual"]
            and row.get("relative_projection_norm_change", 0)
            <= protocol["projection"]["max_relative_norm_increase"]
            and row.get("blocked_face_speed_m_per_s", 0) == 0
        )
        archive = dict(
            liquid_volumes_m3=geometry.volumes_m3,
            liquid_centroids_m=geometry.centroids_m,
            solid_fraction=geometry.solid_fractions,
            x_edges_m=geometry.grid.edges[2],
            y_edges_m=geometry.grid.edges[1],
            z_edges_m=geometry.grid.edges[0],
            solid_vertices_m=geometry.solid_vertices_m,
            wall_cells=geometry.wall_cells,
            wall_face_ids=geometry.wall_face_ids,
            wall_areas_m2=geometry.wall_areas_m2,
            wall_centroids_m=geometry.wall_centroids_m,
            wall_normals=geometry.wall_normals,
            axis_refinement_history=geometry.refinement_history,
            **fields,
        )
        for axis, label in zip((2, 1, 0), ("x", "y", "z")):
            archive[f"{label}_open_areas_m2"] = geometry.open_areas_m2[axis]
            archive[f"{label}_face_centroids_m"] = geometry.face_centroids_m[axis]
        np.savez_compressed(output / f"{name}.npz", **archive)
        rows.append(row)
        print(name, json.dumps(row, allow_nan=False), flush=True)

    for focused in (False, True):
        for n in protocol["sloped_wall"]["counts"]:
            record(
                f"{'focused' if focused else 'uniform'}_poisson_{n}",
                lambda: poisson_case(
                    length,
                    protocol["sloped_wall"],
                    n,
                    protocol["projection"],
                    focused=focused,
                ),
            )

    def project_geometry(geometry, volume, extra=None):
        _operator, fields, phi, metrics = projection_metrics(
            geometry, protocol["projection"]
        )
        metrics.update(
            geometry_volume_relative_error=abs(geometry.volumes_m3.sum() / volume - 1),
            **(extra or {}),
        )
        return (
            geometry,
            metrics,
            dict(
                pressure_potential_m2_per_s=phi,
                projected_u_x_m_per_s=fields[0],
                projected_u_y_m_per_s=fields[1],
                projected_u_z_m_per_s=fields[2],
            ),
        )

    for n in protocol["rotated_cube"]["counts"]:
        record(
            f"rotated_cube_{n}",
            lambda: project_geometry(
                cube_geometry(length, protocol["rotated_cube"], [n] * 3),
                length**3 - protocol["rotated_cube"]["width_m"] ** 3,
                dict(n=n),
            ),
        )
    record(
        "subcell_cube",
        lambda: project_geometry(
            cube_geometry(
                length, protocol["subcell_cube"], protocol["subcell_cube"]["counts_xyz"]
            ),
            length**3 - protocol["subcell_cube"]["width_m"] ** 3,
        ),
    )
    tiny = protocol["tiny_layer"]
    if len(set(tiny["counts_xyz"])) != 1:
        raise ValueError("The tiny-layer verification uses equal counts")
    n = tiny["counts_xyz"][0]
    fraction = tiny["liquid_fraction_in_top_cells"]
    thin_settings = dict(
        protocol["sloped_wall"],
        intercept_m=length - fraction * length / n,
        slope_dz_dy=0,
    )
    record(
        "tiny_layer",
        lambda: project_geometry(
            wall_geometry(length, thin_settings, n),
            length**2 * (length - thin_settings["intercept_m"]),
            dict(geometry_volume_tolerance=tiny["relative_volume_tolerance"]),
        ),
    )
    n = protocol["sloped_wall"]["counts"][0]
    wall_volume = length**2 * (
        length
        - protocol["sloped_wall"]["intercept_m"]
        - protocol["sloped_wall"]["slope_dz_dy"] * length / 2
    )
    record(
        "periodic_wall",
        lambda: project_geometry(
            wall_geometry(
                length, protocol["sloped_wall"], n, periodic=(True, False, False)
            ),
            wall_volume,
        ),
    )
    slab = Crystal.box([length / 2] * 3, [0.2 * length, 1.2 * length, 1.2 * length])
    record(
        "disconnected_regions",
        lambda: project_geometry(
            CutCellGeometry(RectilinearGrid([length] * 3, [10] * 3, [False] * 3), slab),
            0.8 * length**3,
        ),
    )
    split = Crystal.box([length / 2] * 3, [0.08 * length, 1.2 * length, 1.2 * length])
    try:
        CutCellGeometry(RectilinearGrid([length] * 3, [5] * 3, [False] * 3), split)
        rejected = False
    except ValueError as error:
        if "Split liquid cell" not in str(error):
            raise
        rejected = True
    rows.append(dict(case="split_cell_rejection", constraints_pass=rejected))
    gates = dict(
        geometry_and_projection=all(r["constraints_pass"] for r in rows),
        split_cell_rejected=rejected,
    )
    for family in ("uniform", "focused"):
        for metric, tolerance in (
            ("relative_potential_rms_error", "finest_relative_potential_tolerance"),
            (
                "relative_gradient_projection_error",
                "finest_relative_gradient_projection_tolerance",
            ),
        ):
            family_rows = [
                r for r in rows if r["case"].startswith(family + "_poisson_")
            ]
            values = [r[metric] for r in family_rows if metric in r]
            gates[family + "_" + metric] = (
                len(values) == len(protocol["sloped_wall"]["counts"])
                and all(b < a for a, b in zip(values, values[1:]))
                and values[-1] <= protocol["sloped_wall"][tolerance]
            )
    tiny_row = next(r for r in rows if r["case"] == "tiny_layer")
    gates["tiny_capacity_retained"] = (
        "min_liquid_fraction" in tiny_row
        and abs(tiny_row["min_liquid_fraction"] / fraction - 1) < 1e-6
    )
    gates["disconnected_gauges"] = (
        next(r for r in rows if r["case"] == "disconnected_regions").get(
            "connected_liquid_regions"
        )
        == 2
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during verification")
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
        "# Fixed cut-cell geometry/pressure verification",
        "",
        protocol["limits"],
        "",
        "| Case | Potential RMS error | Gradient projection residual | Integral flux residual |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['case']} | {r.get('relative_potential_rms_error', 'n/a')} | {r.get('relative_gradient_projection_error', 'n/a')} | {r.get('relative_integral_flux_residual', 'n/a')} |"
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for family in ("uniform", "focused"):
        rows = [
            r
            for r in report["results"]
            if r["case"].startswith(family + "_poisson_")
            and "relative_potential_rms_error" in r
        ]
        n = [r["n"] for r in rows]
        axes[0].loglog(
            n, [r["relative_potential_rms_error"] for r in rows], "o-", label=family
        )
        axes[1].loglog(
            n,
            [r["relative_gradient_projection_error"] for r in rows],
            "o-",
            label=family,
        )
    for axis, label in zip(
        axes, ("Potential error", "Analytic-gradient projection residual")
    ):
        axis.set(xlabel="Cells per axis", ylabel="Relative weighted RMS", title=label)
        axis.set_xticks(n, labels=[str(v) for v in n])
        axis.set_xticks([], minor=True)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output / "refinement.png", dpi=150)
    plt.close(fig)
    available = [
        r["n"]
        for r in report["results"]
        if r["case"].startswith("focused_poisson_")
        and "relative_potential_rms_error" in r
    ]
    if not available:
        return
    n = available[-1]
    with np.load(output / f"focused_poisson_{n}.npz") as fields:
        x = fields["liquid_volumes_m3"].shape[2] // 2
        active = fields["liquid_volumes_m3"][..., x] > 0
        fraction = 1 - fields["solid_fraction"][..., x]
        phi = np.ma.masked_where(~active, fields["pressure_potential_m2_per_s"][..., x])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for axis, value, label in zip(
            axes,
            (fraction, phi),
            ("Liquid fraction", "Pressure-correction potential (m²/s)"),
        ):
            mesh = axis.pcolormesh(
                fields["y_edges_m"] * 1000,
                fields["z_edges_m"] * 1000,
                value,
                edgecolors=(0, 0, 0, 0.12),
                linewidth=0.25,
            )
            axis.set(xlabel="y (mm)", ylabel="z (mm)", title=label, aspect="equal")
            y = fields["y_edges_m"] * 1000
            wall = report["protocol"]["sloped_wall"]
            axis.plot(
                y,
                wall["intercept_m"] * 1000 + wall["slope_dz_dy"] * y,
                color="black",
                linewidth=0.8,
            )
            fig.colorbar(mesh, ax=axis)
        fig.savefig(output / "sloped_wall.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config", type=Path, default=ROOT / "configs/cutcell_verification.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/cutcell_verification"
    )
    args = parser.parse_args()
    report = run_study(json.loads(args.study_config.read_text()), args.output)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
