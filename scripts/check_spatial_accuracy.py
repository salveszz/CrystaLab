# HOW THIS FILE WORKS
# Freeze the solid and prescribe a known affine concentration field in 3D.
# First compare the engine's facet averages with exact normal-prism integrals.
# Distinguish values sampled at Cartesian centers from true liquid-cell averages.
# Independently integrate clipped convex solids using tetrahedral volume moments.
# Then test internal diffusion fluxes above a flat/sloped impermeable solid plane.
# An affine field tangent to that plane has zero Laplacian and zero solid flux.
# Check interior cut cells only; outer tank walls do not satisfy that test field.
# Exact-gradient corrections are diagnostic references, not a production solver.
# Finally translate a liquid layer with a known affine scalar-transport solution.
# Save source fingerprints, measurements, a report and a static figure.
# No crystal-growth kinetics, AI or physical parameter fitting occurs.

"""Isolate spatial sampling and cut-cell flux errors against analytic fields."""

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.spatial import ConvexHull, QhullError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.crystal import Crystal
from simulation.engine import Simulation
from simulation.physics import _liquid_graph, _centroid_flux_correction, displace_liquid
from simulation.tank import SimulationConfig, Tank


class BoxIntersectionMoments:
    """Independent volume/first-moment reference for a convex solid inside a box.

    Enumerate feasible intersections of triples of planes, rather than reusing
    Crystal.clipped_volume's edge-clipping algorithm. Triangular hull faces and
    an interior anchor partition the result into nonoverlapping tetrahedra.
    """

    def __init__(self, normals, offsets):
        self.normals = np.vstack(
            (np.asarray(normals).reshape(-1, 3), np.eye(3), -np.eye(3))
        )
        self.offsets = np.asarray(offsets)
        triples = np.array(list(combinations(range(len(self.normals)), 3)))
        matrices = self.normals[triples]
        valid = abs(np.linalg.det(matrices)) > 1e-12
        self.triples = triples[valid]
        self.inverses = np.linalg.inv(matrices[valid])

    def integrate(self, lower, upper):
        """Return volume and integral(position dV), in world coordinates."""
        lower, upper = np.asarray(lower), np.asarray(upper)
        if np.any(upper <= lower):
            return 0.0, np.zeros(3)
        # Local coordinates reduce cancellation in translated subcell regions.
        center = (lower + upper) / 2
        offsets = np.r_[self.offsets, upper, -lower] - self.normals @ center
        points = np.einsum("nij,nj->ni", self.inverses, offsets[self.triples])
        tolerance = 1e-11 * max(1.0, float(np.max(upper - lower)))
        points = np.unique(
            points[np.all(points @ self.normals.T <= offsets + tolerance, axis=1)],
            axis=0,
        )
        if len(points) < 4:
            return 0.0, np.zeros(3)
        try:
            hull = ConvexHull(points)
        except QhullError:
            # Contact at a plane/edge has no three-dimensional measure.
            return 0.0, np.zeros(3)
        anchor = points[hull.vertices].mean(axis=0)
        triangles = points[hull.simplices]
        tetrahedra = triangles - anchor
        volumes = abs(np.linalg.det(tetrahedra)) / 6
        centroids = (triangles.sum(axis=1) + anchor) / 4
        volume = float(volumes.sum())
        moment = np.sum(volumes[:, None] * centroids, axis=0) + volume * center
        return volume, moment


def liquid_centroids(tank, solid):
    """Exact liquid-cell first moments by subtracting solid from the wet box."""
    centers = (
        ((tank.geometry.bounds[0] + tank.geometry.bounds[1]) / 2).reshape(-1, 3).copy()
    )
    reference = BoxIntersectionMoments(solid.normals, solid.offsets)
    partial = ((tank.geometry.solid_fraction > 1e-12) & tank.liquid_mask).ravel()
    max_volume_error = 0.0
    for flat in np.flatnonzero(partial):
        iz, iy, ix = np.unravel_index(flat, tank.shape)
        lower = tank.geometry.bounds[0][iz, iy, ix]
        upper = tank.geometry.bounds[1][iz, iy, ix]
        box_volume = float(np.prod(upper - lower))
        solid_volume, solid_moment = reference.integrate(lower, upper)
        volume = box_volume - solid_volume
        capacity = tank.liquid_volumes.ravel()[flat]
        max_volume_error = max(max_volume_error, abs(volume - capacity))
        if volume <= 0.0:
            raise ValueError(
                "Reference liquid volume must agree with wet-cell geometry"
            )
        centers[flat] = (box_volume * (lower + upper) / 2 - solid_moment) / volume
    return centers, max_volume_error


def face_centroid(points):
    """Area centroid from polygon triangles, not an unweighted corner average."""
    triangles = np.stack(
        (np.broadcast_to(points[0], points[1:-1].shape), points[1:-1], points[2:]),
        axis=1,
    )
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    return np.average(triangles.mean(axis=1), weights=areas, axis=0)


def sampling_case(grid, pose="aligned", base=None):
    """Compare the current sampler and exact integrals for the same affine field."""
    base = (
        json.loads((ROOT / "configs/default.json").read_text())
        if base is None
        else dict(base)
    )
    gradient = np.array([0.08, -0.05, 0.06])
    origin = np.array([base["width"] / 2, base["depth"] / 2, base["height"] / 2])
    depth, sigma0 = 0.75, 0.35
    spacing = max(
        base[axis] / count for axis, count in zip(("width", "depth", "height"), grid)
    )
    base.update(
        nx=grid[0],
        ny=grid[1],
        nz=grid[2],
        concentration_noise=0.0,
        temperature_gradient=0.0,
        temperature=25.0,
        solubility_reference=1.0,
        reference_temperature=25.0,
        boundary_layer_cells=depth / spacing,
    )
    if pose == "tilted_shifted":
        a, b = 0.37, 0.29
        rz = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        ry = np.array(
            [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
        )
        base.update(
            seed_rotation=(rz @ ry).tolist(),
            seed_center=(origin + [0.07, 0.03, 0.11]).tolist(),
        )
    elif pose != "aligned":
        raise ValueError("Unknown diagnostic seed pose")
    sim = Simulation(SimulationConfig(**base))
    tank, crystal = sim.tank, sim.crystal
    nominal_spacing = spacing
    spacing = max(tank.dx, tank.dy, tank.dz)
    initial_offsets = crystal.offsets.copy()
    centers, volume_error = liquid_centroids(tank, crystal)
    observed = {}
    for name, positions in (
        ("cartesian_centers", tank.positions),
        ("liquid_averages", centers),
    ):
        tank.concentration = (1.0 + sigma0 + (positions - origin) @ gradient).reshape(
            tank.shape
        )
        tank.concentration[~tank.liquid_mask] = 0.0
        observed[name] = sim._face_conditions()[0]
    rows = []
    for index in np.flatnonzero(sim.coefficients > 0):
        points = crystal.vertices[crystal.faces[index]]
        if len(points) < 3:
            continue
        prism = crystal.sampling_prism(index, depth, tank.liquid_level)
        if np.any(prism.bounds[0] < 0) or np.any(
            prism.bounds[1] > [base["width"], base["depth"], tank.liquid_level]
        ):
            raise ValueError(
                "Analytic prism reference requires an unclipped, fully submerged neighborhood"
            )
        expected = float(
            sigma0
            + (face_centroid(points) + crystal.normals[index] * depth / 2 - origin)
            @ gradient
        )
        weights = tank.sampling_volumes(prism)
        reference = BoxIntersectionMoments(prism.normals, prism.offsets)
        total, moment = 0.0, np.zeros(3)
        for flat in np.flatnonzero(weights > 0):
            iz, iy, ix = np.unravel_index(flat, tank.shape)
            lower = tank.geometry.bounds[0][iz, iy, ix]
            upper = tank.geometry.bounds[1][iz, iy, ix]
            volume, first = reference.integrate(lower, upper)
            volume_error = max(volume_error, abs(volume - weights[flat]))
            total += volume
            moment += first
        exact_integral = float(sigma0 + (moment / total - origin) @ gradient)
        rows.append(
            dict(
                face=int(index),
                normal=crystal.normals[index].tolist(),
                expected=expected,
                cartesian_centers=float(observed["cartesian_centers"][index]),
                liquid_averages=float(observed["liquid_averages"][index]),
                exact_overlap_integral=exact_integral,
            )
        )
    return dict(
        grid=list(grid),
        spacing=spacing,
        nominal_spacing=nominal_spacing,
        pose=pose,
        config=asdict(sim.config),
        gradient=gradient.tolist(),
        origin=origin.tolist(),
        sigma0=sigma0,
        sampling_depth=depth,
        faces=rows,
        max_cartesian_error=max(
            abs(r["cartesian_centers"] - r["expected"]) for r in rows
        ),
        max_liquid_average_error=max(
            abs(r["liquid_averages"] - r["expected"]) for r in rows
        ),
        max_exact_integral_error=max(
            abs(r["exact_overlap_integral"] - r["expected"]) for r in rows
        ),
        max_reference_volume_error=volume_error,
        geometry_unchanged=bool(np.array_equal(initial_offsets, crystal.offsets)),
    )


def transport_case(count, slope):
    """Test local flux consistency above z=intercept+slope*x without advancing time.

    The field gradient [1,0,slope] is tangent to the solid plane. Exclude outer
    boundary cells, because the prescribed field does not obey sealed tank walls.
    """
    config = SimulationConfig(
        width=4,
        depth=4,
        height=4,
        nx=count,
        ny=count,
        nz=count,
        initial_liquid_height=4,
        concentration_noise=0.0,
    )
    intercept = 1.27
    box = Crystal.box([2, 2, 2], [4, 4, 4])
    solid = Crystal(
        np.vstack((box.normals, [-slope, 0, 1])), np.r_[box.offsets, intercept]
    )
    tank = Tank(config, crystal=solid)
    centers, volume_error = liquid_centroids(tank, solid)
    active, i, j, conductance, laplacian, _records = _liquid_graph(
        tank.liquid_volumes, tank.geometry.openings, (tank.dz, tank.dy, tank.dx)
    )
    positions, centroids = tank.positions[active], centers[active]
    gradient = 0.1 * np.array([1.0, 0.0, slope])
    values = 1.35 + (centroids - 2.0) @ gradient
    flux = conductance * (values[i] - values[j])
    expected_flux = -conductance * ((positions[j] - positions[i]) @ gradient)
    # An exact-gradient reference removes the offset between the actual liquid
    # centroid and the nominal Cartesian location. No unknown gradient is fitted.
    shifts = centroids - positions
    corrected_flux = flux - conductance * ((shifts[i] - shifts[j]) @ gradient)
    corrected_balance = np.zeros(len(active))
    np.add.at(corrected_balance, i, corrected_flux)
    np.add.at(corrected_balance, j, -corrected_flux)
    balance = laplacian @ values
    operator, _edge_operator, _first, _second = _centroid_flux_correction(
        tank.liquid_volumes,
        tank.geometry.openings,
        (tank.dz, tank.dy, tank.dx),
        tank.liquid_centroids,
        active,
        i,
        j,
        conductance,
    )
    production_balance = (laplacian + operator) @ values
    partial = tank.liquid_volumes.ravel()[active] < tank.cell_volume * (1 - 1e-10)
    indices = np.array(np.unravel_index(active, tank.shape)).T
    interior = np.all((indices > 0) & (indices < count - 1), axis=1)
    selected = partial & interior
    if not selected.any():
        raise ValueError("Transport diagnostic requires interior cut cells")
    near_cut = partial[i] | partial[j]
    scale = float(np.linalg.norm(gradient) * tank.dx**2)
    return dict(
        grid=[count] * 3,
        slope=slope,
        intercept=intercept,
        spacing=tank.dx,
        gradient=gradient.tolist(),
        interior_cut_cells=int(selected.sum()),
        max_internal_flux_error=float(
            abs(flux[near_cut] - expected_flux[near_cut]).max()
        ),
        max_exact_gradient_flux_error=float(abs(corrected_flux - expected_flux).max()),
        max_cut_cell_balance=float(abs(balance[selected]).max()),
        max_exact_gradient_balance=float(abs(corrected_balance[selected]).max()),
        max_production_balance=float(abs(production_balance[selected]).max()),
        normalized_production_balance=float(
            abs(production_balance[selected]).max() / scale
        ),
        normalized_cut_cell_balance=float(abs(balance[selected]).max() / scale),
        max_reference_volume_error=volume_error,
    )


def remap_case(count):
    """Translate the liquid by 0.05 in z; compare exact affine C/T with remapping."""
    config = SimulationConfig(
        width=4,
        depth=4,
        height=4,
        nx=count,
        ny=count,
        nz=count,
        initial_liquid_height=3.2,
        concentration_noise=0,
    )
    old = Crystal.box([2, 2, 0.6], [4, 4, 1.2])
    new = Crystal.box([2, 2, 0.625], [4, 4, 1.25])
    tank = Tank(config, crystal=old)
    z = tank.liquid_centroids[..., 2]
    tank.concentration = 1 + 0.1 * (z - 1.2)
    tank.temperature = 20 + 2 * (z - 1.2)
    tank.concentration[~tank.liquid_mask] = 0
    tank.temperature[~tank.liquid_mask] = 0
    initial_mass = tank.solute_mass
    new_geometry = tank._geometry(new, tank.level_for(new))
    reference_c, _t, _fluxes, _residual = displace_liquid(
        tank.liquid_volumes,
        new_geometry.volumes,
        tank.geometry.openings,
        new_geometry.openings,
        tank.concentration,
        tank.temperature,
        tank.dx,
        tank.dy,
        tank.dz,
    )
    tank.move_boundary(new, 0.5, tank.concentration.copy())
    wet = tank.liquid_mask
    expected = 1 + 0.1 * (tank.liquid_centroids[..., 2] - 1.25)
    expected_temperature = 20 + 2 * (tank.liquid_centroids[..., 2] - 1.25)
    velocity = tank.displacement_velocity[wet] - np.array([0, 0, 0.1])
    return dict(
        grid=[count] * 3,
        spacing=tank.dx,
        displacement=0.05,
        max_first_order_error=float(abs(reference_c[wet] - expected[wet]).max()),
        max_production_error=float(abs(tank.concentration[wet] - expected[wet]).max()),
        max_temperature_error=float(
            abs(tank.temperature[wet] - expected_temperature[wet]).max()
        ),
        max_velocity_error=float(abs(velocity).max()),
        relative_mass_error=abs(tank.solute_mass - initial_mass) / initial_mass,
        diagnostics=dict(tank.remap_diagnostics),
    )


def plot_report(report, output):
    """Export a scientific figure; diagnostic errors have no new pass/fail gate."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    for pose in ("aligned", "tilted_shifted"):
        rows = [r for r in report["sampling"] if r["pose"] == pose]
        for metric, style, label in (
            ("max_cartesian_error", "--", "Cartesian values"),
            ("max_liquid_average_error", "-", "Liquid averages"),
        ):
            axes[0].plot(
                [r["spacing"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                linestyle=style,
                label=f"{pose}: {label}",
            )
    axes[0].set(
        title="Fixed-crystal affine-field sampling",
        ylabel="Maximum facet supersaturation error",
    )
    for slope in (0.0, 0.4):
        rows = [r for r in report["transport"] if r["slope"] == slope]
        axes[1].plot(
            [r["spacing"] for r in rows],
            [r["normalized_cut_cell_balance"] for r in rows],
            marker="o",
            linestyle="--",
            label=f"Two-point slope={slope:g}",
        )
        axes[1].plot(
            [r["spacing"] for r in rows],
            [r["normalized_production_balance"] for r in rows],
            marker="x",
            label=f"Corrected slope={slope:g}",
        )
    axes[1].set(
        title="Tangent affine-field transport",
        ylabel="Net-flux defect / (|gradient| × cell width²)",
    )
    for metric, label in (
        ("max_first_order_error", "Donor reference"),
        ("max_production_error", "Moment/MUSCL production"),
    ):
        axes[2].semilogy(
            [r["spacing"] for r in report["remap"]],
            [r[metric] for r in report["remap"]],
            marker="o",
            label=label,
        )
    axes[2].set(
        title="Prescribed liquid-layer translation",
        ylabel="Maximum concentration error",
    )
    for axis in axes:
        axis.set_xlabel("Liquid cell width (model units)")
        axis.invert_xaxis()
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Spatial diagnostics: analytic fields and prescribed motion, no growth kinetics"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_diagnostics(output, study_config=None):
    """Record the design/source first, then measure two independent error paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(ROOT / "src/simulation").glob("*.py")]
    protocol = (
        json.loads(Path(study_config).read_text()) if study_config is not None else None
    )
    base = (
        json.loads(
            (
                Path(study_config).resolve().parent / protocol["simulation_config"]
            ).read_text()
        )
        if protocol is not None
        else json.loads((ROOT / "configs/default.json").read_text())
    )
    manifest = dict(
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        environment=dict(
            python=platform.python_version(),
            numpy=version("numpy"),
            scipy=version("scipy"),
        ),
        sampling_grids=protocol["grids"]
        if protocol is not None
        else [[20, 16, 16], [30, 24, 24], [40, 32, 32]],
        base_configuration=base,
        sampling_poses=["aligned", "tilted_shifted"],
        sampling_depth=0.75,
        transport_counts=[8, 12, 16],
        transport_slopes=[0.0, 0.4],
        remap_counts=[8, 12, 16],
        remap_displacement=0.05,
        interpretation="Diagnostic errors only; no new tolerance or coupled convergence claim",
    )
    (output / "study.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sampling, transport = [], []
    for pose in manifest["sampling_poses"]:
        for grid in manifest["sampling_grids"]:
            row = sampling_case(grid, pose, base)
            sampling.append(row)
            print(
                f"Sampling {pose} {grid}: liquid-average error={row['max_liquid_average_error']:.6g}",
                flush=True,
            )
    for slope in manifest["transport_slopes"]:
        for count in manifest["transport_counts"]:
            row = transport_case(count, slope)
            transport.append(row)
            print(
                f"Transport slope={slope:g} n={count}: normalized balance={row['normalized_cut_cell_balance']:.6g}",
                flush=True,
            )
    remap = [remap_case(count) for count in manifest["remap_counts"]]
    report = dict(
        manifest=manifest, sampling=sampling, transport=transport, remap=remap
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Fixed-geometry spatial diagnostics",
        "",
        "These errors isolate sampling and internal flux approximations; they do not measure real KDP accuracy.",
        "",
        "## Facet sampling",
        "",
        "| Pose / grid | Cartesian-value error | Liquid-average error | Exact-integral error |",
        "|---|---:|---:|---:|",
    ]
    for row in sampling:
        lines.append(
            f"| {row['pose']} / {row['grid']} | {row['max_cartesian_error']:.6g} | {row['max_liquid_average_error']:.6g} | {row['max_exact_integral_error']:.3e} |"
        )
    lines += [
        "",
        "Sampling errors are absolute dimensionless supersaturation differences, not growth percentages.",
        "",
        "## Prescribed layer translation",
        "",
        "| Grid count | Donor reference error | Production C error | Production T error | Velocity error | Relative mass error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in remap:
        lines.append(
            f"| {row['grid'][0]} | {row['max_first_order_error']:.6g} | {row['max_production_error']:.3e} | {row['max_temperature_error']:.3e} | {row['max_velocity_error']:.3e} | {row['relative_mass_error']:.3e} |"
        )
    lines += [
        "",
        "Translation errors use model concentration, temperature and length/time units. Mass error is relative.",
        "",
        "## Internal transport",
        "",
        "| Plane slope / grid count | Two-point flux error | Two-point balance | Normalized two-point balance | Production corrected balance | Exact-gradient balance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in transport:
        lines.append(
            f"| {row['slope']:g} / {row['grid'][0]} | {row['max_internal_flux_error']:.6g} | {row['max_cut_cell_balance']:.6g} | {row['normalized_cut_cell_balance']:.6g} | {row['max_production_balance']:.3e} | {row['max_exact_gradient_balance']:.3e} |"
        )
    lines += [
        "",
        "Transport uses unit diffusivity. Only interior cut cells enter the balance metric.",
        "The prescribed field is tangent to the solid plane; outer walls are excluded because their sealed boundary condition does not match it.",
        "Exact-gradient values are oracle references; the production correction estimates gradients from cell means.",
        "The layer-translation case exercises the production moving-boundary remap.",
        "These tests do not apportion all coupled growth error or establish a new convergence gate.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot_report(report, output / "summary.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/spatial_accuracy"
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        help="Use the sampling grids and physical setup of this study",
    )
    args = parser.parse_args()
    run_diagnostics(args.output, args.study_config)
    print(f"Report: {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
