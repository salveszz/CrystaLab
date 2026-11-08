# HOW THIS FILE WORKS
# Declare the scalar protocol before measuring. Compare actual finite-volume
# means with analytic translating/diffusing waves and a periodic Gaussian pulse.
# Quantify pulse broadening relative to the very small physical diffusivity.
# Isolate RK time error with an exact semidiscrete diffusion eigenmode, separately
# from continuous-PDE spatial refinement. Resolve momentum around the fixed cube
# while carrying uniform/nonuniform solute to check actual coupling invariants.
# Archive sources, protocol, reference and environment; reject changed sources.
# Raw SI fields and readable reports/figures go to a fresh ignored results folder.

"""Verify conservative fixed-grid solute transport and its momentum coupling."""

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
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.flow import IncompressibleFlow
from simulation.solute import FlowSoluteSimulation, SoluteTransport
from simulation.units import FluidPropertiesSI, finite_number
from describe_physical_setup import describe_setup


def cell_centers_m(flow):
    z, y, x = np.meshgrid(
        *[(np.arange(n) + 0.5) * h for n, h in zip(flow.shape, flow.spacing)],
        indexing="ij",
    )
    return np.stack((x, y, z), axis=-1)


def _flow(properties, length, counts, speed=(0, 0, 0), *, periodic=True):
    flow = IncompressibleFlow(
        [length] * 3, counts, properties, periodic_xyz=(periodic,) * 3
    )
    flow.set_velocity([np.full_like(v, u) for v, u in zip(flow.velocity, speed)])
    return flow


def _error(actual, exact, mean):
    return float(np.linalg.norm(actual - exact) / np.linalg.norm(exact - mean))


def periodic_case(properties, settings, n):
    L, duration = settings["length_m"], settings["duration_s"]
    speed = np.array(settings["velocity_m_per_s"])
    flow = _flow(properties, L, [n] * 3, speed)
    xyz, k = cell_centers_m(flow), 2 * np.pi / L
    mean, amplitude = settings["mean_kg_per_m3"], settings["amplitude_kg_per_m3"]
    amplitude *= np.prod(np.sinc(k * flow.spacing / (2 * np.pi)))
    initial = mean + amplitude * np.cos(k * xyz.sum(axis=-1))
    scalar = SoluteTransport(flow, initial)
    scalar.advance(duration, max_substep_s=settings["max_substep_s"])
    exact = mean + amplitude * np.exp(
        -3 * properties.solute_diffusivity_m2_per_s * k**2 * duration
    ) * np.cos(k * (xyz - speed * duration).sum(axis=-1))
    return (
        scalar,
        exact,
        dict(
            n=n,
            relative_rms_error=_error(scalar.concentration, exact, mean),
            velocity_mode="prescribed_constant_three_component",
        ),
        initial,
    )


def diffusion_case(properties, settings, n):
    L, duration = settings["length_m"], settings["duration_s"]
    flow = _flow(properties, L, [n] * 3, periodic=False)
    k, mean = np.pi / L, settings["mean_kg_per_m3"]
    mode = np.prod(np.cos(k * cell_centers_m(flow)), axis=-1)
    mode *= np.prod(np.sinc(k * flow.spacing / (2 * np.pi)))
    initial = mean + settings["amplitude_kg_per_m3"] * mode
    scalar = SoluteTransport(flow, initial)
    scalar.advance(duration)
    exact = mean + (initial - mean) * np.exp(
        -3 * properties.solute_diffusivity_m2_per_s * k**2 * duration
    )
    return (
        scalar,
        exact,
        dict(
            n=n,
            relative_rms_error=_error(scalar.concentration, exact, mean),
            velocity_mode="prescribed_zero_neumann_diffusion",
        ),
        initial,
    )


def periodic_gaussian_average(flow, settings, t):
    """Analytic periodic heat-kernel pulse integrated over each x cell."""
    L, sigma0 = settings["length_m"], settings["initial_sigma_m"]
    sigma = np.sqrt(sigma0**2 + 2 * flow.properties.solute_diffusivity_m2_per_s * t)
    center = (settings["initial_center_m"] + settings["velocity_m_per_s"] * t) % L
    x = cell_centers_m(flow)[..., 0]
    h = flow.spacing[2]
    value = np.full(flow.shape, settings["mean_kg_per_m3"], dtype=float)
    for image in range(-3, 4):
        distance = x - center - image * L
        value += (
            settings["amplitude_kg_per_m3"]
            * sigma0
            * np.sqrt(np.pi / 2)
            / h
            * (
                erf((distance + h / 2) / (np.sqrt(2) * sigma))
                - erf((distance - h / 2) / (np.sqrt(2) * sigma))
            )
        )
    return value


def pulse_case(properties, settings, n):
    L, duration = settings["length_m"], settings["duration_s"]
    transverse = settings["transverse_counts"]
    flow = _flow(
        properties, L, [n, transverse, transverse], (settings["velocity_m_per_s"], 0, 0)
    )
    initial = periodic_gaussian_average(flow, settings, 0)
    scalar = SoluteTransport(flow, initial)
    scalar.advance(duration, max_substep_s=settings["max_substep_s"])
    exact = periodic_gaussian_average(flow, settings, duration)
    center = (
        settings["initial_center_m"] + settings["velocity_m_per_s"] * duration
    ) % L
    distance = (cell_centers_m(flow)[..., 0] - center + L / 2) % L - L / 2
    mean = settings["mean_kg_per_m3"]

    def variance(field):
        weights = field - mean
        return float(np.sum(weights * distance**2) / np.sum(weights))

    excess = (variance(scalar.concentration) - variance(exact)) / (2 * duration)
    return (
        scalar,
        exact,
        dict(
            n=n,
            relative_rms_error=_error(scalar.concentration, exact, mean),
            excess_diffusivity_m2_per_s=excess,
            abs_excess_diffusivity_over_physical=abs(excess)
            / properties.solute_diffusivity_m2_per_s,
            solutal_peclet=abs(settings["velocity_m_per_s"])
            * L
            / properties.solute_diffusivity_m2_per_s,
            velocity_mode="prescribed_constant_x_periodic_pulse",
        ),
        initial,
    )


def temporal_case(properties, settings, dt):
    L, duration = settings["length_m"], settings["duration_s"]
    flow = _flow(properties, L, settings["counts_xyz"])
    k = 2 * np.pi * settings["x_cycles"] / L
    mean = settings["mean_kg_per_m3"]
    mode = np.cos(k * cell_centers_m(flow)[..., 0]) * np.sinc(
        k * flow.spacing[2] / (2 * np.pi)
    )
    initial = mean + settings["amplitude_kg_per_m3"] * mode
    scalar = SoluteTransport(flow, initial)
    scalar.advance(duration, max_substep_s=dt)
    eigenvalue = 4 * np.sin(0.5 * k * flow.spacing[2]) ** 2 / flow.spacing[2] ** 2
    exact = mean + (initial - mean) * np.exp(
        -properties.solute_diffusivity_m2_per_s * eigenvalue * duration
    )
    return (
        scalar,
        exact,
        dict(
            dt_s=dt,
            relative_rms_error=_error(scalar.concentration, exact, mean),
            reference="semidiscrete_diffusion_exponential",
            velocity_mode="prescribed_zero_periodic_diffusion",
        ),
        initial,
    )


def obstacle_case(reference, settings, *, uniform=False, rest=False):
    domain, crystal = reference["domain"], reference["crystal"]
    center, size = np.array(crystal["center_m"]), np.array(crystal["size_m"])
    flow = IncompressibleFlow(
        [*domain["size_m"][:2], domain["liquid_height_m"]],
        settings["counts_xyz"],
        FluidPropertiesSI(**reference["fluid"]),
        solid_box_m=[center - size / 2, center + size / 2],
    )
    mean = (
        flow.properties.density_kg_per_m3
        * flow.properties.mass_ratio
        / (1 + flow.properties.mass_ratio)
    )
    initial = np.full(flow.shape, mean)
    if not uniform and not rest:
        initial += (
            mean
            * settings["relative_concentration_amplitude"]
            * np.prod(np.cos(np.pi * cell_centers_m(flow) / flow.lengths), axis=-1)
        )
    model = FlowSoluteSimulation(flow, initial)

    def acceleration(position, _time):
        x, y, z = np.moveaxis(position / flow.lengths, -1, 0)
        envelope = np.sin(np.pi * z)
        return settings["vortical_acceleration_scale_m_per_s2"] * np.stack(
            (-(y - 0.5) * envelope, (x - 0.5) * envelope, np.zeros_like(z)), axis=-1
        )

    duration = reference["duration_s"] if rest else settings["duration_s"]
    model.run_until(
        duration,
        1.0 if rest else settings["max_dt_s"],
        (0, 0, 0) if rest else acceleration,
    )
    return (
        model.solute,
        initial if uniform or rest else None,
        dict(
            velocity_mode="resolved_manufactured_acceleration"
            if not rest
            else "resolved_rest",
            uniform_max_error_kg_per_m3=float(
                np.max(abs(model.solute.concentration[flow.fluid] - mean))
            )
            if uniform or rest
            else None,
        ),
        initial,
    )


def _validate_protocol(protocol):
    for family, key in (
        ("periodic_3d", "counts"),
        ("diffusion_3d", "counts"),
        ("pulse", "x_counts"),
        ("temporal_diffusion", "timesteps_s"),
    ):
        values = protocol[family][key]
        if len(values) < 3:
            raise ValueError(
                "Verification requires at least three positive refinement levels"
            )
        for value in values:
            finite_number(value, key, positive=True)
            if key != "timesteps_s" and (not isinstance(value, int) or value < 3):
                raise ValueError("Grid counts must be integers >=3")
        if not all(
            b < a if key == "timesteps_s" else a < b for a, b in zip(values, values[1:])
        ):
            raise ValueError("Use increasing grid counts and decreasing timesteps")
        finite_number(
            protocol[family]["finest_relative_rms_tolerance"],
            "error tolerance",
            positive=True,
        )
    for key in (
        "max_relative_mass_drift",
        "concentration_bound_tolerance_kg_per_m3",
        "max_scaled_divergence",
        "max_blocked_face_speed_m_per_s",
    ):
        finite_number(protocol[key], key, positive=True)


def run_study(protocol, reference, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh solute-verification output directory")
    describe_setup(reference)
    _validate_protocol(protocol)
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "src/simulation/flow.py",
            "src/simulation/solute.py",
            "src/simulation/units.py",
            "src/simulation/crystal.py",
            "scripts/check_solute.py",
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
        method="Fixed-grid conservative donor+diffusion base, MC advective reconstruction, paired multidimensional flux budgets, SSP RK2, explicit subcycling; old/new mean face flux for coupled momentum.",
    )
    output.mkdir(parents=True)
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for name, digest in sources.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Source changed before solute verification began")
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    results, pulses = [], []
    obstacle = None

    def record(name, builder):
        start = time.perf_counter()
        scalar, exact, metrics, initial = builder()
        flow = scalar.flow
        initial_liquid = initial[flow.fluid]
        values = scalar.concentration[flow.fluid]
        mass_relative = abs(scalar.mass_kg / scalar.initial_mass_kg - 1)
        bound_error = float(
            max(
                initial_liquid.min() - values.min(),
                values.max() - initial_liquid.max(),
                0,
            )
        )
        row = dict(case=name, **flow.measurements())
        row.update(scalar.measurements())
        row.update(
            metrics,
            runtime_s=time.perf_counter() - start,
            relative_mass_drift=mass_relative,
            bound_violation_kg_per_m3=bound_error,
        )
        row["constraints_pass"] = (
            mass_relative <= protocol["max_relative_mass_drift"]
            and bound_error <= protocol["concentration_bound_tolerance_kg_per_m3"]
            and row["scaled_divergence"] <= protocol["max_scaled_divergence"]
            and row["blocked_face_speed_m_per_s"]
            <= protocol["max_blocked_face_speed_m_per_s"]
        )
        archive = dict(
            concentration_kg_per_m3=scalar.concentration,
            initial_concentration_kg_per_m3=initial,
            u_x_m_per_s=flow.velocity[0],
            u_y_m_per_s=flow.velocity[1],
            u_z_m_per_s=flow.velocity[2],
            dynamic_pressure_Pa=flow.pressure,
            fluid=flow.fluid,
            lengths_m=flow.lengths,
            solid_vertices_m=flow.solid_vertices_m,
            time_s=scalar.time_s,
            velocity_mode=row["velocity_mode"],
        )
        if exact is not None:
            archive["reference_concentration_kg_per_m3"] = exact
        np.savez_compressed(output / f"{name}.npz", **archive)
        results.append(row)
        print(name, json.dumps(row, allow_nan=False), flush=True)
        return scalar, exact

    properties = FluidPropertiesSI(**reference["fluid"])
    for family, key, builder in (
        ("periodic_3d", "counts", periodic_case),
        ("diffusion_3d", "counts", diffusion_case),
        ("pulse", "x_counts", pulse_case),
        ("temporal_diffusion", "timesteps_s", temporal_case),
    ):
        for value in protocol[family][key]:
            scalar, exact = record(
                f"{family}_{value}",
                lambda: builder(properties, protocol[family], value),
            )
            if family == "pulse":
                pulses.append((scalar, exact))
    record("rest", lambda: obstacle_case(reference, protocol["obstacle"], rest=True))
    record(
        "obstacle_uniform",
        lambda: obstacle_case(reference, protocol["obstacle"], uniform=True),
    )
    obstacle, _ = record(
        "obstacle_nonuniform", lambda: obstacle_case(reference, protocol["obstacle"])
    )
    gates = dict(
        conservation_bounds_and_flow=all(r["constraints_pass"] for r in results)
    )
    for family in ("periodic_3d", "diffusion_3d", "pulse", "temporal_diffusion"):
        errors = [
            r["relative_rms_error"]
            for r in results
            if r["case"].startswith(family + "_")
        ]
        gates[family] = (
            all(b < a for a, b in zip(errors, errors[1:]))
            and errors[-1] <= protocol[family]["finest_relative_rms_tolerance"]
        )
    pulse = [r for r in results if r["case"].startswith("pulse_")]
    gates["pulse_numerical_diffusion"] = (
        pulse[-1]["abs_excess_diffusivity_over_physical"]
        <= protocol["pulse"]["finest_abs_excess_diffusivity_over_physical_tolerance"]
    )
    rest = next(r for r in results if r["case"] == "rest")
    uniform = next(r for r in results if r["case"] == "obstacle_uniform")
    gates["rest"] = (
        rest["max_speed_m_per_s"] == 0
        and rest["uniform_max_error_kg_per_m3"] == 0
        and rest["time_s"] == reference["duration_s"]
    )
    gates["coupled_uniform"] = (
        uniform["uniform_max_error_kg_per_m3"]
        <= protocol["obstacle"]["uniform_absolute_tolerance_kg_per_m3"]
    )
    gates["nonzero_coupled_flow"] = all(
        r["max_speed_m_per_s"] > 1e-6
        for r in results
        if r["case"].startswith("obstacle_")
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Source changed during verification; results cannot be accepted"
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
        "# Fixed-grid solute verification",
        "",
        protocol["limits"],
        "",
        "| Case | Relative fluctuation RMS error | Relative mass drift | Bound violation (kg/m3) |",
        "|---|---:|---:|---:|",
    ]
    for row in results:
        error = row.get("relative_rms_error")
        lines.append(
            f"| {row['case']} | {error if error is not None else 'n/a'} | {row['relative_mass_drift']:.3e} | {row['bound_violation_kg_per_m3']:.3e} |"
        )
    lines += [
        "",
        f"Finest pulse |D_excess|/D: {pulse[-1]['abs_excess_diffusivity_over_physical']:.6g}",
        "",
        f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}.",
        "",
        *[f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in gates.items()],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    plot_report(report, pulses, obstacle, output)
    return report


def plot_report(report, pulses, obstacle, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, family, label in zip(
        axes.flat,
        ("periodic_3d", "diffusion_3d", "pulse", "temporal_diffusion"),
        (
            "3D translating/diffusing wave",
            "3D no-flux diffusion",
            "High-Peclet Gaussian pulse",
            "Diffusion time error (semidiscrete reference)",
        ),
    ):
        rows = [r for r in report["results"] if r["case"].startswith(family + "_")]
        key = "dt_s" if family == "temporal_diffusion" else "n"
        axis.loglog(
            [r[key] for r in rows], [r["relative_rms_error"] for r in rows], "o-"
        )
        axis.set(
            xlabel="Timestep (s)" if key == "dt_s" else "Cells per refined axis",
            ylabel="Relative fluctuation RMS error",
            title=label,
        )
        axis.grid(alpha=0.25)
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for scalar, exact in pulses:
        x = cell_centers_m(scalar.flow)[0, 0, :, 0] * 1000
        axes[0].plot(x, scalar.concentration[0, 0], label=f"nx={scalar.flow.shape[2]}")
    axes[0].plot(x, exact[0, 0], "k--", label="Analytic cell mean")
    axes[0].set(
        xlabel="x (mm)",
        ylabel="C (kg KDP / m³ solution)",
        title=f"Periodic pulse after {report['protocol']['pulse']['duration_s']:g} s",
    )
    axes[0].legend()
    rows = [r for r in report["results"] if r["case"].startswith("pulse_")]
    axes[1].loglog(
        [r["n"] for r in rows],
        [r["abs_excess_diffusivity_over_physical"] for r in rows],
        "o-",
    )
    axes[1].axhline(
        report["protocol"]["pulse"][
            "finest_abs_excess_diffusivity_over_physical_tolerance"
        ],
        color="black",
        linestyle="--",
        label="Declared finest-grid limit",
    )
    axes[1].set(
        xlabel="x cells",
        ylabel="|Excess diffusivity| / physical D",
        title="Numerical pulse broadening",
    )
    axes[1].legend()
    fig.savefig(output / "pulse.png", dpi=160)
    plt.close(fig)
    flow, concentration = obstacle.flow, obstacle.concentration
    z = flow.shape[0] // 2
    length_mm = flow.lengths.copy() * 1000
    image = np.ma.masked_where(~flow.fluid[z], concentration[z])
    fig, axis = plt.subplots(figsize=(6, 5), layout="constrained")
    mesh = axis.imshow(
        image, origin="lower", extent=(0, length_mm[0], 0, length_mm[1]), cmap="viridis"
    )
    xyz = cell_centers_m(flow)[z] * 1000
    velocity = flow.centered_velocity()[z]
    axis.quiver(
        xyz[::2, ::2, 0],
        xyz[::2, ::2, 1],
        velocity[::2, ::2, 0],
        velocity[::2, ::2, 1],
        color="white",
    )
    axis.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"Resolved fixed-cube flow + solute, t={obstacle.time_s:g} s",
        aspect="equal",
    )
    fig.colorbar(mesh, ax=axis, label="C (kg KDP / m³ solution)")
    fig.savefig(output / "coupled_solute.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config", type=Path, default=ROOT / "configs/solute_verification.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/solute_verification"
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
