# FILE PURPOSE: Run a frozen, constant-property heat/mass/buoyancy verification.
# Save native fields, source snapshots, environment, budgets and failed checks.
# Manufactured properties/solutions are not accepted KDP process training data.
"""Reproduce with --protocol configs/thermal_verification.json --output NEW_DIR."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.flow import IncompressibleFlow
from simulation.thermal import (
    BoussinesqProperties,
    FlowHeatSoluteSimulation,
    HeatPropertiesSI,
    HeatTransport,
    ThermalBoundary,
)
from simulation.units import FluidPropertiesSI

SOURCES = [
    "configs/thermal_verification.json",
    "src/simulation/scalar.py",
    "src/simulation/solute.py",
    "src/simulation/thermal.py",
    "src/simulation/flow.py",
    "src/simulation/mesh.py",
    "src/simulation/pressure.py",
    "src/simulation/linear.py",
    "src/simulation/units.py",
    "scripts/check_thermal.py",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(value**2)))


def shear_truth(protocol, flow, dt):
    """Continuous-in-time reference on the same spatial finite-volume grid.

    w is a face-control-volume mean; x is tangential to its z face. All three
    fields therefore share the x-cell cosine average. There is no advective
    derivative for this z-invariant shear. Spatial eigenvalue is declared here
    to isolate the first-order momentum/coupling time error.
    """
    p = protocol["shear"]
    L, duration = protocol["length_m"], p["duration_s"]
    k, h = 2 * np.pi / L, flow.grid.widths[2][0]
    lam = 4 * np.sin(k * h / 2) ** 2 / h**2
    alpha = protocol["heat"]["conductivity_W_per_m_K"] / (
        protocol["fluid"]["density_kg_per_m3"]
        * protocol["heat"]["heat_capacity_J_per_kg_K"]
    )
    nu = flow.properties.kinematic_viscosity_m2_per_s
    D = flow.properties.solute_diffusivity_m2_per_s
    b = protocol["buoyancy"]

    def response(diffusivity):
        if diffusivity == nu:
            return duration * np.exp(-nu * lam * duration)
        return (
            np.exp(-diffusivity * lam * duration) - np.exp(-nu * lam * duration)
        ) / ((nu - diffusivity) * lam)

    amplitude = -b["gravity_m_per_s2"][2] * (
        b["thermal_expansion_per_K"] * p["temperature_amplitude_K"] * response(alpha)
        - b["solutal_expansion_m3_per_kg"]
        * p["concentration_amplitude_kg_per_m3"]
        * response(D)
    )
    mode = np.sinc(h / L) * np.cos(k * flow.face_positions(2)[..., 0])
    return amplitude * mode


def save_model(path, model, initial_T, initial_C, extra=None):
    np.savez_compressed(
        path,
        initial_temperature_K=initial_T,
        initial_concentration_kg_per_m3=initial_C,
        temperature_K=model.heat.temperature,
        concentration_kg_per_m3=model.solute.concentration,
        u_m_per_s=model.flow.velocity[0],
        v_m_per_s=model.flow.velocity[1],
        w_m_per_s=model.flow.velocity[2],
        pressure_Pa=model.flow.pressure,
        **({} if extra is None else extra),
    )


def run_study(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "source"
    sources = list(SOURCES)
    for relative in sources:
        source = ROOT / relative
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    started = time.time()
    environment = dict(
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        started_unix_s=started,
    )
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    fluid, heat, buoyancy = (
        FluidPropertiesSI(**protocol["fluid"]),
        HeatPropertiesSI(**protocol["heat"]),
        BoussinesqProperties(**protocol["buoyancy"]),
    )
    L, threshold = protocol["length_m"], protocol["acceptance"]
    report = dict(
        purpose=protocol["purpose"],
        protocol_sha256=digest(protocol_path),
        shear=[],
        wall_conduction=[],
        mixing=[],
    )
    for dt in protocol["shear"]["timesteps_s"]:
        p = protocol["shear"]
        flow = IncompressibleFlow(
            [L] * 3, p["counts_xyz"], fluid, periodic_xyz=(True,) * 3
        )
        mode = np.sinc(1 / p["counts_xyz"][0]) * np.cos(
            2 * np.pi * flow.cell_positions()[..., 0] / L
        )
        T = buoyancy.reference_temperature_K + p["temperature_amplitude_K"] * mode
        C = (
            buoyancy.reference_concentration_kg_per_m3
            + p["concentration_amplitude_kg_per_m3"] * mode
        )
        model = FlowHeatSoluteSimulation(flow, T, C, heat, buoyancy)
        model.run_until(p["duration_s"], dt)
        truth = shear_truth(protocol, flow, dt)
        name = f"shear_dt_{dt:g}.npz"
        save_model(output / name, model, T, C, {"reference_w_m_per_s": truth})
        report["shear"].append(
            dict(
                dt_s=dt,
                artifact=name,
                relative_velocity_error=rms(flow.velocity[2] - truth) / rms(truth),
                **model.measurements(),
            )
        )
    for nx in protocol["wall_conduction"]["counts_x"]:
        p = protocol["wall_conduction"]
        flow = IncompressibleFlow([L] * 3, [nx, 4, 4], fluid)
        x = flow.cell_positions()[..., 0]
        base = (
            p["left_temperature_K"]
            + (p["right_temperature_K"] - p["left_temperature_K"]) * x / L
        )
        mode = np.sinc(1 / (2 * nx)) * np.sin(np.pi * x / L)
        T = base + p["sine_amplitude_K"] * mode
        boundaries = {
            f"x_{side}": ThermalBoundary(
                p[f"{label}_temperature_K"], None, "DESIGN-THERMAL-001"
            )
            for side, label in (("min", "left"), ("max", "right"))
        }
        thermal = HeatTransport(flow, T, heat, boundaries=boundaries)
        thermal.advance(p["duration_s"], max_substep_s=p["max_substep_s"])
        exact = base + p["sine_amplitude_K"] * mode * np.exp(
            -thermal.diffusivity_m2_per_s * (np.pi / L) ** 2 * p["duration_s"]
        )
        name = f"wall_nx_{nx}.npz"
        np.savez_compressed(
            output / name,
            initial_temperature_K=T,
            temperature_K=thermal.temperature,
            reference_temperature_K=exact,
        )
        report["wall_conduction"].append(
            dict(
                counts_x=nx,
                artifact=name,
                rms_error_K=rms(thermal.temperature - exact),
                **thermal.measurements(),
            )
        )
    for enabled in (False, True):
        p = protocol["mixing"]
        flow = IncompressibleFlow(
            [L] * 3, p["counts_xyz"], fluid, periodic_xyz=(True,) * 3
        )
        velocity = []
        for index in range(3):
            xyz = 2 * np.pi * flow.face_positions(index) / L
            first, second = ((2, 1), (0, 2), (1, 0))[index]
            velocity.append(
                p["initial_ABC_speed_m_per_s"]
                * (np.sin(xyz[..., first]) + np.cos(xyz[..., second]))
            )
        flow.set_velocity(velocity)
        xyz = 2 * np.pi * flow.cell_positions() / L
        T = buoyancy.reference_temperature_K + p["temperature_amplitude_K"] * np.cos(
            xyz[..., 0]
        ) * np.sin(xyz[..., 1])
        C = buoyancy.reference_concentration_kg_per_m3 + p[
            "concentration_amplitude_kg_per_m3"
        ] * np.sin(xyz[..., 1]) * np.cos(xyz[..., 2])
        coefficients = (
            buoyancy
            if enabled
            else replace(
                buoyancy, thermal_expansion_per_K=0, solutal_expansion_m3_per_kg=0
            )
        )
        model = FlowHeatSoluteSimulation(flow, T, C, heat, coefficients)
        model.run_until(p["duration_s"], p["max_dt_s"])
        name = f"mixing_buoyancy_{str(enabled).lower()}.npz"
        save_model(output / name, model, T, C)
        report["mixing"].append(
            dict(buoyancy_enabled=enabled, artifact=name, **model.measurements())
        )
    with (
        np.load(output / report["mixing"][0]["artifact"]) as off,
        np.load(output / report["mixing"][1]["artifact"]) as on,
    ):
        difference = max(
            float(abs(on[key] - off[key]).max())
            for key in ("u_m_per_s", "v_m_per_s", "w_m_per_s")
        )
    report["mixing_buoyancy_velocity_difference_m_per_s"] = difference
    errors = [r["relative_velocity_error"] for r in report["shear"]]
    ratios = [a / b for a, b in zip(errors, errors[1:])]
    report["shear_temporal_error_ratios"] = ratios
    rows = report["shear"] + report["wall_conduction"] + report["mixing"]
    wall_errors = [r["rms_error_K"] for r in report["wall_conduction"]]
    report["checks"] = dict(
        first_order_coupling=all(
            threshold["shear_temporal_error_ratio_min"]
            <= r
            <= threshold["shear_temporal_error_ratio_max"]
            for r in ratios
        ),
        shear_finest_accuracy=errors[-1]
        < threshold["shear_finest_relative_velocity_error_max"],
        wall_refinement=all(b < a for a, b in zip(wall_errors, wall_errors[1:]))
        and wall_errors[-1] < threshold["wall_finest_rms_error_K_max"],
        heat_budgets=all(
            abs(r["heat_budget_residual_J"]) < threshold["heat_budget_residual_J_max"]
            for r in rows
        ),
        solute_budgets=all(
            abs(r.get("solute_mass_drift_kg", 0))
            < threshold["solute_mass_drift_kg_max"]
            for r in rows
        ),
        incompressibility=all(
            r.get("scaled_divergence", 0) < threshold["scaled_divergence_max"]
            for r in rows
        ),
        buoyancy_feedback=difference
        > threshold["mixing_buoyancy_velocity_difference_m_per_s_min"],
    )
    report["elapsed_s"] = time.time() - started
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    manifest = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                checks=report["checks"],
                shear_errors=errors,
                wall_errors_K=wall_errors,
                elapsed_s=report["elapsed_s"],
            ),
            indent=2,
        )
    )
    if not all(report["checks"].values()):
        raise ArithmeticError(
            "Frozen thermal verification checks failed; artifacts retained"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/thermal_verification.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.protocol, args.output)
