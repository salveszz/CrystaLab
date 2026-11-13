# FILE PURPOSE: Independently reconstruct thermal verification from saved fields.
# No producer/solver import: use cell integrals, matrix exponential, modal RK
# polynomials, face divergence and budgets. This is numerical, not KDP validation.
"""Audit a check_thermal output directory without running its numerical solver."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(value**2)))


def audit(root, destination):
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest.items():
        assert sha(root / relative) == expected, relative
    p = json.loads((root / "protocol.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert sha(root / "protocol.json") == report["protocol_sha256"]
    rho = p["fluid"]["density_kg_per_m3"]
    capacity = rho * p["heat"]["heat_capacity_J_per_kg_K"]
    alpha = p["heat"]["conductivity_W_per_m_K"] / capacity
    D = p["fluid"]["solute_diffusivity_m2_per_s"]
    nu = p["fluid"]["dynamic_viscosity_Pa_s"] / rho
    L, b, limits = p["length_m"], p["buoyancy"], p["acceptance"]
    assert [r["dt_s"] for r in report["shear"]] == p["shear"]["timesteps_s"]
    assert [r["counts_x"] for r in report["wall_conduction"]] == p["wall_conduction"][
        "counts_x"
    ]
    assert [r["buoyancy_enabled"] for r in report["mixing"]] == [False, True]
    rows, shear_errors, wall_errors = [], [], []
    for group in ("shear", "wall_conduction", "mixing"):
        for row in report[group]:
            with np.load(root / row["artifact"]) as data:
                arrays = {key: data[key] for key in data.files}
            T, T0 = arrays["temperature_K"], arrays["initial_temperature_K"]
            nz, ny, nx = T.shape
            expected_counts = (
                [row["counts_x"], 4, 4]
                if group == "wall_conduction"
                else p[group]["counts_xyz"]
            )
            assert [nx, ny, nz] == expected_counts
            assert row["time_s"] == p[group]["duration_s"]
            volume = L**3 / (nx * ny * nz)
            x_left, x_right = np.arange(nx) * L / nx, np.arange(1, nx + 1) * L / nx
            x = (x_left + x_right) / 2
            assert T.shape == T0.shape and np.isfinite(T).all()
            minimum, maximum = float(T0.min()), float(T0.max())
            if group == "wall_conduction":
                minimum = min(
                    minimum,
                    p[group]["left_temperature_K"],
                    p[group]["right_temperature_K"],
                )
                maximum = max(
                    maximum,
                    p[group]["left_temperature_K"],
                    p[group]["right_temperature_K"],
                )
            assert T.min() >= minimum - 1e-10 and T.max() <= maximum + 1e-10
            energy = (
                capacity * volume * np.sum(T - p["heat"]["reference_temperature_K"])
            )
            energy_change = capacity * volume * np.sum(T - T0)
            budget_error = energy_change - row["boundary_heat_J"]
            assert abs(energy - row["sensible_energy_J"]) < 2e-11
            assert abs(budget_error) < limits["heat_budget_residual_J_max"]
            result = dict(
                artifact=row["artifact"], heat_budget_residual_J=float(budget_error)
            )
            if group != "wall_conduction":
                C, C0 = (
                    arrays["concentration_kg_per_m3"],
                    arrays["initial_concentration_kg_per_m3"],
                )
                assert C.shape == C0.shape == T.shape and np.isfinite(C).all()
                assert C.min() >= C0.min() - 1e-10 and C.max() <= C0.max() + 1e-10
                mass = volume * np.sum(C)
                drift = volume * np.sum(C - C0)
                assert abs(mass - row["solute_mass_kg"]) < 1e-16
                assert abs(drift) < limits["solute_mass_drift_kg_max"]
                fields = [
                    arrays[key] for key in ("u_m_per_s", "v_m_per_s", "w_m_per_s")
                ]
                divergence = np.zeros(T.shape)
                for velocity, axis, n in zip(fields, (2, 1, 0), (nx, ny, nz)):
                    shape = list(T.shape)
                    shape[axis] += 1
                    assert (
                        velocity.shape == tuple(shape) and np.isfinite(velocity).all()
                    )
                    np.testing.assert_allclose(
                        np.take(velocity, 0, axis),
                        np.take(velocity, -1, axis),
                        atol=1e-14,
                        rtol=0,
                    )
                    divergence += np.diff(velocity, axis=axis) / (L / n)
                assert np.isfinite(arrays["pressure_Pa"]).all()
                assert abs(divergence).max() < 1e-10
                scale = max(
                    abs(u).max() / (L / n) for u, n in zip(fields, (nx, ny, nz))
                )
                assert (
                    abs(divergence).max() / max(scale, 1e-30)
                    < limits["scaled_divergence_max"]
                )
                result.update(
                    solute_mass_drift_kg=float(drift),
                    max_divergence_per_s=float(abs(divergence).max()),
                )
            if group == "shear":
                case = p[group]
                wave = 2 * np.pi / L
                mode = (np.sin(wave * x_right) - np.sin(wave * x_left)) / (
                    wave * L / nx
                )
                lam = 4 * np.sin(np.pi / nx) ** 2 / (L / nx) ** 2
                g = -b["gravity_m_per_s2"][2]
                operator = np.array(
                    [
                        [-alpha * lam, 0, 0],
                        [0, -D * lam, 0],
                        [
                            g * b["thermal_expansion_per_K"],
                            -g * b["solutal_expansion_m3_per_kg"],
                            -nu * lam,
                        ],
                    ]
                )
                amplitude0 = np.array(
                    [
                        case["temperature_amplitude_K"],
                        case["concentration_amplitude_kg_per_m3"],
                        0,
                    ]
                )
                np.testing.assert_allclose(
                    T0,
                    b["reference_temperature_K"]
                    + amplitude0[0] * np.broadcast_to(mode, T.shape),
                    atol=1e-12,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    C0,
                    b["reference_concentration_kg_per_m3"]
                    + amplitude0[1] * np.broadcast_to(mode, C.shape),
                    atol=1e-12,
                    rtol=0,
                )
                exact = expm(case["duration_s"] * operator) @ amplitude0
                velocity_reference = np.broadcast_to(
                    exact[2] * mode, arrays["w_m_per_s"].shape
                )
                np.testing.assert_allclose(
                    velocity_reference,
                    arrays["reference_w_m_per_s"],
                    atol=1e-16,
                    rtol=0,
                )
                error = rms(arrays["w_m_per_s"] - velocity_reference) / rms(
                    velocity_reference
                )
                assert abs(error - row["relative_velocity_error"]) < 1e-12
                # Independent finite-step modal solution checks both scalar
                # amplitudes and their signed forcing in the momentum update.
                numerical, t = amplitude0.copy(), 0.0
                while t < case["duration_s"]:
                    dt = min(row["dt_s"], case["duration_s"] - t)
                    numerical[2] = (
                        numerical[2] + dt * operator[2, :2] @ numerical[:2]
                    ) / (1 + nu * lam * dt)
                    rates = np.array([alpha, D]) * lam * dt
                    numerical[:2] *= 1 - rates + 0.5 * rates**2
                    t += dt
                np.testing.assert_allclose(
                    T,
                    b["reference_temperature_K"]
                    + numerical[0] * np.broadcast_to(mode, T.shape),
                    atol=2e-11,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    C,
                    b["reference_concentration_kg_per_m3"]
                    + numerical[1] * np.broadcast_to(mode, C.shape),
                    atol=2e-11,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    arrays["w_m_per_s"],
                    numerical[2] * np.broadcast_to(mode, arrays["w_m_per_s"].shape),
                    atol=1e-13,
                    rtol=0,
                )
                assert (
                    max(abs(arrays[k]).max() for k in ("u_m_per_s", "v_m_per_s"))
                    < 1e-12
                )
                result["relative_velocity_error"] = error
                shear_errors.append(error)
            if group == "wall_conduction":
                case = p[group]
                wave = np.pi / L
                mode = (np.cos(wave * x_left) - np.cos(wave * x_right)) / (
                    wave * L / nx
                )
                base = (
                    case["left_temperature_K"]
                    + (case["right_temperature_K"] - case["left_temperature_K"]) * x / L
                )
                np.testing.assert_allclose(
                    T0,
                    np.broadcast_to(base + case["sine_amplitude_K"] * mode, T.shape),
                    atol=1e-12,
                    rtol=0,
                )
                truth = base + case["sine_amplitude_K"] * mode * np.exp(
                    -alpha * wave**2 * case["duration_s"]
                )
                np.testing.assert_allclose(
                    arrays["reference_temperature_K"],
                    np.broadcast_to(truth, T.shape),
                    atol=1e-12,
                    rtol=0,
                )
                error = rms(T - truth)
                assert abs(error - row["rms_error_K"]) < 1e-12
                lam = 4 * np.sin(np.pi / (2 * nx)) ** 2 / (L / nx) ** 2
                count = row["heat_substeps"]
                value = alpha * lam * case["duration_s"] / count
                discrete = (
                    base
                    + case["sine_amplitude_K"]
                    * mode
                    * (1 - value + 0.5 * value**2) ** count
                )
                np.testing.assert_allclose(
                    T, np.broadcast_to(discrete, T.shape), atol=2e-11, rtol=0
                )
                wall_errors.append(error)
                result["rms_error_K"] = error
            rows.append(result)
    errors = np.array(shear_errors)
    ratios = errors[:-1] / errors[1:]
    assert np.all(
        (ratios >= limits["shear_temporal_error_ratio_min"])
        & (ratios <= limits["shear_temporal_error_ratio_max"])
    )
    assert errors[-1] < limits["shear_finest_relative_velocity_error_max"]
    assert all(a > b for a, b in zip(wall_errors, wall_errors[1:]))
    assert wall_errors[-1] < limits["wall_finest_rms_error_K_max"]
    with (
        np.load(root / report["mixing"][0]["artifact"]) as off,
        np.load(root / report["mixing"][1]["artifact"]) as on,
    ):
        difference = max(
            float(abs(on[k] - off[k]).max())
            for k in ("u_m_per_s", "v_m_per_s", "w_m_per_s")
        )
    assert (
        abs(difference - report["mixing_buoyancy_velocity_difference_m_per_s"]) < 1e-15
    )
    assert difference > limits["mixing_buoyancy_velocity_difference_m_per_s_min"]
    result = dict(
        passed=True,
        scope="native-field numerical reconstruction; no KDP calibration or physical risk labels",
        source_manifest_sha256=sha(root / "manifest.json"),
        protocol_sha256=report["protocol_sha256"],
        rows=rows,
        shear_temporal_error_ratios=ratios.tolist(),
        mixing_velocity_difference_m_per_s=difference,
    )
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
