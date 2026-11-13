# FILE PURPOSE: Verify heat budgets, coupled buoyancy signs, convergence and rollback.
# All constants here are numerical test designs, not measured KDP properties.
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_thermal import run_study
from audit_thermal import audit
from simulation.flow import IncompressibleFlow
from simulation.thermal import (
    BoussinesqProperties,
    FlowHeatSoluteSimulation,
    HeatPropertiesSI,
    HeatTransport,
    ThermalBoundary,
)
from simulation.units import FluidPropertiesSI


FLUID = FluidPropertiesSI(303.15, 0, 1000, 0.001, 5e-8, "DESIGN-THERMAL-001")
HEAT = HeatPropertiesSI(4000, 0.4, 303.15, "DESIGN-THERMAL-001")
BUOYANCY = BoussinesqProperties(
    2e-4, 8e-4, 303.15, 200, (0, 0, -9.81), 0.05, "DESIGN-THERMAL-001"
)


def flow(n=6, periodic=True, **kwargs):
    return IncompressibleFlow(
        [0.01] * 3, [n] * 3, FLUID, periodic_xyz=(periodic,) * 3, **kwargs
    )


def wall(temperature, h=None):
    return ThermalBoundary(temperature, h, "DESIGN-THERMAL-001")


def test_periodic_heat_cell_average_diffusion_converges_and_conserves_energy():
    errors = []
    for n in (6, 10, 16):
        f = flow(n)
        x = f.cell_positions()[..., 0]
        k = 2 * np.pi / 0.01
        mode = np.sinc(1 / n) * np.cos(k * x)
        initial = 303.15 + 2 * mode
        heat = HeatTransport(f, initial, HEAT)
        heat.advance(2, max_substep_s=0.02)
        truth = 303.15 + 2 * mode * np.exp(-heat.diffusivity_m2_per_s * k * k * 2)
        errors.append(float(np.sqrt(np.mean((truth - heat.temperature) ** 2))))
        assert abs(heat.measurements()["heat_budget_residual_J"]) < 1e-12
        assert heat.temperature.min() >= initial.min()
        assert heat.temperature.max() <= initial.max()
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.002


def test_dirichlet_heat_budget_and_kelvin_bounds_in_cold_closed_cavity():
    f = flow(periodic=False)
    heat = HeatTransport(f, 300, HEAT, boundaries={"x_min": wall(310)})
    before = heat.temperature.copy()
    row = heat.advance(20)
    assert row["boundary_heat_J"] > 0
    assert abs(row["heat_budget_residual_J"]) < 1e-10
    assert heat.temperature.min() >= 300
    assert heat.temperature.max() <= 310
    assert np.linalg.norm(heat.temperature - before) > 1


def test_robin_resistance_preserves_exact_steady_linear_temperature_on_stretched_grid():
    edges = [[0, 0.001, 0.003, 0.006, 0.01]] * 3
    f = IncompressibleFlow([0.01] * 3, [4] * 3, FLUID, grid_edges_m=edges)
    q = (310 - 300) / (1 / 20 + 0.01 / 0.4 + 1 / 40)
    initial = 310 - q / 20 - q * f.cell_positions()[..., 0] / 0.4
    heat = HeatTransport(
        f, initial, HEAT, boundaries={"x_min": wall(310, 20), "x_max": wall(300, 40)}
    )
    heat.advance(10)
    np.testing.assert_allclose(heat.temperature, initial, atol=2e-12, rtol=0)
    assert abs(heat.measurements()["heat_budget_residual_J"]) < 1e-11


def test_time_dependent_walls_are_integrated_at_both_rk_stages():
    f = flow(periodic=False)
    seen = []

    def ambient(t):
        seen.append(t)
        return 301 + t

    heat = HeatTransport(f, 300, HEAT, boundaries={"x_min": wall(ambient, 100)})
    row = heat.advance(0.1, max_substep_s=0.05)
    assert any(abs(t - 0.05) < 1e-15 for t in seen)
    assert any(abs(t - 0.1) < 1e-15 for t in seen)
    assert abs(row["heat_budget_residual_J"]) < 1e-11
    assert row["boundary_heat_J"] > 0


def test_aligned_solid_is_thermally_insulating_and_zero_outside_liquid():
    f = flow(8, periodic=False, solid_box_m=[[0.00375] * 3, [0.00625] * 3])
    initial = np.full(f.shape, 300.0)
    initial[3, 3, 2] = 310
    heat = HeatTransport(f, initial, HEAT)
    heat.advance(0.1)
    assert heat.temperature[3, 3, 5] == 300
    assert np.all(heat.temperature[~f.fluid] == 0)
    assert abs(heat.measurements()["heat_budget_residual_J"]) < 1e-11


@pytest.mark.parametrize("thermal,solutal,sign", [(1, 0, 1), (0, 1, -1), (4, 1, 0)])
def test_hot_fluid_rises_dense_solute_sinks_and_equal_terms_cancel(
    thermal, solutal, sign
):
    f = flow()
    model = FlowHeatSoluteSimulation(f, 303.15 + thermal, 200 + solutal, HEAT, BUOYANCY)
    force = model.buoyancy_acceleration()[2][..., 2][f.components[2].mask]
    expected = 9.81 * (2e-4 * thermal - 8e-4 * solutal)
    np.testing.assert_allclose(force, expected, atol=1e-16)
    row = model.step(0.05)
    np.testing.assert_allclose(
        f.velocity[2][f.components[2].mask], 0.05 * expected, atol=1e-14
    )
    assert np.sign(expected) == sign
    assert row["time_s"] == model.heat.time_s == model.solute.time_s == 0.05
    assert abs(row["solute_mass_drift_kg"]) < 1e-15
    assert abs(row["heat_budget_residual_J"]) < 1e-11


def test_buoyancy_interpolation_uses_distances_on_stretched_grid():
    edges = [[0, 0.001, 0.003, 0.006, 0.01]] * 3
    f = IncompressibleFlow([0.01] * 3, [4] * 3, FLUID, grid_edges_m=edges)
    xyz = f.cell_positions()
    # A linear Kelvin field must interpolate exactly at every open face.
    model = FlowHeatSoluteSimulation(f, 303.15 + 100 * xyz[..., 2], 200, HEAT, BUOYANCY)
    actual = model.buoyancy_acceleration()[2][..., 2]
    expected = 9.81 * 2e-4 * 100 * f.face_positions(2)[..., 2]
    np.testing.assert_allclose(
        actual[f.components[2].mask], expected[f.components[2].mask], atol=1e-16
    )


def test_failure_after_heat_update_rolls_back_flow_heat_solute_and_budgets():
    f = flow(periodic=False)
    model = FlowHeatSoluteSimulation(
        f, 300, 200, HEAT, BUOYANCY, thermal_boundaries={"x_min": wall(310)}
    )
    before = (f.velocity, f.pressure, model.heat.values, model.solute.values)

    def failure(*args, **kwargs):
        assert model.heat.time_s == 0.1  # An actual earlier scalar update occurred.
        assert model.heat.boundary_integral > 0
        raise ArithmeticError("injected second-scalar failure")

    with (
        patch.object(model.solute, "advance", side_effect=failure),
        pytest.raises(ArithmeticError),
    ):
        model.step(0.1)
    assert f.velocity is before[0] and f.pressure is before[1]
    assert model.heat.values is before[2] and model.solute.values is before[3]
    assert f.time_s == model.heat.time_s == model.solute.time_s == 0
    assert (
        model.heat.boundary_integral
        == model.heat.substeps
        == model.solute.substeps
        == 0
    )


def test_invalid_thermal_physics_clocks_and_density_budget_are_rejected():
    with pytest.raises(ValueError):
        replace(HEAT, conductivity_W_per_m_K=-1)
    with pytest.raises(ValueError):
        replace(BUOYANCY, maximum_density_perturbation_fraction=0.2)
    with pytest.raises(ValueError):
        HeatTransport(flow(), 300, HEAT, boundaries={"x_min": wall(310)})
    with pytest.raises(ValueError):
        HeatTransport(flow(), 0, HEAT)
    with pytest.raises(ValueError):
        FlowHeatSoluteSimulation(flow(), 1000, 200, HEAT, BUOYANCY)
    with pytest.raises(ValueError):
        wall(float("nan"))
    model = FlowHeatSoluteSimulation(flow(), 303.15, 200, HEAT, BUOYANCY)
    model.heat.time_s = 1
    with pytest.raises(ValueError, match="clocks"):
        model.step(0.1)


def test_wall_that_exceeds_buoyancy_budget_rolls_back_complete_step():
    model = FlowHeatSoluteSimulation(
        flow(periodic=False),
        303.15,
        200,
        HEAT,
        replace(BUOYANCY, maximum_density_perturbation_fraction=1e-7),
        thermal_boundaries={"x_min": wall(310)},
    )
    with pytest.raises(ValueError, match="Density perturbation"):
        model.step(1)
    assert model.flow.time_s == model.heat.time_s == model.solute.time_s == 0
    assert model.heat.boundary_integral == 0


def test_frozen_experiment_independent_audit_and_native_field_tamper_detection(
    tmp_path, capsys
):
    import hashlib
    import json

    run = tmp_path / "run"
    protocol = Path(__file__).resolve().parents[1] / "configs/thermal_verification.json"
    report = run_study(protocol, run)
    assert all(report["checks"].values())
    audit(run, tmp_path / "audit.json")
    # Rehash altered fields to get past integrity checks: the independent
    # numerical reconstruction must still reject an incorrect velocity.
    target = run / report["shear"][0]["artifact"]
    with np.load(target) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["w_m_per_s"] *= 1.01
    np.savez_compressed(target, **arrays)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit(run, tmp_path / "altered_audit.json")
    capsys.readouterr()
