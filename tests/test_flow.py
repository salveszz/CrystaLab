# WHAT THESE TESTS CHECK
# Exercise the SI 3D momentum component with analytic solutions and independent
# invariants: exact volume exclusion, projection energy/divergence, viscous decay,
# pressure balance, wall impermeability and nonlinear Beltrami-flow refinement.
# A rest case alone cannot verify Navier-Stokes; nonzero flow and all three
# velocity components must be checked. Unsupported cut cells fail explicitly.

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.flow import IncompressibleFlow
from simulation.units import FluidPropertiesSI
from check_flow import (
    abc_case,
    channel_case,
    decay_case,
    obstacle_case,
    reference_flow,
    plot_report,
)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(
            (ROOT / "configs/physical_reference.json").read_text()
        )
        self.properties = FluidPropertiesSI(**self.reference["fluid"])
        self.protocol = json.loads(
            (ROOT / "configs/flow_verification.json").read_text()
        )

    def flow(self, **kwargs):
        return IncompressibleFlow(
            [0.01, 0.012, 0.008], [6, 6, 6], self.properties, **kwargs
        )

    def test_staggered_components_retain_all_three_dimensions(self):
        flow = self.flow()
        self.assertEqual(flow.pressure.shape, (6, 6, 6))
        self.assertEqual(
            [v.shape for v in flow.velocity], [(6, 6, 7), (6, 7, 6), (7, 6, 6)]
        )
        self.assertEqual(flow.centered_velocity().shape, (6, 6, 6, 3))
        self.assertEqual(flow.face_positions(2).shape, (7, 6, 6, 3))

    def test_reference_rest_preserves_zero_velocity_and_dynamic_pressure(self):
        flow = reference_flow(self.reference, [20, 20, 16])
        result = flow.run_until(0.13, 0.05)
        self.assertEqual(result["time_s"], 0.13)
        self.assertEqual(result["max_speed_m_per_s"], 0)
        self.assertEqual(result["max_divergence_per_s"], 0)
        np.testing.assert_array_equal(flow.pressure, 0)
        self.assertAlmostEqual(
            float(flow.fluid.sum()) * flow.cell_volume, 7.99e-7, delta=1e-20
        )

    def test_grid_aligned_crystal_has_exact_excluded_volume(self):
        flow = reference_flow(self.reference, [20, 20, 16])
        self.assertAlmostEqual(
            float((~flow.fluid).sum()) * flow.cell_volume, 1e-9, delta=1e-22
        )
        self.assertEqual(flow.solid_vertices_m.shape, (8, 3))

    def test_partial_solid_is_rejected_instead_of_voxelized(self):
        modified = deepcopy(self.reference)
        modified["crystal"]["center_m"][0] += 0.0001
        with self.assertRaisesRegex(ValueError, "cut cells"):
            reference_flow(modified, [20, 20, 16])

    def test_projection_removes_random_3d_divergence_and_does_not_add_energy(self):
        flow = self.flow()
        rng = np.random.default_rng(27)
        flow.set_velocity([rng.normal(0, 1e-4, v.shape) for v in flow.velocity])
        original = flow.measurements()
        self.assertGreater(original["max_divergence_per_s"], 0.01)
        flow.set_velocity(flow.velocity, project=True)
        projected = flow.measurements()
        self.assertLess(projected["scaled_divergence"], 1e-9)
        self.assertLessEqual(
            projected["kinetic_energy_J"], original["kinetic_energy_J"]
        )
        self.assertEqual(projected["blocked_face_speed_m_per_s"], 0)
        self.assertAlmostEqual(projected["mean_dynamic_pressure_Pa"], 0, delta=1e-15)

    def test_periodic_endpoint_is_one_unknown_and_remains_consistent(self):
        flow = self.flow(periodic_xyz=(True, True, True))
        rng = np.random.default_rng(13)
        flow.set_velocity(
            [rng.normal(0, 1e-4, v.shape) for v in flow.velocity], project=True
        )
        for value, component in zip(flow.velocity, flow.components):
            np.testing.assert_array_equal(
                np.take(value, 0, axis=component.axis),
                np.take(value, -1, axis=component.axis),
            )
        self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)
        flow.step(0.01)
        self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)

    def test_viscous_operators_are_symmetric_and_dissipative(self):
        flow = self.flow()
        rng = np.random.default_rng(8)
        for component in flow.components:
            matrix = component.laplacian
            self.assertLess(abs(matrix - matrix.T).max(), 1e-9)
            x = rng.normal(size=matrix.shape[0])
            self.assertGreater(float(x @ (matrix @ x)), 0)

    def test_periodic_momentum_transport_preserves_component_means(self):
        flow = self.flow(periodic_xyz=(True, True, True))
        rng = np.random.default_rng(17)
        flow.set_velocity(
            [
                rng.normal((i + 1) * 1e-4, 2e-5, v.shape)
                for i, v in enumerate(flow.velocity)
            ],
            project=True,
        )
        before = [
            float(v[tuple(c.cells.T)].mean())
            for v, c in zip(flow.velocity, flow.components)
        ]
        flow.step(0.01)
        after = [
            float(v[tuple(c.cells.T)].mean())
            for v, c in zip(flow.velocity, flow.components)
        ]
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-17)

    def test_unforced_flow_loses_kinetic_energy(self):
        flow = self.flow()
        rng = np.random.default_rng(5)
        flow.set_velocity(
            [rng.normal(0, 1e-4, v.shape) for v in flow.velocity], project=True
        )
        energy = flow.measurements()["kinetic_energy_J"]
        result = flow.step(0.03)
        self.assertLess(result["kinetic_energy_J"], energy)
        self.assertLess(result["scaled_divergence"], 1e-9)

    def test_initialized_pressure_balances_conservative_acceleration(self):
        flow = self.flow()
        z, y, x = np.meshgrid(
            *[(np.arange(n) + 0.5) * h for n, h in zip(flow.shape, flow.spacing)],
            indexing="ij",
        )
        force = np.array([1e-4, -2e-4, 3e-4])
        pressure = flow.properties.density_kg_per_m3 * (
            force[0] * x + force[1] * y + force[2] * z
        )
        flow.set_pressure(pressure)
        for _ in range(3):
            flow.step(0.02, force)
        self.assertLess(flow.measurements()["max_speed_m_per_s"], 1e-16)
        np.testing.assert_allclose(
            flow.pressure, pressure - pressure.mean(), atol=1e-15
        )

    def test_channel_profile_refines_toward_quadratic_reference(self):
        errors = []
        for n in (4, 8, 12):
            flow, row = channel_case(self.properties, self.protocol["channel"], n)
            errors.append(row["relative_rms_error"])
            self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)
        self.assertLess(errors[1], errors[0] / 3)
        self.assertLess(errors[2], errors[1])

    def test_viscous_eigenmode_time_error_refines(self):
        errors = []
        for dt in (0.5, 0.25, 0.125):
            _flow, row = decay_case(self.properties, self.protocol["decay"], dt)
            errors.append(row["relative_rms_error"])
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[-1], 0.003)

    def test_nonlinear_three_component_abc_solution_refines(self):
        settings = dict(self.protocol["abc"], duration_s=0.1)
        rows = []
        for n in (6, 10):
            flow, row = abc_case(self.properties, settings, n)
            rows.append(row)
            for velocity in flow.velocity:
                self.assertGreater(float(abs(velocity).max()), 1e-5)
            self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)
        self.assertLess(
            rows[-1]["relative_velocity_error"], rows[0]["relative_velocity_error"]
        )
        self.assertLess(
            rows[-1]["relative_pressure_error"], rows[0]["relative_pressure_error"]
        )

    def test_forced_obstacle_flow_is_nonzero_and_impermeable(self):
        settings = dict(self.protocol["obstacle"], duration_s=0.2)
        flow, _row = obstacle_case(self.reference, settings)
        metrics = flow.measurements()
        self.assertGreater(metrics["max_speed_m_per_s"], 1e-6)
        self.assertLess(metrics["scaled_divergence"], 1e-9)
        self.assertEqual(metrics["blocked_face_speed_m_per_s"], 0)
        self.assertTrue(np.isfinite(flow.pressure).all())

    def test_cfl_and_invalid_inputs_fail_before_advancing(self):
        for counts in ([2, 4, 4], [True, 4, 4], [4.5, 4, 4]):
            with self.assertRaises(ValueError):
                IncompressibleFlow([0.01] * 3, counts, self.properties)
        flow = self.flow()
        with self.assertRaises(ValueError):
            flow.set_velocity([flow.velocity[0]])
        with self.assertRaises(ValueError):
            flow.step(-1)
        with self.assertRaises(ValueError):
            flow.step(0.01, (float("nan"), 0, 0))
        self.assertEqual(flow.time_s, 0)
        flow.set_velocity([np.ones_like(v) for v in flow.velocity])
        with self.assertRaisesRegex(ValueError, "CFL"):
            flow.step(1)
        self.assertEqual(flow.time_s, 0)

    def test_report_plot_does_not_modify_physical_domain_or_velocity(self):
        flow = reference_flow(self.reference, [20, 20, 16])
        lengths = flow.lengths.copy()
        velocity = [v.copy() for v in flow.velocity]
        report = dict(
            results=[
                dict(case="channel_8", nz=8, relative_rms_error=0.01),
                dict(case="decay_0.5", dt_s=0.5, relative_rms_error=0.01),
                dict(
                    case="abc_12",
                    n=12,
                    relative_velocity_error=0.01,
                    relative_pressure_error=0.02,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            plot_report(report, flow, Path(directory))
            self.assertTrue((Path(directory) / "fixed_crystal_flow.png").is_file())
        np.testing.assert_array_equal(flow.lengths, lengths)
        for actual, original in zip(flow.velocity, velocity):
            np.testing.assert_array_equal(actual, original)


if __name__ == "__main__":
    unittest.main()
