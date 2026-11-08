# WHAT THESE TESTS CHECK
# Exercise the canonical SI flow/solute solver on genuinely unequal cells.
# Independent geometry integrals, weighted projection energy, viscous symmetry,
# pressure-force balance, analytic channel/wave refinement and periodic momentum
# conservation detect accidental reuse of mean spacings or unit cell volumes.
# Explicitly verify fixed aligned-solid exclusion and rejection of partial cells.

import json
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.sparse import diags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.flow import IncompressibleFlow
from simulation.mesh import RectilinearGrid, focused_axis_edges
from simulation.solute import FlowSoluteSimulation, SoluteTransport
from simulation.units import FluidPropertiesSI
from check_rectilinear_flow import focused_abc_case, run_study


def smooth_edges(length, n, amplitude=0.4):
    coordinate = np.linspace(0, 1, n + 1)
    result = length * (
        coordinate + amplitude * np.sin(2 * np.pi * coordinate) / (2 * np.pi)
    )
    result[0], result[-1] = 0, length
    return result


class RectilinearFlowTests(unittest.TestCase):
    def setUp(self):
        reference = json.loads((ROOT / "configs/physical_reference.json").read_text())
        self.properties = FluidPropertiesSI(**reference["fluid"])

    def flow(self, n=8, *, periodic=True):
        lengths = [0.01, 0.012, 0.008]
        edges = [smooth_edges(L, n, a) for L, a in zip(lengths, (0.4, -0.25, 0.3))]
        return IncompressibleFlow(
            lengths,
            [n] * 3,
            self.properties,
            periodic_xyz=(periodic,) * 3,
            grid_edges_m=edges,
        )

    def test_actual_primal_and_periodic_dual_volumes_integrate_domain(self):
        flow = self.flow()
        expected = np.prod(flow.lengths)
        self.assertAlmostEqual(float(flow.cell_volumes.sum()) / expected, 1, places=14)
        self.assertGreater(float(flow.cell_volumes.max() / flow.cell_volumes.min()), 4)
        for component in flow.components:
            self.assertAlmostEqual(
                float(component.volumes.sum()) / expected, 1, places=14
            )
        flow.set_velocity(
            [np.full_like(v, (i + 1) * 1e-4) for i, v in enumerate(flow.velocity)]
        )
        np.testing.assert_allclose(flow.divergence(), 0, atol=1e-16)
        for field, component in zip(flow.volume_fluxes_m3_per_s(), flow.components):
            self.assertGreater(float(field.max() / field.min()), 1.5)
            np.testing.assert_array_equal(
                np.take(field, 0, axis=component.axis),
                np.take(field, -1, axis=component.axis),
            )

    def test_invalid_edge_definitions_and_mutation_are_rejected(self):
        lengths, counts = [0.01] * 3, [4] * 3
        valid = [np.linspace(0, 0.01, 5)] * 3
        for edge in (
            [0, 0.001, 0.001, 0.009, 0.01],
            [0, 0.004, 0.006, 0.008, 0.011],
            [0, 0.004, 0.008, 0.01],
            [False] * 5,
            [0, 0.004, float("nan"), 0.009, 0.01],
        ):
            with self.assertRaises(ValueError):
                IncompressibleFlow(
                    lengths, counts, self.properties, grid_edges_m=[edge, *valid[1:]]
                )
        flow = self.flow()
        with self.assertRaises(ValueError):
            flow.grid.edges[0][1] = 0
        for count in (True, 2, 4.5):
            with self.assertRaises(ValueError):
                RectilinearGrid(lengths, [count, 4, 4], [False] * 3)

    def test_projection_is_volume_weighted_and_does_not_add_kinetic_energy(self):
        flow = self.flow(periodic=False)
        rng = np.random.default_rng(91)
        flow.set_velocity([rng.normal(0, 1e-4, v.shape) for v in flow.velocity])
        before = flow.measurements()["kinetic_energy_J"]
        flow.set_velocity(flow.velocity, project=True)
        self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)
        self.assertLess(flow.measurements()["kinetic_energy_J"], before)
        self.assertAlmostEqual(
            float(np.sum(flow.pressure * flow.cell_volumes)), 0, delta=1e-22
        )

    def test_periodic_seam_with_different_end_widths_projects_and_conserves(self):
        lengths = [0.01] * 3
        edges = [0.01 * np.linspace(0, 1, 9) ** power for power in (1.3, 1.1, 1.5)]
        for edge in edges:
            edge[-1] = 0.01
        flow = IncompressibleFlow(
            lengths,
            [8] * 3,
            self.properties,
            periodic_xyz=(True,) * 3,
            grid_edges_m=edges,
        )
        rng = np.random.default_rng(92)
        flow.set_velocity(
            [rng.normal(1e-4, 1e-5, v.shape) for v in flow.velocity], project=True
        )
        before = [
            float(np.sum(c.volumes * v[tuple(c.cells.T)]))
            for c, v in zip(flow.components, flow.velocity)
        ]
        flow.step(0.01)
        after = [
            float(np.sum(c.volumes * v[tuple(c.cells.T)]))
            for c, v in zip(flow.components, flow.velocity)
        ]
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-24)
        self.assertLess(flow.measurements()["scaled_divergence"], 1e-9)

    def test_viscosity_is_self_adjoint_in_dual_volume_inner_product(self):
        flow = self.flow(periodic=False)
        rng = np.random.default_rng(93)
        for component in flow.components:
            weighted = diags(component.volumes) @ component.laplacian
            self.assertLess(abs(weighted - weighted.T).max(), 1e-15)
            vector = rng.normal(size=len(component.cells))
            self.assertGreater(float(vector @ (weighted @ vector)), 0)
        flow.set_velocity(
            [rng.normal(0, 1e-4, v.shape) for v in flow.velocity], project=True
        )
        initial = flow.measurements()["kinetic_energy_J"]
        flow.step(0.02)
        self.assertLess(flow.measurements()["kinetic_energy_J"], initial)

    def test_linear_pressure_balances_constant_acceleration_on_actual_centers(self):
        flow = self.flow(periodic=False)
        force = np.array([1e-4, -2e-4, 3e-4])
        pressure = self.properties.density_kg_per_m3 * (flow.cell_positions() @ force)
        flow.set_pressure(pressure)
        expected = flow.pressure.copy()
        flow.run_until(0.13, 0.05, force)
        self.assertLess(flow.measurements()["max_speed_m_per_s"], 1e-16)
        np.testing.assert_allclose(flow.pressure, expected, rtol=0, atol=1e-15)

    def test_stretched_channel_refines_toward_quadratic_solution(self):
        errors = []
        for n in (8, 16, 32):
            lengths = [0.01, 0.01, 0.008]
            edges = [smooth_edges(L, k, 0.4) for L, k in zip(lengths, (4, 4, n))]
            flow = IncompressibleFlow(
                lengths,
                [4, 4, n],
                self.properties,
                periodic_xyz=(True, True, False),
                grid_edges_m=edges,
            )
            nu, H = self.properties.kinematic_viscosity_m2_per_s, lengths[2]
            force = 2 * nu * 0.001 / H**2
            flow.run_until(400, 1, (force, 0, 0))
            z = flow.face_positions(0)[..., 2]
            exact = force * (2 * H * z - z * z) / (2 * nu)
            volumes = flow.grid.momentum_volumes[2]
            errors.append(
                float(
                    np.sqrt(
                        np.sum(volumes * (flow.velocity[0] - exact) ** 2)
                        / np.sum(volumes * exact**2)
                    )
                )
            )
        self.assertLess(errors[1], errors[0] / 3)
        self.assertLess(errors[2], errors[1] / 3)
        self.assertLess(errors[-1], 0.002)

    def test_solute_preserves_true_volume_integral_and_changes_raw_sum(self):
        flow = self.flow()
        rng = np.random.default_rng(94)
        flow.set_velocity(
            [
                np.full_like(v, speed)
                for v, speed in zip(flow.velocity, (0.0003, -0.0002, 0.0001))
            ]
        )
        initial = rng.uniform(100, 300, flow.shape)
        scalar = SoluteTransport(flow, initial)
        scalar.advance(2)
        self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
        self.assertGreater(abs(float(scalar.concentration.sum() - initial.sum())), 1)
        self.assertGreaterEqual(
            float(scalar.concentration.min()), initial.min() - 1e-10
        )
        self.assertLessEqual(float(scalar.concentration.max()), initial.max() + 1e-10)

    def test_cell_average_three_dimensional_wave_refines_on_unequal_cells(self):
        errors = []
        speed = np.array([0.0003, -0.0002, 0.0001])
        duration = 2
        for n in (6, 10, 16):
            flow = self.flow(n)
            flow.set_velocity(
                [np.full_like(v, u) for v, u in zip(flow.velocity, speed)]
            )
            xyz, k = flow.cell_positions(), 2 * np.pi / flow.lengths
            wz, wy, wx = flow.grid.widths
            average = (
                np.sinc(k[2] * wz / (2 * np.pi))[:, None, None]
                * np.sinc(k[1] * wy / (2 * np.pi))[None, :, None]
                * np.sinc(k[0] * wx / (2 * np.pi))[None, None, :]
            )
            phase = xyz @ k
            initial = 270 + 20 * average * np.cos(phase)
            scalar = SoluteTransport(flow, initial)
            scalar.advance(duration, max_substep_s=0.05)
            exact = 270 + 20 * average * np.exp(
                -self.properties.solute_diffusivity_m2_per_s * np.sum(k * k) * duration
            ) * np.cos(phase - np.dot(k, speed) * duration)
            errors.append(
                float(
                    np.sqrt(
                        np.sum(flow.cell_volumes * (scalar.concentration - exact) ** 2)
                        / np.sum(flow.cell_volumes * (exact - 270) ** 2)
                    )
                )
            )
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[-1], 0.05)

    def test_focused_mesh_coupling_preserves_uniform_solute_and_exact_clocks(self):
        edges = [focused_axis_edges(0.01, 10, 0.005, 0.0036)] * 3
        flow = IncompressibleFlow(
            [0.01] * 3, [10] * 3, self.properties, grid_edges_m=edges
        )
        rng = np.random.default_rng(95)
        flow.set_velocity(
            [rng.normal(0, 1e-4, v.shape) for v in flow.velocity], project=True
        )
        model = FlowSoluteSimulation(flow, 270)
        model.run_until(0.13, 0.05)
        np.testing.assert_allclose(model.solute.concentration, 270, atol=1e-10, rtol=0)
        self.assertEqual(model.solute.time_s, flow.time_s)
        self.assertEqual(flow.time_s, 0.13)
        self.assertLess(
            abs(model.solute.mass_kg / model.solute.initial_mass_kg - 1), 1e-13
        )

    def test_aligned_solid_excludes_actual_unequal_cell_volumes(self):
        edge = [0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01]
        flow = IncompressibleFlow(
            [0.01] * 3,
            [6] * 3,
            self.properties,
            grid_edges_m=[edge] * 3,
            solid_box_m=[[0.003] * 3, [0.007] * 3],
        )
        self.assertAlmostEqual(
            float(flow.cell_volumes[~flow.fluid].sum()) / (0.004**3), 1, places=13
        )
        model = FlowSoluteSimulation(flow, 270)
        model.run_until(0.11, 0.05)
        self.assertAlmostEqual(
            model.solute.mass_kg / (270 * (0.01**3 - 0.004**3)), 1, places=13
        )
        with self.assertRaisesRegex(ValueError, "cut cells"):
            IncompressibleFlow(
                [0.01] * 3,
                [6] * 3,
                self.properties,
                grid_edges_m=[edge] * 3,
                solid_box_m=[[0.0032] * 3, [0.007] * 3],
            )

    def test_focused_nonlinear_flow_and_uniform_solute_refine_together(self):
        protocol = json.loads(
            (ROOT / "configs/rectilinear_verification.json").read_text()
        )
        settings = dict(protocol["focused_abc_solute"], duration_s=0.1)
        rows = []
        for n in (8, 12):
            flow, scalar, metrics, _exact = focused_abc_case(
                self.properties, settings, n
            )
            rows.append(metrics)
            self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
            self.assertLess(metrics["uniform_error_kg_per_m3"], 1e-10)
            self.assertEqual(flow.time_s, scalar.time_s)
        self.assertLess(
            rows[1]["relative_velocity_error"], rows[0]["relative_velocity_error"]
        )
        self.assertLess(
            rows[1]["relative_pressure_error"], rows[0]["relative_pressure_error"]
        )

    def test_study_retains_failure_source_identity_and_actual_grid_fields(self):
        protocol = json.loads(
            (ROOT / "configs/rectilinear_verification.json").read_text()
        )
        reference = json.loads((ROOT / "configs/physical_reference.json").read_text())
        protocol["channel"].update(vertical_counts=[4, 6, 8], duration_s=0.1)
        protocol["decay"]["duration_s"] = 0.1
        for family in ("abc", "focused_abc_solute", "wave", "diffusion"):
            protocol[family].update(counts=[4, 6, 8], duration_s=0.1)
        protocol["obstacle"]["duration_s"] = 0.15
        protocol["wave"]["finest_relative_rms_tolerance"] = 1e-30
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            with redirect_stdout(io.StringIO()):
                report = run_study(protocol, reference, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["wave_relative_rms_error"])
            self.assertEqual(len(report["results"]), 20)
            saved = json.loads((output / "study.json").read_text())
            self.assertIn("src/simulation/mesh.py", saved["sources"])
            for name, digest in saved["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            with np.load(output / "aligned_cube_nonuniform.npz") as fields:
                actual = (
                    np.diff(fields["z_edges_m"])[:, None, None]
                    * np.diff(fields["y_edges_m"])[None, :, None]
                    * np.diff(fields["x_edges_m"])[None, None, :]
                )
                np.testing.assert_allclose(
                    fields["cell_volumes_m3"], actual, rtol=1e-14
                )
                self.assertGreater(float(actual.max() / actual.min()), 4)
            self.assertTrue((output / "focused_cube.png").is_file())
            with self.assertRaises(FileExistsError):
                run_study(protocol, reference, output)
        protocol["abc"]["counts"] = [4, 6]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with self.assertRaises(ValueError):
                run_study(protocol, reference, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
