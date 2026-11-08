# WHAT THESE TESTS CHECK
# Check conservative SI scalar transport using analytic cell-average solutions,
# maximum-principle/mass invariants and real coupled nonzero 3D momentum steps.
# Periodic seams must count once; blocked walls must carry no scalar mass.
# Invalid fluxes/failed coupling must leave physical state unchanged.

import json
import hashlib
import io
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.flow import IncompressibleFlow
from simulation.solute import FlowSoluteSimulation, SoluteTransport
from simulation.units import FluidPropertiesSI
from check_solute import pulse_case, temporal_case, run_study


def centers(flow):
    z, y, x = np.meshgrid(
        *[(np.arange(n) + 0.5) * h for n, h in zip(flow.shape, flow.spacing)],
        indexing="ij",
    )
    return np.stack((x, y, z), axis=-1)


class SoluteTests(unittest.TestCase):
    def setUp(self):
        reference = json.loads((ROOT / "configs/physical_reference.json").read_text())
        self.properties = FluidPropertiesSI(**reference["fluid"])

    def flow(self, counts=(8, 8, 8), *, periodic=True, solid=None):
        return IncompressibleFlow(
            [0.01] * 3,
            counts,
            self.properties,
            periodic_xyz=(periodic,) * 3,
            solid_box_m=solid,
        )

    def test_face_volume_flux_uses_actual_area_and_independent_arrays(self):
        flow = IncompressibleFlow(
            [0.01, 0.012, 0.008],
            [4, 6, 8],
            self.properties,
            periodic_xyz=(True,) * 3,
        )
        flow.set_velocity(
            [np.full_like(v, 1e-4 * (i + 1)) for i, v in enumerate(flow.velocity)]
        )
        flux = flow.volume_fluxes_m3_per_s()
        for value, expected in zip(
            flux, (1e-4 * 0.002 * 0.001, 2e-4 * 0.0025 * 0.001, 3e-4 * 0.0025 * 0.002)
        ):
            np.testing.assert_allclose(value, expected, rtol=1e-14)
        flux[0][:] = 0
        self.assertEqual(float(flow.velocity[0].min()), 1e-4)

    def test_reference_uniform_rest_has_exact_constant_concentration(self):
        flow = self.flow(periodic=False, solid=[[0.00375] * 3, [0.00625] * 3])
        model = FlowSoluteSimulation(flow, 269.576)
        result = model.run_until(0.13, 0.05)
        self.assertEqual(result["time_s"], 0.13)
        self.assertEqual(model.solute.time_s, flow.time_s)
        self.assertEqual(result["solute_mass_drift_kg"], 0)
        np.testing.assert_array_equal(model.solute.concentration[flow.fluid], 269.576)
        np.testing.assert_array_equal(model.solute.concentration[~flow.fluid], 0)

    def test_projected_nonuniform_3d_flow_preserves_constant_solute(self):
        flow = self.flow(periodic=False, solid=[[0.00375] * 3, [0.00625] * 3])
        rng = np.random.default_rng(31)
        flow.set_velocity(
            [rng.normal(0, 1e-4, v.shape) for v in flow.velocity], project=True
        )
        scalar = SoluteTransport(flow, 270)
        scalar.advance(1.3)
        np.testing.assert_allclose(
            scalar.concentration[flow.fluid], 270, atol=1e-10, rtol=0
        )
        self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)

    def test_discontinuous_3d_advection_is_bounded_and_conservative(self):
        flow = self.flow(counts=(12, 10, 8))
        flow.set_velocity(
            [
                np.full_like(v, speed)
                for v, speed in zip(flow.velocity, (0.001, -0.0007, 0.0003))
            ]
        )
        xyz = centers(flow)
        initial = np.where(np.all((xyz > 0.002) & (xyz < 0.006), axis=-1), 400.0, 0.0)
        scalar = SoluteTransport(flow, initial, diffusivity_m2_per_s=0)
        scalar.advance(7)
        self.assertGreater(scalar.substeps, 10)
        self.assertGreater(scalar.limited_face_stages, 0)
        self.assertGreaterEqual(float(scalar.concentration.min()), -1e-10)
        self.assertLessEqual(float(scalar.concentration.max()), 400 + 1e-10)
        self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
        self.assertGreater(float(np.linalg.norm(scalar.concentration - initial)), 1)

    def test_periodic_cell_average_solution_refines_on_all_three_axes(self):
        errors = []
        speed = np.array([0.0003, -0.0002, 0.0001])
        duration, k = 2.0, 2 * np.pi / 0.01
        for n in (8, 12, 20):
            flow = self.flow(counts=(n,) * 3)
            flow.set_velocity(
                [np.full_like(v, u) for v, u in zip(flow.velocity, speed)]
            )
            xyz = centers(flow)
            average = np.prod(np.sinc(k * flow.spacing / (2 * np.pi)))
            initial = 270 + 20 * average * np.cos(k * xyz.sum(axis=-1))
            scalar = SoluteTransport(flow, initial)
            scalar.advance(duration, max_substep_s=0.05)
            exact = 270 + 20 * average * np.exp(
                -3 * k * k * self.properties.solute_diffusivity_m2_per_s * duration
            ) * np.cos(k * (xyz - speed * duration).sum(axis=-1))
            errors.append(
                float(
                    np.linalg.norm(scalar.concentration - exact)
                    / np.linalg.norm(exact - 270)
                )
            )
            self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 0.05)

    def test_no_flux_diffusion_refines_toward_neumann_cosine_solution(self):
        errors = []
        k, duration = np.pi / 0.01, 1000.0
        for n in (6, 10, 16):
            flow = self.flow(counts=(n,) * 3, periodic=False)
            xyz = centers(flow)
            average = np.prod(np.sinc(k * flow.spacing / (2 * np.pi)))
            mode = np.prod(np.cos(k * xyz), axis=-1) * average
            scalar = SoluteTransport(flow, 270 + 20 * mode)
            scalar.advance(duration)
            exact = 270 + 20 * mode * np.exp(
                -3 * self.properties.solute_diffusivity_m2_per_s * k * k * duration
            )
            errors.append(
                float(
                    np.linalg.norm(scalar.concentration - exact)
                    / np.linalg.norm(exact - 270)
                )
            )
            self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[-1], 0.001)

    def test_diffusion_cannot_cross_solid_or_outer_walls(self):
        flow = self.flow(
            counts=(8,) * 3, periodic=False, solid=[[0.00375] * 3, [0.00625] * 3]
        )
        initial = np.zeros(flow.shape)
        initial[3, 3, 2] = 400
        scalar = SoluteTransport(flow, initial, diffusivity_m2_per_s=1e-7)
        scalar.advance(0.1)
        self.assertEqual(scalar.concentration[3, 3, 5], 0)
        self.assertEqual(scalar.concentration[0, 0, 0], 0)
        self.assertTrue(np.all(scalar.concentration[~flow.fluid] == 0))
        self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-14)
        self.assertGreater(float(scalar.concentration[3, 3, 1]), 0)

    def test_zero_solute_stays_exactly_zero(self):
        flow = self.flow()
        scalar = SoluteTransport(flow, 0)
        scalar.advance(3)
        self.assertEqual(scalar.mass_kg, 0)
        np.testing.assert_array_equal(scalar.concentration, 0)

    def test_high_peclet_pulse_resolves_physical_diffusion_under_refinement(self):
        protocol = json.loads((ROOT / "configs/solute_verification.json").read_text())
        rows = []
        for n in (64, 128, 256):
            scalar, _exact, row, _initial = pulse_case(
                self.properties, protocol["pulse"], n
            )
            rows.append(row)
            self.assertLess(abs(scalar.mass_kg / scalar.initial_mass_kg - 1), 1e-13)
        self.assertGreater(rows[0]["abs_excess_diffusivity_over_physical"], 1)
        self.assertLess(rows[-1]["abs_excess_diffusivity_over_physical"], 1)
        self.assertLess(rows[-1]["relative_rms_error"], 0.02)
        self.assertTrue(
            all(
                b["relative_rms_error"] < a["relative_rms_error"]
                for a, b in zip(rows, rows[1:])
            )
        )

    def test_diffusion_time_error_is_second_order_against_semidiscrete_reference(self):
        protocol = json.loads((ROOT / "configs/solute_verification.json").read_text())
        errors = []
        for dt in protocol["temporal_diffusion"]["timesteps_s"]:
            _scalar, _exact, row, _initial = temporal_case(
                self.properties, protocol["temporal_diffusion"], dt
            )
            errors.append(row["relative_rms_error"])
        for a, b in zip(errors, errors[1:]):
            self.assertGreater(a / b, 3.8)
            self.assertLess(a / b, 4.2)

    def test_new_forcing_subcycles_solute_using_updated_face_fluxes(self):
        flow = self.flow()
        xyz = centers(flow)
        model = FlowSoluteSimulation(
            flow, 270 + 20 * np.cos(2 * np.pi * xyz[..., 0] / 0.01)
        )
        result = model.step(1, (0.1, 0, 0))
        self.assertGreater(result["solute_substeps"], 20)
        self.assertAlmostEqual(result["max_speed_m_per_s"], 0.1)
        self.assertLess(
            abs(model.solute.mass_kg / model.solute.initial_mass_kg - 1), 1e-13
        )
        self.assertGreaterEqual(result["min_concentration_kg_per_m3"], 250)
        self.assertLessEqual(result["max_concentration_kg_per_m3"], 290)
        self.assertEqual(model.solute.time_s, flow.time_s)

    def test_invalid_concentration_time_and_diffusivity_are_rejected(self):
        flow = self.flow()
        for value in (
            -1,
            True,
            float("nan"),
            complex(1, 1),
            "270",
            [1, 2],
            self.properties.density_kg_per_m3,
        ):
            with self.subTest(concentration=value), self.assertRaises(ValueError):
                SoluteTransport(flow, value)
        for value in (-1, True, float("inf")):
            with self.subTest(diffusivity=value), self.assertRaises(ValueError):
                SoluteTransport(flow, 270, diffusivity_m2_per_s=value)
        scalar = SoluteTransport(flow, 270)
        for value in (0, -1, True, float("inf")):
            with self.subTest(dt=value), self.assertRaises(ValueError):
                scalar.advance(value)
        self.assertEqual(scalar.time_s, 0)

    def test_inconsistent_fluxes_fail_atomically(self):
        flow = self.flow(periodic=False)
        scalar = SoluteTransport(flow, 270)
        original = scalar.concentration.copy()
        invalid = []
        for index in ((0, 0, 0), (0, 0, 1)):
            flux = flow.volume_fluxes_m3_per_s()
            flux[0][index] = 1e-9
            invalid.append(flux)
        invalid += [
            [np.zeros(1)] * 3,
            [np.zeros(1)],
            [v.astype(complex) for v in flow.velocity],
        ]
        for flux in invalid:
            with self.assertRaises(ValueError):
                scalar.advance(1, volume_fluxes_m3_per_s=flux)
            np.testing.assert_array_equal(scalar.concentration, original)
            self.assertEqual(scalar.time_s, 0)
        periodic = self.flow()
        periodic.velocity[0][..., -1] = 1
        with self.assertRaisesRegex(ValueError, "Periodic"):
            SoluteTransport(periodic, 270).advance(1)
        flow.fluid[0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "fixed"):
            scalar.advance(1)

    def test_coupled_failure_rolls_back_momentum_scalar_and_clocks(self):
        flow = self.flow()
        model = FlowSoluteSimulation(flow, 270)
        before = (
            [v.copy() for v in flow.velocity],
            flow.pressure.copy(),
            model.solute.concentration.copy(),
        )
        with patch.object(
            model.solute, "advance", side_effect=ArithmeticError("failure")
        ):
            with self.assertRaises(ArithmeticError):
                model.step(0.03, (1e-4, 0, 0))
        self.assertEqual(flow.time_s, 0)
        self.assertEqual(model.solute.time_s, 0)
        for actual, old in zip(flow.velocity, before[0]):
            np.testing.assert_array_equal(actual, old)
        np.testing.assert_array_equal(flow.pressure, before[1])
        np.testing.assert_array_equal(model.solute.concentration, before[2])

    def test_nonzero_coupled_flow_preserves_mass_and_stops_at_exact_time(self):
        flow = self.flow(counts=(6,) * 3, periodic=False)
        xyz = centers(flow)
        model = FlowSoluteSimulation(
            flow, 270 + 10 * np.cos(np.pi * xyz[..., 0] / 0.01)
        )

        def acceleration(position, _time):
            result = np.zeros_like(position)
            result[..., 0] = 1e-4 * np.sin(np.pi * position[..., 1] / 0.01)
            return result

        result = model.run_until(0.13, 0.05, acceleration)
        self.assertEqual(result["time_s"], 0.13)
        self.assertEqual(model.solute.time_s, flow.time_s)
        self.assertGreater(result["max_speed_m_per_s"], 1e-7)
        self.assertLess(result["scaled_divergence"], 1e-9)
        self.assertEqual(result["blocked_face_speed_m_per_s"], 0)
        self.assertLess(
            abs(model.solute.mass_kg / model.solute.initial_mass_kg - 1), 1e-13
        )
        self.assertGreaterEqual(result["min_concentration_kg_per_m3"], 260)
        self.assertLessEqual(result["max_concentration_kg_per_m3"], 280)
        with self.assertRaises(ValueError):
            model.run_until(0.12, 0.05)
        flow.time_s += 0.1
        with self.assertRaisesRegex(ValueError, "clocks"):
            model.step(0.1)
        with self.assertRaisesRegex(ValueError, "clocks"):
            model.run_until(flow.time_s, 0.1)

    def test_verification_preserves_failed_gate_sources_fields_and_existing_output(
        self,
    ):
        protocol = json.loads((ROOT / "configs/solute_verification.json").read_text())
        reference = json.loads((ROOT / "configs/physical_reference.json").read_text())
        for family in ("periodic_3d", "diffusion_3d"):
            protocol[family].update(counts=[4, 5, 6], duration_s=0.2)
        protocol["pulse"].update(
            x_counts=[8, 12, 16], duration_s=0.2, finest_relative_rms_tolerance=1e-30
        )
        protocol["temporal_diffusion"].update(
            duration_s=0.2, timesteps_s=[0.02, 0.01, 0.005]
        )
        protocol["obstacle"]["duration_s"] = 0.15
        reference["duration_s"] = 0.2
        original = deepcopy(protocol)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            with redirect_stdout(io.StringIO()):
                report = run_study(protocol, reference, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["pulse"])
            self.assertEqual(protocol, original)
            self.assertEqual(
                json.loads((output / "report.json").read_text())["gates"],
                report["gates"],
            )
            manifest = json.loads((output / "study.json").read_text())
            self.assertEqual(manifest["protocol"], original)
            self.assertEqual(len(report["results"]), 15)
            for name, digest in manifest["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            for archive in output.glob("*.npz"):
                with np.load(archive) as fields:
                    for key in fields.files:
                        if fields[key].dtype.kind in "fiu":
                            self.assertTrue(np.isfinite(fields[key]).all())
            self.assertTrue((output / "pulse.png").is_file())
            with self.assertRaises(FileExistsError):
                run_study(protocol, reference, output)
        protocol["periodic_3d"]["counts"] = [4, 5]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with self.assertRaises(ValueError):
                run_study(protocol, reference, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
