# WHAT THESE TESTS CHECK
# Check real cell volumes, [z, y, x] indexing, vertical temperature gradients and
# reproducible initialization. Exercise diffusion along every axis with unequal
# spacings, sealed walls, uniform fields, zero rates and large time intervals.
# Compare grid refinement with a known cosine solution of the diffusion equation,
# rather than relying only on tests that duplicate the implementation's arithmetic.
# The coupled 3D growth engine is tested separately in test_simulation.py.

"""Check 3D tank fields and full-box diffusion; moving-fluid tests live separately."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation.physics import diffuse
from simulation.tank import Tank, SimulationConfig


def full_tank_config(**settings):
    # Field-only tests deliberately fill the entire box and do not insert a solid.
    # Open-surface and solid-displacement behavior is tested in test_fluid_domain.py.
    settings.setdefault("initial_liquid_height", settings.get("height", 8.0))
    return SimulationConfig(**settings)


class Tank3DTests(unittest.TestCase):
    def test_shape_coordinates_volume_and_mass(self):
        config = full_tank_config(
            width=3.0,
            depth=4.0,
            height=5.0,
            nx=6,
            ny=8,
            nz=10,
            concentration=2.0,
            concentration_noise=0.0,
        )
        tank = Tank(config)
        self.assertEqual(tank.concentration.shape, (10, 8, 6))
        self.assertEqual(tank.temperature.shape, (10, 8, 6))
        self.assertEqual(tank.cell_volume, 0.125)
        self.assertEqual(tank.volume, 60.0)
        self.assertEqual(tank.solute_mass, 120.0)
        np.testing.assert_allclose(tank.x, np.arange(6) * 0.5 + 0.25)
        np.testing.assert_allclose(tank.y, np.arange(8) * 0.5 + 0.25)
        np.testing.assert_allclose(tank.z, np.arange(10) * 0.5 + 0.25)

    def test_temperature_gradient_is_vertical_z(self):
        tank = Tank(full_tank_config(nx=5, ny=6, nz=7, temperature_gradient=12.0))
        np.testing.assert_allclose(
            tank.temperature[:, 0, 0], 25.0 + 12.0 * (tank.z / 8.0 - 0.5)
        )
        self.assertGreater(tank.temperature[-1, 0, 0], tank.temperature[0, 0, 0])
        np.testing.assert_array_equal(
            tank.temperature[:, 0, 0], tank.temperature[:, -1, -1]
        )
        self.assertAlmostEqual(float(tank.temperature.mean()), 25.0)

    def test_seeded_noise_and_nonnegative_initial_concentration(self):
        config = full_tank_config(concentration=0.01, concentration_noise=0.1, seed=6)
        a, b = Tank(config), Tank(config)
        np.testing.assert_array_equal(a.concentration, b.concentration)
        self.assertTrue((a.concentration >= 0).all())
        other = Tank(full_tank_config(seed=7))
        self.assertFalse(np.array_equal(a.concentration, other.concentration))

    def test_both_fields_diffuse_conservatively(self):
        tank = Tank(
            full_tank_config(
                nx=8,
                ny=9,
                nz=10,
                temperature_gradient=10.0,
                solute_diffusivity=1.0,
                thermal_diffusivity=2.0,
            )
        )
        mass = tank.solute_mass
        thermal_sum = float(tank.temperature.sum())
        old_temperature_std = float(tank.temperature.std())
        old_concentration_std = float(tank.concentration.std())
        tank.diffuse(20.0)
        self.assertAlmostEqual(tank.solute_mass, mass, places=10)
        self.assertAlmostEqual(float(tank.temperature.sum()), thermal_sum, places=8)
        self.assertLess(tank.temperature.std(), old_temperature_std)
        self.assertLess(tank.concentration.std(), old_concentration_std)
        self.assertTrue((tank.concentration >= 0).all())

    def test_invalid_configs_are_rejected(self):
        for settings in (
            {"width": 0},
            {"depth": -1},
            {"height": np.inf},
            {"nx": 2},
            {"ny": 4.5},
            {"nz": True},
            {"seed": -1},
            {"concentration": -1},
            {"concentration_noise": -1},
            {"thermal_diffusivity": np.nan},
            {"solute_diffusivity": -1},
            {"temperature_gradient": np.inf},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                full_tank_config(**settings)

    def test_failed_diffusion_does_not_partially_update_tank(self):
        tank = Tank()
        previous = tank.temperature.copy()
        tank.concentration[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            tank.diffuse(1.0)
        np.testing.assert_array_equal(tank.temperature, previous)


class Diffusion3DTests(unittest.TestCase):
    def test_impulse_diffuses_on_correct_axes_with_unequal_spacing(self):
        field = np.zeros((5, 5, 5))
        field[2, 2, 2] = 1.0
        result = diffuse(field, 0.1, 0.05, dx=0.5, dy=1.0, dz=2.0)
        # One short explicit step has a known flux through each adjacent face.
        self.assertAlmostEqual(result[2, 2, 3], 0.02)  # x
        self.assertAlmostEqual(result[2, 3, 2], 0.005)  # y
        self.assertAlmostEqual(result[3, 2, 2], 0.00125)  # z
        self.assertAlmostEqual(result[2, 2, 2], 0.9475)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertEqual(field[2, 2, 3], 0.0)  # Input remains unchanged.

    def test_walls_are_sealed_without_periodic_wrap(self):
        field = np.zeros((4, 4, 4))
        field[0, 0, 0] = 1.0
        result = diffuse(field, 0.1, 0.1, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(result[0, 0, 0], 0.97)
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertEqual(result[-1, 0, 0], 0.0)
        self.assertEqual(result[0, -1, 0], 0.0)
        self.assertEqual(result[0, 0, -1], 0.0)

    def test_uniform_field_and_zero_rate_or_time_are_unchanged(self):
        uniform = np.full((3, 4, 5), 25.0)
        np.testing.assert_array_equal(
            diffuse(uniform, 3.0, 10.0, 0.5, 0.7, 0.9), uniform
        )
        varied = np.arange(60.0).reshape(3, 4, 5)
        for coefficient, dt in [(0.0, 10.0), (1.0, 0.0)]:
            result = diffuse(varied, coefficient, dt, 0.5, 0.7, 0.9)
            np.testing.assert_array_equal(result, varied)
            self.assertFalse(np.shares_memory(result, varied))

    def test_large_interval_preserves_mass_and_value_range(self):
        field = np.random.default_rng(9).random((5, 7, 9))
        result = diffuse(field, 3.0, 10.0, 0.3, 0.5, 0.7)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), float(field.min()) - 1e-12)
        self.assertLessEqual(float(result.max()), float(field.max()) + 1e-12)
        self.assertAlmostEqual(float(result.sum()), float(field.sum()), places=10)

    def test_single_layer_has_no_flux_in_missing_direction(self):
        # A one-layer z field can still diffuse horizontally without leaking mass.
        field = np.arange(12.0).reshape(1, 3, 4)
        a = diffuse(field, 0.1, 0.1, 1.0, 1.0, 1.0)
        b = diffuse(field, 0.1, 0.1, 1.0, 1.0, 10.0)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(a.sum()), float(field.sum()))

    def test_diffusion_converges_toward_analytic_cosine_solution(self):
        # This Neumann-boundary mode has an exact continuous decay rate. Refining
        # the grid must reduce the combined spatial/time discretization error.
        errors = []
        for refinement in (1, 2):
            nx, ny, nz = 8 * refinement, 10 * refinement, 12 * refinement
            dx, dy, dz = 2 / nx, 3 / ny, 4 / nz
            x = (np.arange(nx) + 0.5) * dx
            y = (np.arange(ny) + 0.5) * dy
            z = (np.arange(nz) + 0.5) * dz
            mode = (
                np.cos(np.pi * z / 4)[:, None, None]
                * np.cos(np.pi * y / 3)[None, :, None]
                * np.cos(np.pi * x / 2)[None, None, :]
            )
            coefficient, dt = 0.08, 0.5
            exact = 1.0 + 0.1 * mode * np.exp(
                -coefficient * dt * np.pi**2 * (1 / 2**2 + 1 / 3**2 + 1 / 4**2)
            )
            result = diffuse(1.0 + 0.1 * mode, coefficient, dt, dx, dy, dz)
            errors.append(float(np.sqrt(np.mean((result - exact) ** 2))))
        self.assertLess(errors[1], errors[0] * 0.4)

    def test_invalid_diffusion_arguments(self):
        valid = np.ones((3, 4, 5))
        for coefficient, dt, dx, dy, dz in [
            (-1, 1, 1, 1, 1),
            (1, -1, 1, 1, 1),
            (1, np.inf, 1, 1, 1),
            (1, 1, 0, 1, 1),
            (1, 1, 1, np.nan, 1),
            (True, 1, 1, 1, 1),
        ]:
            with self.subTest(dt=dt, dx=dx, dy=dy), self.assertRaises(ValueError):
                diffuse(valid, coefficient, dt, dx, dy, dz)
        for field in (
            np.ones((3, 4)),
            np.ones((0, 3, 4)),
            np.full((3, 4, 5), np.nan),
            np.ones((3, 4, 5), complex),
        ):
            with self.subTest(shape=field.shape), self.assertRaises(ValueError):
                diffuse(field, 1.0, 1.0, 1.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
