# WHAT THESE TESTS CHECK
# Exercise the actual 3D growth engine, not a separate geometry demonstration.
# Check conservation of dissolved + crystal mass, positive concentrations, effects
# of temperature/concentration, reproducibility and explicit final-time stopping.
# Verify arbitrary face motion, growth in depth, wall clipping for sloped solids,
# exhaustion, local solute withdrawal and compatibility with the volumetric grid.
# These are numerical/model checks, not validation of real KDP growth or quality.

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation.engine import Simulation
from simulation.tank import SimulationConfig


class SimulationTests(unittest.TestCase):
    def test_growth_and_mass_conservation(self):
        sim = Simulation(SimulationConfig(duration=20.0))
        initial_mass = sim.total_mass
        initial = sim.crystal.dimensions.copy()
        sim.run()
        self.assertTrue(np.all(sim.crystal.dimensions > initial))
        self.assertLess(
            sim.tank.liquid_mean(sim.tank.concentration), sim.config.concentration
        )
        self.assertAlmostEqual(sim.total_mass, initial_mass, places=9)
        self.assertTrue(np.all(sim.tank.concentration >= 0))
        self.assertEqual(sim.tank.concentration.shape, (16, 16, 20))
        self.assertIn("volume", sim.history[-1])
        self.assertNotIn("area", sim.history[-1])

    def test_temperature_and_concentration_response(self):
        volumes = []
        for temperature, concentration in [(25, 1.35), (40, 1.35), (25, 0.8)]:
            sim = Simulation(
                SimulationConfig(
                    temperature=temperature,
                    concentration=concentration,
                    concentration_noise=0,
                    duration=10.0,
                )
            )
            sim.run()
            volumes.append(sim.crystal.volume)
        self.assertGreater(volumes[0], volumes[1])
        self.assertAlmostEqual(volumes[2], 0.6 * 0.6 * 0.8)

    def test_repeatability_with_defects(self):
        config = SimulationConfig(defect_rate=10, impurity=0.1, duration=10.0)
        a, b = Simulation(config), Simulation(config)
        a.run()
        b.run()
        self.assertEqual(a.history, b.history)
        np.testing.assert_array_equal(a.tank.concentration, b.tank.concentration)

    def test_large_step_stability(self):
        sim = Simulation(
            SimulationConfig(
                solute_diffusivity=1,
                thermal_diffusivity=1,
                temperature_gradient=10,
                concentration_noise=0.5,
            )
        )
        mass = sim.total_mass
        sim.step(5.0)
        self.assertTrue(np.isfinite(sim.tank.temperature).all())
        self.assertTrue(np.isfinite(sim.tank.concentration).all())
        self.assertTrue((sim.tank.concentration >= 0).all())
        self.assertAlmostEqual(sim.total_mass, mass, places=9)

    def test_sloped_seed_exhaustion_and_all_six_walls(self):
        # An octahedral seed has no tank-aligned kinetic faces; every growing plane
        # is sloped. Huge coefficients must still respect the entire 3D tank.
        from itertools import product

        sim = Simulation(
            SimulationConfig(
                seed_normals=list(product([-1.0, 1.0], repeat=3)),
                seed_offsets=[0.7] * 8,
                growth_rate=1e6,
            )
        )
        mass = sim.total_mass
        sim.step()
        self.assertTrue((sim.tank.concentration >= 0).all())
        self.assertTrue((sim.crystal.vertices >= -1e-10).all())
        self.assertTrue(
            (sim.crystal.vertices <= [10.0 + 1e-10, 8.0 + 1e-10, 8.0 + 1e-10]).all()
        )
        self.assertAlmostEqual(sim.total_mass, mass, places=8)

    def test_actual_volume_used_instead_of_bounding_box_volume(self):
        from itertools import product

        sim = Simulation(
            SimulationConfig(
                seed_normals=list(product([-1.0, 1.0], repeat=3)),
                seed_offsets=[0.7] * 8,
            )
        )
        volume = sim.crystal.volume
        mass = sim.tank.solute_mass
        sim.step()
        expected = (sim.crystal.volume - volume) * sim.config.crystal_density
        self.assertAlmostEqual(mass - sim.tank.solute_mass, expected, places=10)
        self.assertLess(sim.crystal.volume, np.prod(sim.crystal.dimensions))

    def test_independent_face_growth_in_depth(self):
        # Box face order is +x,-x,+y,-y,+z,-z. Move only the +y depth face.
        sim = Simulation(
            SimulationConfig(face_growth=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0], duration=5.0)
        )
        initial_bounds = sim.crystal.bounds.copy()
        sim.run()
        np.testing.assert_allclose(sim.crystal.bounds[0], initial_bounds[0], atol=1e-12)
        self.assertGreater(sim.crystal.depth, sim.config.seed_depth)
        self.assertAlmostEqual(sim.crystal.width, sim.config.seed_width)
        self.assertAlmostEqual(sim.crystal.height, sim.config.seed_height)

    def test_local_uptake_does_not_change_distant_cells_without_transport(self):
        sim = Simulation(
            SimulationConfig(
                concentration_noise=0.0,
                solute_diffusivity=0.0,
                rotation=0.0,
                circulation=0.0,
            )
        )
        far = sim.tank.concentration[0, 0, 0]
        sim.step()
        self.assertEqual(sim.tank.concentration[0, 0, 0], far)
        self.assertLess(sim.tank.concentration.min(), far)

    def test_final_time_is_exact_and_further_steps_do_not_grow(self):
        sim = Simulation(SimulationConfig(duration=1.3, dt=0.5))
        sim.run()
        self.assertEqual(sim.time, 1.3)
        np.testing.assert_allclose(
            [row["time"] for row in sim.history], [0.0, 0.5, 1.0, 1.3]
        )
        volume, rows = sim.crystal.volume, len(sim.history)
        sim.step(100.0)
        self.assertEqual(sim.time, 1.3)
        self.assertEqual(sim.crystal.volume, volume)
        self.assertEqual(len(sim.history), rows)

    def test_mass_conservation_with_imported_fortran_order_field(self):
        # Users may import volumetric data with a different memory layout.
        # Local withdrawal must change the stored field even if ravel() copies it.
        sim = Simulation()
        sim.tank.concentration = np.asfortranarray(sim.tank.concentration)
        mass = sim.total_mass
        sim.step()
        self.assertAlmostEqual(sim.total_mass, mass, places=9)

    def test_configuration_validation(self):
        for params in (
            {"dt": 0},
            {"nx": 2},
            {"nz": 4.5},
            {"concentration": -1},
            {"duration": float("nan")},
            {"seed_depth": 50},
            {"rotation": -1},
            {"face_growth": [0.0, 0.1]},
            {"seed_normals": [[1.0, 0.0, 0.0]]},
            {"seed_rotation": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SimulationConfig(**params)

    def test_invalid_steps(self):
        sim = Simulation()
        for dt in (0.0, -1.0, np.nan, np.inf, True):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                sim.step(dt)


if __name__ == "__main__":
    unittest.main()
