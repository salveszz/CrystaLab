# WHAT THESE TESTS CHECK
# Run with: python -m unittest discover -s tests -v
# Check that growth consumes solute while conserving total mass, temperatures
# and concentrations affect growth, and seeded runs reproduce the same results.
# Also check diffusion conservation, stability with large requested timesteps,
# growth limits at walls or solute exhaustion, configuration validation, and
# independent face motion. These verify the implemented numerical rules;
# they do not validate the model against real KDP experiments.

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from simulation.engine import Simulation
from simulation.tank import SimulationConfig
from simulation.physics import diffuse


class SimulationTests(unittest.TestCase):
    def test_growth_and_mass_conservation(self):
        sim = Simulation()
        initial = sim.total_mass
        sim.run()
        self.assertGreater(sim.crystal.width, sim.config.seed_width)
        self.assertGreater(sim.crystal.height, sim.config.seed_height)
        self.assertLess(sim.tank.concentration.mean(), sim.config.concentration)
        self.assertAlmostEqual(sim.total_mass, initial, places=9)
        self.assertTrue(np.all(sim.tank.concentration >= 0))

    def test_temperature_and_concentration_response(self):
        areas = []
        for temperature, concentration in [(25, 1.35), (40, 1.35), (25, 0.8)]:
            sim = Simulation(SimulationConfig(temperature=temperature, concentration=concentration,
                                              concentration_noise=0, duration=20))
            sim.run()
            areas.append(sim.crystal.area)
        self.assertGreater(areas[0], areas[1])
        self.assertAlmostEqual(areas[2], 0.48)

    def test_repeatability_with_defects(self):
        config = SimulationConfig(defect_rate=10, impurity=0.1, duration=20)
        a, b = Simulation(config), Simulation(config)
        a.run()
        b.run()
        self.assertEqual(a.history, b.history)
        np.testing.assert_array_equal(a.tank.concentration, b.tank.concentration)

    def test_large_step_stability(self):
        sim = Simulation(SimulationConfig(solute_diffusivity=1, thermal_diffusivity=1,
                                           temperature_gradient=10, concentration_noise=0.5))
        mass = sim.total_mass
        sim.step(10)
        self.assertTrue(np.isfinite(sim.tank.temperature).all())
        self.assertTrue(np.isfinite(sim.tank.concentration).all())
        self.assertTrue((sim.tank.concentration >= 0).all())
        self.assertAlmostEqual(sim.total_mass, mass, places=9)

    def test_exhaustion_and_walls(self):
        sim = Simulation(SimulationConfig(growth_left=1e6, growth_right=1e6,
                                           growth_top=1e6, growth_bottom=1e6))
        mass = sim.total_mass
        sim.step()
        self.assertTrue((sim.tank.concentration >= 0).all())
        self.assertGreaterEqual(sim.crystal.left, 0)
        self.assertLessEqual(sim.crystal.right, sim.config.width)
        self.assertAlmostEqual(sim.total_mass, mass, places=9)

    def test_diffusion_conserves_field(self):
        field = np.random.default_rng(1).random((10, 10))
        result = diffuse(field, 0.1, 0.1, 1, 1)
        self.assertAlmostEqual(float(result.sum()), float(field.sum()), places=12)

    def test_configuration_validation(self):
        for params in ({'dt': 0}, {'nx': 2}, {'concentration': -1}, {'duration': float('nan')},
                       {'seed_width': 50}, {'nx': 4.5}, {'rotation': -1}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SimulationConfig(**params)

    def test_directional_growth(self):
        sim = Simulation(SimulationConfig(growth_left=0, growth_right=0.1, duration=10))
        left = sim.crystal.left
        sim.run()
        self.assertEqual(sim.crystal.left, left)
        self.assertGreater(sim.crystal.width, sim.config.seed_width)


if __name__ == '__main__':
    unittest.main()
