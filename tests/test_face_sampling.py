# WHAT THESE TESTS CHECK
# A fixed physical face neighborhood must not gain/lose whole voxels when the
# grid changes. Compare exact overlap volume with area * prism thickness, verify
# constant-field averages and translate/rotate the same seed relative to the mesh.
# Uptake must remove exactly the deposited mass while using fractional source
# volumes and leaving unrelated liquid unchanged before displacement remapping.

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.crystal import Crystal
from simulation.engine import Simulation
from simulation.tank import SimulationConfig


class FaceSamplingTests(unittest.TestCase):
    def simulation(self, grid=(20, 16, 16), **overrides):
        base = json.loads((ROOT / "configs/default.json").read_text())
        spacing = max(
            base["width"] / grid[0], base["depth"] / grid[1], base["height"] / grid[2]
        )
        settings = (
            base
            | dict(
                nx=grid[0],
                ny=grid[1],
                nz=grid[2],
                concentration_noise=0.0,
                temperature_gradient=0.0,
                boundary_layer_cells=0.75 / spacing,
            )
            | overrides
        )
        return Simulation(SimulationConfig(**settings))

    def test_prism_is_bounded_by_face_and_has_analytic_volume(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        prism = cube.sampling_prism(0, 0.75, 10.0)
        self.assertAlmostEqual(prism.volume, 4.0 * 0.75, places=12)
        np.testing.assert_allclose(prism.bounds, [[1.0, -1.0, -1.0], [1.75, 1.0, 1.0]])
        self.assertFalse(cube.contains(prism.interior_point))
        with self.assertRaises(ValueError):
            cube.sampling_prism(0, 0.0, 10.0)

    def test_physical_sampling_volume_and_constant_fields_across_grids(self):
        for grid in ((20, 16, 16), (30, 24, 24), (40, 32, 32)):
            with self.subTest(grid=grid):
                sim = self.simulation(grid)
                values, volumes = sim._face_conditions()
                expected = sim.crystal.surface_area * 0.75
                self.assertAlmostEqual(float(volumes.sum()), expected, places=10)
                np.testing.assert_allclose(values[:12], 0.35, atol=1e-12)
                self.assertTrue((volumes >= 0.0).all())
                self.assertTrue(
                    (volumes <= sim.tank.liquid_volumes.ravel() + 1e-12).all()
                )

    def test_grid_alignment_and_seed_rotation_do_not_change_sampling_volume(self):
        angle = 0.37
        rotation = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        for center in ([5.0, 4.0, 4.0], [5.07, 4.03, 4.11]):
            with self.subTest(center=center):
                sim = self.simulation(seed_center=center, seed_rotation=rotation)
                values, volumes = sim._face_conditions()
                self.assertAlmostEqual(
                    float(volumes.sum()), sim.crystal.surface_area * 0.75, places=10
                )
                np.testing.assert_allclose(values[:12], 0.35, atol=1e-12)

    def test_partial_wetting_clips_the_facet_before_extrusion(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        prism = cube.sampling_prism(0, 0.75, 0.25)
        self.assertAlmostEqual(cube.wetted_face_area(0, 0.25), 2.5, places=12)
        self.assertAlmostEqual(prism.volume, 2.5 * 0.75, places=12)
        self.assertAlmostEqual(prism.bounds[1, 2], 0.25, places=12)
        with self.assertRaises(ValueError):
            cube.sampling_prism(4, 0.75, 0.25)

    def test_sampling_volume_stops_at_tank_walls_and_free_surface(self):
        # Each prism extends 0.75, but only 0.2 remains before its physical bound.
        for center, face, area in (
            ([3.5, 2, 2], 0, 0.6 * 0.8),
            ([2, 2, 2.4], 4, 0.6 * 0.6),
        ):
            with self.subTest(face=face):
                sim = Simulation(
                    SimulationConfig(
                        width=4,
                        depth=4,
                        height=4,
                        nx=8,
                        ny=8,
                        nz=8,
                        initial_liquid_height=3.0,
                        seed_center=center,
                        concentration_noise=0.0,
                    )
                )
                prism = sim.crystal.sampling_prism(face, 0.75, 3.0)
                volumes = sim.tank.sampling_volumes(prism)
                self.assertAlmostEqual(float(volumes.sum()), area * 0.2, places=12)
                self.assertTrue(
                    (volumes <= sim.tank.liquid_volumes.ravel() + 1e-12).all()
                )

    def test_nearly_coincident_growth_corners_do_not_make_invalid_prisms(self):
        # Symmetric faces may meet at Qhull corners split only by floating error.
        # Their extrusion must still have a strict interior and the proper volume.
        sim = self.simulation(grid=(10, 10, 10), duration=0.5, dt=0.25)
        sim.step()
        values, volumes = sim._face_conditions()
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(
            float(volumes.sum()), sim.crystal.surface_area * 0.75, places=8
        )
        sim.run()

    def test_temperature_gradient_and_nearly_symmetric_apex_keep_valid_prisms(self):
        # Slightly unequal cap growth creates projected interior corners near a
        # shared apex. Convex facet reconstruction must discard those corners.
        sim = self.simulation(temperature_gradient=2.0, duration=3.0, dt=1.0)
        sim.run()
        values, volumes = sim._face_conditions()
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(
            float(volumes.sum()), sim.crystal.surface_area * 0.75, places=7
        )

    def test_fractional_uptake_conserves_mass_before_and_after_displacement(self):
        sim = self.simulation()
        _values, volumes = sim._face_conditions()
        old_c = sim.tank.concentration.copy()
        old_capacity = sim.tank.liquid_volumes.copy()
        old_volume, mass = sim.crystal.volume, sim.total_mass
        captured = []
        move = sim.tank.move_boundary

        def capture(crystal, dt, concentration):
            captured.append(concentration.copy())
            return move(crystal, dt, concentration)

        with patch.object(sim.tank, "move_boundary", side_effect=capture):
            sim._grow(0.2)
        used = (sim.crystal.volume - old_volume) * sim.config.crystal_density
        self.assertGreater(used, 0.0)
        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(
            float(np.sum((old_c - captured[0]) * old_capacity)), used, places=11
        )
        np.testing.assert_array_equal(
            captured[0].ravel()[volumes == 0.0], old_c.ravel()[volumes == 0.0]
        )
        self.assertTrue((captured[0] >= 0.0).all())
        self.assertAlmostEqual(sim.total_mass, mass, places=10)


if __name__ == "__main__":
    unittest.main()
