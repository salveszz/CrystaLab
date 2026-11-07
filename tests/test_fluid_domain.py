# WHAT THESE TESTS CHECK
# Exact solid intersections resolve partial cells and blocked face areas. An open
# tank conserves liquid carrier volume while its free surface rises with growth.
# Solute/heat follow displaced liquid, diffusion cannot pass through solid/air,
# and the displacement flux satisfies discrete volume continuity. These tests
# validate the quasistatic numerical model, not Navier-Stokes or real KDP physics.

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation.crystal import Crystal
from simulation.tank import SimulationConfig, Tank
from simulation.engine import Simulation
from simulation.physics import diffuse_liquid, displace_liquid


class SolidIntersectionTests(unittest.TestCase):
    def test_exact_box_intersection_volume(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        self.assertAlmostEqual(cube.clipped_volume([0, 0, 0], [2, 2, 2]), 1.0)
        self.assertAlmostEqual(cube.clipped_volume([-2, -2, -2], [2, 2, 2]), 8.0)
        self.assertEqual(cube.clipped_volume([1, 0, 0], [2, 1, 1]), 0.0)
        self.assertEqual(cube.clipped_volume([3, 3, 3], [4, 4, 4]), 0.0)

    def test_rotated_solid_cut_cells_sum_to_exact_volume(self):
        a = 0.43
        rotation = np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        solid = Crystal.box([0, 0, 0], [1.0, 1.2, 1.4]).rotated(rotation)
        total = 0.0
        for z in np.arange(-1.0, 1.0, 0.5):
            for y in np.arange(-1.0, 1.0, 0.5):
                for x in np.arange(-1.0, 1.0, 0.5):
                    lo = np.array([x, y, z])
                    total += solid.clipped_volume(lo, lo + 0.5)
        self.assertAlmostEqual(total, solid.volume, places=10)

    def test_sloped_face_section_and_wetted_area(self):
        from itertools import product

        octahedron = Crystal(list(product([-1.0, 1.0], repeat=3)), [1.0] * 8)
        rectangle = [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]]
        self.assertAlmostEqual(octahedron.section_area(rectangle), 2.0)
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        self.assertAlmostEqual(cube.wetted_face_area(0, 0.0), 2.0)
        self.assertEqual(cube.wetted_face_area(4, 0.0), 0.0)
        self.assertEqual(cube.wetted_face_area(5, 0.0), 4.0)


class OpenTankTests(unittest.TestCase):
    def test_emerging_crystal_displaces_only_its_submerged_volume(self):
        config = SimulationConfig(
            width=4.0,
            depth=4.0,
            height=4.0,
            nx=8,
            ny=8,
            nz=8,
            initial_liquid_height=2.5,
            concentration_noise=0.0,
        )
        old = config.make_crystal().bounded_by_box([4.0, 4.0, 4.0])
        tank = Tank(config, crystal=old)
        # Only the upper face advances, through the free surface into the air.
        new = old.advanced(np.r_[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.zeros(6)])
        expected = (tank.carrier_volume - 0.6 * 0.6 * old.bounds[0, 2]) / (
            16.0 - 0.6 * 0.6
        )
        tank.move_boundary(new, 1.0, tank.concentration.copy())
        self.assertAlmostEqual(tank.liquid_level, expected, places=10)
        self.assertLess(tank.liquid_level, new.bounds[1, 2])
        self.assertAlmostEqual(tank.liquid_volume, tank.carrier_volume, places=9)

    def test_seed_displaces_liquid_at_initialization(self):
        sim = Simulation(
            SimulationConfig(seed_width=2.0, seed_depth=2.0, seed_height=2.0)
        )
        tank = sim.tank
        solid_volume = float(tank.geometry.solid_fraction.sum() * tank.cell_volume)
        self.assertAlmostEqual(solid_volume, sim.crystal.volume, places=10)
        self.assertAlmostEqual(
            tank.liquid_volume, 10.0 * 8.0 * 6.0 - sim.crystal.volume, places=10
        )
        solid = tank.geometry.solid_fraction > 1.0 - 1e-10
        self.assertTrue(solid.any())
        self.assertTrue((tank.concentration[solid] == 0.0).all())
        self.assertTrue((tank.temperature[solid] == 0.0).all())
        self.assertTrue((tank.liquid_volumes[solid] == 0.0).all())

    def test_subcell_seed_is_not_lost_to_voxel_center_sampling(self):
        config = SimulationConfig(seed_width=0.1, seed_depth=0.12, seed_height=0.14)
        sim = Simulation(config)
        occupied = sim.tank.geometry.solid_fraction
        self.assertGreater(occupied.max(), 0.0)
        self.assertLess(occupied.max(), 1.0)
        self.assertAlmostEqual(
            float(occupied.sum() * sim.tank.cell_volume), 0.1 * 0.12 * 0.14, places=12
        )

    def test_moving_boundary_conserves_carrier_solute_and_heat(self):
        config = SimulationConfig(temperature_gradient=8.0, concentration_noise=0.0)
        old = config.make_crystal().bounded_by_box([10.0, 8.0, 8.0])
        tank = Tank(config, crystal=old)
        volume, mass = tank.liquid_volume, tank.solute_mass
        heat = float(np.sum(tank.temperature * tank.liquid_volumes))
        level = tank.liquid_level
        new = old.advanced(np.r_[np.full(6, 0.12), np.zeros(6)])
        tank.move_boundary(new, 1.0, tank.concentration.copy())
        self.assertAlmostEqual(tank.liquid_volume, volume, places=10)
        self.assertAlmostEqual(tank.solute_mass, mass, places=9)
        self.assertAlmostEqual(
            float(np.sum(tank.temperature * tank.liquid_volumes)), heat, places=8
        )
        self.assertAlmostEqual(
            (tank.liquid_level - level) * 80.0, new.volume - old.volume, places=10
        )
        np.testing.assert_allclose(
            tank.concentration[tank.liquid_mask], config.concentration, atol=1e-9
        )
        self.assertGreater(
            float(np.linalg.norm(tank.displacement_velocity, axis=-1).max()), 0.0
        )
        self.assertLess(tank.continuity_error, 1e-10)

    def test_coupled_growth_preserves_mass_and_carrier_volume(self):
        sim = Simulation(SimulationConfig(duration=3.0))
        mass, volume, level = (
            sim.total_mass,
            sim.tank.liquid_volume,
            sim.tank.liquid_level,
        )
        sim.run()
        self.assertAlmostEqual(sim.total_mass, mass, places=8)
        self.assertAlmostEqual(sim.tank.liquid_volume, volume, places=9)
        self.assertGreater(sim.tank.liquid_level, level)
        self.assertTrue((sim.tank.concentration >= -1e-12).all())
        self.assertTrue((sim.tank.concentration[~sim.tank.liquid_mask] == 0.0).all())

    def test_rim_rejects_untracked_overflow(self):
        config = SimulationConfig(initial_liquid_height=8.0)
        old = config.make_crystal()
        tank = Tank(config, crystal=old)
        with self.assertRaisesRegex(ValueError, "headspace"):
            tank.level_for(old.advanced(np.ones(6) * 0.1))

    def test_growth_pauses_at_a_full_rim_without_consuming_solute(self):
        sim = Simulation(SimulationConfig(initial_liquid_height=8.0, duration=1.0))
        volume, mass = sim.crystal.volume, sim.total_mass
        sim.run()
        self.assertEqual(sim.crystal.volume, volume)
        self.assertAlmostEqual(sim.total_mass, mass, places=9)


class LiquidTransportTests(unittest.TestCase):
    def test_iterative_transport_solve_matches_direct_and_has_safe_fallback(self):
        from unittest.mock import patch
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
        from simulation.physics import _solve_spd

        n = 300
        matrix = diags(
            [-np.ones(n - 1), np.full(n, 3.0), -np.ones(n - 1)], [-1, 0, 1]
        ).tocsr()
        rhs = np.sin(np.arange(n) * 0.1) + 1.0
        reference = spsolve(matrix.tocsc(), rhs)
        np.testing.assert_allclose(
            _solve_spd(matrix, rhs), reference, rtol=1e-12, atol=1e-12
        )
        with patch("scipy.sparse.linalg.cg", return_value=(np.zeros(n), 1)):
            np.testing.assert_array_equal(_solve_spd(matrix, rhs), reference)

    def test_invalid_capacities_do_not_enter_the_displacement_solve(self):
        old = np.array([[[1.0, 1.0]]])
        openings = (np.zeros((0, 1, 2)), np.zeros((1, 0, 2)), np.ones((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            displace_liquid(
                old,
                np.array([[[-1.0, 3.0]]]),
                openings,
                openings,
                old,
                old,
                1.0,
                1.0,
                1.0,
            )

    def test_no_diffusion_across_solid_barrier(self):
        volumes = np.ones((3, 3, 5))
        volumes[:, :, 2] = 0.0
        openings = (np.ones((2, 3, 5)), np.ones((3, 2, 5)), np.ones((3, 3, 4)))
        values = np.zeros_like(volumes)
        values[:, :, :2] = 1.0
        result = diffuse_liquid(values, volumes, openings, 1.0, 100.0, 1.0, 1.0, 1.0)
        np.testing.assert_allclose(result[:, :, :2], 1.0, atol=1e-12)
        self.assertTrue((result[:, :, 2:] == 0.0).all())
        self.assertAlmostEqual(float(np.sum(result * volumes)), 18.0)

    def test_implicit_small_cut_cell_is_positive_and_conservative(self):
        volumes = np.array([[[1e-8, 1.0]]])
        openings = (np.zeros((0, 1, 2)), np.zeros((1, 0, 2)), np.ones((1, 1, 1)))
        initial = np.array([[[1.0, 0.0]]])
        result = diffuse_liquid(initial, volumes, openings, 1.0, 100.0, 1.0, 1.0, 1.0)
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= 1.0).all())
        self.assertAlmostEqual(float(np.sum(result * volumes)), 1e-8, places=15)

    def test_displacement_upwind_transport_with_new_liquid_cell(self):
        old = np.array([[[1.0, 1.0, 0.0]]])
        new = np.array([[[0.0, 1.0, 1.0]]])
        openings = (np.zeros((0, 1, 3)), np.zeros((1, 0, 3)), np.ones((1, 1, 2)))
        c, t, flux, error = displace_liquid(
            old,
            new,
            openings,
            openings,
            np.array([[[2.0, 4.0, 0.0]]]),
            np.array([[[10.0, 20.0, 0.0]]]),
            1.0,
            1.0,
            1.0,
        )
        np.testing.assert_allclose(c, [[[0.0, 3.0, 3.0]]])
        np.testing.assert_allclose(t, [[[0.0, 15.0, 15.0]]])
        np.testing.assert_allclose(flux[2], [[[1.0, 1.0]]])
        self.assertLess(error, 1e-12)

    def test_trapped_fluid_is_not_pumped_through_solid(self):
        old = np.array([[[1.0, 0.0, 1.0]]])
        new = np.array([[[0.5, 0.0, 1.5]]])
        openings = (np.zeros((0, 1, 3)), np.zeros((1, 0, 3)), np.zeros((1, 1, 2)))
        with self.assertRaisesRegex(ValueError, "trapped"):
            displace_liquid(
                old,
                new,
                openings,
                openings,
                np.ones_like(old),
                np.ones_like(old),
                1.0,
                1.0,
                1.0,
            )

    def test_liquid_diffusion_converges_with_grid_and_timestep_refinement(self):
        errors = []
        for refinement in (1, 2):
            nx, ny, nz = 6 * refinement, 8 * refinement, 10 * refinement
            dx, dy, dz = 2 / nx, 3 / ny, 4 / nz
            x = (np.arange(nx) + 0.5) * dx
            y = (np.arange(ny) + 0.5) * dy
            z = (np.arange(nz) + 0.5) * dz
            mode = (
                np.cos(np.pi * z / 4)[:, None, None]
                * np.cos(np.pi * y / 3)[None, :, None]
                * np.cos(np.pi * x / 2)[None, None, :]
            )
            volumes = np.full((nz, ny, nx), dx * dy * dz)
            openings = (
                np.full((nz - 1, ny, nx), dx * dy),
                np.full((nz, ny - 1, nx), dx * dz),
                np.full((nz, ny, nx - 1), dy * dz),
            )
            values = 1.0 + 0.1 * mode
            steps = 5 * refinement**2
            for _ in range(steps):
                values = diffuse_liquid(
                    values, volumes, openings, 0.1, 0.1 / steps, dx, dy, dz
                )
            exact = 1.0 + 0.1 * mode * np.exp(
                -0.1 * 0.1 * np.pi**2 * (1 / 2**2 + 1 / 3**2 + 1 / 4**2)
            )
            errors.append(float(np.sqrt(np.mean((values - exact) ** 2))))
        self.assertLess(errors[1], errors[0] * 0.4)


if __name__ == "__main__":
    unittest.main()
