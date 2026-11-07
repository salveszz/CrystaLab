# WHAT THESE TESTS CHECK
# A rectilinear grid resolves the crystal neighborhood with smaller cells while
# preserving the same 3D tank, seed and physical sampling depth. Every capacity,
# aperture, distance and flux must use actual coordinates rather than a mean dx.
# Independent moment/affine references and a cosine diffusion solution verify
# geometry, sampling, transport and moving-boundary conservation on this mesh.

from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.crystal import Crystal
from simulation.engine import Simulation
from simulation.tank import Tank, SimulationConfig
from simulation.physics import diffuse_liquid, nonnegative_gradient
from check_spatial_accuracy import sampling_case, liquid_centroids


class FocusedGridTests(unittest.TestCase):
    def config(self, **kw):
        values = dict(
            nx=24,
            ny=24,
            nz=24,
            grid_focus_width=3.6,
            concentration_noise=0,
            initial_liquid_height=6.0,
            temperature_gradient=0,
        )
        values.update(kw)
        return SimulationConfig(**values)

    def test_edges_are_monotone_and_small_cells_surround_the_seed(self):
        tank = Tank(self.config())
        for edges, length, center in zip(
            (tank.x_edges, tank.y_edges, tank.z_edges), (10, 8, 8), (5, 4, 4)
        ):
            self.assertEqual(len(edges), 25)
            self.assertEqual(edges[0], 0)
            self.assertEqual(edges[-1], length)
            self.assertTrue((np.diff(edges) > 0).all())
            inside = (edges[:-1] >= center - 1.8 - 1e-12) & (
                edges[1:] <= center + 1.8 + 1e-12
            )
            widths = np.diff(edges)
            self.assertLess(widths[inside].max(), widths[~inside].max())
        self.assertAlmostEqual(float(tank.cell_volumes.sum()), 640.0, places=10)
        self.assertAlmostEqual(tank.liquid_volume, 480.0, places=10)

    def test_solid_volume_and_liquid_centroids_match_independent_geometry(self):
        config = self.config(seed_center=[5.07, 4.03, 4.11])
        solid = config.make_crystal()
        tank = Tank(config, crystal=solid)
        self.assertAlmostEqual(
            float(np.sum(tank.geometry.solid_fraction * tank.cell_volumes)),
            solid.volume,
            places=11,
        )
        reference, error = liquid_centroids(tank, solid)
        wet = tank.liquid_mask
        np.testing.assert_allclose(
            tank.liquid_centroids[wet],
            reference.reshape((*tank.shape, 3))[wet],
            atol=1e-10,
        )
        self.assertLess(error, 1e-10)

    def test_affine_sampling_is_exact_on_aligned_and_tilted_focused_cells(self):
        base = asdict(self.config())
        for pose in ("aligned", "tilted_shifted"):
            result = sampling_case([24, 24, 24], pose, base)
            self.assertLess(result["max_liquid_average_error"], 1e-10)
            self.assertAlmostEqual(result["sampling_depth"], 0.75, places=12)

    def test_nonuniform_diffusion_is_bounded_and_conservative(self):
        c = self.config(nx=12, ny=12, nz=12)
        tank = Tank(c, crystal=c.make_crystal())
        original = np.random.default_rng(8).random(tank.shape)
        before = float(np.sum(original * tank.liquid_volumes))
        result = diffuse_liquid(
            original,
            tank.liquid_volumes,
            tank.geometry.openings,
            0.03,
            10.0,
            tank.dx,
            tank.dy,
            tank.dz,
            centroids=tank.liquid_centroids,
            distances=tank.distances,
            nominal_positions=tank.positions,
        )
        wet = tank.liquid_mask
        self.assertGreaterEqual(result[wet].min(), original[wet].min() - 1e-11)
        self.assertLessEqual(result[wet].max(), original[wet].max() + 1e-11)
        self.assertAlmostEqual(
            float(np.sum(result * tank.liquid_volumes)), before, places=9
        )

    def test_nonnegative_reconstruction_uses_actual_focused_cell_bounds(self):
        tank = Tank(self.config(nx=40, ny=40, nz=40))
        field = np.full(tank.shape, 0.01)
        gradient = np.zeros((*tank.shape, 3))
        # This cell is near x=5. A fabricated uniform box at index*min(dx)
        # lies near x=3 and misses the negative value at the real right face.
        cell = (20, 20, 20)
        gradient[cell] = [-1, 0, 0]
        limited = nonnegative_gradient(
            field,
            gradient,
            tank.liquid_centroids,
            np.array([tank.dx, tank.dy, tank.dz]),
            tank.liquid_level,
            bounds=tank.geometry.bounds,
        )
        low, high = tank.geometry.bounds
        minimum = field + np.sum(
            limited
            * np.where(
                limited >= 0, low - tank.liquid_centroids, high - tank.liquid_centroids
            ),
            axis=-1,
        )
        self.assertGreaterEqual(float(minimum[tank.liquid_mask].min()), -1e-14)
        self.assertAlmostEqual(limited[cell][0], -0.01 / 0.075, places=12)
        np.testing.assert_array_equal(field, 0.01)

    def test_nonuniform_layer_translation_preserves_affine_concentration(self):
        config = SimulationConfig(
            width=4,
            depth=4,
            height=4,
            nx=12,
            ny=12,
            nz=12,
            grid_focus_width=2.0,
            initial_liquid_height=3.2,
            concentration_noise=0,
        )
        tank = Tank(config, crystal=Crystal.box([2, 2, 0.6], [4, 4, 1.2]))
        tank.concentration = 1 + 0.1 * (tank.liquid_centroids[..., 2] - 1.2)
        tank.concentration[~tank.liquid_mask] = 0
        mass = tank.solute_mass
        tank.move_boundary(
            Crystal.box([2, 2, 0.625], [4, 4, 1.25]), 0.5, tank.concentration.copy()
        )
        expected = 1 + 0.1 * (tank.liquid_centroids[..., 2] - 1.25)
        np.testing.assert_allclose(
            tank.concentration[tank.liquid_mask], expected[tank.liquid_mask], atol=1e-10
        )
        self.assertAlmostEqual(tank.solute_mass, mass, places=10)

    def test_cosine_diffusion_refines_on_focused_grid(self):
        errors = []
        for nx in (12, 24, 48):
            tank = Tank(
                SimulationConfig(
                    width=4,
                    depth=4,
                    height=4,
                    nx=nx,
                    ny=3,
                    nz=3,
                    initial_liquid_height=4,
                    grid_focus_width=1.8,
                    concentration_noise=0,
                )
            )
            k = np.pi / 4
            D = 0.03
            dt = 0.01
            steps = 10
            average = (np.sin(k * tank.x_edges[1:]) - np.sin(k * tank.x_edges[:-1])) / (
                k * np.diff(tank.x_edges)
            )
            initial = np.broadcast_to(
                1 + 0.1 * average[None, None, :], tank.shape
            ).copy()
            field = initial
            for _ in range(steps):
                field = diffuse_liquid(
                    field,
                    tank.liquid_volumes,
                    tank.geometry.openings,
                    D,
                    dt,
                    tank.dx,
                    tank.dy,
                    tank.dz,
                    centroids=tank.liquid_centroids,
                    distances=tank.distances,
                    nominal_positions=tank.positions,
                )
            exact = 1 + (initial - 1) * np.exp(-D * k * k * dt * steps)
            errors.append(
                float(
                    np.sqrt(
                        np.sum((field - exact) ** 2 * tank.liquid_volumes)
                        / tank.liquid_volume
                    )
                )
            )
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_one_growth_step_conserves_mass_and_fills_only_liquid(self):
        sim = Simulation(
            self.config(duration=1.0, boundary_layer_cells=0.75 / (10 / 24))
        )
        mass = sim.total_mass
        sim.step()
        self.assertAlmostEqual(sim.total_mass, mass, places=9)
        self.assertTrue(np.isfinite(sim.tank.concentration).all())
        self.assertTrue((sim.tank.concentration >= -1e-12).all())


if __name__ == "__main__":
    unittest.main()
