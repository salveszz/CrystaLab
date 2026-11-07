# WHAT THESE TESTS CHECK
# Verify the diagnostic's independent volume moments against analytic boxes,
# a tetrahedron and a trapezoid before trusting its concentration references.
# A partially solid cell's liquid centroid differs from its Cartesian center.
# Exact affine prism integrals must agree with analytic facet averages for
# aligned and tilted seeds. Transport checks use a tangent affine field and
# exclude incompatible outer walls; their oracle correction is not runtime code.
# Runtime moments and local reconstruction are checked independently, including
# rank-deficient stencils, solid barriers and nonnegative concentration slopes.
# Corrected implicit diffusion must preserve bounds/mass with solver fallback;
# moving-boundary transport must recover analytic layer motion and evacuate cells.

from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_spatial_accuracy import (
    BoxIntersectionMoments,
    face_centroid,
    liquid_centroids,
    sampling_case,
    transport_case,
)
from simulation.crystal import Crystal
from simulation.tank import SimulationConfig, Tank
from simulation.physics import (
    liquid_gradient,
    diffuse_liquid,
    _bounded_diffusion_correction,
    _solve_corrected,
)


class SpatialReferenceTests(unittest.TestCase):
    def test_unclipped_box_volume_and_first_moment(self):
        reference = BoxIntersectionMoments([], [])
        volume, moment = reference.integrate([0, 0, 0], [2, 3, 4])
        self.assertAlmostEqual(volume, 24.0, places=12)
        np.testing.assert_allclose(moment, [24, 36, 48], atol=1e-12)

    def test_half_box_volume_and_centroid(self):
        reference = BoxIntersectionMoments([[1, 0, 0]], [0.5])
        volume, moment = reference.integrate([0, 0, 0], [1, 1, 1])
        self.assertAlmostEqual(volume, 0.5, places=12)
        np.testing.assert_allclose(moment / volume, [0.25, 0.5, 0.5], atol=1e-12)

    def test_tetrahedron_volume_and_centroid(self):
        reference = BoxIntersectionMoments([[1, 1, 1]], [1.0])
        volume, moment = reference.integrate([0, 0, 0], [1, 1, 1])
        self.assertAlmostEqual(volume, 1 / 6, places=12)
        np.testing.assert_allclose(moment / volume, [0.25] * 3, atol=1e-12)

    def test_empty_and_flat_intersections_have_zero_moments(self):
        for offset in (-1.0, 0.0):
            with self.subTest(offset=offset):
                reference = BoxIntersectionMoments([[1, 0, 0]], [offset])
                volume, moment = reference.integrate([0, 0, 0], [1, 1, 1])
                self.assertEqual(volume, 0.0)
                np.testing.assert_array_equal(moment, np.zeros(3))

    def test_polygon_centroid_uses_area_instead_of_corner_mean(self):
        trapezoid = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 1, 0]])
        np.testing.assert_allclose(
            face_centroid(trapezoid), [7 / 9, 4 / 9, 0], atol=1e-12
        )

    def test_liquid_centroid_subtracts_the_occupied_corner(self):
        config = SimulationConfig(
            width=3,
            depth=3,
            height=3,
            nx=3,
            ny=3,
            nz=3,
            initial_liquid_height=3,
            concentration_noise=0.0,
        )
        solid = Crystal.box([0.25] * 3, [0.5] * 3)
        tank = Tank(config, crystal=solid)
        centroids, volume_error = liquid_centroids(tank, solid)
        # Unit cell minus its lower 1/8 corner: each centroid coordinate is 15/28.
        self.assertAlmostEqual(tank.liquid_volumes[0, 0, 0], 7 / 8, places=12)
        np.testing.assert_allclose(centroids[0], [15 / 28] * 3, atol=1e-12)
        self.assertLess(volume_error, 1e-12)

    def test_affine_facet_reference_for_aligned_and_tilted_geometry(self):
        base = asdict(
            SimulationConfig(width=6, depth=6, height=6, initial_liquid_height=5.5)
        )
        for pose in ("aligned", "tilted_shifted"):
            with self.subTest(pose=pose):
                result = sampling_case([12, 12, 12], pose, base)
                self.assertTrue(result["geometry_unchanged"])
                self.assertEqual(len(result["faces"]), 6)
                self.assertLess(result["max_exact_integral_error"], 1e-10)
                self.assertLess(result["max_liquid_average_error"], 1e-10)
                self.assertLess(result["max_reference_volume_error"], 1e-10)
                self.assertTrue(np.isfinite(result["max_liquid_average_error"]))
                expected = [row["expected"] for row in result["faces"]]
                self.assertGreater(np.ptp(expected), 0.01)

    def test_tangent_field_transport_reference_and_boundary_selection(self):
        for slope in (0.0, 0.4):
            with self.subTest(slope=slope):
                result = transport_case(8, slope)
                self.assertGreater(result["interior_cut_cells"], 0)
                self.assertLess(result["max_exact_gradient_flux_error"], 1e-12)
                self.assertLess(result["max_exact_gradient_balance"], 1e-12)
                self.assertLess(result["max_production_balance"], 1e-12)
                self.assertLess(result["max_reference_volume_error"], 1e-10)
                if slope == 0.0:
                    self.assertLess(result["max_cut_cell_balance"], 1e-12)

    def test_runtime_clipped_moments_match_independent_rotated_reference(self):
        a = 0.37
        rotation = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        solid = (
            Crystal.box([0, 0, 0], [0.7, 0.9, 1.2])
            .rotated(rotation)
            .translated([2.07, 1.93, 2.11])
        )
        reference = BoxIntersectionMoments(solid.normals, solid.offsets)
        for lower, upper in (
            ([1.5] * 3, [2.0] * 3),
            ([2.0] * 3, [2.5] * 3),
            ([0] * 3, [4] * 3),
        ):
            expected_v, expected_m = reference.integrate(lower, upper)
            volume, moment = solid.clipped_moments(lower, upper)
            self.assertAlmostEqual(volume, expected_v, places=12)
            np.testing.assert_allclose(moment, expected_m, atol=1e-12)

    def test_runtime_liquid_centroids_and_initial_temperature_use_actual_wet_volume(
        self,
    ):
        config = SimulationConfig(
            width=4,
            depth=4,
            height=4,
            nx=8,
            ny=8,
            nz=8,
            initial_liquid_height=2.37,
            seed_center=[2.07, 1.93, 1.8],
            temperature_gradient=2.0,
            concentration_noise=0,
        )
        solid = config.make_crystal()
        tank = Tank(config, crystal=solid)
        reference, _error = liquid_centroids(tank, solid)
        wet = tank.liquid_mask
        np.testing.assert_allclose(
            tank.liquid_centroids[wet],
            reference.reshape((*tank.shape, 3))[wet],
            atol=1e-11,
        )
        expected = config.temperature + config.temperature_gradient * (
            reference[:, 2] / config.height - 0.5
        )
        np.testing.assert_allclose(
            tank.temperature[wet], expected.reshape(tank.shape)[wet], atol=1e-11
        )

    def test_local_gradients_recover_affine_field_without_an_exact_gradient_input(self):
        config = SimulationConfig(
            nx=6, ny=7, nz=8, initial_liquid_height=8, concentration_noise=0
        )
        tank = Tank(config)
        expected = np.array([0.08, -0.05, 0.06])
        field = 1.35 + tank.liquid_centroids @ expected
        gradient = tank.field_gradient(field)
        np.testing.assert_allclose(
            gradient, np.broadcast_to(expected, gradient.shape), atol=1e-12
        )

    def test_rank_deficient_stencil_recovers_only_observable_gradient(self):
        values = np.arange(4, dtype=float).reshape(1, 1, 4)
        centers = np.zeros((1, 1, 4, 3))
        centers[0, 0, :, 0] = np.arange(4, dtype=float)
        openings = (np.zeros((0, 1, 4)), np.zeros((1, 0, 4)), np.ones((1, 1, 3)))
        result = liquid_gradient(values, np.ones_like(values), centers, openings)
        np.testing.assert_allclose(result[..., 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(result[..., 1:], 0.0, atol=1e-12)

    def test_reconstruction_stencil_cannot_cross_a_solid_wall(self):
        config = SimulationConfig(
            width=4,
            depth=4,
            height=4,
            nx=8,
            ny=8,
            nz=8,
            initial_liquid_height=4,
            concentration_noise=0,
        )
        wall = Crystal.box([2, 2, 2], [1, 4, 4])
        tank = Tank(config, crystal=wall)
        field = np.where(tank.liquid_centroids[..., 0] < 2, 1.0, 10.0)
        field[~tank.liquid_mask] = 0
        np.testing.assert_allclose(tank.field_gradient(field), 0.0, atol=1e-12)

    def test_nonnegative_reconstruction_preserves_cell_means(self):
        tank = Tank(
            SimulationConfig(
                width=3,
                depth=3,
                height=3,
                nx=3,
                ny=3,
                nz=3,
                initial_liquid_height=3,
                concentration_noise=0,
            )
        )
        field = np.full(tank.shape, 10.0)
        field[0, 0, 0] = 1e-8
        before = field.copy()
        gradient = tank.field_gradient(field, nonnegative=True)
        for x in (-0.5, 0.5):
            for y in (-0.5, 0.5):
                for z in (-0.5, 0.5):
                    values = field + gradient @ np.array([x, y, z])
                    self.assertGreaterEqual(float(values.min()), -1e-12)
        np.testing.assert_array_equal(field, before)

    def test_corrected_diffusion_preserves_range_and_mass_for_cut_cells(self):
        config = SimulationConfig(
            width=4,
            depth=4,
            height=4,
            nx=8,
            ny=8,
            nz=8,
            initial_liquid_height=4,
            concentration_noise=0,
        )
        box = Crystal.box([2, 2, 2], [4, 4, 4])
        solid = Crystal(
            np.vstack((box.normals, [-0.4, 0, 1])), np.r_[box.offsets, 1.27]
        )
        tank = Tank(config, crystal=solid)
        wet = tank.liquid_mask
        rng = np.random.default_rng(7)
        random = rng.random(tank.shape)
        impulse = np.zeros(tank.shape)
        wet_indices = np.flatnonzero(wet)
        smallest = wet_indices[np.argmin(tank.liquid_volumes.ravel()[wet_indices])]
        impulse.ravel()[smallest] = 1
        for original in (random, impulse, random * 50 - 15):
            for dt in (0.05, 100.0):
                with self.subTest(dt=dt, minimum=float(original[wet].min())):
                    before = original.copy()
                    diagnostics = {}
                    after = diffuse_liquid(
                        before,
                        tank.liquid_volumes,
                        tank.geometry.openings,
                        0.05,
                        dt,
                        tank.dx,
                        tank.dy,
                        tank.dz,
                        centroids=tank.liquid_centroids,
                        diagnostics=diagnostics,
                    )
                    self.assertTrue(np.isfinite(after).all())
                    self.assertGreaterEqual(after[wet].min(), before[wet].min() - 1e-11)
                    self.assertLessEqual(after[wet].max(), before[wet].max() + 1e-11)
                    self.assertAlmostEqual(
                        float(np.sum(after * tank.liquid_volumes)),
                        float(np.sum(before * tank.liquid_volumes)),
                        places=10,
                    )
                    np.testing.assert_array_equal(before, original)

    def test_bounded_edge_transfers_preserve_mass_without_clipping(self):
        from scipy.sparse import csr_matrix

        matrix = csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
        old = np.array([0.0, 1.0])
        capacity = np.ones(2)
        for edge_flux in (10.0, -10.0):
            diagnostics = {}
            result = _bounded_diffusion_correction(
                matrix,
                capacity * old,
                old,
                capacity,
                np.array([edge_flux]),
                np.array([0]),
                np.array([1]),
                1.0,
                diagnostics,
            )
            self.assertGreaterEqual(result.min(), 0)
            self.assertLessEqual(result.max(), 1)
            self.assertAlmostEqual(float(result.sum()), 1.0, places=12)
            self.assertEqual(diagnostics["bounded_solves"], 1)
            self.assertEqual(diagnostics["limited_edges"], 1)

    def test_nonsymmetric_solver_has_direct_fallback(self):
        from unittest.mock import patch
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve

        matrix = csr_matrix([[2.0, -0.8], [-1.2, 2.0]])
        rhs = np.array([1.0, 2.0])
        diagnostics = {}
        with patch("scipy.sparse.linalg.gmres", return_value=(np.zeros(2), 1)):
            result = _solve_corrected(matrix, rhs, np.zeros(2), diagnostics)
        np.testing.assert_allclose(result, spsolve(matrix, rhs), atol=1e-12)
        self.assertEqual(diagnostics["direct_fallbacks"], 1)

    def test_open_aperture_centroid_excludes_the_solid_section(self):
        config = SimulationConfig(
            width=3,
            depth=3,
            height=3,
            nx=3,
            ny=3,
            nz=3,
            initial_liquid_height=3,
            concentration_noise=0,
        )
        solid = Crystal.box([0.75, 0.25, 0.25], [1.5, 0.5, 0.5])
        tank = Tank(config, crystal=solid)
        self.assertAlmostEqual(tank.geometry.openings[2][0, 0, 0], 0.75, places=12)
        np.testing.assert_allclose(
            tank.geometry.opening_centroids[2][0, 0, 0],
            [1.0, 7 / 12, 7 / 12],
            atol=1e-12,
        )

    def test_partial_free_surface_aperture_uses_its_actual_midpoint(self):
        tank = Tank(
            SimulationConfig(
                width=4,
                depth=4,
                height=4,
                nx=8,
                ny=8,
                nz=8,
                initial_liquid_height=2.37,
                seed_center=[2, 2, 1.8],
                concentration_noise=0,
            )
        )
        np.testing.assert_allclose(
            tank.geometry.opening_centroids[2][4, 0, 0], [0.5, 0.25, 2.185], atol=1e-12
        )

    def test_translating_liquid_layer_preserves_affine_fields_and_velocity(self):
        for n in (8, 12, 16):
            with self.subTest(n=n):
                tank = self.translating_tank(n)
                mass = tank.solute_mass
                heat = float(np.sum(tank.temperature * tank.liquid_volumes))
                solid = Crystal.box([2, 2, 0.625], [4, 4, 1.25])
                tank.move_boundary(solid, 0.5, tank.concentration.copy())
                wet = tank.liquid_mask
                z = tank.liquid_centroids[..., 2]
                np.testing.assert_allclose(
                    tank.concentration[wet], (1 + 0.1 * (z - 1.25))[wet], atol=1e-10
                )
                np.testing.assert_allclose(
                    tank.temperature[wet], (20 + 2 * (z - 1.25))[wet], atol=1e-10
                )
                np.testing.assert_allclose(
                    tank.displacement_velocity[wet],
                    np.broadcast_to(
                        [0.0, 0.0, 0.1], tank.displacement_velocity[wet].shape
                    ),
                    atol=1e-10,
                )
                self.assertAlmostEqual(tank.solute_mass, mass, places=10)
                self.assertAlmostEqual(
                    float(np.sum(tank.temperature * tank.liquid_volumes)),
                    heat,
                    places=9,
                )

    def translating_tank(self, n):
        config = SimulationConfig(
            width=4,
            depth=4,
            height=4,
            nx=n,
            ny=n,
            nz=n,
            initial_liquid_height=3.2,
            concentration_noise=0,
        )
        tank = Tank(config, crystal=Crystal.box([2, 2, 0.6], [4, 4, 1.2]))
        z = tank.liquid_centroids[..., 2]
        tank.concentration = 1 + 0.1 * (z - 1.2)
        tank.temperature = 20 + 2 * (z - 1.2)
        tank.concentration[~tank.liquid_mask] = 0
        tank.temperature[~tank.liquid_mask] = 0
        return tank

    def test_large_translation_closing_cells_stays_bounded_and_conservative(self):
        tank = self.translating_tank(8)
        mass = tank.solute_mass
        heat = float(np.sum(tank.temperature * tank.liquid_volumes))
        tank.move_boundary(
            Crystal.box([2, 2, 0.825], [4, 4, 1.65]), 1.0, tank.concentration.copy()
        )
        wet = tank.liquid_mask
        self.assertGreaterEqual(tank.concentration[wet].min(), 0.98 - 1e-10)
        self.assertLessEqual(tank.concentration[wet].max(), 1.2 + 1e-10)
        self.assertGreaterEqual(tank.temperature[wet].min(), 19.6 - 1e-10)
        self.assertLessEqual(tank.temperature[wet].max(), 24.0 + 1e-10)
        self.assertAlmostEqual(tank.solute_mass, mass, places=10)
        self.assertAlmostEqual(
            float(np.sum(tank.temperature * tank.liquid_volumes)), heat, places=9
        )


if __name__ == "__main__":
    unittest.main()
