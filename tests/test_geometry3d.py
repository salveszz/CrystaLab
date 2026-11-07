# WHAT THESE TESTS CHECK
# Compare reconstructed solids with analytic cube, box, tetrahedron and octahedron
# results. Verify rotations, translations, outward face displacement, face ordering
# and redundant planes. Reject open, impossible, flat and malformed plane sets.
# These checks validate geometry, not KDP growth physics or crystal quality.

"""Run with: python -m unittest discover -s tests -v."""

from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation.crystal import Crystal


class Geometry3DTests(unittest.TestCase):
    def test_cube_and_box_analytic_geometry(self):
        for size, volume, area in [([2, 2, 2], 8.0, 24.0), ([2, 4, 6], 48.0, 88.0)]:
            with self.subTest(size=size):
                solid = Crystal.box(center=[3, 4, 5], size=size)
                self.assertAlmostEqual(solid.volume, volume, places=10)
                self.assertAlmostEqual(solid.surface_area, area, places=10)
                self.assertEqual(len(solid.vertices), 8)
                self.assertEqual([len(face) for face in solid.faces], [4] * 6)
                self.assertAlmostEqual(solid.face_areas.sum(), area, places=10)
                np.testing.assert_allclose(np.ptp(solid.vertices, axis=0), size)

    def test_tetrahedron_with_sloped_face(self):
        # First octant, bounded by x/2 + y/3 + z/4 <= 1: volume = 2*3*4/6.
        normals = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0.5, 1 / 3, 0.25]]
        solid = Crystal(normals, [0, 0, 0, 1])
        area = 3 + 4 + 6 + 0.5 * np.sqrt(12**2 + 8**2 + 6**2)
        self.assertAlmostEqual(solid.volume, 4.0, places=10)
        self.assertAlmostEqual(solid.surface_area, area, places=10)
        self.assertEqual(len(solid.vertices), 4)
        self.assertEqual([len(face) for face in solid.faces], [3] * 4)
        self.assertAlmostEqual(solid.face_areas.sum(), area, places=10)

    def test_octahedron_all_faces_sloped(self):
        # |x|+|y|+|z| <= 1 has eight triangular faces and six axial vertices.
        solid = Crystal(list(product([-1.0, 1.0], repeat=3)), np.ones(8))
        self.assertAlmostEqual(solid.volume, 4 / 3, places=10)
        self.assertAlmostEqual(solid.surface_area, 4 * np.sqrt(3), places=10)
        self.assertEqual(len(solid.vertices), 6)
        self.assertEqual([len(face) for face in solid.faces], [3] * 8)
        self.assertTrue(solid.contains([0, 0, 0]))
        self.assertFalse(solid.contains([0.5, 0.5, 0.5]))

    def test_plane_scaling_preserves_geometry(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        scales = np.arange(1.0, 7.0)
        rescaled = Crystal(cube.normals * scales[:, None], cube.offsets * scales)
        self.assertAlmostEqual(rescaled.volume, cube.volume, places=10)
        np.testing.assert_allclose(np.linalg.norm(rescaled.normals, axis=1), 1.0)

    def test_rotation_translation_and_outward_face_order(self):
        angle = 0.63
        rotation = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        box = Crystal.box([2, 3, 4], [2, 3, 4])
        moved = box.rotated(rotation, origin=[1, 1, 1]).translated([-5, 2, 1])
        expected = (box.vertices - 1) @ rotation.T + 1 + [-5, 2, 1]
        self.assertTrue(moved.contains(expected).all())
        self.assertAlmostEqual(moved.volume, box.volume, places=9)
        self.assertAlmostEqual(moved.surface_area, box.surface_area, places=9)
        for normal, face in zip(moved.normals, moved.faces):
            points = moved.vertices[face]
            cross = np.cross(points[1] - points[0], points[2] - points[0])
            self.assertGreater(float(cross @ normal), 0.0)

    def test_prescribed_face_displacement_and_zero_change(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        self.assertAlmostEqual(cube.advanced(np.zeros(6)).volume, 8.0)
        self.assertAlmostEqual(cube.advanced(np.full(6, 0.25)).volume, 2.5**3)
        one_face = cube.advanced([0.5, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(one_face.volume, 10.0)
        self.assertAlmostEqual(cube.volume, 8.0)  # The original solid is unchanged.

    def test_redundant_and_duplicate_planes_have_no_extra_area(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        solid = Crystal(
            np.vstack((cube.normals, [1, 0, 0], [0, 0, 1])),
            np.r_[cube.offsets, 2.0, 1.0],
        )
        self.assertAlmostEqual(solid.volume, 8.0)
        self.assertAlmostEqual(solid.face_areas.sum(), 24.0)
        self.assertEqual(len(solid.faces[-2]), 0)
        self.assertEqual(len(solid.faces[-1]), 0)

    def test_face_can_disappear_under_prescribed_motion(self):
        # Cut one cube corner, then move that sloped plane beyond the cube.
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        cut = Crystal(np.vstack((cube.normals, [1, 1, 1])), np.r_[cube.offsets, 2.0])
        self.assertLess(cut.volume, cube.volume)
        self.assertEqual(len(cut.faces[-1]), 3)
        grown = cut.advanced([0, 0, 0, 0, 0, 0, 1.0])
        self.assertAlmostEqual(grown.volume, 8.0)
        self.assertEqual(len(grown.faces[-1]), 0)

    def test_contains_batched_points_and_readonly_geometry(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        points = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 0, 0], [0, 0, -2]]])
        np.testing.assert_array_equal(
            cube.contains(points), [[True, True], [False, False]]
        )
        with self.assertRaises(ValueError):
            cube.offsets[0] = 10.0

    def test_unbounded_flat_and_impossible_solids_are_rejected(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        cases = [
            (cube.normals[:4], cube.offsets[:4]),  # Infinite along z.
            (cube.normals, [1, 1, 1, 1, 0, 0]),  # Zero height.
            (cube.normals, [-2, 1, 1, 1, 1, 1]),
        ]  # x <= -2 and x >= -1.
        for normals, offsets in cases:
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                Crystal(normals, offsets)

    def test_invalid_geometry_arguments(self):
        cube = Crystal.box([0, 0, 0], [2, 2, 2])
        for normals, offsets in [
            ([[1, 0]] * 4, [1] * 4),
            ([[0, 0, 0]] * 4, [1] * 4),
            (cube.normals, [1] * 5),
            (cube.normals, [np.nan] * 6),
        ]:
            with self.subTest(normals=normals), self.assertRaises(ValueError):
                Crystal(normals, offsets)
        for distances in ([1], [-1] * 6, [np.inf] * 6):
            with self.subTest(distances=distances), self.assertRaises(ValueError):
                cube.advanced(distances)
        with self.assertRaises(ValueError):
            cube.rotated(np.diag([-1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            Crystal.box([0, 0, 0], [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
