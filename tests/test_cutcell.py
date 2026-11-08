# WHAT THESE TESTS CHECK
# Verify exact SI cut capacities/apertures/wall moments against analytic solids
# and divergence-theorem identities. Reject split liquid pockets, retain a tiny
# resolved layer, and test pressure projection/gauges on real sloped geometry.
# Manufactured Poisson sources are independently integrated over the liquid;
# they are not produced by applying the discrete operator to its own solution.
# These tests establish geometry/pressure, not embedded no-slip momentum dynamics.

import json
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.crystal import Crystal
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid, focused_axis_edges
from simulation.pressure import PressureProjection
from check_cutcell import (
    poisson_case,
    integrated_source,
    exact_potential,
    exact_gradient,
    _polynomial,
    wall_geometry,
    cube_geometry,
    run_study,
)


def slope_geometry(
    n=8, *, periodic=(False, False, False), edges=None, intercept=0.002, slope=0.5
):
    L = 0.01
    box = Crystal.box([L / 2] * 3, [L] * 3)
    solid = Crystal(
        np.vstack((box.normals, [0, -slope, 1])), np.r_[box.offsets, intercept]
    )
    return CutCellGeometry(RectilinearGrid([L] * 3, [n] * 3, periodic, edges), solid)


class CutCellTests(unittest.TestCase):
    def test_clipped_original_facets_have_analytic_areas_and_moments(self):
        solid = Crystal.box([0.5] * 3, [1] * 3)
        areas, moments = solid.clipped_face_moments([0.5, 0, 0], [1, 1, 1])
        np.testing.assert_allclose(areas, [1, 0, 0.5, 0.5, 0.5, 0.5], atol=1e-14)
        np.testing.assert_allclose(moments[0], [1, 0.5, 0.5], atol=1e-14)
        np.testing.assert_allclose(moments[2], [0.375, 0.5, 0.25], atol=1e-14)
        empty, first = solid.clipped_face_moments([2] * 3, [3] * 3)
        np.testing.assert_array_equal(empty, 0)
        np.testing.assert_array_equal(first, 0)

    def test_sloped_geometry_integrates_exact_volume_surface_and_boundary_moments(self):
        geometry = slope_geometry()
        metrics = geometry.measurements()
        self.assertAlmostEqual(
            metrics["liquid_volume_m3"] / (0.01**2 * (0.01 - 0.002 - 0.5 * 0.01 / 2)),
            1,
            places=13,
        )
        self.assertAlmostEqual(
            metrics["embedded_wall_area_m2"] / (0.01**2 * np.sqrt(1.25)), 1, places=13
        )
        self.assertGreater(metrics["cut_cell_count"], 0)
        self.assertLess(metrics["normalized_area_closure"], 1e-11)
        self.assertLess(metrics["normalized_moment_closure"], 1e-10)
        for area, full in zip(geometry.open_areas_m2, geometry.grid.face_areas):
            self.assertTrue(np.any((area > 0) & (area < full)))

    def test_rotated_subcell_solid_is_not_voxelized(self):
        angle = 0.31
        rotation = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )
        for width in (0.003, 0.0002):
            solid = (
                Crystal.box([0] * 3, [width] * 3)
                .rotated(rotation)
                .translated([0.0051, 0.0052, 0.0049])
            )
            grid = RectilinearGrid([0.01] * 3, [8] * 3, [False] * 3)
            geometry = CutCellGeometry(grid, solid)
            self.assertAlmostEqual(
                (grid.volumes.sum() - geometry.volumes_m3.sum()) / solid.volume,
                1,
                delta=1e-10,
            )
            self.assertAlmostEqual(
                geometry.wall_areas_m2.sum() / solid.surface_area, 1, places=11
            )
            self.assertLess(geometry.measurements()["normalized_moment_closure"], 1e-10)

    def test_full_fluid_and_aligned_faces_assign_wall_area_once(self):
        grid = RectilinearGrid([0.01] * 3, [8] * 3, [False] * 3)
        full = CutCellGeometry(grid)
        np.testing.assert_array_equal(full.volumes_m3, grid.volumes)
        self.assertLess(full.measurements()["normalized_moment_closure"], 1e-11)
        solid = Crystal.box([0.005] * 3, [0.0025] * 3)
        geometry = CutCellGeometry(grid, solid)
        self.assertAlmostEqual(
            geometry.wall_areas_m2.sum() / solid.surface_area, 1, places=12
        )
        self.assertEqual(geometry.measurements()["cut_cell_count"], 0)
        self.assertLess(geometry.measurements()["normalized_moment_closure"], 1e-10)

    def test_disconnected_pockets_in_one_cell_are_rejected(self):
        grid = RectilinearGrid([0.01] * 3, [5] * 3, [False] * 3)
        slab = Crystal.box([0.005] * 3, [0.0008, 0.012, 0.012])
        with self.assertRaisesRegex(ValueError, "Split liquid cell"):
            CutCellGeometry(grid, slab)

    def test_bounded_refinement_resolves_rotated_crystal_without_changing_solid(self):
        protocol = json.loads((ROOT / "configs/cutcell_verification.json").read_text())
        settings = protocol["rotated_cube"]
        geometry = cube_geometry(protocol["length_m"], settings, [8] * 3)
        self.assertGreater(len(geometry.refinement_history), 0)
        self.assertLessEqual(
            len(geometry.refinement_history), settings["max_axis_refinements"]
        )
        self.assertAlmostEqual(
            geometry.volumes_m3.sum()
            / (protocol["length_m"] ** 3 - settings["width_m"] ** 3),
            1,
            places=13,
        )
        self.assertLess(geometry.measurements()["normalized_moment_closure"], 1e-9)
        grid = RectilinearGrid([0.01] * 3, [5] * 3, [False] * 3)
        slab = Crystal.box([0.005] * 3, [0.0008, 0.012, 0.012])
        edges = [e.copy() for e in grid.edges]
        with self.assertRaisesRegex(ValueError, "refinement limit"):
            CutCellGeometry.resolved(grid, slab, max_axis_refinements=0)
        for actual, original in zip(grid.edges, edges):
            np.testing.assert_array_equal(actual, original)

    def test_study_preserves_failed_gate_real_geometry_sources_and_existing_output(
        self,
    ):
        protocol = json.loads((ROOT / "configs/cutcell_verification.json").read_text())
        protocol["sloped_wall"]["counts"] = [4, 6, 8]
        protocol["sloped_wall"]["finest_relative_potential_tolerance"] = 1e-30
        protocol["rotated_cube"]["counts"] = [4, 6, 8]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            with redirect_stdout(io.StringIO()):
                report = run_study(protocol, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["uniform_relative_potential_rms_error"])
            self.assertEqual(len(report["results"]), 14)
            manifest = json.loads((output / "study.json").read_text())
            for name, digest in manifest["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            with np.load(output / "rotated_cube_8.npz") as fields:
                self.assertGreater(len(fields["axis_refinement_history"]), 0)
                self.assertEqual(
                    len(fields["wall_face_ids"]), len(fields["wall_areas_m2"])
                )
                self.assertTrue(np.isfinite(fields["liquid_volumes_m3"]).all())
            self.assertTrue((output / "sloped_wall.png").is_file())
            with self.assertRaises(FileExistsError):
                run_study(protocol, output)
            protocol["rotated_cube"]["max_axis_refinements"] = 0
            failed = Path(directory) / "failed_topology"
            with redirect_stdout(io.StringIO()):
                failed_report = run_study(protocol, failed)
            self.assertFalse(failed_report["overall_pass"])
            failures = json.loads((failed / "case_failures.json").read_text())
            self.assertTrue(any("refinement limit" in row["error"] for row in failures))

    def test_tiny_layer_is_retained_and_pressure_projection_is_stable(self):
        fraction = 1e-7
        geometry = slope_geometry(intercept=0.01 - fraction * 0.01 / 8, slope=0)
        metrics = geometry.measurements()
        self.assertAlmostEqual(metrics["min_liquid_fraction"] / fraction, 1, delta=1e-7)
        self.assertEqual(int(geometry.fluid.sum()), 64)
        self.assertGreater(metrics["liquid_volume_m3"], 0)
        self._check_projection(geometry)

    def _check_projection(self, geometry):
        operator = PressureProjection(
            geometry.grid, geometry.volumes_m3, geometry.open_areas_m2
        )
        rng = np.random.default_rng(71)
        fields = []
        for face in operator.faces:
            value = np.zeros(face.mask.shape)
            value[tuple(face.cells.T)] = rng.normal(0, 1e-4, len(face.cells))
            operator._periodic_alias(value, face.axis)
            fields.append(value)
        before = [v.copy() for v in fields]
        norm = operator.projection_norm_squared(fields)
        flux_scale = max(
            float(abs(q).max(initial=0)) for q in operator.volume_fluxes(fields)
        )
        projected, phi = operator.project(fields)
        self.assertLess(
            float(abs(operator.net_flux(projected)).max(initial=0)) / flux_scale, 1e-10
        )
        self.assertLessEqual(
            operator.projection_norm_squared(projected), norm * (1 + 1e-12)
        )
        for new, face, old, current in zip(projected, operator.faces, before, fields):
            self.assertEqual(float(abs(new[~face.mask]).max(initial=0)), 0)
            np.testing.assert_array_equal(old, current)
        values = phi[operator.fluid]
        for nodes, _matrix, _direct in operator.groups:
            self.assertAlmostEqual(
                float(np.average(values[nodes], weights=operator._volumes[nodes])),
                0,
                delta=1e-20,
            )
        return operator

    def test_sloped_and_focused_projection_preserve_flux_and_decrease_norm(self):
        self._check_projection(slope_geometry())
        edge = focused_axis_edges(0.01, 8, 0.005, 0.0036)
        self._check_projection(slope_geometry(edges=[edge] * 3))

    def test_periodic_cut_geometry_has_one_end_face_unknown(self):
        geometry = slope_geometry(periodic=(True, False, False))
        operator = self._check_projection(geometry)
        face = operator.faces[0]
        self.assertFalse(
            np.any(face.cells[:, face.axis] == geometry.grid.shape[face.axis])
        )
        areas = [a.copy() for a in geometry.open_areas_m2]
        areas[2][..., -1] *= 0.9
        with self.assertRaisesRegex(ValueError, "Periodic end apertures"):
            PressureProjection(geometry.grid, geometry.volumes_m3, areas)

    def test_each_disconnected_liquid_region_has_its_own_gauge_and_source_check(self):
        grid = RectilinearGrid([0.01] * 3, [10] * 3, [False] * 3)
        slab = Crystal.box([0.005] * 3, [0.002, 0.012, 0.012])
        geometry = CutCellGeometry(grid, slab)
        operator = self._check_projection(geometry)
        self.assertEqual(len(operator.groups), 2)
        source = np.ones(grid.shape)
        source[..., 6:] = -1
        self.assertAlmostEqual(
            float(np.sum(source * geometry.volumes_m3)), 0, delta=1e-20
        )
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            operator.solve_poisson_source(source)

    def test_projection_rejects_invalid_or_leaking_inputs_atomically(self):
        geometry = slope_geometry()
        operator = PressureProjection(
            geometry.grid, geometry.volumes_m3, geometry.open_areas_m2
        )
        fields = [np.zeros(f.mask.shape) for f in operator.faces]
        fields[0][0, 0, 0] = 1e-4
        original = fields[0].copy()
        with self.assertRaisesRegex(ValueError, "blocked-face"):
            operator.project(fields)
        np.testing.assert_array_equal(fields[0], original)
        with self.assertRaises(ValueError):
            operator.project(fields[:1])
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            operator.solve_poisson_source(np.ones(geometry.grid.shape))
        with self.assertRaises(ValueError):
            geometry.volumes_m3[0, 0, 0] = 1
        with self.assertRaises(ValueError):
            operator.faces[0].mask[0, 0, 0] = True

    def test_manufactured_potential_matches_independent_differences_and_wall_conditions(
        self,
    ):
        protocol = json.loads((ROOT / "configs/cutcell_verification.json").read_text())
        settings, L = protocol["sloped_wall"], protocol["length_m"]
        points = L * np.array([[0.3, 0.35, 0.65], [0.7, 0.55, 0.8], [0.2, 0.25, 0.55]])
        h = L * 1e-4
        numerical = np.zeros(3)
        for direction in np.eye(3):
            numerical -= (
                exact_potential(points + h * direction, L, settings)
                - 2 * exact_potential(points, L, settings)
                + exact_potential(points - h * direction, L, settings)
            ) / h**2
        x, y, z = (points / L).T
        p, lap, _gy, _gz = _polynomial(
            y, z, settings["intercept_m"] / L, settings["slope_dz_dy"]
        )
        source = (
            settings["potential_scale_m2_per_s"]
            / L**2
            * np.cos(np.pi * x)
            * (np.pi**2 * p - lap)
        )
        np.testing.assert_allclose(numerical, source, rtol=1e-6, atol=1e-11)
        points[:, 2] = settings["intercept_m"] + settings["slope_dz_dy"] * points[:, 1]
        np.testing.assert_allclose(exact_gradient(points, L, settings), 0, atol=1e-20)
        geometry = wall_geometry(L, settings, 8)
        first = integrated_source(geometry, L, settings)
        second = integrated_source(geometry, L, dict(settings, quadrature_order=10))
        np.testing.assert_allclose(first, second, rtol=1e-11, atol=1e-18)
        self.assertLess(abs(float(np.sum(first * geometry.volumes_m3))), 1e-20)

    def test_manufactured_neumann_poisson_refines_on_true_cut_volumes(self):
        protocol = json.loads((ROOT / "configs/cutcell_verification.json").read_text())
        errors = []
        for n in (6, 10, 14):
            _geometry, metrics, _fields = poisson_case(
                protocol["length_m"], protocol["sloped_wall"], n, protocol["projection"]
            )
            errors.append(metrics["relative_potential_rms_error"])
            self.assertLess(metrics["relative_integral_flux_residual"], 1e-10)
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.05)


if __name__ == "__main__":
    unittest.main()
