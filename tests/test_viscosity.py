# WHAT THESE TESTS CHECK
# Check independent oblique-wall shear means/forcing, exact affine no slip,
# conservative dissipative implicit solves, tiny capacities and disconnected
# regions. Verify periodic eigenmodes and normal-face export compatibility.
# Ensure verification failures retain their protocol/sources and reject overwrite.
# This suite does not claim a coupled pressure/viscosity cut-cell flow step.

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from simulation.crystal import Crystal
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid
from simulation.pressure import PressureProjection
from simulation.viscosity import CutCellViscosity
from check_cutcell import wall_geometry, cube_geometry
from check_viscosity import (
    manufactured_means,
    shear_case,
    decay_case,
    temporal_case,
    run_study,
)


def protocol():
    return json.loads((ROOT / "configs/viscosity_verification.json").read_text())


class ViscosityTests(unittest.TestCase):
    def test_affine_shear_is_exact_at_actual_sloped_wall_on_both_meshes(self):
        p = protocol()
        for focused in (False, True):
            op, state, metrics, _fields = shear_case(p, 8, focused)
            self.assertLess(metrics["affine_relative_max_error"], 1e-11)
            np.testing.assert_array_equal(state.trace_velocity_m_per_s[op.wall_ids], 0)
            self.assertGreater(np.max(abs(state.cell_velocity_m_per_s)), 0)
            for _ids, stiffness, _b, _total in op._local:
                np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-16)
                self.assertGreater(np.linalg.eigvalsh(stiffness)[0], 0)

    def test_independent_quadrature_matches_closed_form_volume_mean(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 6, focused=True)
        op = CutCellViscosity(g)
        cell, force, traces = manufactured_means(op, p)
        a = p["sloped_wall"]["intercept_m"] / p["length_m"]
        b = p["sloped_wall"]["slope_dz_dy"]

        def integral_h(power):
            return ((1 - a) ** (power + 1) - (1 - a - b) ** (power + 1)) / (
                b * (power + 1)
            )

        expected = (
            p["speed_m_per_s"]
            * (integral_h(2) / 2 - integral_h(3) / 3)
            / (1 - a - b / 2)
        )
        mean = (
            np.sum(g.volumes_m3[..., None] * cell, axis=(0, 1, 2)) / g.volumes_m3.sum()
        )
        np.testing.assert_allclose(
            mean, expected * np.array([1, 2 / np.pi, b * 2 / np.pi]), rtol=1e-13
        )
        p["sloped_wall"]["quadrature_order"] = 4
        other = manufactured_means(op, p)
        for value, reference in zip(other, (cell, force, traces)):
            np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-18)

    def test_manufactured_velocity_converges_under_spatial_refinement(self):
        errors = [
            shear_case(protocol(), n)[2]["relative_volume_rms_error"] for n in (4, 6, 8)
        ]
        self.assertTrue(all(b < a for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.03)

    def test_implicit_energy_work_and_flux_balance_on_cut_geometries(self):
        p = protocol()
        geometries = [wall_geometry(p["length_m"], p["sloped_wall"], 6, focused=True)]
        for name in ("rotated_cube", "subcell_cube"):
            geometries.append(
                cube_geometry(p["length_m"], p[name], p[name]["counts_xyz"])
            )
        for g in geometries:
            op, state, metrics, _fields = decay_case(p, g)
            self.assertLessEqual(metrics["relative_energy_increase"], 0)
            self.assertLess(metrics["relative_balance_error"], 1e-10)
            self.assertGreaterEqual(metrics["dissipation_m5_per_s3"], 0)
            self.assertEqual(len(op.cells), np.count_nonzero(g.fluid))
            self.assertFalse(state.cell_velocity_m_per_s.flags.writeable)
            self.assertFalse(state.trace_velocity_m_per_s.flags.writeable)

    def test_forced_step_accounts_for_body_work(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 5)
        op = CutCellViscosity(g)
        old = np.zeros((*g.grid.shape, 3))
        force = np.zeros_like(old)
        force[g.fluid] = [1e-5, 2e-5, -3e-5]
        state = op.implicit(
            old,
            p["dt_s"],
            p["kinematic_viscosity_m2_per_s"],
            acceleration_m_per_s2=force,
        )
        balance = op.balances(
            state,
            p["kinematic_viscosity_m2_per_s"],
            previous=old,
            dt_s=p["dt_s"],
            acceleration=force,
        )
        self.assertGreater(balance["body_power_m5_per_s3"], 0)
        self.assertLess(
            abs(balance["work_residual_m5_per_s3"]) / balance["body_power_m5_per_s3"],
            1e-11,
        )

    def test_tiny_layer_is_retained_without_clipping_or_cell_merging(self):
        p = protocol()
        n, fraction = p["tiny_layer"]["n"], p["tiny_layer"]["fraction"]
        settings = dict(
            p["sloped_wall"],
            intercept_m=p["length_m"] * (1 - fraction / n),
            slope_dz_dy=0,
        )
        g = wall_geometry(p["length_m"], settings, n)
        _op, _state, metrics, _fields = decay_case(p, g)
        self.assertEqual(np.count_nonzero(g.fluid), n * n)
        self.assertAlmostEqual(
            g.measurements()["min_liquid_fraction"] / fraction, 1, delta=1e-6
        )
        self.assertLess(metrics["relative_balance_error"], 1e-8)
        self.assertGreater(metrics["max_relative_moment_defect"], 1e-10)

    def test_periodic_constants_and_analytic_discrete_eigenmode(self):
        p = protocol()
        n, length = 6, p["length_m"]
        g = CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
        op = CutCellViscosity(g)
        old = np.broadcast_to([1e-4, -2e-4, 3e-4], (*g.grid.shape, 3)).copy()
        state = op.implicit(old, 0.2, 1e-6)
        np.testing.assert_allclose(
            state.cell_velocity_m_per_s, old, rtol=1e-12, atol=1e-18
        )
        with self.assertRaisesRegex(ValueError, "Dirichlet wall"):
            op.steady(np.zeros_like(old), 1e-6)
        p["temporal"]["n"] = n
        p["temporal"]["duration_s"] = 0.2
        op, state, _metrics, fields = temporal_case(p, 1)
        rate = 4 * 1e-6 * np.sin(np.pi / n) ** 2 / (length / n) ** 2
        expected = fields["initial_velocity_m_per_s"] / (1 + 0.2 * rate)
        np.testing.assert_allclose(
            state.cell_velocity_m_per_s, expected, rtol=1e-11, atol=1e-18
        )

    def test_normal_trace_export_matches_pressure_layout_and_periodic_aliases(self):
        p = protocol()
        g = wall_geometry(
            p["length_m"], p["sloped_wall"], 6, periodic=(True, False, False)
        )
        op, state, _metrics, _fields = decay_case(p, g)
        normal = op.normal_face_velocity(state)
        pressure = PressureProjection(g.grid, g.volumes_m3, g.open_areas_m2)
        pressure.volume_fluxes(normal)
        for values, faces in zip(normal, pressure.faces):
            np.testing.assert_array_equal(values[~faces.mask], 0)
        np.testing.assert_array_equal(normal[0][..., 0], normal[0][..., -1])

    def test_disconnected_liquid_regions_exchange_no_viscous_flux(self):
        length = 0.01
        slab = Crystal.box([length / 2] * 3, [0.002, 0.012, 0.012])
        g = CutCellGeometry(RectilinearGrid([length] * 3, [10] * 3, [False] * 3), slab)
        op = CutCellViscosity(g)
        old = np.zeros((*g.grid.shape, 3))
        left = g.fluid & (g.centroids_m[..., 0] < length / 2)
        old[left] = 1e-4
        a = op.implicit(old, 0.1, 1e-6)
        old[g.fluid & ~left] = -3e-4
        b = op.implicit(old, 0.1, 1e-6)
        np.testing.assert_array_equal(a.cell_velocity_m_per_s[~left], 0)
        np.testing.assert_allclose(
            a.cell_velocity_m_per_s[left],
            b.cell_velocity_m_per_s[left],
            rtol=1e-12,
            atol=1e-18,
        )

    def test_invalid_values_and_nonstationary_embedded_walls_are_rejected(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 4)
        op = CutCellViscosity(g)
        zero = np.zeros((*g.grid.shape, 3))
        for dt in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                op.implicit(zero, dt, 1e-6)
        for nu in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                op.implicit(zero, 0.1, nu)
        for bad in (
            np.zeros(g.grid.shape),
            np.full(zero.shape, np.nan),
            np.ones(zero.shape),
        ):
            with self.assertRaises(ValueError):
                op.implicit(bad, 0.1, 1e-6)
        with self.assertRaisesRegex(ValueError, "zero velocity"):
            op.steady(
                zero, 1e-6, boundary_velocity_m_per_s=np.ones((len(op.boundary_ids), 3))
            )
        g = wall_geometry(
            p["length_m"], p["sloped_wall"], 4, periodic=(False, True, False)
        )
        with self.assertRaisesRegex(ValueError, "Periodic end"):
            CutCellViscosity(g)

    def test_failed_gate_sources_fields_and_overwrite_protection(self):
        p = protocol()
        p["sloped_wall"]["counts"] = [4, 6, 8]
        p["gates"]["affine_relative_max_error"] = 1e-30
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed_gate"
            with redirect_stdout(io.StringIO()):
                report = run_study(p, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["uniform_affine_exactness"])
            self.assertEqual(len(report["results"]), 13)
            manifest = json.loads((output / "study.json").read_text())
            for name, digest in manifest["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            with np.load(output / "tiny_layer.npz") as fields:
                self.assertGreater(np.count_nonzero(fields["liquid_volumes_m3"]), 0)
                self.assertTrue(
                    all(np.isfinite(fields[name]).all() for name in fields.files)
                )
            self.assertTrue((output / "refinement.png").is_file())
            with self.assertRaises(FileExistsError):
                run_study(p, output)
            failed = copy.deepcopy(p)
            failed["rotated_cube"]["max_axis_refinements"] = 0
            with redirect_stdout(io.StringIO()):
                report = run_study(failed, Path(directory) / "failed_geometry")
            self.assertFalse(report["overall_pass"])
            self.assertTrue(
                any("refinement limit" in r.get("error", "") for r in report["results"])
            )


if __name__ == "__main__":
    unittest.main()
