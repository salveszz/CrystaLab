# WHAT THESE TESTS CHECK
# Verify conservative, dissipative fixed cut-cell pressure/viscosity coupling,
# stationary no-slip traces, independent manufactured pressure/velocity accuracy,
# periodic modes, pressure gauges, density units and disconnected regions.
# Exercise forced/outer-boundary work and retained tiny capacities. Ensure the
# verification runner preserves failures, exact sources and existing results.
# These tests cover linear Stokes, not nonlinear momentum or moving geometry.

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
from simulation.stokes import CutCellStokes
from check_cutcell import wall_geometry
from check_stokes import (
    operator_for,
    shear_case,
    decay_case,
    periodic_time_case,
    coupled_trajectory,
    run_study,
)


def protocol():
    return json.loads((ROOT / "configs/stokes_verification.json").read_text())


class StokesTests(unittest.TestCase):
    def test_rest_is_exact_with_zero_pressure_and_no_slip(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 4)
        op = operator_for(g, p)
        zero = np.zeros((*g.grid.shape, 3))
        for state in (op.step(zero, 0.1), op.steady(zero)):
            np.testing.assert_array_equal(state.cell_velocity_m_per_s, 0)
            np.testing.assert_array_equal(state.trace_velocity_m_per_s, 0)
            np.testing.assert_array_equal(state.pressure_pa, 0)

    def test_random_step_has_active_pressure_conservation_and_energy_identity(self):
        p = protocol()
        for focused in (False, True):
            g = wall_geometry(p["length_m"], p["sloped_wall"], 6, focused=focused)
            op, state, metrics, fields = decay_case(p, g)
            self.assertGreater(abs(state.pressure_pa).max(), 1e-6)
            self.assertLess(metrics["relative_integral_flux"], 1e-11)
            self.assertLess(metrics["relative_pressure_work"], 1e-11)
            self.assertLess(metrics["relative_balance_error"], 1e-10)
            self.assertLessEqual(metrics["relative_energy_increase"], 0)
            np.testing.assert_array_equal(
                state.trace_velocity_m_per_s[op.viscosity.boundary], 0
            )
            np.testing.assert_allclose(
                op.net_flux(state),
                op.projection.net_flux(op.normal_face_velocity(state)),
                rtol=0,
                atol=1e-23,
            )
            self.assertFalse(state.pressure_pa.flags.writeable)
            np.testing.assert_array_equal(
                fields["previous_velocity_m_per_s"][~g.fluid], 0
            )

    def test_pressure_divergence_adjoint_and_only_region_constant_nullspace(self):
        g = CutCellGeometry(RectilinearGrid([0.01] * 3, [3] * 3, [True] * 3))
        op = operator_for(g, protocol())
        d = op.divergence_integral[:, op._free_dofs].toarray()
        self.assertEqual(np.linalg.matrix_rank(d), len(op.viscosity.cells) - 1)
        np.testing.assert_allclose(
            d.T @ np.ones(len(op.viscosity.cells)), 0, atol=1e-20
        )
        rng = np.random.default_rng(23)
        trace, pressure = rng.normal(size=d.shape[1]), rng.normal(size=d.shape[0])
        self.assertAlmostEqual(
            float(pressure @ (d @ trace)), float(trace @ (d.T @ pressure)), delta=1e-17
        )
        saddle = op._system(0.1 * op.nu)[2]
        np.testing.assert_allclose((saddle - saddle.T).data, 0, atol=1e-18)

    def test_independent_velocity_and_pressure_converge_with_gradient_leakage(self):
        p = protocol()
        rows = [shear_case(p, n)[2] for n in (4, 6, 8)]
        for name in (
            "relative_velocity_rms_error",
            "pressure_rms_over_scale",
            "gradient_leakage_over_speed",
        ):
            self.assertTrue(
                all(b[name] < a[name] for a, b in zip(rows, rows[1:])), name
            )
        self.assertLess(rows[-1]["max_pressure_gauge_over_scale"], 1e-12)
        self.assertGreater(rows[-1]["gradient_leakage_over_speed"], 0)

    def test_pressure_is_in_pa_and_density_does_not_change_velocity(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 5)
        rng = np.random.default_rng(7)
        old = rng.normal(size=(*g.grid.shape, 3)) * 1e-4
        old[~g.fluid] = 0
        a = operator_for(g, p).step(old, 0.1)
        p["density_kg_per_m3"] *= 2
        b = operator_for(g, p).step(old, 0.1)
        np.testing.assert_array_equal(a.cell_velocity_m_per_s, b.cell_velocity_m_per_s)
        np.testing.assert_array_equal(a.pressure_pa * 2, b.pressure_pa)

    def test_body_and_prescribed_outer_boundary_work_are_accounted_for(self):
        p = protocol()
        p["temporal"]["n"] = 4
        op, state, metrics, _fields = coupled_trajectory(p, 4)
        self.assertGreater(abs(metrics["body_power_m5_per_s3"]), 0)
        self.assertGreater(abs(metrics["boundary_power_m5_per_s3"]), 0)
        self.assertLess(metrics["relative_balance_error"], 1e-10)
        self.assertLess(metrics["relative_pressure_work"], 1e-11)
        np.testing.assert_array_equal(
            state.trace_velocity_m_per_s[op.viscosity.wall_ids], 0
        )

    def test_periodic_constant_preservation_aliases_and_analytic_eigenmode(self):
        p = protocol()
        n, length = p["temporal"]["n"], p["length_m"]
        g = CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
        op = operator_for(g, p)
        old = np.broadcast_to([1e-4, -2e-4, 3e-4], (*g.grid.shape, 3)).copy()
        state = op.step(old, 0.2)
        np.testing.assert_allclose(
            state.cell_velocity_m_per_s, old, rtol=1e-12, atol=1e-18
        )
        for values, axis in zip(op.normal_face_velocity(state), (2, 1, 0)):
            np.testing.assert_array_equal(values.take(0, axis), values.take(-1, axis))
        with self.assertRaisesRegex(ValueError, "Dirichlet wall"):
            op.steady(np.zeros_like(old))
        _op, state, _metrics, fields = periodic_time_case(p, 1)
        rate = 4 * op.nu * np.sin(np.pi / n) ** 2 / (length / n) ** 2
        expected = fields["initial_velocity_m_per_s"] / (
            1 + p["temporal"]["duration_s"] * rate
        )
        np.testing.assert_allclose(
            state.cell_velocity_m_per_s, expected, rtol=1e-11, atol=1e-18
        )

    def test_coupled_temporal_refinement_with_nonzero_pressure(self):
        p = protocol()
        p["temporal"]["n"] = 4
        _op, fine, _metrics, _fields = coupled_trajectory(p, 64)
        errors = []
        for steps in (4, 8, 16):
            op, state, _metrics, _fields = coupled_trajectory(p, steps)
            self.assertGreater(abs(state.pressure_pa).max(), 1e-7)
            errors.append(
                np.sqrt(
                    np.sum(
                        op.geometry.volumes_m3[..., None]
                        * (state.cell_velocity_m_per_s - fine.cell_velocity_m_per_s)
                        ** 2
                    )
                )
            )
        self.assertTrue(all(b < 0.7 * a for a, b in zip(errors, errors[1:])))

    def test_disconnected_regions_have_independent_pressure_gauges_and_motion(self):
        length = 0.01
        solid = Crystal.box([length / 2] * 3, [0.002, 0.012, 0.012])
        g = CutCellGeometry(RectilinearGrid([length] * 3, [10] * 3, [False] * 3), solid)
        op = operator_for(g, protocol())
        self.assertEqual(len(op._groups), 2)
        old = np.zeros((*g.grid.shape, 3))
        left = g.fluid & (g.centroids_m[..., 0] < length / 2)
        old[left] = [1e-4, 2e-4, -1e-4]
        a = op.step(old, 0.1)
        old[g.fluid & ~left] = [-3e-4, 1e-4, 2e-4]
        b = op.step(old, 0.1)
        np.testing.assert_array_equal(a.cell_velocity_m_per_s[~left], 0)
        np.testing.assert_allclose(
            a.cell_velocity_m_per_s[left],
            b.cell_velocity_m_per_s[left],
            rtol=1e-11,
            atol=1e-18,
        )
        np.testing.assert_allclose(
            a.pressure_pa[left], b.pressure_pa[left], rtol=1e-11, atol=1e-18
        )
        for nodes in op._groups:
            self.assertLess(
                abs(
                    np.average(
                        b.pressure_pa[g.fluid][nodes],
                        weights=op.viscosity.volumes[nodes],
                    )
                ),
                1e-15,
            )

    def test_tiny_capacity_is_retained_with_conservative_coupled_step(self):
        p = protocol()
        n, fraction = p["tiny_layer"]["n"], p["tiny_layer"]["fraction"]
        settings = dict(
            p["sloped_wall"],
            intercept_m=p["length_m"] * (1 - fraction / n),
            slope_dz_dy=0,
        )
        g = wall_geometry(p["length_m"], settings, n)
        op, state, metrics, _fields = decay_case(p, g)
        self.assertEqual(len(op.viscosity.cells), n * n)
        self.assertAlmostEqual(
            g.measurements()["min_liquid_fraction"] / fraction, 1, delta=1e-6
        )
        self.assertLess(metrics["relative_balance_error"], 1e-8)
        self.assertLess(metrics["max_divergence_per_s"], 1e-10)
        self.assertTrue(np.isfinite(state.pressure_pa).all())

    def test_invalid_inputs_and_incompatible_boundary_flux_are_rejected(self):
        p = protocol()
        g = wall_geometry(p["length_m"], p["sloped_wall"], 4)
        for name in ("kinematic_viscosity_m2_per_s", "density_kg_per_m3"):
            for bad in (0, -1, True, float("nan")):
                options = dict(
                    kinematic_viscosity_m2_per_s=1e-6, density_kg_per_m3=1170
                )
                options[name] = bad
                with self.assertRaises(ValueError):
                    CutCellStokes(g, **options)
        op = operator_for(g, p)
        zero = np.zeros((*g.grid.shape, 3))
        for bad in (0, -1, True, float("inf")):
            with self.assertRaises(ValueError):
                op.step(zero, bad)
        with self.assertRaises(ValueError):
            op.step(np.full(zero.shape, np.nan), 0.1)
        boundary = np.zeros((len(op.viscosity.boundary_ids), 3))
        positions = op.viscosity.face_centroids_m[op.viscosity.boundary_ids]
        boundary[positions[:, 0] == 0, 0] = 1e-4
        with self.assertRaisesRegex(ValueError, "Incompatible prescribed flux"):
            op.steady(zero, boundary_velocity_m_per_s=boundary)
        with self.assertRaisesRegex(ValueError, "zero velocity"):
            op.steady(zero, boundary_velocity_m_per_s=np.ones_like(boundary))

    def test_failed_study_retains_sources_fields_and_rejects_overwrite(self):
        p = protocol()
        p["sloped_wall"]["counts"] = [4, 6, 8]
        p["temporal"]["n"] = 4
        p["temporal"]["reference_steps"] = 64
        p["gates"]["finest_relative_velocity_rms_error"] = 1e-30
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed_gate"
            with redirect_stdout(io.StringIO()):
                report = run_study(p, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["uniform_relative_velocity_rms_error"])
            self.assertEqual(len(report["results"]), 17)
            manifest = json.loads((output / "study.json").read_text())
            for name, digest in manifest["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            with np.load(output / "tiny_layer.npz") as fields:
                self.assertTrue(
                    all(np.isfinite(fields[name]).all() for name in fields.files)
                )
            self.assertTrue((output / "refinement.png").is_file())
            with self.assertRaises(FileExistsError):
                run_study(p, output)
            p["rotated_cube"]["max_axis_refinements"] = 0
            with redirect_stdout(io.StringIO()):
                report = run_study(p, Path(directory) / "failed_geometry")
            self.assertFalse(report["overall_pass"])
            self.assertTrue(
                any("refinement limit" in r.get("error", "") for r in report["results"])
            )


if __name__ == "__main__":
    unittest.main()
