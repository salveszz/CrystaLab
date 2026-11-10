# WHAT THESE TESTS CHECK
# Check implicit conservative momentum transport against an analytic discrete
# Fourier mode, constant preservation, signed-vector bounds and exact budgets.
# Exercise tiny capacities at Courant 1000, periodic/disconnected topology and
# coupling to no-slip Stokes without losing momentum or producing energy.
# Independently check polynomial curl/source references and retained evidence.
# The suite does not enable the canonical cut-cell or growing-crystal runtime.

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.polynomial import polynomial as poly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from simulation.crystal import Crystal
from simulation.cutcell import CutCellGeometry
from simulation.mesh import RectilinearGrid
from simulation.momentum import ImplicitMomentumTransport, CutCellMomentum
from simulation.stokes import StokesState
from simulation.stokes import CutCellStokes
from check_cutcell import wall_geometry
from check_stokes import operator_for
from check_momentum import (
    polynomial_fields,
    polynomial_means,
    wall_reference,
    abc_reference,
    tiny_cycle,
    tiny_coupled_case,
    momentum_wall_geometry,
    combined_metrics,
    transport_wave_case,
    run_study,
)


def protocol():
    return json.loads((ROOT / "configs/momentum_verification.json").read_text())


def periodic_operator(n=4):
    return operator_for(
        CutCellGeometry(RectilinearGrid([0.01] * 3, [n] * 3, [True] * 3)), protocol()
    )


class MomentumTests(unittest.TestCase):
    def test_focused_background_cap_preserves_domain_and_wall_with_actual_counts(self):
        p = protocol()
        for n in (6, 10):
            original = wall_geometry(p["length_m"], p["sloped_wall"], n, focused=True)
            resolved = momentum_wall_geometry(p, n, focused=True)
            self.assertGreater(resolved.grid.shape[0], n)
            self.assertLessEqual(
                max(w.max() for w in resolved.grid.widths),
                p["length_m"] / n * (1 + 1e-12),
            )
            self.assertAlmostEqual(
                resolved.volumes_m3.sum() / original.volumes_m3.sum(), 1, places=12
            )
            self.assertAlmostEqual(
                resolved.wall_areas_m2.sum() / original.wall_areas_m2.sum(),
                1,
                places=12,
            )
            np.testing.assert_array_equal(resolved.grid.lengths, original.grid.lengths)

    def test_schur_and_direct_stokes_solve_the_same_equations(self):
        p = protocol()
        geometries = [
            periodic_operator().geometry,
            wall_geometry(0.01, p["sloped_wall"], 5),
            tiny_cycle(p)[0].geometry,
        ]
        for geometry in geometries:
            old = np.random.default_rng(3).normal(size=(*geometry.grid.shape, 3)) * 1e-4
            old[~geometry.fluid] = 0
            states = []
            for backend in ("direct", "schur"):
                op = CutCellStokes(
                    geometry,
                    kinematic_viscosity_m2_per_s=1e-6,
                    density_kg_per_m3=1170,
                    linear_solver=backend,
                )
                states.append(op.step(old, 0.1))
                # Also exercise the steady Schur/mass preconditioner on a
                # region with walls (periodic steady velocity has a nullspace).
                if not all(geometry.grid.periodic):
                    forced = np.zeros_like(old)
                    forced[geometry.fluid] = [1e-5, -2e-5, 3e-5]
                    op.steady(forced)
            np.testing.assert_allclose(
                states[0].cell_velocity_m_per_s,
                states[1].cell_velocity_m_per_s,
                rtol=1e-9,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                states[0].pressure_pa, states[1].pressure_pa, rtol=1e-7, atol=1e-11
            )

    def test_coupled_tiny_cell_step_retains_large_courant_and_energy_budget(self):
        op, state, metrics, _fields = tiny_coupled_case(protocol())
        self.assertAlmostEqual(metrics["max_outgoing_courant"], 1000, places=8)
        self.assertLess(metrics["relative_work_error"], 1e-8)
        self.assertLess(metrics["relative_momentum_error"], 1e-9)
        self.assertLess(metrics["relative_volume_flux"], 1e-10)
        self.assertLess(metrics["relative_energy_increase"], 0)
        self.assertLess(op.geometry.measurements()["min_liquid_fraction"], 1e-8)
        self.assertTrue(np.isfinite(state.pressure_pa).all())

    def test_implicit_periodic_fourier_mode_matches_exact_discrete_amplification(self):
        n = 8
        op = periodic_operator(n)
        g = op.geometry
        transport = ImplicitMomentumTransport(op.projection)
        native = tuple(
            np.full(f.mask.shape, 1e-4 if i == 0 else 0)
            for i, f in enumerate(op.projection.faces)
        )
        x = g.grid.positions()[..., 0]
        phase = 2 * np.pi * x / 0.01
        old = (
            np.stack((np.sin(phase), np.cos(phase), -0.5 * np.sin(phase)), axis=-1)
            * 1e-4
        )
        dt = 3.0
        result, metrics = transport.step(old, native, dt)
        factor = 1 / (1 + dt * 1e-4 / (0.01 / n) * (1 - np.exp(-2j * np.pi / n)))
        expected = (
            np.stack(
                (
                    np.imag(np.exp(1j * phase) * factor),
                    np.real(np.exp(1j * phase) * factor),
                    -0.5 * np.imag(np.exp(1j * phase) * factor),
                ),
                axis=-1,
            )
            * 1e-4
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-18)
        self.assertLess(metrics["bound_violation_m_per_s"], 1e-16)
        self.assertLess(abs(metrics["work_residual_m5_per_s3"]), 1e-27)

    def test_constant_and_zero_flux_preservation_with_large_step(self):
        op = periodic_operator()
        g = op.geometry
        t = ImplicitMomentumTransport(op.projection)
        old = np.broadcast_to([1, -0.5, 0.25], (*g.grid.shape, 3)).copy()
        native = tuple(
            np.full(f.mask.shape, 1e-4 if i == 0 else 0)
            for i, f in enumerate(op.projection.faces)
        )
        result, _metrics = t.step(old, native, 100)
        np.testing.assert_allclose(result, old, rtol=1e-12, atol=1e-14)
        zeros = tuple(np.zeros_like(v) for v in native)
        varying = np.random.default_rng(4).normal(size=old.shape)
        result, _metrics = t.step(varying, zeros, 10)
        np.testing.assert_allclose(result, varying, rtol=1e-15, atol=1e-15)

    def test_tiny_cycle_retains_capacity_bounds_momentum_and_work_at_courant_1000(self):
        op, native, old, dt = tiny_cycle(protocol())
        result, metrics = ImplicitMomentumTransport(op.projection).step(old, native, dt)
        self.assertAlmostEqual(metrics["max_outgoing_courant"], 1000, places=9)
        self.assertGreater(op.geometry.measurements()["min_liquid_fraction"], 0)
        self.assertLess(op.geometry.measurements()["min_liquid_fraction"], 1e-8)
        self.assertEqual(np.count_nonzero(op.geometry.fluid), len(op.viscosity.cells))
        self.assertLess(metrics["bound_violation_m_per_s"], 1e-12)
        self.assertLess(np.max(np.abs(metrics["momentum_change_m4_per_s"])), 1e-30)
        power = metrics["old_energy_m5_per_s2"] / dt
        self.assertLess(abs(metrics["work_residual_m5_per_s3"]) / power, 1e-10)
        self.assertGreater(metrics["donor_dissipation_m5_per_s3"], 0)
        self.assertFalse(result.flags.writeable)

    def test_signed_vector_transport_is_bounded_and_periodic_momentum_conserved(self):
        op = periodic_operator()
        g = op.geometry
        rng = np.random.default_rng(61)
        old = rng.normal(size=(*g.grid.shape, 3)) * 1e-4
        raw = tuple(rng.normal(size=f.mask.shape) * 1e-4 for f in op.projection.faces)
        for value, f in zip(raw, op.projection.faces):
            ix = [slice(None)] * 3
            ix[f.axis] = -1
            value[tuple(ix)] = value.take(0, axis=f.axis)
        native, _pi = op.projection.project(raw)
        original = old.copy()
        result, metrics = ImplicitMomentumTransport(op.projection).step(
            old, native, 2.0
        )
        self.assertLess(metrics["bound_violation_m_per_s"], 1e-16)
        self.assertLess(np.max(np.abs(metrics["momentum_change_m4_per_s"])), 1e-23)
        self.assertLess(metrics["energy_m5_per_s2"], metrics["old_energy_m5_per_s2"])
        np.testing.assert_array_equal(old, original)
        self.assertTrue(np.isfinite(result).all())

    def test_reference_curl_has_zero_boundary_and_analytic_integral(self):
        p = protocol()
        g = wall_geometry(0.01, p["sloped_wall"], 6)
        op = operator_for(g, p)
        state, _linear, advection = wall_reference(op, p)
        np.testing.assert_array_equal(
            state.trace_velocity_m_per_s[op.viscosity.boundary], 0
        )
        self.assertLess(abs(op.net_flux(state)).max(), 1e-23)
        integral = np.sum(
            g.volumes_m3[..., None] * state.cell_velocity_m_per_s, axis=(0, 1, 2)
        )
        self.assertLess(abs(integral).max(), 1e-17)
        self.assertGreater(np.max(abs(advection)), 1e-9)
        coefficients, _, _ = polynomial_fields(p)
        direct = polynomial_means(op, p, coefficients, faces=True)
        self.assertLess(abs(direct - state.trace_velocity_m_per_s).max(), 1e-10)
        p["sloped_wall"]["quadrature_order"] = 14
        other, other_linear, other_advection = wall_reference(op, p)
        np.testing.assert_allclose(
            other.cell_velocity_m_per_s,
            state.cell_velocity_m_per_s,
            rtol=1e-6,
            atol=1e-12,
        )
        self.assertTrue(
            np.isfinite(other_linear).all() and np.isfinite(other_advection).all()
        )

    def test_manufactured_advection_is_independent_analytic_velocity_derivative(self):
        p = protocol()
        u, linear, advection = polynomial_fields(p)
        point = np.array([0.43, 0.46, 0.69])
        epsilon = 1e-5

        def evaluate(coefficients, position):
            return np.array([poly.polyval3d(*position, c) for c in coefficients])

        value = evaluate(u, point)
        jac = np.column_stack(
            [
                (
                    evaluate(u, point + epsilon * np.eye(3)[j])
                    - evaluate(u, point - epsilon * np.eye(3)[j])
                )
                / (2 * epsilon * p["length_m"])
                for j in range(3)
            ]
        )
        exact = evaluate(advection, point)
        np.testing.assert_allclose(exact, jac @ value, rtol=1e-4, atol=1e-12)
        self.assertLess(abs(np.trace(jac)), 1e-7)
        self.assertTrue(np.isfinite(evaluate(linear, point)).all())

    def test_split_rest_and_periodic_constant_velocity_are_preserved(self):
        op = periodic_operator()
        g = op.geometry
        zero = StokesState(
            np.zeros((*g.grid.shape, 3)),
            np.zeros((len(op.viscosity.boundary), 3)),
            np.zeros(g.grid.shape),
        )
        momentum = CutCellMomentum(op)
        result = momentum.advance(zero, 0.1)
        np.testing.assert_array_equal(result.state.cell_velocity_m_per_s, 0)
        initial = abc_reference(op, protocol(), 0)
        values = np.broadcast_to(
            [1e-4, -2e-4, 0.5e-4], initial.cell_velocity_m_per_s.shape
        ).copy()
        traces = np.broadcast_to(
            [1e-4, -2e-4, 0.5e-4], initial.trace_velocity_m_per_s.shape
        ).copy()
        state = StokesState(values, traces, np.zeros(g.grid.shape))
        result = momentum.advance(state, 2.0)
        np.testing.assert_allclose(
            result.state.cell_velocity_m_per_s, values, rtol=1e-12, atol=1e-18
        )

    def test_split_energy_momentum_wall_and_pressure_budgets(self):
        p = protocol()
        for periodic in (False, True):
            g = (
                CutCellGeometry(RectilinearGrid([0.01] * 3, [4] * 3, [True] * 3))
                if periodic
                else wall_geometry(0.01, p["sloped_wall"], 5)
            )
            op = operator_for(g, p)
            old = np.random.default_rng(9).normal(size=(*g.grid.shape, 3)) * 1e-4
            old[~g.fluid] = 0
            state = op.step(old, 0.1)
            result = CutCellMomentum(op).advance(state, 0.1)
            metrics = combined_metrics(result, op, p, 0.1)
            self.assertLess(metrics["relative_work_error"], 1e-10)
            self.assertLess(metrics["relative_momentum_error"], 1e-10)
            self.assertLess(metrics["relative_volume_flux"], 1e-11)
            self.assertLess(metrics["relative_energy_increase"], 0)
            self.assertLess(metrics["relative_bound_violation"], 1e-11)
            self.assertTrue(metrics["wall_trace_exactly_zero"])

    def test_body_force_impulse_and_work_are_accounted_for(self):
        op = periodic_operator()
        g = op.geometry
        state = op.step(np.zeros((*g.grid.shape, 3)), 0.1)
        force = np.broadcast_to([1e-5, -2e-5, 3e-5], (*g.grid.shape, 3)).copy()
        result = CutCellMomentum(op).advance(state, 0.2, acceleration_m_per_s2=force)
        expected = 0.2 * force
        np.testing.assert_allclose(
            result.state.cell_velocity_m_per_s, expected, rtol=1e-12, atol=1e-18
        )
        self.assertLess(
            np.max(np.abs(result.diagnostics["combined_momentum_residual_m4_per_s"])),
            1e-23,
        )
        self.assertLess(
            abs(result.diagnostics["combined_work_residual_m5_per_s3"]), 1e-28
        )

    def test_disconnected_regions_do_not_exchange_advective_momentum(self):
        length = 0.01
        solid = Crystal.box([length / 2] * 3, [0.002, 0.012, 0.012])
        op = operator_for(
            CutCellGeometry(
                RectilinearGrid([length] * 3, [10] * 3, [False] * 3), solid
            ),
            protocol(),
        )
        g = op.geometry
        initial = np.zeros((*g.grid.shape, 3))
        left = g.fluid & (g.centroids_m[..., 0] < length / 2)
        initial[left] = 1e-4
        state = op.step(initial, 0.1)
        result = CutCellMomentum(op).advance(state, 0.1)
        np.testing.assert_array_equal(result.state.cell_velocity_m_per_s[~left], 0)

    def test_transport_reference_spatial_refinement_is_first_order(self):
        rows = [transport_wave_case(protocol(), n)[2] for n in (8, 16, 24)]
        self.assertTrue(
            all(
                b["relative_rms_error"] < a["relative_rms_error"]
                for a, b in zip(rows, rows[1:])
            )
        )
        self.assertLess(rows[-1]["relative_rms_error"], 0.08)

    def test_invalid_inputs_boundary_flow_and_divergence_are_rejected(self):
        op = periodic_operator()
        g = op.geometry
        t = ImplicitMomentumTransport(op.projection)
        old = np.zeros((*g.grid.shape, 3))
        native = [np.zeros(f.mask.shape) for f in op.projection.faces]
        for bad in (0, -1, True, float("nan")):
            with self.assertRaises(ValueError):
                t.step(old, native, bad)
        with self.assertRaises(ValueError):
            t.step(np.full(old.shape, np.nan), native, 0.1)
        native[0][1, 1, 1] = 1e-4
        with self.assertRaisesRegex(ValueError, "divergence-free"):
            t.step(old, native, 0.1)
        state = op.step(old, 0.1)
        with self.assertRaises(ValueError):
            CutCellMomentum(op).advance(old, 0.1)
        with self.assertRaises(ValueError):
            CutCellMomentum(op).advance(
                state, 0.1, acceleration_m_per_s2=np.full(old.shape, np.inf)
            )
        p = protocol()
        op = operator_for(wall_geometry(0.01, p["sloped_wall"], 4), p)
        state = op.step(np.zeros((*op.geometry.grid.shape, 3)), 0.1)
        traces = state.trace_velocity_m_per_s.copy()
        traces[op.viscosity.boundary] = 1e-4
        with self.assertRaisesRegex(ValueError, "stationary no-slip"):
            CutCellMomentum(op).advance(
                StokesState(state.cell_velocity_m_per_s, traces, state.pressure_pa), 0.1
            )

    def test_failed_gates_sources_and_existing_output_are_preserved(self):
        p = protocol()
        for name in ("transport_wave", "abc", "sloped_wall"):
            p[name]["counts"] = [3, 4, 5]
        for name in ("abc", "sloped_wall"):
            p[name]["duration_s"] = 0.1
            p[name]["dt_s"] = 0.05
        p["temporal"]["n"] = 3
        p["temporal"]["reference_steps"] = 64
        p["unforced"]["steps"] = 1
        p["gates"]["finest_abc_relative_velocity_error"] = 1e-30
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed_gate"
            with redirect_stdout(io.StringIO()):
                report = run_study(p, output)
            self.assertFalse(report["overall_pass"])
            self.assertFalse(report["gates"]["abc_relative_velocity_error"])
            self.assertEqual(len(report["results"]), 20)
            manifest = json.loads((output / "study.json").read_text())
            for name, digest in manifest["sources"].items():
                self.assertEqual(
                    hashlib.sha256(
                        (output / "sources" / name).read_bytes()
                    ).hexdigest(),
                    digest,
                )
            with np.load(output / "tiny_cycle.npz") as fields:
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
