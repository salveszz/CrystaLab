"""Check backend scoping, failure propagation and the shared residual gate."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import diags

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulation.linear import solve_spd, using_product_backend
from simulation.vulkan_spmv import canonical_csr


class LinearBackendTests(unittest.TestCase):
    def test_scope_restores_after_failure_and_never_falls_back(self):
        class Broken:
            def solve(self, *_):
                raise RuntimeError("backend failed")
        a, rhs = diags([2., 3.]), np.array([4., 9.])
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            with using_product_backend(Broken()):
                solve_spd(a, rhs)
        np.testing.assert_allclose(solve_spd(a, rhs), [2, 3], rtol=1e-12)

    def test_backend_status_and_independent_residual_are_checked(self):
        class Wrong:
            def __init__(self, status):
                self.status = status
            def solve(self, a, rhs, guess):
                return np.ones_like(rhs), self.status
        a, rhs = diags([2., 3.]), np.array([4., 9.])
        for status in (0, 1):
            with self.subTest(status=status), using_product_backend(Wrong(status)):
                with self.assertRaisesRegex(ArithmeticError, "residual check"):
                    solve_spd(a, rhs)

    def test_csr_identity_tracks_mutation_and_validates_payload(self):
        a = diags([2., 3.]).tocsr()
        _, _, before = canonical_csr(a)
        a.data[0] = 4
        _, _, after = canonical_csr(a)
        self.assertNotEqual(before, after)
        for invalid in (np.ones((2, 3)), np.eye(2)*np.nan, np.eye(2)*1j):
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                canonical_csr(invalid)

    def test_nested_scope_restores_outer_backend(self):
        class Exact:
            def __init__(self):
                self.calls = 0
            def solve(self, a, rhs, guess):
                self.calls += 1
                return rhs / a.diagonal(), 0
        outer, inner = Exact(), Exact()
        a, rhs = diags([2., 3.]), np.array([4., 9.])
        with using_product_backend(outer):
            solve_spd(a, rhs)
            with using_product_backend(inner):
                solve_spd(a, rhs)
            solve_spd(a, rhs)
        self.assertEqual((outer.calls, inner.calls), (2, 1))
