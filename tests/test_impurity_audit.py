# FILE PURPOSE: Check independent fermion ordering and spin before auditing real runs.
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "scripts/audit_impurity_quantum.py"
SPEC = importlib.util.spec_from_file_location("impurity_audit", MODULE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IndependentFermionTests(unittest.TestCase):
    def test_single_orbital_hubbard_energy_counts_pair_once(self):
        matrix = AUDIT.fermion_matrix(np.array([[-.4]]), np.array([[[[.7]]]]))
        np.testing.assert_allclose(matrix, np.diag([0, -.4, -.4, -.1]), atol=1e-15)

    def test_fermionic_creation_order_has_exchange_sign(self):
        forward = AUDIT.apply_ladder(0, [(0, True), (2, True)])
        reverse = AUDIT.apply_ladder(0, [(2, True), (0, True)])
        self.assertEqual(forward[0], reverse[0])
        self.assertEqual(forward[1], -reverse[1])
        self.assertEqual(AUDIT.apply_ladder(1, [(0, True)]), (None, 0))

    def test_spin_operator_distinguishes_double_occupation_and_triplet(self):
        single = AUDIT.spin_matrix(1)
        np.testing.assert_allclose(single, np.diag([0, .75, .75, 0]))
        two = AUDIT.spin_matrix(2)
        self.assertEqual(two[3, 3], 2.)  # two alpha electrons
        self.assertEqual(two[5, 5], 0.)  # paired alpha/beta in orbital zero
