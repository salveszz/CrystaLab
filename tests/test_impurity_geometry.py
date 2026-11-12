# FILE PURPOSE: Check cluster stoichiometry, protonation and strict scientific record output.
"""Geometry checks need no optional quantum runtime and create no experimental data."""

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.impurity_quantum import make_geometry,save


class ImpurityGeometryTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/"configs/kdp_impurity_quantum.json").read_text())

    def test_formal_ligand_exchange_preserves_atoms_and_charge(self):
        counts=[]
        for case in self.protocol["cases"]:
            atoms,_=make_geometry(self.protocol,case)
            counts.append(Counter(e for e,p in atoms))
        self.assertEqual(counts[0],Counter(Al=1,O=6,H=12))
        self.assertEqual(counts[1],Counter(Al=1,P=1,O=8,H=10))
        # Al(H2O)6^3+ + H2PO4^- -> Al(H2O)4(H2PO4)^2+ + 2 H2O
        self.assertEqual(counts[0]+Counter(H=2,P=1,O=4),counts[1]+Counter(H=4,O=2))
        self.assertEqual(self.protocol["cases"][0]["charge"]-1,self.protocol["cases"][1]["charge"])

    def test_six_oxygen_ligands_and_bonded_hydrogens_are_present(self):
        for case in self.protocol["cases"]:
            with self.subTest(case=case["case_id"]):
                atoms,donors=make_geometry(self.protocol,case)
                metal=atoms[0][1]
                oxygen=np.array([p for e,p in atoms if e=="O"])
                distances=np.linalg.norm(oxygen-metal,axis=1)
                self.assertEqual(np.count_nonzero(np.isclose(distances,1.9)),6)
                for element,point in atoms:
                    if element=="H":
                        self.assertAlmostEqual(float(np.linalg.norm(oxygen-point,axis=1).min()),.96,places=10)
                self.assertTrue(all(atoms[i][0]=="O" for i in donors))

    def test_charge_or_spin_changes_cannot_silently_change_electron_sector(self):
        case=copy.deepcopy(self.protocol["cases"][1])
        case["charge"]+=1
        with self.assertRaisesRegex(ValueError,"electron count"):
            make_geometry(self.protocol,case)

    def test_numpy_scalar_serialization_keeps_nonfinite_values_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"record.json"
            save(path,dict(index=np.int64(2),valid=np.bool_(True),energy=np.float64(-1.25)))
            self.assertEqual(json.loads(path.read_text()),dict(index=2,valid=True,energy=-1.25))
            with self.assertRaises(ValueError):
                save(Path(directory)/"invalid.json",dict(energy=np.float64(np.nan)))
