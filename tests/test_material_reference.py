# FILE PURPOSE: Check the pinned COD reference against independent cell-mass and
# reciprocal-lattice identities, and reject source/transcription corruption.

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.material import load_kdp_structure


class MaterialReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(
            (ROOT / "configs/kdp_material_reference.json").read_text()
        )
        self.structure = load_kdp_structure(ROOT / "data/COD.zip", self.reference)

    def test_cell_mass_and_si_density(self):
        expected = 4 * 0.1360855 / (6.02214076e23 * 387.293e-30)
        self.assertAlmostEqual(self.structure.density_kg_per_m3, expected)
        self.assertAlmostEqual(self.structure.crystal_mass_kg(1e-9), expected * 1e-9)
        self.assertEqual(self.structure.reported_density_kg_per_m3, 2334)
        self.assertEqual(self.structure.temperature_K, 295)
        self.assertIsNone(self.structure.pressure_Pa)

    def test_face_normal_uses_reciprocal_lattice(self):
        normal = self.structure.face_normal([1, 0, 1])
        self.assertAlmostEqual(normal[2] / normal[0], 7.4521 / 6.974)
        self.assertAlmostEqual(np.linalg.norm(normal), 1)
        self.assertGreater(abs(normal[2] - 2**-0.5), 0.02)
        np.testing.assert_allclose(self.structure.face_normal([-1, 0, -1]), -normal)

    def test_invalid_miller_indices_and_volume(self):
        for hkl in ([0, 0, 0], [1, 0], [True, False, True], [1.0, 0, 1.0]):
            with self.assertRaises(ValueError):
                self.structure.face_normal(hkl)
        for volume in (-1, float("nan"), True):
            with self.assertRaises(ValueError):
                self.structure.crystal_mass_kg(volume)

    def test_archive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.zip"
            path.write_bytes((ROOT / "data/COD.zip").read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "archive SHA256"):
                load_kdp_structure(path, self.reference)

    def test_transcription_identity(self):
        changed = deepcopy(self.reference)
        changed["cif_scalars"]["_cell_length_c"] = "7.0"
        with self.assertRaisesRegex(ValueError, "Transcription mismatch"):
            load_kdp_structure(ROOT / "data/COD.zip", changed)
        changed = deepcopy(self.reference)
        changed["doi"] = "wrong-paper"
        with self.assertRaisesRegex(ValueError, "DOI differs"):
            load_kdp_structure(ROOT / "data/COD.zip", changed)


if __name__ == "__main__":
    unittest.main()
