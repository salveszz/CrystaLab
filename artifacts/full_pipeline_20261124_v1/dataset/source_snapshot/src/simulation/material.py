# FILE PURPOSE: Verify one archived KDP structural reference and derive its SI
# density and crystal-frame facet normals. This bounded reader checks only the
# selected scalar CIF fields; it is not a general crystallographic importer.
# These reference calculations do not change the illustrative growth runtime,
# supply growth kinetics, or extend a 295 K record to other temperatures.

"""A traceable, fixed-condition tetragonal KDP structural reference."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import shlex
from zipfile import ZipFile

import numpy as np

from .units import KDP_MOLAR_MASS_KG_PER_MOL, finite_number


@dataclass(frozen=True)
class KDPStructure:
    source_id: str
    a_m: float
    c_m: float
    density_kg_per_m3: float
    reported_density_kg_per_m3: float
    temperature_K: float
    pressure_Pa: float | None

    def face_normal(self, hkl):
        """Return a unit normal in crystal axes using the reciprocal lattice."""
        values = np.asarray(hkl)
        if values.shape != (3,) or values.dtype.kind not in "iu" or not np.any(values):
            raise ValueError("hkl must contain three integers, not all zero")
        normal = values.astype(float) / [self.a_m, self.a_m, self.c_m]
        return normal / np.linalg.norm(normal)

    def crystal_mass_kg(self, physical_volume_m3):
        """Reference-point mass; the argument must already be physical SI volume."""
        return self.density_kg_per_m3 * finite_number(
            physical_volume_m3, "physical_volume_m3", nonnegative=True
        )


def _scalar(text, tag):
    """Read one required, single-line scalar in the hash-pinned reference CIF."""
    matches = re.findall(r"^" + re.escape(tag) + r"[ \t]+([^\n]+)$", text, re.M)
    if len(matches) != 1:
        raise ValueError(f"Expected one single-line {tag}")
    tokens = shlex.split(matches[0], comments=True)
    if len(tokens) != 1:
        raise ValueError(f"Expected one scalar value for {tag}")
    return tokens[0]


def load_kdp_structure(archive_path, reference):
    """Check archive identity, transcribed fields and density consistency."""
    if (
        type(reference.get("schema_version")) is not int
        or reference["schema_version"] != 1
    ):
        raise ValueError("Use structural reference schema v1")
    if (
        reference.get("entry") != "9010000.cif"
        or reference.get("source_id") != "COD-9010000"
    ):
        raise ValueError("This bounded check supports COD-9010000 only")
    archive_bytes = Path(archive_path).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != reference["archive_sha256"]:
        raise ValueError("COD archive SHA256 differs from the reviewed reference")
    with ZipFile(archive_path) as archive:
        data = archive.read(reference["entry"])
    if hashlib.sha256(data).hexdigest() != reference["entry_sha256"]:
        raise ValueError("COD entry SHA256 differs from the reviewed reference")
    text = data.decode("utf-8")
    if _scalar(text, "_journal_paper_doi") != reference["doi"]:
        raise ValueError("Reference DOI differs from the archived CIF")
    for tag, value in reference["cif_scalars"].items():
        if _scalar(text, tag) != value:
            raise ValueError(f"Transcription mismatch for {tag}")
    if (
        _scalar(text, "_chemical_formula_sum") != "H2 K O4 P"
        or _scalar(text, "_space_group_IT_number") != "122"
        or any(
            float(_scalar(text, f"_cell_angle_{a}")) != 90
            for a in ("alpha", "beta", "gamma")
        )
    ):
        raise ValueError("Reference must be tetragonal KH2PO4, space group 122")
    a, b, c = [
        finite_number(float(_scalar(text, f"_cell_length_{axis}")), axis, positive=True)
        * 1e-10
        for axis in "abc"
    ]
    volume = float(_scalar(text, "_cell_volume")) * 1e-30
    count = float(_scalar(text, "_cell_formula_units_Z"))
    reported = float(_scalar(text, "_exptl_crystal_density_diffrn")) * 1000
    temperature = float(_scalar(text, "_diffrn_ambient_temperature"))
    if a != b or count != 4 or not math.isclose(a * b * c, volume, rel_tol=1e-5):
        raise ValueError("Tetragonal cell dimensions, volume or Z disagree")
    density = count * KDP_MOLAR_MASS_KG_PER_MOL / (6.02214076e23 * volume)
    if not math.isclose(density, reported, rel_tol=1e-3):
        raise ValueError("Reported density disagrees with formula and unit cell")
    # Missing measurement pressure is preserved; it is not assumed atmospheric.
    if re.search(r"^_(?:cell_measurement|diffrn_ambient)_pressure\b", text, re.M):
        raise ValueError("Review newly supplied pressure before using this reference")
    return KDPStructure(
        reference["source_id"], a, c, density, reported, temperature, None
    )
