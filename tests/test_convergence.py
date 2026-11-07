# WHAT THESE TESTS CHECK
# The study changes resolution without changing the world-space seed, physical
# coefficients or sampling thickness. Whole histories, not just final numbers,
# participate in its gate. Exercise export/resume with real tiny simulations;
# altered settings must not reuse cached results. Tests do not require the
# expensive production-resolution convergence study to pass its numerical gate.
# Independent process workers must reproduce serial histories and face motion;
# scheduling may change runtime, while experiment results retain protocol order.

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_convergence import assess, build_cases, compare_cases, run_study


class ConvergenceStudyTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/default.json").read_text())
        self.protocol = json.loads((ROOT / "configs/convergence.json").read_text())

    def test_only_resolution_changes_and_sampling_stays_physical(self):
        original = deepcopy(self.base)
        cases, spatial, temporal = build_cases(self.base, self.protocol)
        self.assertEqual(self.base, original)
        self.assertEqual(len(cases), 5)
        self.assertEqual(spatial[-1], temporal[0])
        controlled = {"nx", "ny", "nz", "dt", "boundary_layer_cells"}
        reference = cases[0]["config"]
        for case in cases:
            c = case["config"]
            spacing = max(
                c[name] / c[count]
                for name, count in [("width", "nx"), ("depth", "ny"), ("height", "nz")]
            )
            self.assertAlmostEqual(c["boundary_layer_cells"] * spacing, 0.75)
            self.assertEqual(c["concentration_noise"], 0.0)
            self.assertEqual(c["defect_rate"], 0.0)
            self.assertEqual(
                {k: v for k, v in c.items() if k not in controlled},
                {k: v for k, v in reference.items() if k not in controlled},
            )

    def test_invalid_refinement_protocol_is_rejected(self):
        for key, value in [
            ("duration", 0),
            ("boundary_layer_thickness", -1),
            ("grids", [[8, 8, 8], [7, 8, 8], [10, 10, 10]]),
            ("timesteps", [0.5, 1.0, 0.25]),
            ("relative_change_tolerance", float("nan")),
        ]:
            protocol = deepcopy(self.protocol)
            protocol[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_cases(self.base, protocol)

    def test_real_study_export_and_safe_resume(self):
        protocol = deepcopy(self.protocol)
        protocol.update(
            duration=0.5,
            grids=[[6, 6, 6], [8, 8, 8], [10, 10, 10]],
            timesteps=[0.5, 0.25, 0.125],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_study(self.base, protocol, output)
            self.assertEqual(len(report["results"]), 5)
            self.assertTrue(report["conservation_pass"])
            self.assertTrue((output / "study.json").exists())
            self.assertTrue((output / "report.json").exists())
            self.assertIn("refinement", (output / "report.md").read_text())
            for case in report["results"]:
                self.assertEqual(case["final"]["time"], 0.5)
                self.assertTrue((output / case["id"] / "measurements.csv").exists())
                with np.load(output / case["id"] / "final_fields.npz") as fields:
                    self.assertEqual(fields["concentration"].ndim, 3)
            with patch(
                "check_convergence.run_case",
                side_effect=AssertionError("Should reuse cached cases"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    resumed = run_study(self.base, protocol, output, resume=True)
                self.assertEqual(
                    [r["id"] for r in resumed["results"]],
                    [r["id"] for r in report["results"]],
                )
            protocol["boundary_layer_thickness"] = 0.5
            with self.assertRaisesRegex(ValueError, "resume"):
                run_study(self.base, protocol, output, resume=True)

    def test_parallel_cases_match_serial_measurements(self):
        protocol = deepcopy(self.protocol)
        protocol.update(
            duration=0.25,
            grids=[[6, 6, 6], [8, 8, 8], [10, 10, 10]],
            timesteps=[0.25, 0.125, 0.0625],
        )
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                serial = run_study(self.base, protocol, Path(directory) / "serial")
                parallel = run_study(
                    self.base, protocol, Path(directory) / "parallel", jobs=2
                )
            for a, b in zip(serial["results"], parallel["results"]):
                self.assertEqual(a["id"], b["id"])
                np.testing.assert_allclose(
                    a["face_offsets"], b["face_offsets"], atol=1e-12
                )
                for key in ("volume", "temperature", "concentration", "liquid_level"):
                    np.testing.assert_allclose(
                        [row[key] for row in a["history"]],
                        [row[key] for row in b["history"]],
                        atol=1e-12,
                    )

    def _case(self, name, final_volume=2.0, middle_volume=1.5):
        # Initial solid/level offsets are large enough that normalizing by final
        # volume/level instead of GAIN would incorrectly hide a growth discrepancy.
        initial = dict(
            time=0.0, volume=1.0, surface_area=6.0, liquid_level=6.0, concentration=1.35
        )
        middle = dict(
            time=0.5,
            volume=middle_volume,
            surface_area=6.0,
            liquid_level=6.005,
            concentration=1.34,
        )
        final = dict(
            time=1.0,
            volume=final_volume,
            surface_area=6.0,
            liquid_level=6.01,
            concentration=1.33,
        )
        return dict(
            id=name,
            initial=initial,
            final=final,
            history=[initial, middle, final],
            face_offsets=[[5.0], [5.05], [5.1]],
            initial_face_offsets=[5.0],
            active_faces=[0],
            max_mass_relative_error=0.0,
            max_carrier_relative_error=0.0,
            max_continuity_cell_error=0.0,
            min_concentration=0.0,
            exact_final_time=True,
        )

    def test_comparison_normalizes_growth_and_catches_transient_error(self):
        fine = self._case("fine")
        coarse = self._case("coarse", final_volume=2.1)
        comparison = compare_cases(coarse, fine)
        self.assertAlmostEqual(comparison["volume_gain_relative_change"], 0.1)
        transient = self._case("transient", middle_volume=1.8)
        comparison = compare_cases(transient, fine)
        self.assertEqual(comparison["volume_gain_relative_change"], 0.0)
        self.assertAlmostEqual(comparison["volume_trajectory_relative_change"], 0.3)

    def test_failed_gate_is_preserved_and_independent_of_conservation(self):
        cases = [
            self._case("coarse", 2.2),
            self._case("middle", 2.1),
            self._case("fine"),
            self._case("time_middle"),
            self._case("time_fine"),
        ]
        report = assess(
            cases,
            ["coarse", "middle", "fine"],
            ["fine", "time_middle", "time_fine"],
            self.protocol,
        )
        self.assertTrue(report["conservation_pass"])
        self.assertFalse(report["spatial_pass"])
        self.assertTrue(report["temporal_pass"])
        self.assertFalse(report["overall_pass"])


if __name__ == "__main__":
    unittest.main()
