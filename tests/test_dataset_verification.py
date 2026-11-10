# FILE PURPOSE: Check the real synthetic-data refinement runner's evidence and gates.
# Small actual 3D runs verify fixed sampling depth, exact horizon comparisons,
# shared-case reuse and identity-checked resume; no model or surrogate is fitted.
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from check_dataset import study  # noqa: E402
from environment.dataset import read_json, verify_episode  # noqa: E402
from test_dataset import protocol  # noqa: E402


class DatasetVerificationTests(unittest.TestCase):
    def test_actual_refinement_evidence_and_unchanged_resume(self):
        design = protocol()
        design["simulation"].update(
            nx=5,
            ny=5,
            nz=5,
            duration=1.5,
            dt=0.5,
            concentration_noise=0,
            boundary_layer_cells=0.375,
        )
        design["recording"].update(
            sample_interval=0.5,
            decision_interval=0.5,
            forecast_horizon=0.5,
            sensor_delay=0.5,
        )
        design["recipes"] = design["recipes"][:1]
        settings = dict(
            groups=["a"],
            recipes=["baseline"],
            grids=[[3, 3, 3], [4, 4, 4], [5, 5, 5]],
            timesteps=[0.5, 0.25, 0.125],
            boundary_layer_thickness=0.75,
            relative_tolerance=0.02,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study"
            result = study(design, settings, output)
            self.assertEqual(result["cases"], 5)
            self.assertEqual(len(result["comparisons"]), 4)
            comparisons = result["comparisons"]
            gate = all(
                c["pass_gate"]
                for c in comparisons
                if c["pair"] == [1, 2] or c["kind"] == "dt"
            )
            self.assertEqual(result["status"], "pass" if gate else "fail")
            gains = []
            for suffix in ["grid_1", "grid_2"]:
                folder = output / "episodes" / ("a__baseline__" + suffix)
                metadata = read_json(folder / "episode.json")
                case = metadata["case"]
                cfg = case["simulation"]
                self.assertAlmostEqual(
                    cfg["boundary_layer_cells"]
                    * max(
                        cfg["width"] / cfg["nx"],
                        cfg["depth"] / cfg["ny"],
                        cfg["height"] / cfg["nz"],
                    ),
                    0.75,
                )
                rows = verify_episode(folder, case)
                gains.append(
                    np.array([r["labels"]["volume_gain"] for r in rows["samples"]])
                )
            expected = np.max(
                abs(gains[0] - gains[1]) / np.maximum(abs(gains[1]), 1e-12)
            )
            self.assertEqual(comparisons[1]["max_relative_label_error"], expected)
            source = output / "source_snapshot/src/simulation/engine.py"
            self.assertEqual(
                source.read_bytes(), (ROOT / "src/simulation/engine.py").read_bytes()
            )
            self.assertEqual(result, study(design, settings, output, resume=True))
            changed = dict(settings, relative_tolerance=0.03)
            with self.assertRaisesRegex(ValueError, "resume requires"):
                study(design, changed, output, resume=True)
            with self.assertRaises(FileExistsError):
                study(design, settings, output)


if __name__ == "__main__":
    unittest.main()
