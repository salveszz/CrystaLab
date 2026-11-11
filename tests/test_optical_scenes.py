# FILE PURPOSE: Verify deterministic scene geometry, ancestry splits and rendered-mask auditing.
# Pure preparation checks run without Blender; the actual pinned Cycles smoke run
# is recorded separately after visual inspection rather than replaced with mocks.

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rendering.optical import prepare_scenes, select_proxy_mask


class OpticalSceneTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/synthetic_vision_smoke.json"
            ).read_text()
        )

    def test_deterministic_geometry_and_temporal_ancestry(self):
        scenes = prepare_scenes(self.protocol)
        self.assertEqual(scenes, prepare_scenes(self.protocol))
        self.assertEqual(len(scenes), 6)
        for scene in scenes:
            self.assertEqual(len(scene["frames"]), 3)
            volumes = [r["geometry_volume"] for r in scene["frames"]]
            self.assertTrue(all(b > a for a, b in zip(volumes, volumes[1:])))
            self.assertFalse(scene["frames"][0]["secondary_visible"])
            self.assertTrue(scene["frames"][1]["secondary_visible"])
            self.assertEqual(
                scene["frames"][0]["time_semantics"],
                "illustrative_geometry_step_not_physical_time",
            )
        self.assertNotEqual(
            scenes[0]["frames"][0]["vertices"], scenes[1]["frames"][0]["vertices"]
        )

    def test_related_scenes_cannot_cross_splits(self):
        d = deepcopy(self.protocol)
        d["scenes"][-1]["family_id"] = d["scenes"][0]["family_id"]
        with self.assertRaisesRegex(ValueError, "family"):
            prepare_scenes(d)

    def test_optical_mask_uses_geometry_anchor_instead_of_largest_reflection(self):
        proxy = np.zeros((30, 30), np.uint8)
        proxy[2:12, 2:12] = 255
        proxy[20:25, 20:25] = 200
        geometry = np.zeros_like(proxy)
        geometry[20:25, 20:25] = 255
        mask, discarded = select_proxy_mask(proxy, geometry, 32)
        self.assertEqual(int(np.count_nonzero(mask)), 25)
        self.assertEqual(discarded, 100)
        self.assertTrue(np.array_equal(mask > 0, geometry > 0))
        geometry.fill(0)
        mask, discarded = select_proxy_mask(proxy, geometry, 32)
        self.assertFalse(mask.any())
        self.assertEqual(discarded, 125)

    def test_bad_ids_seeds_classes_and_counts_are_rejected(self):
        for key, value in [
            ("id", "../escape"),
            ("seed", True),
            ("secondary_count", -1),
        ]:
            d = deepcopy(self.protocol)
            d["scenes"][0][key] = value
            with self.assertRaises(ValueError):
                prepare_scenes(d)
        d = deepcopy(self.protocol)
        d["classes"] = ["optical_quality"]
        with self.assertRaises(ValueError):
            prepare_scenes(d)


if __name__ == "__main__":
    unittest.main()
