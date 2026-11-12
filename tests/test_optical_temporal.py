# FILE PURPOSE: Verify temporal scene schedules, legacy compatibility and independent target reconstruction.
"""Pure checks do not stand in for a real Blender render or event evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from rendering.optical import prepare_scenes
from rendering.optical_temporal import SCHEMA, temporal_profile, constructed_targets


class OpticalTemporalTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/synthetic_vision_smoke.json"
            ).read_text()
        )
        self.protocol["frames_per_scene"] = 8
        self.protocol["temporal_schema"] = SCHEMA
        for case in self.protocol["scenes"]:
            case["face_offset_by_frame"] = [0, 0.01, 0.02, 0.02, 0.02, 0.04, 0.06, 0.08]
            case["presence_intervals"] = {
                f"{kind}_{i}": [3, 8]
                for kind, key in (
                    ("secondary", "secondary_count"),
                    ("inclusion", "inclusion_count"),
                    ("bubble", "bubble_count"),
                )
                for i in range(case[key])
            }

    def test_growth_plateau_and_exact_scheduled_presence(self):
        scenes = prepare_scenes(self.protocol)
        for scene in scenes:
            volumes = [f["geometry_volume"] for f in scene["frames"]]
            self.assertAlmostEqual(volumes[2], volumes[3])
            self.assertAlmostEqual(volumes[3], volumes[4])
            self.assertGreater(volumes[5], volumes[4])
            self.assertEqual(scene["frames"][0]["present_object_ids"], ["seed"])
            for index in range(8):
                wanted = constructed_targets(
                    self.protocol, scene, scene["frames"][index]
                )
                self.assertEqual(
                    len(wanted),
                    1
                    + (
                        scene["secondary_count"] + scene["inclusion_count"]
                        if index >= 3
                        else 0
                    ),
                )
        self.assertEqual(scenes, prepare_scenes(self.protocol))

    def test_object_end_and_disabled_presence_are_not_silently_extended(self):
        case = self.protocol["scenes"][0]
        for name in case["presence_intervals"]:
            case["presence_intervals"][name] = [1, 4]
        case["presence_intervals"]["secondary_0"] = None
        scene = prepare_scenes(self.protocol)[0]
        for index in (0, 4, 7):
            self.assertEqual(scene["frames"][index]["present_object_ids"], ["seed"])
        self.assertNotIn("secondary_0", scene["frames"][2]["present_object_ids"])

    def test_rehashed_frame_presence_or_growth_cannot_override_protocol(self):
        scene = prepare_scenes(self.protocol)[0]
        for change in (
            dict(present_object_ids=["seed"]),
            dict(face_offset=0.49),
            dict(secondary_visible=False),
        ):
            frame = dict(scene["frames"][4], **change)
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "differs"),
            ):
                constructed_targets(self.protocol, scene, frame)

    def test_bad_growth_and_presence_contracts_fail_before_rendering(self):
        for offsets in (
            [0] * 7,
            [1] * 8,
            [0, 0, 0, 0, 0, 0, 0, float("nan")],
            [0, 0, 0, 0, 0, 0, 0, True],
        ):
            protocol = deepcopy(self.protocol)
            protocol["scenes"][0]["face_offset_by_frame"] = offsets
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                prepare_scenes(protocol)
        for interval in (
            [4, 3],
            [-1, 3],
            [1, 9],
            [True, 5],
            [1.5, 5],
            [[1, 3], [4, 8]],
        ):
            protocol = deepcopy(self.protocol)
            protocol["scenes"][0]["presence_intervals"]["secondary_0"] = interval
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                prepare_scenes(protocol)
        protocol = deepcopy(self.protocol)
        protocol["scenes"][0]["presence_intervals"]["seed"] = [0, 8]
        with self.assertRaises(ValueError):
            prepare_scenes(protocol)

    def test_legacy_protocol_retains_original_geometry_and_presence(self):
        legacy = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/synthetic_vision_smoke.json"
            ).read_text()
        )
        scenes = prepare_scenes(legacy)
        self.assertIsNone(temporal_profile(legacy, legacy["scenes"][0]))
        self.assertNotIn("present_object_ids", scenes[0]["frames"][0])
        self.assertNotIn("face_offset", scenes[0]["frames"][0])
        for scene in scenes:
            for frame in scene["frames"]:
                wanted = constructed_targets(legacy, scene, frame)
                self.assertEqual(
                    len(wanted),
                    1
                    + scene["inclusion_count"]
                    + (scene["secondary_count"] if frame["index"] > 0 else 0),
                )
        broken = deepcopy(legacy)
        broken["scenes"][0]["face_offset_by_frame"] = [0, 0.1, 0.2]
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare_scenes(broken)


if __name__ == "__main__":
    unittest.main()
