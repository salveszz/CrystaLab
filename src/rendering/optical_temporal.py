# FILE PURPOSE: Define bounded illustrative growth and object-presence schedules for optical clips.
# Appearance means constructed presence, not proven optical visibility or nucleation.
# Legacy three-frame protocols retain their original geometry and object behavior.
"""Pure temporal scene contracts shared by preparation and artifact auditing."""

import math

SCHEMA = "crystalab_optical_temporal_v1"


def temporal_profile(protocol, case):
    schema = protocol.get("temporal_schema")
    if schema is None:
        if "face_offset_by_frame" in case or "presence_intervals" in case:
            raise ValueError("Temporal schedules require an explicit temporal schema")
        return None
    if schema != SCHEMA:
        raise ValueError("Unsupported optical temporal schema")
    count = protocol["frames_per_scene"]
    offsets = case.get("face_offset_by_frame")
    if (
        not isinstance(offsets, list)
        or len(offsets) != count
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 0.5
            for v in offsets
        )
        or offsets[0] != 0
    ):
        raise ValueError(
            "Declare one finite nonnegative face offset per frame, beginning at zero"
        )
    allowed = {
        f"{kind}_{i}"
        for kind, key in (
            ("secondary", "secondary_count"),
            ("inclusion", "inclusion_count"),
            ("bubble", "bubble_count"),
        )
        for i in range(case[key])
    }
    intervals = case.get("presence_intervals")
    if not isinstance(intervals, dict) or set(intervals) != allowed:
        raise ValueError(
            "Declare exactly one presence interval for each non-seed object"
        )
    for interval in intervals.values():
        if interval is None:
            continue
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(v) is not int for v in interval)
            or not 0 <= interval[0] < interval[1] <= count
        ):
            raise ValueError(
                "Presence intervals use bounded inclusive-start/exclusive-end frame indices"
            )
    return dict(offsets=list(offsets), intervals=intervals)


def present_ids(profile, index):
    return sorted(
        ["seed"]
        + [
            name
            for name, interval in profile["intervals"].items()
            if interval is not None and interval[0] <= index < interval[1]
        ]
    )


def constructed_targets(protocol, scene, frame):
    """Reconstruct target identities from the declared protocol, not model predictions."""
    profile = temporal_profile(protocol, scene)
    wanted = {"seed": 0}
    if profile is None:
        wanted.update({f"inclusion_{i}": 2 for i in range(scene["inclusion_count"])})
        if frame["secondary_visible"]:
            wanted.update(
                {f"secondary_{i}": 1 for i in range(scene["secondary_count"])}
            )
        return wanted
    expected = present_ids(profile, frame["index"])
    if (
        frame.get("present_object_ids") != expected
        or frame.get("face_offset") != profile["offsets"][frame["index"]]
        or frame["secondary_visible"]
        != any(name.startswith("secondary_") for name in expected)
    ):
        raise ValueError(
            "Frame geometry/presence schedule differs from the temporal protocol"
        )
    for name in expected:
        if name.startswith("secondary_"):
            wanted[name] = 1
        elif name.startswith("inclusion_"):
            wanted[name] = 2
    return wanted
