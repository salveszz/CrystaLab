# FILE PURPOSE: Construct independent pixel/event labels and intentionally imperfect fixture predictions.
# These masks are supplied test inputs, not learned-model outputs or physical
# growth data. Startup false positives, transient noise and occlusion stay explicit.
"""Generate a reproducible temporal candidate integration fixture."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.learned import LearnedObservationConfig, LearnedSeedTracker
from vision.replay import replay
from vision.training_data import save_json
from vision.video import file_sha256


def geometry(index):
    seed = np.zeros((240, 320), np.uint8)
    seed[60:100, 40:90] = 255
    startup = np.zeros_like(seed)
    cv2.circle(startup, (240, 160), 8, 255, -1)
    values = [("seed", 0, seed), ("startup_bubble_false_positive", 1, startup)]
    if index == 5:
        transient = np.zeros_like(seed)
        transient[35:44, 185:194] = 255
        values.append(("transient_false_positive", 1, transient))
    if index >= 10:
        crystal = np.zeros_like(seed)
        crystal[105:127, 145:165] = 255
        values.append(("new_crystal_A", 1, crystal))
    if index >= 22:
        crystal = np.zeros_like(seed)
        crystal[180:200, 135:151] = 255
        values.append(("crystal_C_after_unknown", 1, crystal))
    if index >= 28:
        inclusion = np.zeros_like(seed)
        inclusion[72:78, 52:58] = 255
        values.append(("new_inclusion_B", 2, inclusion))
    return [] if 20 <= index < 24 else values


def generate_candidate_fixture(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "truth_masks").mkdir()
    frames = 48
    fps = 10.0
    writer = cv2.VideoWriter(
        str(output / "sequence.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240)
    )
    if not writer.isOpened():
        raise OSError("Fixture video writer unavailable")
    try:
        for i in range(frames):
            image = np.full((240, 320, 3), 220, np.uint8)
            if 20 <= i < 24:
                image[:] = 180
            else:
                for name, class_id, mask in geometry(i):
                    image[mask > 0] = 0 if class_id == 2 else 40
            writer.write(image)
    finally:
        writer.release()
    maps = [
        dict(
            output_frame_index=i,
            display_time_s=i / fps,
            source_sample_id=f"candidate_fixture:{i}",
            fixture_step=i,
            acquisition_time_s=None,
            time_semantics="software_fixture_display_clock",
        )
        for i in range(frames)
    ]
    (output / "frame_map.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in maps)
    )
    info = dict(
        evidence_kind="fixture",
        material="none",
        source_id="candidate_policy_fixture_v1",
        split="development",
        frame_map_sha256=file_sha256(output / "frame_map.jsonl"),
        limitations="Analytic RGB and supplied imperfect masks; no learned detector or physical defect/clock evidence",
    )
    save_json(output / "source_info.json", info)
    config = LearnedObservationConfig((65, 80))

    class FixtureObserver(LearnedSeedTracker):
        def measure(self, image, timestamp):
            record, mask = super().measure(image, timestamp)
            record["method"] = "fixture_supplied_candidate_masks"
            record["display_method"] = "FIXTURE / supplied masks / no model accuracy"
            return record, mask

    def factory(folder):
        index = 0

        def predict(image):
            nonlocal index
            values = [
                dict(
                    class_id=class_id,
                    class_name=["seed", "secondary_crystal", "inclusion_candidate"][
                        class_id
                    ],
                    mask=mask,
                    score=0.9,
                    score_is_calibrated=False,
                    fixture_object_id=name,
                )
                for name, class_id, mask in geometry(index)
            ]
            index += 1
            return values

        observer = FixtureObserver(config, predict)
        observer.info = dict(
            method="fixture_supplied_candidate_masks", model_accuracy_claim=False
        )
        observer.source_files = [Path(__file__).resolve()]
        return observer

    observation = replay(
        output / "sequence.avi",
        config,
        output / "observations",
        run_id="candidate_policy_fixture_v1",
        source_info=info,
        observer_factory=factory,
        frame_map=output / "frame_map.jsonl",
    )
    expected = [
        (
            "new_crystal_A",
            10,
            0.9,
            1.0,
            "crystal_candidate",
            "appeared_after_clear_view",
        ),
        (
            "crystal_C_after_unknown",
            24,
            1.9,
            2.4,
            "crystal_candidate",
            "appeared_after_unknown_interval",
        ),
        (
            "new_inclusion_B",
            28,
            2.7,
            2.8,
            "inclusion_candidate",
            "appeared_after_clear_view",
        ),
    ]
    events = []
    for name, index, low, high, kind, origin in expected:
        mask = next(mask for label, _, mask in geometry(index) if label == name)
        relative = f"truth_masks/{name}.png"
        if not cv2.imwrite(str(output / relative), mask):
            raise OSError("Cannot save independent event mask")
        events.append(
            dict(
                event_id=name,
                candidate_type=kind,
                origin=origin,
                onset_interval_source_s=[low, high],
                first_visible_frame_index=index,
                first_visible_mask_path=relative,
                mask_sha256=file_sha256(output / relative),
                uncertain=False,
            )
        )
    truth = dict(
        schema="crystalab_visible_event_annotations_v1",
        evidence_kind="fixture",
        clock="source_video_time_s",
        source_video_sha256=file_sha256(output / "sequence.avi"),
        events=events,
        negative_intervals=[
            dict(candidate_type="crystal_candidate", start_s=0.0, end_s=0.9),
            dict(candidate_type="inclusion_candidate", start_s=0.0, end_s=2.7),
        ],
        label_semantics="Independent constructed visible shapes; full-view occlusion steps 20-23, startup bubble/transient noise are not positive targets",
    )
    save_json(output / "annotations.json", truth)
    save_json(
        output / "manifest.json",
        dict(
            schema="crystalab_candidate_fixture_v1",
            status="complete",
            frames=frames,
            evidence_kind="fixture",
            source_video_sha256=truth["source_video_sha256"],
            independent_visible_events=len(events),
            supplied_observation_valid_frames=observation["valid_observations"],
        ),
    )
    return dict(
        status="complete",
        frames=frames,
        events=len(events),
        observation_valid=observation["valid_observations"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(generate_candidate_fixture(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
