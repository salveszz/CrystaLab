# FILE PURPOSE: Observe a file/selected camera with a baseline or verified local segmentation model.
# Export source timestamps, masks, validity events and overlays without training,
# quality inference or actuator commands. Existing output directories are refused.
"""Run timestamped baseline or local learned seed observation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.baseline import BaselineConfig
from vision.replay import replay
from vision.learned import LearnedObservationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--camera", type=int)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-info", type=Path)
    parser.add_argument(
        "--model-run", type=Path, help="Verified local YOLOv8-seg training run"
    )
    parser.add_argument(
        "--frame-map", type=Path, help="Hashed derivative-to-original frame map"
    )
    parser.add_argument("--clock", choices=("container", "nominal_fps", "arrival"))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--experiment-origin-s", type=float)
    parser.add_argument("--experiment-seconds-per-video-second", type=float)
    args = parser.parse_args()
    camera = args.camera is not None
    settings = json.loads(args.config.read_text())
    observer_factory = None
    if args.model_run is None:
        config = BaselineConfig(**settings)
    else:
        from models.observation import YOLOObserver

        config = LearnedObservationConfig(**settings)

        def observer_factory(output):
            return YOLOObserver(config, args.model_run, output / "settings")

    result = replay(
        args.camera if camera else args.video,
        config,
        args.output,
        run_id=args.run_id,
        camera=camera,
        clock=args.clock or ("arrival" if camera else "container"),
        fps=args.fps,
        max_frames=args.max_frames,
        sample_every=args.sample_every,
        source_info=None
        if args.source_info is None
        else json.loads(args.source_info.read_text()),
        experiment_origin_s=args.experiment_origin_s,
        experiment_seconds_per_video_second=args.experiment_seconds_per_video_second,
        observer_factory=observer_factory,
        frame_map=args.frame_map,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "processed_frames",
                    "valid_observations",
                    "invalid_observations",
                    "inspection_observations",
                    "raw_candidate_instances",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
