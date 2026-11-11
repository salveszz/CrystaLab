# FILE PURPOSE: Analyze saved learned observations for persistent candidate inspection events.
# Original clocks, uncertainty and immutable source identities remain explicit.
"""Track candidate masks and preserve startup/new/gap distinctions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.candidates import CandidateConfig
from vision.candidate_replay import analyze_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze_candidates(
        a.replay, CandidateConfig(**json.loads(a.config.read_text())), a.output
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "status",
                    "processed_frames",
                    "persistent_candidate_events",
                    "event_kinds",
                    "issues",
                    "usable_source_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
