# FILE PURPOSE: Produce train-only RGB/mask review panels for a completed optical dataset.
# Preview generation never creates an acceptance receipt or inspects held-out
# images for tuning. Geometric and refracted emission masks retain their meanings.
"""Inspect first/last training frames before accepting synthetic proxy labels."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rendering.optical import audit_optical
from vision.video import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit = audit_optical(a.source)
    rows = [
        json.loads(line)
        for line in (a.source / "samples.jsonl").read_text().splitlines()
    ]
    scenes = {}
    for row in rows:
        if row["split"] == "train":
            scenes.setdefault(row["scene_id"], []).append(row)
    selected = []
    for name in sorted(scenes):
        frames = sorted(scenes[name], key=lambda row: row["frame_index"])
        selected.append(frames[0])
        if len(frames) > 1:
            selected.append(frames[-1])
    if not selected:
        raise ValueError("No training scenes to preview")
    a.output.mkdir(parents=True, exist_ok=False)
    panels = []
    records = []
    for index, row in enumerate(selected):
        image = cv2.imread(str(a.source / row["image_path"]))
        overlay = image.copy()
        hashes = {row["image_path"]: file_sha256(a.source / row["image_path"])}
        for obj in row["objects"]:
            mask = cv2.imread(str(a.source / obj["mask_path"]), 0)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                overlay,
                contours,
                -1,
                [(0, 190, 0), (0, 210, 255), (255, 100, 0)][obj["class_id"]],
                1,
            )
            hashes[obj["mask_path"]] = file_sha256(a.source / obj["mask_path"])
        cv2.putText(
            overlay,
            row["sample_id"],
            (3, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (20, 20, 20),
            1,
        )
        path = f"{index:03d}.png"
        if not cv2.imwrite(str(a.output / path), overlay):
            raise OSError("Cannot save review panel")
        panels.append(overlay)
        records.append(
            dict(
                sample_id=row["sample_id"],
                preview_path=path,
                source_artifact_sha256=hashes,
            )
        )
    sheets = []
    for start in range(0, len(panels), 12):
        page = panels[start : start + 12]
        while len(page) % 3:
            page.append(np.full_like(page[0], 240))
        contact = np.concatenate(
            [np.concatenate(page[i : i + 3], axis=1) for i in range(0, len(page), 3)],
            axis=0,
        )
        name = f"training_overlays_{start // 12:02d}.png"
        if not cv2.imwrite(str(a.output / name), contact):
            raise OSError("Cannot save contact sheet")
        sheets.append(name)
    report = dict(
        schema="crystalab_optical_preview_v1",
        source_manifest_sha256=file_sha256(a.source / "manifest.json"),
        audit=audit,
        records=records,
        contact_sheets=sheets,
        decision="not_reviewed_by_preview",
        legend=dict(
            seed="green", secondary_crystal="yellow", inclusion_candidate="blue"
        ),
        scope="Training frames only; preview does not accept labels or establish real optics",
    )
    (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="preview_generated", images=len(records), contact_sheets=sheets
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
