# FILE PURPOSE: Build explicitly artificial image bundles for annotation UI/API checks.
"""No downloaded image or actual human review is used by this fixture."""

import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from vision.annotations import prepare_bundle
from vision.baseline import BaselineConfig
from vision.data import SCHEMA
from vision.video import file_sha256


def prepare_review_fixture(root, count=3):
    root = Path(root)
    index = root / "index"
    index.mkdir()
    archive_path = root / "fixture.zip"
    rows = []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for number in range(count):
            image = np.full((480, 640, 3), 224, np.uint8)
            points = np.array([[200, 130], [370, 100], [430, 300], [260, 350]])
            points[:, 0] += number * 5
            cv2.fillPoly(image, [points], (90, 100, 105))
            cv2.polylines(image, [points], True, (35, 40, 45), 3)
            cv2.circle(image, (530, 130), 25, (160, 170, 175), 3)
            cv2.putText(
                image,
                "AUTOMATED QA FIXTURE - NOT REAL DATA",
                (18, 445),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 80, 80),
                1,
            )
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise OSError("Cannot encode fixture")
            data = encoded.tobytes()
            member = f"frame_{number}.jpg"
            archive.writestr(member, data)
            rows.append(
                dict(
                    sample_id=f"fixture:run:{number}",
                    run_id="run",
                    family_id="fixture_family",
                    split="train",
                    frame_index=number,
                    evidence_kind="fixture",
                    archive_member=member,
                    image_sha256=hashlib.sha256(data).hexdigest(),
                    image_width=640,
                    image_height=480,
                )
            )
    (index / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (index / "manifest.json").write_text(
        json.dumps(
            dict(
                schema=SCHEMA,
                status="complete",
                archive_path=str(archive_path),
                index_sha256=file_sha256(index / "images.jsonl"),
                protocol=dict(
                    archive_sha256=file_sha256(archive_path),
                    material="QA_fixture",
                    license="Generated test pixels only",
                ),
            )
        )
    )
    bundle = root / "fixture_bundle"
    prepare_bundle(
        index,
        "run",
        bundle,
        frame_indices=list(range(count)),
        baseline=BaselineConfig((310, 230), threshold=120, morphology_size=1),
        proposal_width=640,
    )
    return bundle
