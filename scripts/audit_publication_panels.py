# FILE PURPOSE: Independently verify publication crop pixels, metadata and uncertain sheets.
"""Pillow reconstruction does not import the OpenCV producer or its audit."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(index, bundle, protocol_path):
    index, bundle = Path(index), Path(bundle)
    protocol = json.loads(Path(protocol_path).read_text())
    manifest = json.loads((index / "manifest.json").read_text())
    sheets = json.loads((bundle / "manifest.json").read_text())
    cache = Path(manifest["cache_path"])
    check(manifest["status"] == "complete" and sheets["status"] == "ready_for_review", "Incomplete import")
    check(protocol == manifest["protocol"] == sheets["source_protocol"], "External publication protocol differs")
    check(sha(protocol_path) == manifest["protocol_sha256"] == sha(index / "protocol.json"), "Protocol hash differs")
    check(sha(index / "images.jsonl") == manifest["index_sha256"] == sheets["source_index_sha256"], "Index changed")
    check(sha(index / "source_snapshot/publication_panels.py") == manifest["source_sha256"], "Producer snapshot changed")
    for asset in protocol["metadata"] + protocol["figures"]:
        check(sha(cache / asset["file"]) == asset["sha256"], "Publication source bytes differ")
    xml = ET.parse(cache / "article.xml")
    doi = xml.find('.//article-id[@pub-id-type="doi"]')
    license_element = xml.find('.//license')
    check(doi is not None and doi.text == protocol["doi"], "Source DOI differs")
    check(license_element is not None and protocol["license_url"] in ET.tostring(license_element, encoding="unicode"), "Source license differs")
    official = json.loads((cache / "pmc_metadata.json").read_text())
    check(official["doi"] == protocol["doi"] and official["license_code"] == "CC BY" and not official["is_retracted"], "Official metadata differs")
    for url in official["media_urls"] + [official["xml_url"]]:
        parsed = urlparse(url)
        filename = "article.xml" if parsed.path.endswith(".xml") else Path(parsed.path).name
        path = cache / filename
        if path.exists():
            check(hashlib.md5(path.read_bytes()).hexdigest() == parse_qs(parsed.query)["md5"][0],
                  "Payload differs from official publisher MD5")
    figures = {f["figure_id"]: f for f in protocol["figures"]}
    rows = [json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()]
    check(len(rows) == len(protocol["panels"]) == len(sheets["samples"]) == manifest["panels"], "Panel count differs")
    reports = []
    for order, (panel, row, sheet) in enumerate(zip(protocol["panels"], rows, sheets["samples"])):
        source = figures[panel["figure_id"]]
        with Image.open(cache / source["file"]) as image:
            check(image.size == (source["width"], source["height"]), "Figure size differs")
            expected = np.asarray(image.convert("RGB"))[panel["crop_xyxy"][1]:panel["crop_xyxy"][3], panel["crop_xyxy"][0]:panel["crop_xyxy"][2]]
        with Image.open(index / row["image_path"]) as image:
            actual = np.asarray(image.convert("RGB"))
        check(actual.shape == expected.shape, "Crop dimensions differ")
        difference = int(np.max(np.abs(actual.astype(int) - expected.astype(int))))
        check(difference == 0, "Independent JPEG decoder/crop pixels differ")
        check(hashlib.sha256(expected[:, :, ::-1].copy().tobytes()).hexdigest() == row["native_BGR_sha256"], "Pixel hash differs")
        check(sheet["source_record"] == row, "Bundle source record differs")
        check(sha(index / row["image_path"]) == sheet["image_sha256"] == sha(bundle / sheet["image_path"]), "Bundle pixels differ")
        for key, value in dict(panel_id=panel["panel_id"], frame_index=order, data_kind="publication_panel",
            crop_xyxy=panel["crop_xyxy"], domain=panel["domain"], phase_context=panel["context"],
            source_id=protocol["source_id"], material=protocol["material"], split="train", run_id=protocol["run_id"],
            source_url=protocol["source_url"], license=protocol["license"], dependency_family_id=protocol["dependency_family_id"],
            published_relative_time_s=panel["published_relative_time_s"]).items():
            check(row[key] == value, f"Publication provenance differs: {key}")
        for key in ("acquisition_timestamp", "growth_run_start", "first_hour_membership", "pixel_scale_m", "segmentation_labels", "event_labels", "sensor_values", "applied_commands", "quality_labels", "physical_run_id"):
            check(row[key] is None, f"Unsupported measured field: {key}")
        annotation_path = bundle / sheet["annotation_path"]
        annotation = json.loads(annotation_path.read_text())
        check(sha(annotation_path) == sheet["initial_proposal_sha256"], "Initial annotation changed")
        check(annotation["flags"] == dict(reviewed=False, uncertain=True) and annotation["shapes"] == [], "Expected unreviewed unknown sheet")
        check(not (bundle / sheet["review_path"]).exists(), "A review was recorded during acquisition")
        reports.append(dict(panel_id=panel["panel_id"], width=actual.shape[1], height=actual.shape[0], cross_decoder_max_pixel_difference=difference))
    return dict(status="passed", protocol_sha256=sha(protocol_path), panels=reports, families=1,
                accepted_labels=0, human_reviews=0, scope="Independent original-figure crop/pixel and critical-provenance reconstruction; no segmentation or event accuracy claim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kdp_gel_panels.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Audit output already exists")
    report = audit(args.index, args.bundle, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
