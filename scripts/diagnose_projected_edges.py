# FILE PURPOSE: Execute the frozen reference/edge diagnostic and a real-data refusal check.
# Construction labels stay separate from RGB-derived masks and measurements.
"""This is an artificial measurement experiment, not a trained face detector."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.baseline import BaselineConfig, SeedTracker
from vision.projected_edges import FRACTION_ROUNDOFF, LENGTH_ROUNDOFF_PX, ProjectedEdgeTracker
from vision.video import file_sha256


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture_frame(protocol, case, step):
    width, height = protocol["image_size"]
    cx, cy = protocol["seed_point"]
    law = case["law"]
    translation = 2 * step if law == "symmetric_translation" else 0
    if law == "unequal":
        left, right = cx - protocol["initial_half_width"], cx + protocol["initial_half_width"] + 4 * step
    else:
        half_width = protocol["initial_half_width"] + 2 * step
        left, right = cx + translation - half_width, cx + translation + half_width
    top, bottom = cy - protocol["initial_half_height"] - step, cy + protocol["initial_half_height"] + step
    if case["context"] == "pose_unknown_after_3" and step >= 4:
        angle = np.deg2rad(8 * (step - 3))
        # Orthographic width of a yawing box; the external mount markers do not
        # rotate with the object. Registration cannot certify its object pose.
        extent = (protocol["initial_half_width"] + 2 * step) * abs(np.cos(angle)) + 20 * abs(np.sin(angle))
        left, right = cx - extent, cx + extent
    cut = protocol["corner_cut_px"]
    polygon = np.rint([[left + cut, top], [right - cut, top], [right, top + cut], [right, bottom - cut],
                      [right - cut, bottom], [left + cut, bottom], [left, bottom - cut], [left, top + cut]]).astype(np.int32)
    source_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(source_mask, [polygon], 255)
    image = np.full((height, width, 3), 235, np.uint8)
    image[source_mask > 0] = 50
    radius = protocol["marker_radius_px"]
    for center, color in zip(protocol["marker_centers"], protocol["marker_BGR"]):
        x, y = center
        cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), color, -1)
    matrix = np.array([[1., 0, 0], [0, 1., 0]])
    if case["camera"] == "similarity":
        matrix = cv2.getRotationMatrix2D((cx, cy), 1.25 * step, 1 + .005 * step)
        matrix[:, 2] += [2 * step, -.25 * step]
    if case["camera"] == "shear_after_3" and step >= 4:
        shear = .08 * (step - 3)
        matrix = np.array([[1., shear, -shear * cy], [0, 1., 0]])
    image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=(235, 235, 235))
    visible = cv2.warpAffine(source_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    if case["camera"] == "missing_marker_4_to_7" and 4 <= step <= 7:
        x, y = protocol["marker_centers"][0]
        image[y - radius:y + radius + 1, x - radius:x + radius + 1] = 235
    if case["context"] == "occluded_4_to_7" and 4 <= step <= 7:
        image[cy - 20:cy + 55, cx - 10:cx + 100] = 235
        visible[cy - 20:cy + 55, cx - 10:cx + 100] = 0
    timestamp = step / protocol["fps"]
    context = dict(run_id=case["case_id"], frame_index=step, evidence_kind="fixture",
        observed_at_source_s=timestamp, available_at_source_s=timestamp,
        reference_id="constructed_planar_mount_reference", pose_id="constructed_object_pose_0",
        attachment_supported=case["context"] != "unattached",
        pose_supported=not (case["context"] == "pose_unknown_after_3" and step >= 4),
        full_visibility_supported=not (case["context"] == "occluded_4_to_7" and 4 <= step <= 7),
        provenance="Frozen artificial fixture construction assertion; no human review or physical calibration")
    construction = dict(frame_index=step, source_time_s=timestamp, unregistered_polygon_px=polygon.tolist(),
        canonical_extents_px=[float(left), float(right), float(top), float(bottom)], canonical_to_image=matrix.tolist(),
        constructed_object_translation_px=[translation, 0],
        constructed_object_relative_advances_px=[4 * step, 0, step, step] if law == "unequal" else [2 * step, 2 * step, step, step],
        law=law, physical_interpretation="Unequal +x/−x advance on an asserted fixed mount" if law == "unequal" else
            "Symmetric expansion plus translation gives identical images to the unequal case" if law == "symmetric_translation" else
            "Symmetric constructed expansion; no measured KDP kinetics")
    return image, visible, context, construction


def baseline_supports(mask, origin):
    if mask is None:
        return None
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    cx, cy = origin
    return [float(xx.max() - cx), float(cx - xx.min()), float(yy.max() - cy), float(cy - yy.min())]


def baseline_proposals(records, key, config):
    streak = [0, 0]; active = [False, False]; proposals = []
    for i, row in enumerate(records):
        earlier = [r for r in records[:i] if r["source_time_s"] <= row["source_time_s"] - config["horizon_source_s"]]
        if not earlier or row[key] is None or earlier[-1][key] is None:
            streak, active = [0, 0], [False, False]
            continue
        delta = np.array(row[key]) - earlier[-1][key]
        for axis in range(2):
            p, n = delta[axis * 2:axis * 2 + 2];total = p + n;difference = abs(p - n)
            triggered = bool(min(p, n) + LENGTH_ROUNDOFF_PX >= -config["maximum_negative_advance_px"] and total + LENGTH_ROUNDOFF_PX >= config["minimum_total_advance_px"] and
                difference + LENGTH_ROUNDOFF_PX >= config["imbalance_difference_px"] and difference / total + FRACTION_ROUNDOFF >= config["imbalance_fraction"])
            streak[axis] = streak[axis] + 1 if triggered else 0
            if not triggered: active[axis] = False
            if streak[axis] >= config["persistence_windows"] and not active[axis]:
                active[axis] = True;proposals.append(dict(frame_index=row["frame_index"], axis=axis))
    return proposals


def run(protocol_path, output):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "crystalab_projected_edge_diagnostic_v1" or protocol["evidence_kind"] != "fixture":
        raise ValueError("Use the declared artificial measurement diagnostic")
    real_bundle = ROOT / protocol["frozen_checks"]["real_bundle"]
    if file_sha256(real_bundle / "manifest.json") != protocol["frozen_checks"]["real_bundle_manifest_sha256"]:
        raise ValueError("Real refusal-check source changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot").mkdir()
    sources = [Path(__file__), ROOT / "src/vision/projected_edges.py", ROOT / "src/vision/baseline.py"]
    for path in sources: shutil.copyfile(path, output / "source_snapshot" / path.name)
    shutil.copyfile(protocol_path, output / "protocol.json")
    manifest = dict(status="running", schema=protocol["schema"], protocol_sha256=file_sha256(protocol_path),
        source_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in sources},
        scope=protocol["purpose"], evidence_kind="fixture", cases=[], real_data=None)
    save(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        case_rows = {}
        for case in protocol["cases"]:
            folder = output / case["case_id"];folder.mkdir();(folder / "images").mkdir();(folder / "masks").mkdir()
            edge = ProjectedEdgeTracker(protocol["measurement"], protocol["marker_BGR"], run_id=case["case_id"],
                clock_id=case["case_id"] + ":fixture_source", evidence_kind="fixture")
            seed = SeedTracker(BaselineConfig(seed_point=tuple(protocol["seed_point"]), threshold=100, morphology_size=1,
                minimum_area_px=100, minimum_iou=.01, maximum_gap_s=20))
            records, labels = [], []
            for step in range(protocol["frames_per_case"]):
                image, visible, context, construction = fixture_frame(protocol, case, step)
                source_time = step / protocol["fps"]
                observation, mask = seed.measure(image, source_time)
                row = edge.measure(image, mask, timestamp=source_time, frame_index=step, context=context)
                row.update(seed_observation=observation, source_image_path=f"images/{step:04d}.png",
                    observed_mask_path=None if mask is None else f"masks/{step:04d}.png",
                    observed_mask_pixels_sha256=None if mask is None else hashlib.sha256(mask.tobytes()).hexdigest(),
                    unregistered_supports_px=baseline_supports(mask, protocol["seed_point"]))
                if mask is not None:
                    yy, xx = np.nonzero(mask)
                    row["centroid_centered_supports_px"] = baseline_supports(mask, [xx.mean(), yy.mean()])
                    if not cv2.imwrite(str(folder / row["observed_mask_path"]), mask): raise OSError("Mask write failed")
                else: row["centroid_centered_supports_px"] = None
                if not cv2.imwrite(str(folder / row["source_image_path"]), image): raise OSError("Image write failed")
                construction["visible_construction_mask_area_px2"] = int(np.count_nonzero(visible))
                records.append(row);labels.append(construction)
            for name, values in (("measurements.jsonl", records), ("construction.jsonl", labels)):
                (folder / name).write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in values))
            proposals = [p for r in records for p in r["proposals"]]
            comparisons = [r["comparison"] for r in records if r["comparison"]]
            summary = dict(case_id=case["case_id"], frames=len(records), valid_frames=sum(r["valid"] for r in records),
                seed_valid_frames=sum(r["seed_observation"]["valid"] for r in records),
                reasons=dict(Counter(r["reason"] for r in records)), proposals=proposals,
                naive_fixed_origin_proposals=baseline_proposals(records, "unregistered_supports_px", protocol["measurement"]),
                naive_centroid_proposals=baseline_proposals(records, "centroid_centered_supports_px", protocol["measurement"]),
                maximum_registered_imbalance_px=max((a["difference_px"] for c in comparisons for a in c["axes"]), default=None))
            manifest["cases"].append(summary);case_rows[case["case_id"]] = records
        real_manifest = json.loads((real_bundle / "manifest.json").read_text())
        original_files = {str(p.relative_to(real_bundle)):file_sha256(p) for p in real_bundle.rglob("*") if p.is_file()}
        real = []
        for i, sample in enumerate(real_manifest["samples"]):
            path = real_bundle / sample["image_path"]
            if file_sha256(path) != sample["image_sha256"]: raise ValueError("Real source pixels changed")
            record = sample["source_record"]
            tracker = ProjectedEdgeTracker(protocol["measurement"], protocol["marker_BGR"], run_id=record["run_id"],
                clock_id="published_video_display:" + record["run_id"], evidence_kind="real")
            measured = tracker.measure(cv2.imread(str(path)), None, timestamp=record["display_elapsed_s"],
                frame_index=record["frame_index"], context=None)
            measured["source_sample_id"] = record["sample_id"]
            real.append(measured)
        if original_files != {str(p.relative_to(real_bundle)):file_sha256(p) for p in real_bundle.rglob("*") if p.is_file()}:
            raise ValueError("Read-only real source changed")
        (output / "real_unavailable.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in real))
        manifest["real_data"] = dict(source_bundle=str(real_bundle.relative_to(ROOT)), source_manifest_sha256=file_sha256(real_bundle / "manifest.json"),
            samples=len(real), numerical_projected_measurements=sum(r["valid"] for r in real), proposals=sum(len(r["proposals"]) for r in real),
            reasons=dict(Counter(r["reason"] for r in real)), files_unchanged=True, human_reviews_created=0)
        unequal = case_rows["unequal_anchored"]
        free = case_rows["symmetric_free_translation"]
        counterexample = all(a["image_BGR_sha256"] == b["image_BGR_sha256"] and a["observed_mask_pixels_sha256"] == b["observed_mask_pixels_sha256"]
                             for a, b in zip(unequal, free))
        frozen = protocol["frozen_checks"]
        first = [p["source_frame_index"] for r in unequal for p in r["proposals"]]
        checks = dict(unequal_first_proposal=bool(first and first[0] == frozen["expected_unequal_first_proposal_frame"]),
            other_cases_no_proposals=all(not r["proposals"] for key, values in case_rows.items() if key != "unequal_anchored" for r in values),
            same_pixels_different_physics=counterexample,
            camera_motion_bounded=max(a["difference_px"] for r in case_rows["symmetric_camera_motion"] if r["comparison"] for a in r["comparison"]["axes"]) <= frozen["maximum_symmetric_camera_registered_imbalance_px"],
            real_data_not_fabricated=len(real) == frozen["real_samples_expected"] and not any(r["valid"] or r["proposals"] for r in real))
        manifest.update(status="complete", elapsed_s=time.perf_counter() - started, checks=checks,
            identical_image_counterexample=counterexample, physical_commands=0,
            files={str(p.relative_to(output)):file_sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"})
        save(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/projected_edge_diagnostic.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.protocol, args.output)
    print(json.dumps({k:v for k,v in report.items() if k != "files"}, indent=2))
