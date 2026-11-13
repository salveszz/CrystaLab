# FILE PURPOSE: Independently reconstruct image references, supports, causal decisions and ambiguity.
"""Pillow pixel decoding and SVD registration; no producer/tracker imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def registration(current, reference):
    x, y = current - current.mean(axis=0), reference - reference.mean(axis=0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    scale = np.sum(singular * np.diag(correction)) / np.sum(x * x)
    linear = scale * rotation
    shift = reference.mean(axis=0) - current.mean(axis=0) @ linear
    matrix = np.column_stack([linear.T, shift])
    rms = np.sqrt(np.mean(np.sum((current @ linear + shift - reference) ** 2, axis=1)))
    return matrix, float(rms)


def baseline_decisions(values, rows, config):
    result = [];streak = [0, 0];active = [False, False]
    for step, value in enumerate(values):
        eligible = [i for i in range(step) if rows[i]["source_time_s"] <= rows[step]["source_time_s"] - config["horizon_source_s"]]
        if not eligible: continue
        delta = value - values[eligible[-1]]
        for axis in range(2):
            positive, negative = delta[2 * axis:2 * axis + 2]
            total, difference = positive + negative, abs(positive - negative)
            exceeds = bool(min(positive, negative) + 1e-9 >= -config["maximum_negative_advance_px"] and
                total + 1e-9 >= config["minimum_total_advance_px"] and difference + 1e-9 >= config["imbalance_difference_px"] and
                difference / total + 1e-12 >= config["imbalance_fraction"])
            streak[axis] = streak[axis] + 1 if exceeds else 0
            if not exceeds: active[axis] = False
            if streak[axis] >= config["persistence_windows"] and not active[axis]:
                active[axis] = True;result.append(dict(frame_index=step, axis=axis))
    return result


def audit(run, protocol_path):
    run = Path(run)
    protocol = json.loads(Path(protocol_path).read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    check(manifest["status"] == "complete", "Incomplete experiment")
    check(manifest["protocol_sha256"] == digest(protocol_path) == digest(run / "protocol.json"), "External frozen protocol differs")
    for name, expected in manifest["files"].items(): check(digest(run / name) == expected, "Artifact changed: " + name)
    for name, expected in manifest["source_sha256"].items():
        check(digest(run / "source_snapshot" / Path(name).name) == expected, "Producer snapshot changed")
    config = protocol["measurement"]
    summaries, all_rows, maximum_support_error, maximum_matrix_error = [], {}, 0., 0.
    for case in protocol["cases"]:
        folder = run / case["case_id"]
        rows = read_rows(folder / "measurements.jsonl")
        check(len(rows) == protocol["frames_per_case"], "Missing diagnostic frames")
        reference = None;history = [];signature = None;streak = [0, 0];active = [False, False];proposals = []
        fixed_supports, centered_supports = [], []
        for step, row in enumerate(rows):
            check(row["physical_face_rate_available"] is False and row["physical_commands"] == [], "Unsupported physical output")
            check(row["frame_index"] == step and row["source_time_s"] == step / protocol["fps"], "Fixture source clock differs")
            image = pixels(folder / row["source_image_path"])
            check(hashlib.sha256(image.tobytes()).hexdigest() == row["image_BGR_sha256"], "Image pixel hash differs")
            with Image.open(folder / row["observed_mask_path"]) as decoded:
                mask = np.asarray(decoded)
            # Fixture body is gray 50; colored references are separate objects.
            expected_mask = np.all(image == 50, axis=2).astype(np.uint8) * 255
            check(np.array_equal(mask, expected_mask), "RGB-derived seed differs from observed mask")
            check(hashlib.sha256(mask.tobytes()).hexdigest() == row["observed_mask_pixels_sha256"], "Observed mask hash differs")
            yy, xx = np.nonzero(mask)
            for origin, field, values in [(protocol["seed_point"], "unregistered_supports_px", fixed_supports),
                                          ([xx.mean(), yy.mean()], "centroid_centered_supports_px", centered_supports)]:
                cx, cy = origin
                support = np.array([xx.max() - cx, cx - xx.min(), yy.max() - cy, cy - yy.min()])
                check(np.max(abs(support - row[field])) < 1e-8, "Naive baseline supports differ")
                values.append(support)
            context = row["context"]
            check(hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest() == row["context_sha256"], "View assertions changed")
            check(context["frame_index"] == step and context["evidence_kind"] == "fixture" and
                  0 <= context["observed_at_source_s"] <= context["available_at_source_s"] <= row["source_time_s"], "Noncausal source assertions")
            expected_assertions = dict(attachment_supported=case["context"] != "unattached",
                pose_supported=not (case["context"] == "pose_unknown_after_3" and step >= 4),
                full_visibility_supported=not (case["context"] == "occluded_4_to_7" and 4 <= step <= 7))
            check(all(context[key] == value for key, value in expected_assertions.items()), "Fixture assertions differ from frozen construction case")
            check(context["reference_id"] == "constructed_planar_mount_reference" and context["pose_id"] == "constructed_object_pose_0" and
                  row["run_id"] == case["case_id"] and row["clock_id"] == case["case_id"] + ":fixture_source",
                  "Fixture reference/run/clock identity differs")
            reason = None
            for key, missing in [("attachment_supported", "attachment_not_established"),
                                 ("pose_supported", "object_pose_not_comparable"),
                                 ("full_visibility_supported", "incomplete_silhouette_view")]:
                if context[key] is not True:
                    reason = missing;break
            centers = []
            if reason is None:
                for color in protocol["marker_BGR"]:
                    yy, xx = np.nonzero(np.all(image == color, axis=2))
                    if not len(xx): reason = "reference_markers_missing_or_ambiguous";break
                    centers.append([xx.mean(), yy.mean()])
            if reason is None:
                centers = np.array(centers)
                key = (context["reference_id"], context["pose_id"])
                if signature != key: reference = centers.copy();history = [];signature = key
                matrix, rms = registration(centers, reference)
                matrix_error = float(np.max(abs(matrix - row["registration"]["current_to_reference"])))
                maximum_matrix_error = max(maximum_matrix_error, matrix_error)
                check(matrix_error < 1e-8 and abs(rms - row["registration"]["RMS_px"]) < 1e-8, "Independent SVD registration differs")
                if rms > config["maximum_registration_RMS_px"]: reason = "reference_is_not_a_supported_similarity"
            if reason is not None:
                check(not row["valid"] and row["reason"] == reason and row["comparison"] is None and
                      row["projected_supports_px"] is None and row["proposals"] == [], "Unsupported view produced a measurement or proposal")
                history = [];streak = [0, 0];active = [False, False]
                continue
            yy, xx = np.nonzero(mask)
            anchor = np.asarray(config["anchor_barycentric_weights"]) @ reference
            coordinates = np.column_stack([xx, yy]) @ matrix[:, :2].T + matrix[:, 2] - anchor
            support = np.array([coordinates[:, 0].max(), -coordinates[:, 0].min(), coordinates[:, 1].max(), -coordinates[:, 1].min()])
            error = float(np.max(abs(support - row["projected_supports_px"])))
            maximum_support_error = max(error, maximum_support_error)
            check(row["valid"] and error < 1e-8, "Projected supports differ")
            earlier = [r for r in history if r["source_time_s"] <= row["source_time_s"] - config["horizon_source_s"] + 1e-10]
            expected_proposals = []
            if not earlier: check(row["comparison"] is None, "History was invented")
            else:
                baseline = earlier[-1]
                check(row["comparison"]["baseline_frame_index"] == baseline["frame_index"], "Comparison crosses a gap or future window")
                check(row["comparison"]["baseline_source_time_s"] == baseline["source_time_s"] and
                      row["comparison"]["horizon_source_s"] == row["source_time_s"] - baseline["source_time_s"], "Comparison clock differs")
                for field in ("context_sha256", "image_BGR_sha256", "mask_pixels_sha256"):
                    check(row["comparison"]["baseline_" + field] == baseline[field], "Comparison evidence pointer differs")
                delta = support - baseline["independent_supports"]
                for axis, result in enumerate(row["comparison"]["axes"]):
                    p, n = delta[axis * 2:axis * 2 + 2];total = p + n;difference = abs(p - n)
                    check(max(abs(p - result["positive_advance_px"]), abs(n - result["negative_advance_px"]), abs(difference - result["difference_px"])) < 1e-8, "Directional advance differs")
                    available = bool(min(p, n) + 1e-9 >= -config["maximum_negative_advance_px"] and total + 1e-9 >= config["minimum_total_advance_px"])
                    triggered = bool(available and difference + 1e-9 >= config["imbalance_difference_px"] and difference / total + 1e-12 >= config["imbalance_fraction"])
                    check(available == result["comparable_growth"] and triggered == result["threshold_exceeded"],
                          f"Directional threshold decision differs: {case['case_id']} frame {step} axis {axis}; independent total={total!r}, difference={difference!r}; stored={result}")
                    streak[axis] = streak[axis] + 1 if triggered else 0
                    if not triggered: active[axis] = False
                    check(streak[axis] == result["persistent_windows"], "Persistence differs")
                    if streak[axis] >= config["persistence_windows"] and not active[axis]:
                        active[axis] = True;expected_proposals.append(axis)
            check(len(expected_proposals) == len(row["proposals"]), "Proposal count differs")
            for axis, proposal in zip(expected_proposals, row["proposals"]):
                check(proposal["source_frame_index"] == step and proposal["axis"] == ("image_reference_x", "image_reference_y")[axis] and
                      proposal["requires_operator_review"] and proposal["physical_command"] is None and proposal["physical_defect_confirmed"] is False,
                      "Proposal became a command or confirmed defect")
                check(proposal["run_id"] == row["run_id"] and proposal["clock_id"] == row["clock_id"] and
                      proposal["proposed_at_source_s"] == row["source_time_s"] and proposal["baseline_frame_index"] == baseline["frame_index"],
                      "Proposal source/clock differs")
                for field in ("context_sha256", "image_BGR_sha256", "mask_pixels_sha256"):
                    check(proposal[field] == row[field], "Proposal current evidence differs")
                for field in ("image_BGR_sha256", "mask_pixels_sha256"):
                    check(proposal["baseline_" + field] == baseline[field], "Proposal baseline evidence differs")
                proposals.append(step)
            check(row["physical_face_rate_available"] is False and row["physical_commands"] == [], "Unsupported physical output")
            check(row["mask_pixels_sha256"] == row["observed_mask_pixels_sha256"], "Qualified mask identity differs")
            history.append(dict(frame_index=step, source_time_s=row["source_time_s"], independent_supports=support,
                                **{field:row[field] for field in ("context_sha256", "image_BGR_sha256", "mask_pixels_sha256")}))
        fixed_proposals = baseline_decisions(fixed_supports, rows, config)
        centered_proposals = baseline_decisions(centered_supports, rows, config)
        saved = next(value for value in manifest["cases"] if value["case_id"] == case["case_id"])
        check(fixed_proposals == saved["naive_fixed_origin_proposals"] and centered_proposals == saved["naive_centroid_proposals"], "Naive baseline decisions differ")
        summaries.append(dict(case_id=case["case_id"], frames=len(rows), valid=sum(r["valid"] for r in rows), proposal_frames=proposals,
                              naive_fixed_origin_proposals=fixed_proposals, naive_centroid_proposals=centered_proposals))
        all_rows[case["case_id"]] = rows
    for step in range(protocol["frames_per_case"]):
        first, second = [run / case / "images" / f"{step:04d}.png" for case in ("unequal_anchored", "symmetric_free_translation")]
        check(np.array_equal(pixels(first), pixels(second)), "Identical-image counterexample differs")
    a = read_rows(run / "unequal_anchored/construction.jsonl")
    b = read_rows(run / "symmetric_free_translation/construction.jsonl")
    for step, (anchored, free) in enumerate(zip(a, b)):
        check(anchored["constructed_object_relative_advances_px"][:2] == [4 * step, 0] and
              free["constructed_object_relative_advances_px"][:2] == [2 * step, 2 * step] and
              free["constructed_object_translation_px"] == [2 * step, 0], "Underlying counterexample laws differ")
    real_root = ROOT / protocol["frozen_checks"]["real_bundle"]
    check(digest(real_root / "manifest.json") == protocol["frozen_checks"]["real_bundle_manifest_sha256"], "Real source identity differs")
    real_bundle = json.loads((real_root / "manifest.json").read_text())
    real = read_rows(run / "real_unavailable.jsonl")
    check(len(real) == len(real_bundle["samples"]) == 6, "Real source count differs")
    for row, sample in zip(real, real_bundle["samples"]):
        original = real_root / sample["image_path"]
        check(digest(original) == sample["image_sha256"] and hashlib.sha256(pixels(original).tobytes()).hexdigest() == row["image_BGR_sha256"], "Real source image differs")
        check(row["context"] is None and not row["valid"] and row["projected_supports_px"] is None and not row["proposals"] and
              row["physical_face_rate_available"] is False and row["physical_commands"] == [] and
              row["reason"] == "missing_external_view_reference", "Real view assertions or measurements were invented")
    check(summaries[1]["proposal_frames"] == [protocol["frozen_checks"]["expected_unequal_first_proposal_frame"]], "Unequal case failed frozen proposal check")
    check(all(not row["proposal_frames"] for row in summaries if row["case_id"] != "unequal_anchored"), "Other case produced a proposal")
    camera = all_rows["symmetric_camera_motion"]
    camera_imbalance = max(a["difference_px"] for r in camera if r["comparison"] for a in r["comparison"]["axes"])
    check(camera_imbalance <= protocol["frozen_checks"]["maximum_symmetric_camera_registered_imbalance_px"], "Camera-motion criterion failed")
    return dict(status="passed", protocol_sha256=digest(protocol_path), run_manifest_sha256=digest(run / "manifest.json"),
        fixture_frames=sum(r["frames"] for r in summaries), cases=summaries, maximum_support_error_px=maximum_support_error,
        maximum_similarity_matrix_error=maximum_matrix_error, symmetric_camera_maximum_imbalance_px=camera_imbalance,
        identical_image_counterexample_pairs=12, real_unavailable_samples=len(real), physical_commands=0,
        scope="Independent pixel, SVD/support and causal-decision consistency; external attachment/pose assertions are not inferred or physically validated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/projected_edge_diagnostic.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Audit output exists")
    report = audit(args.run, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
