# FILE PURPOSE: Independently reconstruct process candidates from original public
# CSV/JPEG bytes with Pillow, connected components, polygon algebra and past-only
# integration. No producer import, model fit, source controller or hardware call.
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while part := stream.read(4 * 1024**2):
            digest.update(part)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def polygon_moments(points):
    xy = np.asarray(points, dtype=float)
    shifted = np.roll(xy, -1, axis=0)
    cross = xy[:, 0] * shifted[:, 1] - shifted[:, 0] * xy[:, 1]
    signed_area = cross.sum() / 2
    center = ((xy + shifted) * cross[:, None]).sum(axis=0) / (6 * signed_area)
    return abs(float(signed_area)), center


def reconstruct_smoothing(values, window):
    """Local quadratic regression, without scipy.signal's SG implementation."""
    x = np.linspace(-1, 1, window)
    design = np.column_stack([np.ones(window), x, x**2])
    weights = np.linalg.solve(design.T @ design, [1.0, 0.0, 0.0]) @ design.T
    half = window // 2
    result = np.convolve(values, weights, mode="same")
    left = np.linalg.lstsq(design, values[:window], rcond=None)[0]
    right = np.linalg.lstsq(design, values[-window:], rcond=None)[0]
    result[:half] = design[:half] @ left
    result[-half:] = design[-half:] @ right
    return result


def audit(result, protocol_path, index, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    assert protocol["schema"] == "crystalab_crystalcv_process_audit_v2"
    assert (
        protocol["candidate_samples"]["reported_rate_availability"] == "next_source_row"
    )
    assert sha(result / "protocol.json") == sha(protocol_path)
    for relative, digest in json.loads((result / "manifest.json").read_text()).items():
        assert sha(result / relative) == digest, relative
    for relative in (
        "scripts/check_crystalcv_process.py",
        "src/vision/process_series.py",
    ):
        assert sha(result / "source" / relative) == sha(ROOT / relative)
    assert sha(index / "manifest.json") == protocol["source_index_manifest_sha256"]
    for relative, digest in protocol["source_code_sha256"].items():
        assert sha(ROOT / relative) == digest
    manifest = json.loads((index / "manifest.json").read_text())
    assert sha(index / "images.jsonl") == manifest["file_sha256"]["images.jsonl"]
    images = read_jsonl(index / "images.jsonl")
    images = sorted(
        (
            r
            for r in images
            if r["archive_id"] == protocol["archive_id"]
            and r["source_folder"] == protocol["image_directory"]
        ),
        key=lambda r: r["filename_elapsed_s"],
    )
    asset = next(
        r
        for r in manifest["protocol"]["archives"]
        if r["archive_id"] == protocol["archive_id"]
    )
    archive_path = Path(manifest["cache_path"]) / asset["filename"]
    assert sha(archive_path) == protocol["archive_sha256"]
    observations = read_jsonl(result / "observations.jsonl")
    assert len(observations) == len(images) == protocol["expected_images"]
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(protocol["series_member"])
        assert (
            hashlib.sha256(payload).hexdigest()
            == sha(result / "source_series.csv")
            == protocol["series_sha256"]
        )
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        assert len(rows) == protocol["expected_series_rows"]
        seconds, errors = [], []
        for row in rows:
            source = Decimal(row["Time (h)"]) * 3600
            rounded = int(source.to_integral_value(rounding=ROUND_HALF_EVEN))
            errors.append(float(abs(source - rounded)))
            assert (
                errors[-1] <= protocol["clock"]["integer_second_roundtrip_tolerance_s"]
            )
            seconds.append(rounded)
        assert seconds[0] == 0 and all(b > a for a, b in zip(seconds, seconds[1:]))
        rates = [float(r["Infusion Rate (mL/h)"]) for r in rows]
        workbook_path = ROOT / protocol["original_workbook"]["local_file"]
        assert sha(workbook_path) == protocol["original_workbook"]["sha256"]
        columns = list(rows[0])[:13]
        matched_cells = 0
        with zipfile.ZipFile(workbook_path) as workbook:
            with workbook.open("xl/worksheets/sheet1.xml") as stream:
                for _, cell in ET.iterparse(stream, events=("end",)):
                    if cell.tag.rsplit("}", 1)[-1] != "c":
                        continue
                    address = cell.attrib["r"]
                    column = "".join(c for c in address if c.isalpha())
                    rownum = int("".join(c for c in address if c.isdigit()))
                    if rownum >= 2:
                        assert len(column) == 1 and "A" <= column <= "M"
                        assert not any(c.tag.rsplit("}", 1)[-1] == "f" for c in cell)
                        stored = next(
                            c.text for c in cell if c.tag.rsplit("}", 1)[-1] == "v"
                        )
                        source = rows[rownum - 2][columns[ord(column) - ord("A")]]
                        assert math.isclose(
                            float(stored), float(source), rel_tol=1e-13, abs_tol=1e-13
                        )
                        matched_cells += 1
                    cell.clear()
        assert matched_cells == 13 * len(rows)
        previous = None
        maximum_centroid_error, native_count = 0.0, 0
        settings = protocol["image_measurement"]
        for observed, image in zip(observations, images):
            assert observed["sample_id"] == image["sample_id"]
            assert observed["original_member"] == image["member"]
            raw = archive.read(image["member"])
            assert (
                hashlib.sha256(raw).hexdigest()
                == image["sha256"]
                == observed["source_image_sha256"]
            )
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            assert (
                hashlib.sha256(
                    np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()
                ).hexdigest()
                == observed["native_BGR_sha256"]
                == image["native_BGR_sha256"]
            )
            h, m, s = map(
                int,
                re.fullmatch(
                    r"(\d+)h (\d+)m (\d+)s", Path(image["member"]).stem
                ).groups(),
            )
            time = h * 3600 + m * 60 + s
            assert time == observed["filename_elapsed_s"] == image["filename_elapsed_s"]
            log_index = bisect_right(seconds, time) - 1
            assert (
                time == seconds[log_index]
            )  # This specific source, not a universal clock guarantee.
            assert (
                observed["source_log_row_index"] == log_index
                and observed["source_log_age_s"] == 0
            )
            assert observed["source_log_area_mm2"] == float(
                rows[log_index]["Area (mm2)"]
            )
            yy, xx = np.indices(rgb.shape[:2])
            cx, cy = settings["analysis_circle_xy"]
            roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= settings["analysis_radius_px"] ** 2
            selected = (
                np.all(
                    (rgb >= settings["lower_RGB"]) & (rgb <= settings["upper_RGB"]),
                    axis=2,
                )
                & roi
            )
            labels, count = ndimage.label(selected, structure=np.ones((3, 3)))
            points = np.array(observed["contour_xy"])
            chosen = int(labels[points[0, 1], points[0, 0]])
            assert chosen > 0
            component = ndimage.binary_fill_holes(labels == chosen)
            mask = np.asarray(Image.open(result / observed["mask_path"]))
            assert sha(result / observed["mask_path"]) == observed["mask_file_sha256"]
            assert (
                hashlib.sha256(mask.tobytes()).hexdigest()
                == observed["mask_pixels_sha256"]
            )
            assert np.array_equal(mask > 0, component)
            assert component.sum() == observed["filled_mask_pixels"]
            area, center = polygon_moments(points)
            assert area == observed["contour_area_px2"]
            maximum_centroid_error = max(
                maximum_centroid_error,
                float(abs(center - observed["centroid_xy"]).max()),
            )
            assert np.allclose(center, observed["centroid_xy"], atol=1e-9, rtol=0)
            # Independent pixel-component ranking checks identity. Degenerate
            # one-pixel specks have no polygon area and are not candidates.
            candidates = []
            for label, box in enumerate(ndimage.find_objects(labels), 1):
                if box is None:
                    continue
                ys, xs = box
                if ys.stop - ys.start < 2 or xs.stop - xs.start < 2:
                    continue
                rr, cc = np.nonzero(labels[box] == label)
                if len(rr) < 3:
                    continue
                candidates.append(
                    (
                        label,
                        len(rr),
                        np.array([cc.mean() + xs.start, rr.mean() + ys.start]),
                    )
                )
            picked = (
                max(candidates, key=lambda r: r[1])
                if previous is None
                else min(candidates, key=lambda r: np.linalg.norm(r[2] - previous))
            )[0]
            assert picked == chosen
            assert not np.any(
                component & ~ndimage.binary_erosion(roi, structure=np.ones((3, 3)))
            )
            assert (
                previous is None
                or np.linalg.norm(center - previous)
                <= settings["maximum_centroid_step_px"]
            )
            assert (
                observed["integrity_valid"]
                and not observed["accepted_mask_ground_truth"]
                and not observed["physical_calibration_verified"]
            )
            previous = center
            native_count += 1
    raw_length = np.array([float(r["Length (mm)"]) for r in rows])
    smoothed = reconstruct_smoothing(
        raw_length, protocol["smoothing_diagnostic"]["candidate_effective_window"]
    )
    exported = np.array([float(r["Smoothed Length (mm)"]) for r in rows])
    smoothing_error = float(abs(smoothed - exported).max())
    assert (
        smoothing_error < protocol["smoothing_diagnostic"]["exact_match_tolerance_mm"]
    )
    changed = raw_length.copy()
    cutoff = protocol["smoothing_diagnostic"]["perturbation_cutoff_row"]
    changed[cutoff + 1 :] += protocol["smoothing_diagnostic"]["perturbation_delta_mm"]
    changed_smooth = reconstruct_smoothing(
        changed, protocol["smoothing_diagnostic"]["candidate_effective_window"]
    )
    effect = abs(changed_smooth[: cutoff + 1] - smoothed[: cutoff + 1])
    assert np.count_nonzero(effect > 1e-10) == 252
    samples = read_jsonl(result / "candidate_samples.jsonl")
    saved = {r["sample_id"]: r for r in samples}
    assert len(saved) == len(samples)
    settings = protocol["candidate_samples"]
    times = [r["filename_elapsed_s"] for r in observations]
    available_at = seconds[1:]
    count, rejections = 0, []
    for i, observed in enumerate(observations):
        time = times[i]
        future = bisect_left(times, time + settings["horizon_s"])
        if time - times[0] < settings["history_s"]:
            rejections.append(
                dict(
                    sample_id=observed["sample_id"],
                    cutoff_s=float(time),
                    reason="insufficient_image_history",
                )
            )
            continue
        start = time - settings["history_s"]
        if start < available_at[0]:
            rejections.append(
                dict(
                    sample_id=observed["sample_id"],
                    cutoff_s=float(time),
                    reason="missing_or_stale_past_process_state",
                )
            )
            continue
        if (
            future == len(times)
            or times[future]
            > time + settings["horizon_s"] + settings["target_lateness_s"]
        ):
            rejections.append(
                dict(
                    sample_id=observed["sample_id"],
                    cutoff_s=float(time),
                    reason="missing_comparable_future_proxy",
                )
            )
            continue
        sample = saved[observed["sample_id"]]
        history = [j for j in range(i + 1) if start <= times[j] <= time]
        assert len(history) >= settings["minimum_history_images"]
        log_first, log_last = (
            bisect_right(available_at, start) - 1,
            bisect_right(available_at, time) - 1,
        )
        intervals = [
            (
                max(start, available_at[j]),
                min(time, available_at[j + 1]) if j < log_last else time,
                rates[j],
            )
            for j in range(log_first, log_last + 1)
        ]
        mean_rate = (
            math.fsum((b - a) * r for a, b, r in intervals) / settings["history_s"]
        )
        history_times = np.array([times[j] - time for j in history]) / 3600
        history_values = np.array(
            [math.sqrt(observations[j]["contour_area_px2"]) for j in history]
        )
        slope = np.linalg.lstsq(
            np.column_stack([np.ones(len(history)), history_times]),
            history_values,
            rcond=None,
        )[0][1]
        expected = [
            observed["contour_area_px2"],
            math.sqrt(observed["contour_area_px2"]),
            float(slope),
            rates[log_last],
            mean_rate,
        ]
        assert list(sample["inputs"]) == settings["allowed_input_fields"]
        np.testing.assert_allclose(
            list(sample["inputs"].values()), expected, atol=1e-10, rtol=1e-12
        )
        assert sample["input_image_sample_ids"] == [
            observations[j]["sample_id"] for j in history
        ]
        assert sample["input_log_row_range"] == [log_first, log_last]
        assert sample["maximum_input_time_s"] == sample["cutoff_s"] == time
        assert sample["last_input_log_time_s"] == seconds[log_last] < time
        assert (
            sample["last_input_availability_bound_s"] == available_at[log_last] <= time
        )
        assert sample["target_sample_id"] == observations[future]["sample_id"]
        assert sample["target_time_s"] == times[future] > time
        assert sample["actual_horizon_s"] == times[future] - time
        assert sample["target_sqrt_area_px"] == math.sqrt(
            observations[future]["contour_area_px2"]
        )
        assert (
            not sample["independent_ground_truth"]
            and sample["split"] == "development"
            and sample["physical_command"] is None
        )
        count += 1
    assert count == len(samples)
    assert rejections == read_jsonl(result / "rejected_candidates.jsonl")
    report = json.loads((result / "report.json").read_text())
    prefix_end = protocol["log_scale_diagnostic"]["calibration_prefix_s"]
    prefix = [r for r in observations if r["filename_elapsed_s"] <= prefix_end]
    conversion = float(
        np.median([r["contour_area_px2"] / r["source_log_area_mm2"] for r in prefix])
    )
    for name, cohort in [
        ("prefix", prefix),
        ("later", [r for r in observations if r["filename_elapsed_s"] > prefix_end]),
    ]:
        diagnostic = report["scale_diagnostics"][name]
        assert diagnostic["images"] == len(cohort)
        assert diagnostic["estimated_area_divisor_px2_per_log_mm2"] == conversion
        comparisons = [
            (
                "source_scaled_MAPE",
                [
                    abs(
                        r["contour_area_px2"] / conversion / r["source_log_area_mm2"]
                        - 1
                    )
                    for r in cohort
                ],
            ),
            (
                "sidecar_area_divisor_MAPE",
                [
                    abs(
                        r["contour_area_px2"]
                        / protocol["log_scale_diagnostic"]["sidecar_reported_scale"]
                        / r["source_log_area_mm2"]
                        - 1
                    )
                    for r in cohort
                ],
            ),
            (
                "sidecar_linear_mm_per_px_MAPE",
                [
                    abs(
                        r["contour_area_px2"]
                        * protocol["log_scale_diagnostic"]["sidecar_reported_scale"]
                        ** 2
                        / r["source_log_area_mm2"]
                        - 1
                    )
                    for r in cohort
                ],
            ),
        ]
        for field, values in comparisons:
            assert math.isclose(
                diagnostic[field],
                math.fsum(values) / len(cohort),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
    assert (
        report["local_human_reviews_added"]
        == report["independent_evaluation_groups"]
        == 0
    )
    assert (
        not report["physical_calibration_verified"]
        and not report["hardware_synchronization_verified"]
        and not report["new_model_fit"]
    )
    assert (
        report["physical_command"] is None
        and report["reported_rate_availability"] == "next_source_row"
    )
    assert report["candidate_examples"] == count and report[
        "candidate_rejections"
    ] == dict(Counter(r["reason"] for r in rejections))
    assert report["smoothing"]["affected_rows_at_or_before_cutoff"] == 252
    assert math.isclose(
        report["smoothing"]["maximum_earlier_change_mm"],
        float(effect.max()),
        abs_tol=1e-10,
    )
    record = dict(
        passed=True,
        protocol_sha256=sha(protocol_path),
        result_manifest_sha256=sha(result / "manifest.json"),
        auditor_sha256=sha(Path(__file__)),
        native_images_reconstructed=native_count,
        workbook_numeric_cells_reconstructed=matched_cells,
        candidates_reconstructed=count,
        rejection_count=len(rejections),
        smoothing_reconstruction_max_error_mm=smoothing_error,
        future_dependent_rows_before_or_at_cutoff=252,
        future_perturbation_maximum_earlier_change_mm=float(effect.max()),
        maximum_centroid_difference_px=maximum_centroid_error,
        hardware_synchronization_verified=False,
        physical_calibration_verified=False,
        reviewed_masks=False,
        independent_evaluation_groups=0,
        new_model_fit=False,
        physical_command=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_suffix(".auditor.py")
    if snapshot.exists():
        raise FileExistsError(snapshot)
    shutil.copyfile(Path(__file__), snapshot)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/crystalcv_process_audit_v2.json",
    )
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data/real_vision_crystalcv_v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result, args.protocol, args.index, args.output)
