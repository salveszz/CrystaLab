# FILE PURPOSE: Audit source clock, units and retrospective fields before a forecasting fit.
# Read the original workbook without recalculation/editing. Extract native threshold
# observation proxies and explicitly past-only candidate inputs; train no model.
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
import scipy
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision.process_series import (
    INPUT_FIELDS,
    candidate_samples,
    integer_log_seconds,
    primary_contour,
    asof_index,
)
from vision.video import file_sha256

SOURCE_COLUMNS = [
    "Unnamed: 0",
    "Time (h)",
    "Infusion Rate (mL/h)",
    "Total Infusion (mL)",
    "Number of Crystals",
    "Length (mm)",
    "Smoothed Length (mm)",
    "Area (mm2)",
    "Est-Mass (g)",
    "Growth Rate (mm/h)",
    "Smoothed Growth Rate (mm/h)",
    "e(t)",
    "de/dt",
]
WORKBOOK_HEADERS = [
    None,
    "Time (h)",
    "Inf-rate (mL/h)",
    "Total Inf",
    "Crystal Number",
    "Length(mm)",
    "Smth_L (mm)",
    "Area (mm2)",
    "Est-Mass (g)",
    "Growth rate (mm/h)",
    "Smth_G (mm/h)",
    "e(t)",
    "de(t)/dt",
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_stored_workbook(path):
    """Read the pinned simple source workbook's native values, not recalculated values.

    This source has no formulas in its Data/Parameters/Information sheets. Refuse
    formulas rather than evaluating another file or treating a cache as a new result.
    """
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        if sum(v.file_size for v in archive.infolist()) > 128 * 1024**2:
            raise ValueError("Source workbook exceeds audit bound")
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(v.itertext())
                for v in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        result = {}
        for number, name in [(1, "Data"), (2, "Parameters"), (3, "Information")]:
            rows, special = [], {}
            with archive.open(f"xl/worksheets/sheet{number}.xml") as stream:
                for event, row in ET.iterparse(stream, events=("end",)):
                    if row.tag != "{" + ns["m"] + "}row":
                        continue
                    values = {}
                    for cell in row:
                        if cell.tag != "{" + ns["m"] + "}c":
                            continue
                        if cell.find("m:f", ns) is not None:
                            raise ValueError(
                                "Unexpected source formula; original-value review required"
                            )
                        value = cell.find("m:v", ns)
                        text = None if value is None else value.text
                        kind = cell.attrib.get("t")
                        if kind == "s" and text is not None:
                            value = shared[int(text)]
                        elif kind == "inlineStr":
                            value = "".join(cell.find("m:is", ns).itertext())
                        elif text is None:
                            value = None
                        elif kind in (None, "n"):
                            value = float(text)
                        else:
                            raise ValueError("Unsupported source workbook value type")
                        address = cell.attrib["r"]
                        column = "".join(c for c in address if c.isalpha())
                        values[column] = value
                        if name != "Data":
                            special[address] = value
                    if name == "Data":
                        rows.append([values.get(chr(65 + i)) for i in range(13)])
                    row.clear()
            result[name] = rows if name == "Data" else special
    return result


def run(index, protocol_path, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol["schema"] != "crystalab_crystalcv_process_audit_v2"
        or tuple(protocol["candidate_samples"]["allowed_input_fields"]) != INPUT_FIELDS
    ):
        raise ValueError("Require the frozen source task and input whitelist")
    shutil.copyfile(protocol_path, output / "protocol.json")
    for relative in (
        "scripts/check_crystalcv_process.py",
        "src/vision/process_series.py",
    ):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    save(
        output / "environment.json",
        dict(
            python=sys.version,
            numpy=np.__version__,
            opencv=cv2.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
    )
    if file_sha256(index / "manifest.json") != protocol["source_index_manifest_sha256"]:
        raise ValueError("Source index identity changed")
    manifest = json.loads((index / "manifest.json").read_text())
    if file_sha256(index / "images.jsonl") != manifest["file_sha256"]["images.jsonl"]:
        raise ValueError("Source image index changed")
    images = [
        json.loads(line) for line in (index / "images.jsonl").read_text().splitlines()
    ]
    images = sorted(
        (
            r
            for r in images
            if r["archive_id"] == protocol["archive_id"]
            and r["source_folder"] == protocol["image_directory"]
        ),
        key=lambda r: r["filename_elapsed_s"],
    )
    if len(images) != protocol["expected_images"] or any(
        r["role"] != "source_growth_image" for r in images
    ):
        raise ValueError("Source image cohort changed")
    asset = next(
        v
        for v in manifest["protocol"]["archives"]
        if v["archive_id"] == protocol["archive_id"]
    )
    archive_path = Path(manifest["cache_path"]) / asset["filename"]
    if file_sha256(archive_path) != protocol["archive_sha256"]:
        raise ValueError("Source archive changed")
    workbook_path = ROOT / protocol["original_workbook"]["local_file"]
    if file_sha256(workbook_path) != protocol["original_workbook"]["sha256"]:
        raise ValueError("Original workbook changed")
    for relative, digest in protocol["source_code_sha256"].items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError("Controller source reference changed")
    workbook = read_stored_workbook(workbook_path)
    if workbook["Data"][0] != WORKBOOK_HEADERS:
        raise ValueError("Source workbook columns differ")
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(protocol["series_member"])
        if hashlib.sha256(payload).hexdigest() != protocol["series_sha256"]:
            raise ValueError("Source series changed")
        (output / "source_series.csv").write_bytes(payload)
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        if (
            len(rows) != protocol["expected_series_rows"]
            or len(workbook["Data"]) != len(rows) + 1
        ):
            raise ValueError("Original workbook/CSV row counts differ")
        values = np.array([[float(r[k]) for k in SOURCE_COLUMNS] for r in rows])
        original = np.array(workbook["Data"][1:], dtype=float)
        if not np.isfinite(values).all() or not np.allclose(
            values, original, rtol=1e-13, atol=1e-13
        ):
            raise ValueError("CSV differs from original workbook values")
        workbook_difference = {
            k: float(abs(values[:, i] - original[:, i]).max())
            for i, k in enumerate(SOURCE_COLUMNS)
        }
        source_clock = protocol["clock"]
        seconds, clock_error = integer_log_seconds(
            values[:, 1], source_clock["integer_second_roundtrip_tolerance_s"]
        )
        rates = values[:, 2]
        observations, previous = [], None
        (output / "masks").mkdir()
        for frame, image_row in enumerate(images):
            payload = archive.read(image_row["member"])
            if hashlib.sha256(payload).hexdigest() != image_row["sha256"]:
                raise ValueError("Source image bytes changed")
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if (
                image is None
                or hashlib.sha256(image.tobytes()).hexdigest()
                != image_row["native_BGR_sha256"]
            ):
                raise ValueError("Source image pixels changed")
            measurement, mask = primary_contour(
                image, protocol["image_measurement"], previous
            )
            if measurement["integrity_valid"]:
                previous = measurement["centroid_xy"]
            mask_path = f"masks/{frame:06d}.png"
            if not cv2.imwrite(str(output / mask_path), mask):
                raise OSError("Cannot save observation mask")
            time_s = image_row["filename_elapsed_s"]
            match = asof_index(seconds, time_s, source_clock["maximum_asof_age_s"])
            observations.append(
                dict(
                    sample_id=image_row["sample_id"],
                    source_frame_index=frame,
                    original_member=image_row["member"],
                    source_image_sha256=image_row["sha256"],
                    native_BGR_sha256=image_row["native_BGR_sha256"],
                    filename_elapsed_s=time_s,
                    source_log_row_index=match,
                    source_log_time_s=None if match is None else int(seconds[match]),
                    source_log_age_s=None
                    if match is None
                    else int(time_s - seconds[match]),
                    source_log_area_mm2=None
                    if match is None
                    else float(values[match, 7]),
                    mask_path=mask_path,
                    mask_file_sha256=file_sha256(output / mask_path),
                    mask_pixels_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    **measurement,
                )
            )
    settings = protocol["smoothing_diagnostic"]
    smooth = savgol_filter(
        values[:, 5],
        settings["candidate_effective_window"],
        settings["candidate_polynomial_order"],
    )
    cutoff = settings["perturbation_cutoff_row"]
    changed = values[:, 5].copy()
    changed[cutoff + 1 :] += settings["perturbation_delta_mm"]
    changed_smooth = savgol_filter(
        changed,
        settings["candidate_effective_window"],
        settings["candidate_polynomial_order"],
    )
    errors = abs(smooth - values[:, 6])
    future_effect = abs(changed_smooth[: cutoff + 1] - smooth[: cutoff + 1])
    valid = [
        r
        for r in observations
        if r["integrity_valid"]
        and r["source_log_row_index"] is not None
        and r["source_log_area_mm2"] > 0
    ]
    prefix = [
        r
        for r in valid
        if r["filename_elapsed_s"]
        <= protocol["log_scale_diagnostic"]["calibration_prefix_s"]
    ]
    if not prefix:
        raise ValueError("No source-scale diagnostic prefix")
    conversion = float(
        np.median([r["contour_area_px2"] / r["source_log_area_mm2"] for r in prefix])
    )
    groups = {}
    sidecar_scale = protocol["log_scale_diagnostic"]["sidecar_reported_scale"]
    for name, subset in [
        ("prefix", prefix),
        ("later", [r for r in valid if r not in prefix]),
    ]:
        area = np.array([r["contour_area_px2"] for r in subset])
        logged = np.array([r["source_log_area_mm2"] for r in subset])
        groups[name] = dict(
            images=len(subset),
            estimated_area_divisor_px2_per_log_mm2=conversion,
            source_scaled_MAPE=float(np.mean(abs(area / conversion / logged - 1))),
            sidecar_area_divisor_MAPE=float(
                np.mean(abs(area / sidecar_scale / logged - 1))
            ),
            sidecar_linear_mm_per_px_MAPE=float(
                np.mean(abs(area * sidecar_scale**2 / logged - 1))
            ),
        )
    samples, rejected = candidate_samples(
        observations,
        seconds,
        rates,
        protocol["candidate_samples"],
        source_clock["maximum_asof_age_s"],
    )
    for name, collection in [
        ("observations", observations),
        ("candidate_samples", samples),
        ("rejected_candidates", rejected),
    ]:
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in collection)
        )
    reference = dict(
        workbook_sha256=file_sha256(workbook_path),
        data_sheet_range="Data!A1:M28812",
        matched_numeric_rows=len(rows),
        maximum_numeric_difference=workbook_difference,
        parameters={
            key: workbook["Parameters"].get(key)
            for key in [
                "C4",
                "C5",
                "C6",
                "F4",
                "F5",
                "F8",
                "F9",
                "F12",
                "F13",
                "F14",
                "F15",
                "I5",
                "I6",
                "I8",
                "I9",
                "I10",
            ]
        },
        source_workbook_unchanged=file_sha256(workbook_path)
        == protocol["original_workbook"]["sha256"],
    )
    save(output / "workbook_reference.json", reference)
    report = dict(
        protocol_sha256=file_sha256(protocol_path),
        source_series_rows=len(rows),
        observed_images=len(observations),
        integrity_valid_observations=sum(r["integrity_valid"] for r in observations),
        source_clock_representation_max_error_s=clock_error,
        source_association_matches=sum(
            r["source_log_row_index"] is not None for r in observations
        ),
        source_association_max_age_s=max(
            r["source_log_age_s"]
            for r in observations
            if r["source_log_age_s"] is not None
        ),
        origin_offset_assumed_s=0,
        hardware_synchronization_verified=False,
        smoothing=dict(
            effective_window=settings["candidate_effective_window"],
            polynomial_order=settings["candidate_polynomial_order"],
            maximum_reconstruction_error_mm=float(errors.max()),
            exactly_reconstructs_exported_history=bool(
                errors.max() < settings["exact_match_tolerance_mm"]
            ),
            future_only_perturbation_changes_earlier_values=bool(
                future_effect.max() > 1e-6
            ),
            maximum_earlier_change_mm=float(future_effect.max()),
            affected_rows_at_or_before_cutoff=int(
                np.count_nonzero(future_effect > 1e-10)
            ),
        ),
        scale_diagnostics=groups,
        physical_calibration_verified=False,
        candidate_examples=len(samples),
        candidate_rejections=dict(Counter(r["reason"] for r in rejected)),
        candidate_input_fields=list(INPUT_FIELDS),
        reported_rate_availability=protocol["candidate_samples"][
            "reported_rate_availability"
        ],
        excluded_log_input_fields=protocol["candidate_samples"][
            "excluded_log_input_fields"
        ],
        candidate_target_kind="unreviewed source threshold-contour proxy",
        independent_evaluation_groups=0,
        local_human_reviews_added=0,
        new_model_fit=False,
        physical_command=None,
    )
    save(output / "report.json", report)
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data/real_vision_crystalcv_v1"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/crystalcv_process_audit_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.index, args.protocol, args.output)
