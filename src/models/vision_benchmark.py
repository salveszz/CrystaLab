# FILE PURPOSE: Compare fixed-view OpenCV and learned seed measurements on identical development images.
# Missing/ambiguous measurements remain in accuracy denominators. Baseline seed
# coordinates are declared independently of ground-truth masks and held-out tests.
"""Audited seed-silhouette comparison without temporal or physical-quality claims."""

from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from vision.baseline import BaselineConfig, SeedTracker
from vision.training_data import audit_segmentation, polygon_masks, safe_path, save_json
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def measurement_error(truth, mask):
    """Score absent measurements as zero overlap, while leaving area error unavailable."""
    if truth is None:
        return dict(
            truth_present=False,
            valid=mask is not None,
            iou=None,
            dice=None,
            signed_area_error_px=None,
        )
    expected = truth > 0
    if mask is None:
        return dict(
            truth_present=True,
            valid=False,
            iou=0.0,
            dice=0.0,
            signed_area_error_px=None,
        )
    if mask.shape != truth.shape:
        raise ValueError("Measurement mask shape differs from truth")
    observed = mask > 0
    intersection = int(np.count_nonzero(expected & observed))
    union = int(np.count_nonzero(expected | observed))
    size = int(np.count_nonzero(expected) + np.count_nonzero(observed))
    return dict(
        truth_present=True,
        valid=True,
        iou=intersection / union if union else 0.0,
        dice=2 * intersection / size if size else 0.0,
        signed_area_error_px=int(
            np.count_nonzero(observed) - np.count_nonzero(expected)
        ),
    )


def summarize_errors(errors):
    truth = [r for r in errors if r["truth_present"]]
    area = [
        r["signed_area_error_px"]
        for r in truth
        if r["signed_area_error_px"] is not None
    ]
    return dict(
        images=len(errors),
        truth_images=len(truth),
        valid_on_truth=sum(r["valid"] for r in truth),
        valid_coverage_on_truth=sum(r["valid"] for r in truth) / len(truth)
        if truth
        else None,
        false_valid_on_negative=sum(
            r["valid"] for r in errors if not r["truth_present"]
        ),
        mean_iou_all_truth=float(np.mean([r["iou"] for r in truth])) if truth else None,
        mean_dice_all_truth=float(np.mean([r["dice"] for r in truth]))
        if truth
        else None,
        area_error_valid_count=len(area),
        area_mae_px_valid=float(np.mean(np.abs(area))) if area else None,
        area_bias_px_valid=float(np.mean(area)) if area else None,
    )


def select_learned_seed(predictions, shape, root):
    """Require one seed prediction; never choose by overlap with ground truth."""
    seeds = [p for p in predictions if p["class_id"] == 0]
    if len(seeds) != 1:
        return None, "missing_seed" if not seeds else "ambiguous_seed"
    mask = cv2.imread(str(safe_path(root, seeds[0]["mask_path"])), 0)
    if mask is None or mask.shape != shape or not set(np.unique(mask)) <= {0, 255}:
        raise ValueError("Invalid learned seed mask")
    if not np.any(mask):
        return None, "empty_seed"
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        return None, "clipped_seed"
    return mask, "valid"


def calibrate_seed_baseline(dataset, config, protocol, output):
    """Choose a bounded threshold/polarity grid using training polygons only."""
    if protocol.get("schema") != "crystalab_seed_calibration_v1":
        raise ValueError("Declare a baseline calibration protocol")
    if (
        config.get("schema") != "crystalab_seed_benchmark_v1"
        or config.get("mode") != "single_frame_fixed_seed_point"
    ):
        raise ValueError("Declare the fixed-view baseline contract")
    thresholds = protocol["thresholds"]
    polarities = protocol["polarities"]
    if (
        not thresholds
        or not polarities
        or any(v not in {"dark", "bright"} for v in polarities)
    ):
        raise ValueError("Empty/invalid calibration grid")
    if any(
        v is not None and (type(v) is not int or not 0 <= v <= 255) for v in thresholds
    ):
        raise ValueError("Invalid threshold candidate")
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    audit = audit_segmentation(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = [r for r in manifest["records"] if r["split"] == "train"]
    if not rows:
        raise ValueError("Calibration needs training images")
    frames = []
    for row in rows:
        image = cv2.imread(str(dataset / row["image_path"]))
        if list(image.shape[:2]) != config["image_shape"]:
            raise ValueError("Calibration image shape mismatch")
        targets = [
            r["mask"]
            for r in polygon_masks(dataset / row["label_path"], image.shape[:2])
            if r["class_id"] == 0
        ]
        if len(targets) > 1:
            raise ValueError("Calibration needs unambiguous truth seed identity")
        frames.append((image, targets[0] if targets else None))
    if not any(target is not None for _, target in frames):
        raise ValueError("Calibration needs seed truth")
    candidates = []
    for polarity in polarities:
        for threshold in thresholds:
            settings = dict(config["baseline"], polarity=polarity, threshold=threshold)
            baseline = BaselineConfig(**settings)
            errors = []
            for image, truth in frames:
                _, mask = SeedTracker(baseline).measure(image, 0.0)
                errors.append(measurement_error(truth, mask))
            candidates.append(dict(settings=settings, report=summarize_errors(errors)))
    # Primary metric includes missing observations; deterministic grid order resolves ties.
    best = max(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["report"]["mean_iou_all_truth"],
            -candidates[i]["report"]["false_valid_on_negative"],
        ),
    )
    selected = copy.deepcopy(config)
    selected["baseline"] = candidates[best]["settings"]
    selected["calibration"] = dict(
        dataset_manifest_sha256=audit["manifest_sha256"],
        split="train",
        images=len(rows),
        families=len({r["family_key"] for r in rows}),
        objective="mean seed IoU over all training truth including misses as zero",
        candidate_count=len(candidates),
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "selected_config.json", selected)
    report = dict(
        schema="crystalab_seed_calibration_v1",
        status="complete",
        protocol=protocol,
        input_config=config,
        dataset_audit=audit,
        selected_index=best,
        selected_config_sha256=file_sha256(output / "selected_config.json"),
        training_sample_ids=[r["sample_id"] for r in rows],
        candidates=candidates,
        source_sha256={
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in [
                Path(__file__),
                ROOT / "src/vision/baseline.py",
                ROOT / "src/vision/training_data.py",
            ]
        },
        scope="Training-only fixed-view calibration. No validation/test tuning or physical calibration.",
    )
    save_json(output / "manifest.json", report)
    return report


def benchmark_seed(dataset, config, output, *, split="validation", learned=None):
    if split not in {"train", "validation"}:
        raise ValueError("Development seed benchmark excludes final test")
    if (
        config.get("schema") != "crystalab_seed_benchmark_v1"
        or config.get("mode") != "single_frame_fixed_seed_point"
    ):
        raise ValueError("Declare the fixed-view seed benchmark contract")
    baseline = BaselineConfig(**config["baseline"])
    if len(config["image_shape"]) != 2 or any(
        type(v) is not int or v < 1 for v in config["image_shape"]
    ):
        raise ValueError("Declare expected fixed image dimensions")
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    audit = audit_segmentation(dataset)
    data = json.loads((dataset / "manifest.json").read_text())
    rows = [r for r in data["records"] if r["split"] == split]
    if not rows:
        raise ValueError("No images in requested development split")
    prediction_rows = {}
    learned_manifest = None
    if learned is not None:
        learned = Path(learned).resolve()
        learned_manifest = json.loads((learned / "manifest.json").read_text())
        if (
            learned_manifest.get("schema") != "crystalab_segmentation_development_v1"
            or learned_manifest["status"] != "complete"
            or learned_manifest["split"] != split
            or learned_manifest["dataset_audit"]["manifest_sha256"]
            != audit["manifest_sha256"]
        ):
            raise ValueError(
                "Learned predictions do not match the exact dataset and split"
            )
        for path, digest in learned_manifest["artifact_sha256"].items():
            if file_sha256(safe_path(learned, path)) != digest:
                raise ValueError("Learned prediction artifact identity changed")
        predictions = [
            json.loads(line)
            for line in (learned / "predictions.jsonl").read_text().splitlines()
        ]
        prediction_rows = {r["sample_id"]: r for r in predictions}
        if len(prediction_rows) != len(predictions) or set(prediction_rows) != {
            r["sample_id"] for r in rows
        }:
            raise ValueError("Missing/duplicate learned development image identity")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    (output / "overlays").mkdir()
    manifest = dict(
        schema="crystalab_seed_benchmark_v1",
        status="running",
        config=config,
        split=split,
        dataset_audit=audit,
        learned_manifest_sha256=file_sha256(learned / "manifest.json")
        if learned
        else None,
        learned_path=str(learned) if learned else None,
        learned_weights_sha256=learned_manifest["weights_sha256"] if learned else None,
        scope="Single-frame projected seed accuracy. OpenCV receives a fixed declared seed point; learned seed selection requires exactly one class-0 mask. No physical calibration, temporal tracking or test scoring.",
    )
    sources = [
        Path(__file__),
        ROOT / "src/vision/baseline.py",
        ROOT / "src/vision/training_data.py",
    ]
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): file_sha256(p) for p in sources
    }
    save_json(output / "manifest.json", manifest)
    try:
        for relative, digest in manifest["source_sha256"].items():
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            if file_sha256(target) != digest:
                raise ValueError("Benchmark source changed during snapshot")
        records = []
        for index, row in enumerate(rows):
            image = cv2.imread(str(dataset / row["image_path"]))
            if list(image.shape[:2]) != config["image_shape"]:
                raise ValueError(
                    "Image shape differs from declared fixed-view benchmark"
                )
            truth = [
                r["mask"]
                for r in polygon_masks(dataset / row["label_path"], image.shape[:2])
                if r["class_id"] == 0
            ]
            if len(truth) > 1:
                raise ValueError("Seed identity needs one truth seed per image")
            truth = truth[0] if truth else None
            started = time.perf_counter()
            measured, mask = SeedTracker(baseline).measure(image, 0.0)
            baseline_seconds = time.perf_counter() - started
            errors = measurement_error(truth, mask)
            record = dict(
                sample_id=row["sample_id"],
                family_key=row["family_key"],
                material=row["material"],
                evidence_kind=row["evidence_kind"],
                image_path=row["image_path"],
                fixed_seed_point_inside_truth=bool(
                    truth[baseline.seed_point[1], baseline.seed_point[0]]
                )
                if truth is not None
                else None,
                baseline=dict(
                    measured=measured,
                    error=errors,
                    measure_wall_seconds=baseline_seconds,
                ),
            )
            if mask is not None:
                mask_path = f"masks/{index:06d}_baseline.png"
                if not cv2.imwrite(str(output / mask_path), mask):
                    raise OSError("Cannot save baseline mask")
                record["baseline"]["mask_path"] = mask_path
            learned_mask = None
            if learned is not None:
                prediction = prediction_rows[row["sample_id"]]
                if (
                    prediction["family_key"] != row["family_key"]
                    or prediction["source_image"] != row["image_path"]
                ):
                    raise ValueError("Learned source image/family mismatch")
                learned_mask, status = select_learned_seed(
                    prediction["predictions"], image.shape[:2], learned
                )
                record["learned"] = dict(
                    status=status,
                    error=measurement_error(truth, learned_mask),
                    predict_wall_seconds=prediction["predict_wall_seconds"],
                )
            overlay = image.copy()
            for item, color in [
                (truth, (0, 180, 0)),
                (mask, (255, 0, 0)),
                (learned_mask, (0, 0, 255)),
            ]:
                if item is None:
                    continue
                contours, _ = cv2.findContours(
                    item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, 1)
            cv2.putText(
                overlay,
                "GT green / CV blue / YOLO red",
                (3, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 20, 20),
                1,
            )
            if not cv2.imwrite(str(output / f"overlays/{index:06d}.png"), overlay):
                raise OSError("Cannot save benchmark overlay")
            records.append(record)
        report = {
            name: summarize_errors([r[name]["error"] for r in records])
            for name in ["baseline"] + (["learned"] if learned else [])
        }
        per_family = {
            family: {
                name: summarize_errors(
                    [r[name]["error"] for r in records if r["family_key"] == family]
                )
                for name in report
            }
            for family in sorted({r["family_key"] for r in records})
        }
        (output / "measurements.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
        )
        manifest.update(
            status="complete",
            images=len(records),
            families=len(per_family),
            report=report,
            truth_images_with_fixed_point_inside=sum(
                r["fixed_seed_point_inside_truth"] is True for r in records
            ),
            per_family=per_family,
            images_by_evidence=dict(Counter(r["evidence_kind"] for r in rows)),
            baseline_invalid_reasons=dict(
                Counter(
                    r["baseline"]["measured"]["reason"]
                    for r in records
                    if not r["baseline"]["measured"]["valid"]
                )
            ),
            baseline_median_measure_seconds=float(
                np.median([r["baseline"]["measure_wall_seconds"] for r in records])
            ),
            artifact_sha256={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "manifest.json", manifest)
        raise
