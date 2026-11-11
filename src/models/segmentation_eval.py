# FILE PURPOSE: Inspect local segmentation weights on development images with explicit mask metrics.
# Synthetic polygons remain proxies. Test-split scoring and temporal/physical claims
# are intentionally outside this development evaluation entry point.
"""Save prediction masks, evidence overlays and class-wise instance matching."""

import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from models.segmentation import local_yolo, load_run_metadata
from vision.training_data import (
    CLASSES,
    audit_segmentation,
    polygon_masks,
    save_json,
)
from vision.video import file_sha256


def match_instances(truth, predictions, threshold=0.5):
    """Maximize same-class matches above IoU gate, then total IoU; keep all misses/FPs."""
    scores = np.zeros((len(truth), len(predictions)))
    ious, dices = np.zeros_like(scores), np.zeros_like(scores)
    for i, target in enumerate(truth):
        for j, prediction in enumerate(predictions):
            if target["class_id"] != prediction["class_id"]:
                continue
            a, b = target["mask"] > 0, prediction["mask"] > 0
            intersection = np.count_nonzero(a & b)
            union = np.count_nonzero(a | b)
            ious[i, j] = intersection / union if union else 0.0
            size = np.count_nonzero(a) + np.count_nonzero(b)
            dices[i, j] = 2 * intersection / size if size else 0.0
            if ious[i, j] >= threshold:
                scores[i, j] = 1 + ious[i, j] / (max(scores.shape) + 1)
    matches = []
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        matches = [
            dict(
                truth_index=int(i),
                prediction_index=int(j),
                iou=float(ious[i, j]),
                dice=float(dices[i, j]),
            )
            for i, j in zip(rows, columns)
            if scores[i, j] > 0
        ]
    return matches


def evaluate_development(
    run, dataset, output, *, split="validation", confidence=0.25, iou_gate=0.5
):
    if split not in {"train", "validation"}:
        raise ValueError("This development command excludes the final test split")
    if not 0 < confidence < 1 or not 0 < iou_gate <= 1:
        raise ValueError("Invalid confidence/IoU gate")
    run, dataset, output = (
        Path(run).resolve(),
        Path(dataset).resolve(),
        Path(output).resolve(),
    )
    model_manifest, weights = load_run_metadata(run)
    audit = audit_segmentation(dataset)
    source = json.loads((dataset / "manifest.json").read_text())
    rows = [r for r in source["records"] if r["split"] == split]
    if not rows:
        raise ValueError("No development images in requested split")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema="crystalab_segmentation_development_v1",
        status="running",
        model_manifest_sha256=file_sha256(run / "manifest.json"),
        weights_sha256=file_sha256(weights),
        dataset_audit=audit,
        split=split,
        confidence=confidence,
        iou_gate=iou_gate,
        source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[2])): file_sha256(
                path
            )
            for path in [
                Path(__file__),
                Path(__file__).with_name("segmentation.py"),
                Path(__file__).resolve().parents[2] / "src/vision/training_data.py",
            ]
        },
        scope="Image-instance diagnostic against YOLO visible polygons; synthetic labels are proxies. No temporal events, final-test or real KDP control claim.",
    )
    save_json(output / "manifest.json", manifest)
    try:
        project_root = Path(__file__).resolve().parents[2]
        for relative, digest in manifest["source_sha256"].items():
            target = output / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project_root / relative, target)
            if file_sha256(target) != digest:
                raise ValueError("Evaluation source changed during snapshot")
        YOLO, versions = local_yolo(output / "settings", model_manifest["runtime"])
        import torch

        torch.set_num_threads(model_manifest["config"]["threads"])
        model = YOLO(str(weights), task="segment")
        if [model.names[i] for i in range(len(CLASSES))] != CLASSES:
            raise ValueError("Trained class map differs from dataset")
        totals = {
            i: dict(tp=0, fp=0, fn=0, truth=0, iou_sum=0.0, dice_sum=0.0)
            for i in range(len(CLASSES))
        }
        latencies, diagnostics = [], []
        (output / "masks").mkdir()
        (output / "overlays").mkdir()
        for index, row in enumerate(rows):
            image = cv2.imread(str(dataset / row["image_path"]))
            truth = polygon_masks(dataset / row["label_path"], image.shape[:2])
            started = time.perf_counter()
            result = model.predict(
                image,
                device="cpu",
                imgsz=model_manifest["config"]["imgsz"],
                conf=confidence,
                retina_masks=True,
                verbose=False,
                save=False,
            )[0]
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            predictions = []
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                ids = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()
                if masks.shape[1:] != image.shape[:2]:
                    raise ValueError(
                        "Prediction masks did not preserve original image shape"
                    )
                for j, (mask, class_id, score) in enumerate(zip(masks, ids, scores)):
                    mask = (mask > 0.5).astype(np.uint8) * 255
                    mask_path = f"masks/{index:06d}_{j:03d}.png"
                    if not cv2.imwrite(str(output / mask_path), mask):
                        raise OSError("Cannot save prediction mask")
                    predictions.append(
                        dict(
                            class_id=int(class_id),
                            confidence=float(score),
                            mask=mask,
                            mask_path=mask_path,
                        )
                    )
            matches = match_instances(truth, predictions, iou_gate)
            matched_truth = {m["truth_index"] for m in matches}
            matched_predictions = {m["prediction_index"] for m in matches}
            for i, t in enumerate(truth):
                total = totals[t["class_id"]]
                total["truth"] += 1
                total["fn"] += int(i not in matched_truth)
            for j, p in enumerate(predictions):
                totals[p["class_id"]]["fp"] += int(j not in matched_predictions)
            for match in matches:
                total = totals[truth[match["truth_index"]]["class_id"]]
                total["tp"] += 1
                total["iou_sum"] += match["iou"]
                total["dice_sum"] += match["dice"]
            overlay = image.copy()
            for items, color in [(truth, (0, 200, 0)), (predictions, (0, 0, 255))]:
                for obj in items:
                    contours, _ = cv2.findContours(
                        obj["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(overlay, contours, -1, color, 1)
            cv2.putText(
                overlay,
                "GT green / model red",
                (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (30, 30, 30),
                1,
            )
            overlay_path = f"overlays/{index:06d}.png"
            if not cv2.imwrite(str(output / overlay_path), overlay):
                raise OSError("Cannot save prediction overlay")
            diagnostics.append(
                dict(
                    sample_id=row["sample_id"],
                    family_key=row["family_key"],
                    evidence_kind=row["evidence_kind"],
                    material=row["material"],
                    source_image=row["image_path"],
                    truth_objects=len(truth),
                    predictions=[
                        {k: v for k, v in p.items() if k != "mask"} for p in predictions
                    ],
                    matches=matches,
                    predict_wall_seconds=elapsed,
                    overlay_path=overlay_path,
                )
            )
        per_class = {}
        for i, total in totals.items():
            tp, fp, fn = total["tp"], total["fp"], total["fn"]
            per_class[CLASSES[i]] = dict(
                total,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                mean_iou_per_truth_with_misses_zero=total["iou_sum"] / total["truth"]
                if total["truth"]
                else None,
                mean_dice_per_truth_with_misses_zero=total["dice_sum"] / total["truth"]
                if total["truth"]
                else None,
            )
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in diagnostics)
        )
        manifest.update(
            status="complete",
            images=len(rows),
            families=len({r["family_key"] for r in rows}),
            images_by_evidence={
                k: sum(r["evidence_kind"] == k for r in rows)
                for k in {r["evidence_kind"] for r in rows}
            },
            packages=versions,
            training_environment=model_manifest.get("environment"),
            per_class=per_class,
            latency=dict(
                first_predict_seconds=latencies[0],
                subsequent_median_seconds=float(np.median(latencies[1:]))
                if len(latencies) > 1
                else None,
                all_predict_p95_seconds=float(np.percentile(latencies, 95)),
                scope="Model predict call on CPU including first-call setup; excludes data read, audit and artifact writing",
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
