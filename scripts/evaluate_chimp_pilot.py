# FILE PURPOSE: Evaluate the frozen last checkpoint against original published
# selected-instance masks, preserving native coordinates, misses and unmatched
# predictions. Source-selected agreement is not exhaustive KDP defect accuracy.
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.segmentation import load_run_metadata, local_yolo
from vision.published_mask_training import audit_training
from vision.training_data import save_json
from vision.video import file_sha256


def native_ious(truth, predictions):
    """Exact native-pixel intersections through packed bits, bounded by one prediction."""
    result = np.zeros((len(truth), len(predictions)))
    if not truth or not predictions:
        return result
    shape = truth[0]["mask"].shape
    if any(r["mask"].shape != shape for r in truth + predictions):
        raise ValueError("Instance masks must share native image coordinates")
    packed_truth = np.array([np.packbits(r["mask"].ravel() > 0) for r in truth])
    sizes = np.array([np.count_nonzero(r["mask"]) for r in truth])
    counts = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    for j, prediction in enumerate(predictions):
        packed = np.packbits(prediction["mask"].ravel() > 0)
        intersection = counts[np.bitwise_and(packed_truth, packed)].sum(
            axis=1, dtype=np.int64
        )
        union = sizes + np.count_nonzero(prediction["mask"]) - intersection
        result[:, j] = np.divide(
            intersection, union, out=np.zeros(len(truth)), where=union > 0
        )
    return result


def source_matches(truth, predictions, gate):
    matrix = native_ious(truth, predictions)
    matches = []
    for class_id in range(2):
        ti = [i for i, r in enumerate(truth) if r["class_id"] == class_id]
        pi = [i for i, r in enumerate(predictions) if r["class_id"] == class_id]
        if ti and pi:
            selected = matrix[np.ix_(ti, pi)]
            rr, cc = linear_sum_assignment(-selected)
            matches.extend(
                dict(
                    truth_index=ti[i], prediction_index=pi[j], iou=float(selected[i, j])
                )
                for i, j in zip(rr, cc)
                if selected[i, j] >= gate
            )
    return matches


def summarize(rows, classes):
    output = {}
    for class_id, name in enumerate(classes):
        targets = sum(r["truth_classes"].count(class_id) for r in rows)
        predictions = sum(
            sum(p["class_id"] == class_id for p in r["predictions"]) for r in rows
        )
        overlaps = [
            m["iou"]
            for r in rows
            for m in r["matches"]
            if r["truth_classes"][m["truth_index"]] == class_id
        ]
        matched = len(overlaps)
        output[name] = dict(
            selected_targets=targets,
            predictions=predictions,
            matched_selected_targets=matched,
            missed_selected_targets=targets - matched,
            unmatched_predictions=predictions - matched,
            selected_target_recall=matched / targets if targets else None,
            source_task_precision=matched / predictions if predictions else None,
            matched_mean_native_iou=sum(overlaps) / matched if matched else None,
        )
    return dict(images=len(rows), classes=output)


def evaluate(dataset, run, protocol_path, output):
    dataset, run, output = (
        Path(dataset).resolve(),
        Path(run).resolve(),
        Path(output).resolve(),
    )
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["evaluation"]
    if settings["checkpoint"] != "last":
        raise ValueError("Use the frozen final epoch, not selected validation weights")
    data_audit = audit_training(dataset)
    if file_sha256(dataset / "protocol.json") != file_sha256(protocol_path):
        raise ValueError("Dataset differs from frozen pilot")
    trained, _ = load_run_metadata(run)
    if trained["dataset_task"] != "published_selected_protein_instances" or trained[
        "dataset_path"
    ] != str(dataset):
        raise ValueError("Require the separate published-mask task and source dataset")
    weights = run / trained["last_weights"]
    if file_sha256(weights) != trained["last_weights_sha256"]:
        raise ValueError("Last-checkpoint identity changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "evaluator_snapshot.py")
    records = [
        r for r in json.loads((dataset / "records.json").read_text()) if r["eligible"]
    ]
    model_class, versions = local_yolo(output / "settings", trained["runtime"])
    import torch

    torch.set_num_threads(trained["config"]["threads"])
    model = model_class(str(weights), task="segment")
    if list(model.names.values()) != protocol["classes"]:
        raise ValueError("Source-selected class map changed")
    manifest = dict(
        status="running",
        dataset_manifest_sha256=file_sha256(dataset / "manifest.json"),
        run_manifest_sha256=file_sha256(run / "manifest.json"),
        last_weights_sha256=file_sha256(weights),
        protocol_sha256=file_sha256(protocol_path),
        evaluator_sha256=file_sha256(Path(__file__)),
        runtime=versions,
        scope=protocol["class_semantics"],
        physical_run_grouping_verified=False,
        independent_KDP_evaluation_groups=0,
        model_promoted=False,
        physical_command=None,
    )
    save_json(output / "manifest.json", manifest)
    scores = []
    for i, row in enumerate(records):
        image = cv2.imread(str(dataset / row["image_path"]))
        truth = [
            dict(
                class_id=protocol["classes"].index(o["source_class"]),
                mask=cv2.imread(str(dataset / o["mask_path"]), 0),
            )
            for o in row["objects"]
        ]
        started = time.perf_counter()
        result = model.predict(
            image,
            device="cpu",
            imgsz=settings["imgsz"],
            conf=settings["confidence"],
            iou=settings["nms_iou"],
            max_det=settings["max_det"],
            retina_masks=settings["retina_masks"],
            verbose=False,
            save=False,
        )[0]
        elapsed = time.perf_counter() - started
        predicted = []
        if result.masks is not None:
            for j, (mask, class_id, confidence) in enumerate(
                zip(
                    result.masks.data.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                )
            ):
                binary = (mask > 0.5).astype(np.uint8) * 255
                if binary.shape != image.shape[:2] or int(class_id) not in [0, 1]:
                    raise ValueError(
                        "Predictions differ from native source/task coordinates"
                    )
                path = f"predictions/{i:05d}/{j:04d}.png"
                (output / path).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(output / path), binary):
                    raise OSError("Cannot save native prediction")
                predicted.append(
                    dict(
                        class_id=int(class_id),
                        confidence=float(confidence),
                        mask=binary,
                        mask_path=path,
                    )
                )
        matches = source_matches(truth, predicted, settings["match_iou"])
        scores.append(
            dict(
                sample_id=row["sample_id"],
                split=row["split"],
                source_dataset=row["source_dataset"],
                truth_classes=[r["class_id"] for r in truth],
                predictions=[
                    {k: v for k, v in r.items() if k != "mask"} for r in predicted
                ],
                matches=matches,
                prediction_wall_seconds=elapsed,
            )
        )
        if (i + 1) % 16 == 0:
            print(
                json.dumps(dict(images_evaluated=i + 1, total_images=len(records))),
                flush=True,
            )
    report = dict(
        dataset_audit=data_audit,
        total_images=len(scores),
        metrics={
            split: summarize(
                [r for r in scores if r["split"] == split], protocol["classes"]
            )
            for split in ["train", "validation"]
        },
        per_source_validation={
            name: summarize(
                [
                    r
                    for r in scores
                    if r["split"] == "validation" and r["source_dataset"] == name
                ],
                protocol["classes"],
            )
            for name in protocol["selection"]["datasets"]
        },
        source_empty_annotation_images=dict(
            Counter(r["split"] for r in scores if not r["truth_classes"])
        ),
        independent_KDP_evaluation_groups=0,
        new_human_review_receipts=0,
        model_promoted=False,
        physical_command=None,
    )
    save_json(output / "scores.json", scores)
    save_json(output / "report.json", report)
    manifest.update(
        status="complete",
        files={
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )
    save_json(output / "manifest.json", manifest)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/chimp_mask_pilot.json"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    evaluate(a.dataset, a.run, a.protocol, a.output)
