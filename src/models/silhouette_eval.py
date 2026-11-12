# FILE PURPOSE: Compare fixed-budget source-fit silhouette models on common proxy targets.
# Original mask boundaries remain weak author-shape projections; no real defect accuracy is inferred.
"""Last-checkpoint evaluation with a training-only template and all misses retained."""

from collections import defaultdict
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from models.segmentation import load_run_metadata, local_yolo
from vision.silhouette_training import CLASSES, audit_silhouette
from vision.training_data import save_json
from vision.video import file_sha256


def choose_instance(instances, shape):
    if not instances:
        return np.zeros(shape, np.uint8), None, 0
    selected = max(instances, key=lambda v: v["score"])
    if selected["mask"].shape != shape or selected["mask"].dtype != np.uint8:
        raise ValueError("Predicted mask must retain original image dimensions")
    return selected["mask"], selected["score"], max(0, len(instances) - 1)


def mask_score(predicted, target):
    p, t = predicted > 0, target > 0
    if p.shape != t.shape or not t.any():
        raise ValueError("A nonempty common target is required")
    intersection = int(np.count_nonzero(p & t))
    union = int(np.count_nonzero(p | t))
    return dict(
        proxy_mask_iou=intersection / union,
        proxy_dice=2 * intersection / (int(p.sum()) + int(t.sum())),
        target_area_px2=int(t.sum()),
        predicted_area_px2=int(p.sum()),
        area_absolute_error_px2=abs(int(p.sum()) - int(t.sum())),
        area_absolute_error_image_fraction=abs(int(p.sum()) - int(t.sum())) / t.size,
        missing_prediction=not p.any(),
    )


def summarize(rows):
    def metrics(rr):
        families = defaultdict(list)
        for r in rr:
            families[r["family_key"]].append(r["proxy_mask_iou"])
        return dict(
            images=len(rr),
            families=len(families),
            proxy_mask_iou_mean=float(np.mean([r["proxy_mask_iou"] for r in rr])),
            proxy_mask_iou_family_macro=float(
                np.mean([np.mean(v) for v in families.values()])
            ),
            proxy_dice_mean=float(np.mean([r["proxy_dice"] for r in rr])),
            missing_predictions=sum(r["missing_prediction"] for r in rr),
            extra_predictions=sum(r["extra_predictions"] for r in rr),
            area_mae_image_fraction=float(
                np.mean([r["area_absolute_error_image_fraction"] for r in rr])
            ),
        )

    return dict(
        all=metrics(rows),
        by_evidence={
            kind: metrics([r for r in rows if r["evidence_kind"] == kind])
            for kind in sorted({r["evidence_kind"] for r in rows})
        },
        per_family={
            family: metrics([r for r in rows if r["family_key"] == family])
            for family in sorted({r["family_key"] for r in rows})
        },
    )


def evaluate_silhouette(study_path, output):
    study_path = Path(study_path).resolve()
    study = json.loads(study_path.read_text())
    output = Path(output).resolve()
    if (
        study["schema"] != "crystalab_source_silhouette_experiment_v1"
        or study["selection"]["evaluation_checkpoint"] != "last"
    ):
        raise ValueError("Frozen last-checkpoint silhouette experiment required")
    for path, digest in study["source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("Frozen silhouette producer changed")
    datasets = {}
    validation = None
    for case in study["cases"]:
        if file_sha256(case["config"]) != case["config_sha256"]:
            raise ValueError("Frozen silhouette training configuration changed")
        root = Path(case["dataset"])
        audit_silhouette(root)
        if file_sha256(root / "manifest.json") != case["dataset_manifest_sha256"]:
            raise ValueError("Frozen silhouette cohort changed")
        rows = json.loads((root / "records.json").read_text())
        datasets[case["mode"]] = (root, rows)
        current = sorted(
            (
                r["sample_id"],
                r["family_key"],
                r["source_image_sha256"],
                r["source_mask_sha256"],
            )
            for r in rows
            if r["split"] == "validation"
        )
        if validation is not None and current != validation:
            raise ValueError("All models must use exactly the same validation targets")
        validation = current
    real_root, real_rows = datasets["source_projection_only"]
    training = [r for r in real_rows if r["split"] == "train"]
    if not training or any(r["evidence_kind"] != "real" for r in training):
        raise ValueError("Template requires only real training source projections")
    size = study["selection"]["template_size"]
    masks = [
        cv2.resize(
            cv2.imread(str(real_root / r["mask_path"]), 0),
            (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
        for r in training
    ]
    template = (np.mean(masks, axis=0) >= 0.5).astype(np.uint8) * 255
    output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output / "real_training_mean_template.png"), template)
    source_target = output / "source_snapshot/models/silhouette_eval.py"
    source_target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, source_target)
    manifest = dict(
        schema="crystalab_source_silhouette_evaluation_v1",
        status="running",
        study_sha256=file_sha256(study_path),
        source_sha256=file_sha256(source_target),
        evaluation_checkpoint="last",
        final_test_scored=False,
        pixel_reviewed=False,
        scope="Proxy agreement with source-author 3D projections and synthetic optical masks; not reviewed real pixels, KDP defect accuracy or physical benefit",
    )
    save_json(output / "manifest.json", manifest)
    scores = []
    reports = {}
    targets = [r for r in real_rows if r["split"] == "validation"]
    for r in targets:
        target = cv2.imread(str(real_root / r["mask_path"]), 0)
        prediction = cv2.resize(
            template,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        relative = f"predictions/template/{r['sample_id'].replace(':', '_')}.png"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), prediction)
        scores.append(
            dict(
                method="real_training_mean_template",
                sample_id=r["sample_id"],
                family_key=r["family_key"],
                evidence_kind=r["evidence_kind"],
                target_mask=r["source_mask"],
                target_sha256=r["source_mask_sha256"],
                predicted_mask=relative,
                score=None,
                extra_predictions=0,
                **mask_score(prediction, target),
            )
        )
    reports["real_training_mean_template"] = summarize(scores)
    YOLO = None
    models = {}
    for case in study["cases"]:
        run = Path(case["run"])
        trained, _ = load_run_metadata(run)
        if trained.get("dataset_task") != "source_fit_silhouette":
            raise ValueError("A separate silhouette fit is required")
        weights = run / trained["last_weights"]
        if file_sha256(weights) != trained["last_weights_sha256"]:
            raise ValueError("Frozen last weights changed")
        if trained["dataset_path"] != str(Path(case["dataset"]).resolve()):
            raise ValueError("Training run belongs to another dataset")
        if trained["config"] != json.loads(Path(case["config"]).read_text()):
            raise ValueError("Training differs from the frozen budget")
        if YOLO is None:
            YOLO, _ = local_yolo(output / "settings", trained["runtime"])
            import torch

            torch.set_num_threads(trained["config"]["threads"])
            torch.set_num_interop_threads(1)
        model = YOLO(str(weights), task="segment")
        if list(model.names.values()) != CLASSES:
            raise ValueError("Silhouette model class map changed")
        local = []
        dataset, rows = datasets[case["mode"]]
        for r in rows:
            if r["split"] != "validation":
                continue
            image = cv2.imread(str(dataset / r["image_path"]))
            target = cv2.imread(str(dataset / r["mask_path"]), 0)
            inference = study["inference"]
            prediction = model.predict(
                source=image,
                device="cpu",
                imgsz=inference["imgsz"],
                conf=inference["confidence"],
                iou=inference["nms_iou"],
                max_det=inference["maximum_detections"],
                retina_masks=True,
                verbose=False,
                save=False,
            )[0]
            instances = []
            if prediction.masks is not None:
                for mask, box in zip(
                    prediction.masks.data.cpu().numpy(), prediction.boxes
                ):
                    if int(box.cls.item()) != 0:
                        raise ValueError("Unknown silhouette class")
                    instances.append(
                        dict(
                            mask=(mask > 0.5).astype(np.uint8) * 255,
                            score=float(box.conf.item()),
                        )
                    )
            selected, score, extra = choose_instance(instances, target.shape)
            relative = (
                f"predictions/{case['mode']}/{r['sample_id'].replace(':', '_')}.png"
            )
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), selected)
            local.append(
                dict(
                    method=case["mode"],
                    sample_id=r["sample_id"],
                    family_key=r["family_key"],
                    evidence_kind=r["evidence_kind"],
                    target_mask=r["source_mask"],
                    target_sha256=r["source_mask_sha256"],
                    predicted_mask=relative,
                    score=score,
                    extra_predictions=extra,
                    **mask_score(selected, target),
                )
            )
        scores.extend(local)
        reports[case["mode"]] = summarize(local)
        models[case["mode"]] = dict(
            run=str(run),
            manifest_sha256=file_sha256(run / "manifest.json"),
            last_weights_sha256=trained["last_weights_sha256"],
        )
    save_json(output / "scores.json", scores)
    save_json(output / "metrics.json", reports)
    manifest.update(
        status="complete",
        models=models,
        images_per_method=len(targets),
        score_records=len(scores),
        metrics=reports,
        template_training_sample_ids=[r["sample_id"] for r in training],
        files={
            str(p.relative_to(output)): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p != output / "manifest.json"
        },
    )
    save_json(output / "manifest.json", manifest)
    return manifest
