# FILE PURPOSE: Train the pitch's YOLOv8n-seg locally with pinned inputs and bounded experiments.
# Input data is independently audited; settings disable cloud logging and automatic
# downloads. Real accuracy and physical control are outside a synthetic first fit.
"""Reproducible local YOLOv8-seg training with retained failure manifests."""

from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

from vision.training_data import audit_segmentation, safe_path, save_json
from vision.video import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def load_run_metadata(run):
    """Verify a completed training run and its exact weight/source artifacts before loading."""
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest.get("schema") != "crystalab_yolov8_seg_run_v1"
        or manifest["status"] != "complete"
    ):
        raise ValueError("A completed local training run is required")
    weights = safe_path(run, manifest["best_weights"])
    if file_sha256(weights) != manifest["best_weights_sha256"]:
        raise ValueError("Trained weight identity changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_sha256(safe_path(run / "source_snapshot", relative)) != digest:
            raise ValueError("Training source snapshot changed")
    return manifest, weights


def training_options(config):
    if config.get("schema") != "crystalab_yolov8_seg_train_v1" or config.get(
        "purpose"
    ) not in {"plumbing_check", "overfit_check", "grouped_experiment"}:
        raise ValueError("Declare a supported training experiment purpose")
    for name in ("epochs", "imgsz", "batch", "threads"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"Invalid training {name}")
    if config["imgsz"] < 64 or config["imgsz"] % 32:
        raise ValueError("Image size must be a multiple of 32 and at least 64")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    if config["device"] != "cpu":
        raise ValueError("This initial runtime is characterized for CPU only")
    learning_rate = config["learning_rate"]
    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate < 1
    ):
        raise ValueError("Invalid learning rate")
    if config["purpose"] == "overfit_check" and (
        type(config.get("overfit_samples")) is not int or config["overfit_samples"] < 1
    ):
        raise ValueError(
            "Overfit check needs an explicit positive training-image count"
        )
    return dict(
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        seed=config["seed"],
        device="cpu",
        workers=0,
        deterministic=True,
        amp=False,
        cache=False,
        plots=False,
        save=True,
        save_period=-1,
        patience=config["epochs"],
        optimizer="AdamW",
        lr0=float(learning_rate),
        nbs=config["batch"],
        pretrained=True,
        val=True,
        verbose=False,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
    )


def local_yolo(config_dir, runtime):
    """Initialize optional dependencies only when training/inference is explicitly called."""
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("ultralytics", "torch", "torchvision")
    }
    for name, version in versions.items():
        if version.split("+", 1)[0] != runtime[name]:
            raise ValueError(f"Pinned {name} runtime mismatch: {version}")
    from ultralytics import YOLO, settings

    settings.update(
        {
            key: False
            for key in (
                "sync",
                "clearml",
                "comet",
                "dvc",
                "mlflow",
                "raytune",
                "tensorboard",
                "wandb",
            )
        }
    )
    return YOLO, versions


def train_segmentation(dataset, config, runtime, output):
    options = training_options(config)
    dataset = Path(dataset).resolve()
    audit = audit_segmentation(dataset)
    weights = (ROOT / runtime["local_path"]).resolve()
    if (
        runtime["architecture"] != "YOLOv8n-seg"
        or file_sha256(weights) != runtime["sha256"]
    ):
        raise ValueError("Explicit YOLOv8n-seg pretrained-weight identity required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path(__file__),
        ROOT / "src/vision/training_data.py",
        ROOT / "scripts/train_vision.py",
    ]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        source_hashes[str(relative)] = file_sha256(path)
    manifest = dict(
        schema="crystalab_yolov8_seg_run_v1",
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        runtime=runtime,
        train_options=options,
        dataset_path=str(dataset),
        dataset_audit=audit,
        source_sha256=source_hashes,
        evaluation_scope="training and validation only; final test split is not scored",
        scope="Synthetic first fits establish software behavior; real performance requires independently reviewed real observations",
    )
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        YOLO, versions = local_yolo(output / "settings", runtime)
        import torch

        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        manifest["environment"] = dict(
            python=platform.python_version(),
            packages=versions,
            platform=platform.platform(),
            processor=platform.processor(),
            cpu_count=os.cpu_count(),
            training_threads=config["threads"],
            cuda_available=torch.cuda.is_available(),
            supporting_packages={
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "opencv-python",
                    "pillow",
                    "scipy",
                    "matplotlib",
                    "ultralytics-thop",
                    "polars",
                )
            },
        )
        data_path = dataset / "dataset.yaml"
        if config["purpose"] == "overfit_check":
            source_manifest = json.loads((dataset / "manifest.json").read_text())
            train_rows = [
                r for r in source_manifest["records"] if r["split"] == "train"
            ]
            if config["overfit_samples"] > len(train_rows):
                raise ValueError("Overfit subset exceeds available training images")
            selected = train_rows[: config["overfit_samples"]]
            image_list = output / "overfit_images.txt"
            image_list.write_text(
                "\n".join(str(dataset / r["image_path"]) for r in selected) + "\n"
            )
            data = json.loads(data_path.read_text())
            data.update(train=str(image_list), val=str(image_list))
            data.pop("test", None)
            data_path = output / "overfit_dataset.yaml"
            save_json(data_path, data)
            manifest["evaluation_scope"] = (
                "overfit diagnostic: same training images reused for validation; no holdout accuracy"
            )
            manifest["overfit_sample_ids"] = [r["sample_id"] for r in selected]
        save_json(output / "manifest.json", manifest)
        model = YOLO(str(weights), task="segment")
        if model.task != "segment":
            raise ValueError("Expected an instance-segmentation model")
        metrics = model.train(
            data=str(data_path),
            project=str(output / "ultralytics"),
            name="train",
            exist_ok=False,
            **options,
        )
        run_dir = Path(model.trainer.save_dir)
        best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
        if not best.exists() or not last.exists():
            raise ValueError(
                "Training ended without both expected local weight artifacts"
            )
        manifest.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            metrics={k: float(v) for k, v in metrics.results_dict.items()},
            best_weights=str(best.relative_to(output)),
            best_weights_sha256=file_sha256(best),
            last_weights=str(last.relative_to(output)),
            last_weights_sha256=file_sha256(last),
            results_csv=str((run_dir / "results.csv").relative_to(output)),
        )
        if audit_segmentation(dataset) != audit:
            raise ValueError("Dataset changed during training")
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise
