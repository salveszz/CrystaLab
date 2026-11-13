# FILE PURPOSE: Independently check published-mask pilot lineage and native scores
# using Pillow and Python integer bitsets, plus four fixed neural replays.
# No producer/evaluator imports or new human review receipts.
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(4 * 1024**2):
            h.update(data)
    return h.hexdigest()


def plane(path):
    return np.asarray(Image.open(path).convert("L")) > 0


def bits(mask):
    return int.from_bytes(np.packbits(mask.ravel()).tobytes(), "little")


def audit(dataset, run, evaluation, protocol_path, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    assert protocol["pitch_class_mapping"] is None and protocol["classes"] == [
        "crystal",
        "drop",
    ]
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert (
        sha(dataset / "protocol.json")
        == sha(protocol_path)
        == manifest["protocol_sha256"]
    )
    for name, digest in manifest["files"].items():
        assert sha(dataset / name) == digest
    index = Path(manifest["source_index"])
    assert (
        sha(index / "manifest.json") == protocol["source_manifest_sha256"]
        and sha(index / "images.jsonl") == protocol["source_index_sha256"]
    )
    source_manifest = json.loads((index / "manifest.json").read_text())
    assert (
        sha(source_manifest["archive_path"])
        == source_manifest["protocol"]["archive_sha256"]
    )
    original = {
        r["sample_id"]: r
        for r in [
            json.loads(line)
            for line in (index / "images.jsonl").read_text().splitlines()
        ]
    }
    selection, chosen = protocol["selection"], []
    for split, n in [
        ("train", selection["train_per_dataset"]),
        ("validation", selection["validation_per_dataset"]),
    ]:
        for name in selection["datasets"]:
            available = [
                r
                for r in original.values()
                if r["source_split"] == split and r["source_dataset"] == name
            ]
            chosen.extend(
                sorted(
                    available,
                    key=lambda r: hashlib.sha256(
                        (selection["hash_prefix"] + r["sample_id"]).encode()
                    ).hexdigest(),
                )[:n]
            )
    records = json.loads((dataset / "records.json").read_text())
    assert [r["sample_id"] for r in records] == [r["sample_id"] for r in chosen]
    target_bits, checked_masks, minimum_polygon_iou = {}, 0, 1.0
    for row in records:
        source = original[row["sample_id"]]
        assert sha(dataset / row["image_path"]) == source["image_sha256"]
        rgb = np.asarray(Image.open(dataset / row["image_path"]).convert("RGB"))
        assert (
            hashlib.sha256(np.ascontiguousarray(rgb[:, :, ::-1]).tobytes()).hexdigest()
            == source["decoded_pixels_sha256"]
        )
        assert (
            row["split"] == source["source_split"]
            and row["source_dataset"] == source["source_dataset"]
        )
        assert len(row["objects"]) == len(source["source_objects"])
        line_rows = (
            (dataset / row["label_path"]).read_text().splitlines()
            if row["eligible"]
            else []
        )
        if row["eligible"]:
            assert len(line_rows) == len(row["objects"])
        else:
            assert row["label_path"] is None and row["training_image_path"] is None
        objects = []
        for i, (obj, expected) in enumerate(
            zip(row["objects"], source["source_objects"])
        ):
            native = plane(dataset / obj["mask_path"])
            assert (
                hashlib.sha256(native.tobytes()).hexdigest()
                == expected["bool_plane_sha256"]
            )
            assert (
                obj["source_class"] == expected["source_class"]
                and int(native.sum()) == expected["pixels"]
            )
            value = bits(native)
            objects.append((protocol["classes"].index(obj["source_class"]), value))
            if row["eligible"]:
                tokens = line_rows[i].split()
                assert int(tokens[0]) == objects[-1][0]
                xy = np.array([float(v) for v in tokens[1:]]).reshape(-1, 2)
                xy = np.rint(
                    xy * [source["image_width"], source["image_height"]]
                ).astype(int)
                raster = Image.new("1", (source["image_width"], source["image_height"]))
                ImageDraw.Draw(raster).polygon([tuple(v) for v in xy], fill=1)
                exported = bits(np.asarray(raster))
                overlap = (value & exported).bit_count() / (
                    value | exported
                ).bit_count()
                assert overlap >= protocol["polygon_export"]["minimum_native_iou"]
                minimum_polygon_iou = min(minimum_polygon_iou, overlap)
            checked_masks += 1
        target_bits[row["sample_id"]] = objects
    evaluated = json.loads((evaluation / "manifest.json").read_text())
    assert evaluated["status"] == "complete" and evaluated[
        "dataset_manifest_sha256"
    ] == sha(dataset / "manifest.json")
    for name, digest in evaluated["files"].items():
        assert sha(evaluation / name) == digest
    trained = json.loads((run / "manifest.json").read_text())
    assert (
        trained["status"] == "complete"
        and trained["dataset_task"] == "published_selected_protein_instances"
    )
    assert trained["config"] == json.loads(
        (ROOT / "configs/chimp_mask_pilot_train.json").read_text()
    )
    weights = run / trained["last_weights"]
    assert (
        sha(weights)
        == trained["last_weights_sha256"]
        == evaluated["last_weights_sha256"]
    )
    scores = json.loads((evaluation / "scores.json").read_text())
    assert [r["sample_id"] for r in scores] == [
        r["sample_id"] for r in records if r["eligible"]
    ]
    summaries = {}
    for row in scores:
        truth = target_bits[row["sample_id"]]
        assert row["truth_classes"] == [c for c, _ in truth]
        predicted = [
            (p["class_id"], bits(plane(evaluation / p["mask_path"])))
            for p in row["predictions"]
        ]
        independent = []
        for class_id in [0, 1]:
            ti = [i for i, (c, _) in enumerate(truth) if c == class_id]
            pi = [i for i, (c, _) in enumerate(predicted) if c == class_id]
            if ti and pi:
                matrix = np.array(
                    [
                        [
                            (truth[i][1] & predicted[j][1]).bit_count()
                            / (truth[i][1] | predicted[j][1]).bit_count()
                            for j in pi
                        ]
                        for i in ti
                    ]
                )
                rr, cc = linear_sum_assignment(-matrix)
                independent.extend(
                    dict(
                        truth_index=ti[i],
                        prediction_index=pi[j],
                        iou=float(matrix[i, j]),
                    )
                    for i, j in zip(rr, cc)
                    if matrix[i, j] >= protocol["evaluation"]["match_iou"]
                )
        assert independent == row["matches"]
    report = json.loads((evaluation / "report.json").read_text())
    for split in ["train", "validation"]:
        subset = [r for r in scores if r["split"] == split]
        summaries[split] = {}
        for class_id, name in enumerate(protocol["classes"]):
            target_count = sum(r["truth_classes"].count(class_id) for r in subset)
            prediction_count = sum(
                sum(p["class_id"] == class_id for p in r["predictions"]) for r in subset
            )
            matched = [
                m["iou"]
                for r in subset
                for m in r["matches"]
                if r["truth_classes"][m["truth_index"]] == class_id
            ]
            actual = report["metrics"][split]["classes"][name]
            assert (
                actual["selected_targets"] == target_count
                and actual["predictions"] == prediction_count
            )
            assert (
                actual["matched_selected_targets"] == len(matched)
                and actual["missed_selected_targets"] == target_count - len(matched)
                and actual["unmatched_predictions"] == prediction_count - len(matched)
            )
            assert actual["selected_target_recall"] == (
                len(matched) / target_count if target_count else None
            )
            assert actual["source_task_precision"] == (
                len(matched) / prediction_count if prediction_count else None
            )
            assert actual["matched_mean_native_iou"] == (
                sum(matched) / len(matched) if matched else None
            )
            summaries[split][name] = dict(
                targets=target_count,
                matched=len(matched),
                unmatched_predictions=prediction_count - len(matched),
            )
    # Four fixed replays: first eligible source image per original domain/split.
    replay_rows = []
    for split in ["train", "validation"]:
        for name in selection["datasets"]:
            replay_rows.append(
                next(
                    r
                    for r in records
                    if r["eligible"]
                    and r["split"] == split
                    and r["source_dataset"] == name
                )
            )
    os.environ.update(
        YOLO_CONFIG_DIR=str(output.with_suffix(".settings")),
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
    )
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    from ultralytics import YOLO, settings

    settings.update(
        {
            k: False
            for k in [
                "sync",
                "clearml",
                "comet",
                "dvc",
                "mlflow",
                "raytune",
                "tensorboard",
                "wandb",
            ]
        }
    )
    import torch

    torch.set_num_threads(trained["config"]["threads"])
    model = YOLO(str(weights), task="segment")
    options = protocol["evaluation"]
    replayed = 0
    for row in replay_rows:
        image = np.ascontiguousarray(
            np.asarray(Image.open(dataset / row["image_path"]).convert("RGB"))[
                :, :, ::-1
            ]
        )
        result = model.predict(
            image,
            device="cpu",
            imgsz=options["imgsz"],
            conf=options["confidence"],
            iou=options["nms_iou"],
            max_det=options["max_det"],
            retina_masks=True,
            verbose=False,
            save=False,
        )[0]
        saved = next(r for r in scores if r["sample_id"] == row["sample_id"])[
            "predictions"
        ]
        assert len(saved) == (0 if result.masks is None else len(result.masks.data))
        if saved:
            for stored, native, c, confidence in zip(
                saved,
                result.masks.data.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                assert (
                    stored["class_id"] == int(c)
                    and abs(stored["confidence"] - float(confidence)) < 1e-6
                )
                assert np.array_equal(
                    plane(evaluation / stored["mask_path"]), native > 0.5
                )
        replayed += 1
    result = dict(
        passed=True,
        source_images=len(records),
        source_masks_verified=checked_masks,
        eligible_images_scored=len(scores),
        minimum_independent_polygon_iou=minimum_polygon_iou,
        fixed_neural_replays=replayed,
        metrics=summaries,
        protocol_sha256=sha(protocol_path),
        auditor_sha256=sha(Path(__file__)),
        evaluation_manifest_sha256=sha(evaluation / "manifest.json"),
        last_weights_sha256=sha(weights),
        independent_KDP_evaluation_groups=0,
        model_promoted=False,
        physical_command=None,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/chimp_mask_pilot.json"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.dataset, a.run, a.evaluation, a.protocol, a.output)
