# FILE PURPOSE: Reconstruct weak-target scores, training budgets and artifact lineage independently.
# Neural inference is not rerun; selected prediction masks are the audited model outputs.
"""Verify source-fit geometry and common-window metrics without calling the evaluator or trainer."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.source_shape_audit import audit_projected_shapes


def audit(study_path, evaluation):
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    study = read(study_path)
    root = Path(evaluation)
    manifest = read(root / "manifest.json")
    assert manifest["status"] == "complete" and manifest["study_sha256"] == sha(
        study_path
    )
    assert (
        manifest["evaluation_checkpoint"] == "last"
        and manifest["final_test_scored"] is False
    )
    for path, value in manifest["files"].items():
        assert sha(root / path) == value
    geometry = audit_projected_shapes(study["projection"]["path"])
    source_rows = {}
    common = None
    fits = []
    for case in study["cases"]:
        dataset = Path(case["dataset"])
        assert sha(dataset / "manifest.json") == case["dataset_manifest_sha256"]
        assert sha(case["config"]) == case["config_sha256"]
        data = read(dataset / "manifest.json")
        rows = read(dataset / "records.json")
        assert sum(r["split"] == "train" for r in rows) == case["training_images"]
        for path, value in data["files"].items():
            assert sha(dataset / path) == value
        val = {
            r["sample_id"]: (
                r["family_key"],
                r["evidence_kind"],
                r["source_image_sha256"],
                r["source_mask_sha256"],
            )
            for r in rows
            if r["split"] == "validation"
        }
        if common is not None:
            assert val == common
        common = val
        for row in rows:
            assert (
                row["split"] in {"train", "validation"}
                and row["pixel_reviewed"] is False
            )
            assert (
                sha(row["source_image"]) == row["source_image_sha256"]
                and sha(row["source_mask"]) == row["source_mask_sha256"]
            )
            source_rows[row["sample_id"]] = row
        run = Path(case["run"])
        trained = read(run / "manifest.json")
        assert trained["status"] == "complete"
        assert trained["dataset_task"] == "source_fit_silhouette" and trained[
            "config"
        ] == read(case["config"])
        assert (
            trained["dataset_audit"]["manifest_sha256"]
            == case["dataset_manifest_sha256"]
        )
        assert (
            sha(run / trained["last_weights"])
            == trained["last_weights_sha256"]
            == manifest["models"][case["mode"]]["last_weights_sha256"]
        )
        options = trained["train_options"]
        assert (
            options["batch"] == options["nbs"] == 4
            and options["warmup_epochs"] == 0
            and options["lrf"] == 1
        )
        with (run / trained["results_csv"]).open() as f:
            epochs = list(csv.DictReader(f))
        assert len(epochs) == case["epochs"] and [
            int(float(r["epoch"])) for r in epochs
        ] == list(range(1, case["epochs"] + 1))
        assert (
            math.ceil(case["training_images"] / 4) * len(epochs)
            == case["planned_optimizer_updates"]
            == 210
        )
        assert all(
            math.isclose(float(r[k]), 0.001, abs_tol=1e-12)
            for r in epochs
            for k in r
            if k.startswith("lr/pg")
        )
        fits.append(
            dict(
                mode=case["mode"],
                epochs=len(epochs),
                planned_optimizer_updates=210,
                last_weights_sha256=trained["last_weights_sha256"],
            )
        )
    real = Path(
        next(
            c["dataset"]
            for c in study["cases"]
            if c["mode"] == "source_projection_only"
        )
    )
    training = [r for r in read(real / "records.json") if r["split"] == "train"]
    assert {r["sample_id"] for r in training} == set(
        manifest["template_training_sample_ids"]
    )
    shape = (study["selection"]["template_size"],) * 2
    template = (
        np.mean(
            [
                cv2.resize(
                    cv2.imread(str(real / r["mask_path"]), 0),
                    shape,
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
                for r in training
            ],
            axis=0,
        )
        >= 0.5
    ).astype(np.uint8) * 255
    assert np.array_equal(
        template, cv2.imread(str(root / "real_training_mean_template.png"), 0)
    )
    scores = read(root / "scores.json")
    reports = read(root / "metrics.json")
    methods = {c["mode"] for c in study["cases"]} | {"real_training_mean_template"}
    assert len(scores) == len(common) * len(methods) == manifest["score_records"]
    assert {(r["method"], r["sample_id"]) for r in scores} == {
        (method, sample) for method in methods for sample in common
    }
    for row in scores:
        original = source_rows[row["sample_id"]]
        target_pixels = cv2.imread(original["source_mask"], 0)
        prediction_pixels = cv2.imread(str(root / row["predicted_mask"]), 0)
        assert set(np.unique(target_pixels)) <= {0, 255} and set(
            np.unique(prediction_pixels)
        ) <= {0, 255}
        target, prediction = target_pixels > 0, prediction_pixels > 0
        assert (
            sha(row["target_mask"])
            == row["target_sha256"]
            == original["source_mask_sha256"]
        )
        assert (
            row["evidence_kind"] == original["evidence_kind"]
            and row["family_key"] == original["family_key"]
        )
        assert target.shape == prediction.shape
        intersection = int(np.count_nonzero(target & prediction))
        union = int(np.count_nonzero(target | prediction))
        a, b = int(target.sum()), int(prediction.sum())
        for key, value in dict(
            proxy_mask_iou=intersection / union,
            proxy_dice=2 * intersection / (a + b),
            target_area_px2=a,
            predicted_area_px2=b,
            area_absolute_error_px2=abs(a - b),
            area_absolute_error_image_fraction=abs(a - b) / target.size,
        ).items():
            assert math.isclose(row[key], value, rel_tol=1e-10, abs_tol=1e-12)
        assert row["missing_prediction"] == (b == 0)
        if row["method"] == "real_training_mean_template":
            assert np.array_equal(
                prediction,
                cv2.resize(
                    template,
                    (target.shape[1], target.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0,
            )

    def verify(rr, metric):
        families = {r["family_key"] for r in rr}
        assert metric["images"] == len(rr) and metric["families"] == len(families)
        expected = dict(
            proxy_mask_iou_mean=mean(r["proxy_mask_iou"] for r in rr),
            proxy_mask_iou_family_macro=mean(
                mean(r["proxy_mask_iou"] for r in rr if r["family_key"] == family)
                for family in families
            ),
            proxy_dice_mean=mean(r["proxy_dice"] for r in rr),
            missing_predictions=sum(r["missing_prediction"] for r in rr),
            extra_predictions=sum(r["extra_predictions"] for r in rr),
            area_mae_image_fraction=mean(
                r["area_absolute_error_image_fraction"] for r in rr
            ),
        )
        for key, value in expected.items():
            assert math.isclose(metric[key], value, rel_tol=1e-10, abs_tol=1e-12)

    for method in methods:
        rr = [r for r in scores if r["method"] == method]
        verify(rr, reports[method]["all"])
        for field, key in [
            ("evidence_kind", "by_evidence"),
            ("family_key", "per_family"),
        ]:
            for value in {r[field] for r in rr}:
                verify(
                    [r for r in rr if r[field] == value], reports[method][key][value]
                )
    return dict(
        status="pass",
        study_sha256=sha(study_path),
        auditor_sha256=sha(__file__),
        geometry=geometry,
        fits=fits,
        common_validation_images=len(common),
        selected_prediction_score_records=len(scores),
        metrics_reconstructed=True,
        training_only_template_reconstructed=True,
        neural_inference_rerun=False,
        final_test_scored=False,
        actual_human_reviews_created=0,
        scope="Independent source-fit geometry, selected mask scores and configured training-budget reconstruction; not independently reviewed pixel accuracy or a neural-inference re-execution",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study", type=Path, default=Path("configs/source_silhouette_experiment.json")
    )
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.evaluation.resolve()):
        p.error("Write the audit outside the immutable evaluation")
    result = audit(a.study, a.evaluation)
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
