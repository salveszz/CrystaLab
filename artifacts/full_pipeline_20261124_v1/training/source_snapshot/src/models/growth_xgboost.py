# FILE PURPOSE: Train and evaluate tabular synthetic volume-gain forecasts with
# fixed family splits, causal features, training-only baselines and a frozen test.
"""Complete XGBoost growth benchmark; no physical/quality/control promotion."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from environment.dataset import (
    FEATURES,
    audit_dataset,
    file_hash,
    read_json,
    read_rows,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
INPUTS = (*FEATURES, *("missing__" + name for name in FEATURES))
METHODS = ("persistence", "train_mean", "recent_rate", "xgboost")


def validate_config(config):
    if (
        config.get("schema") != "crystalab_growth_xgboost_v1"
        or config.get("target") != "volume_gain"
        or config.get("selection_metric") != "validation_mae"
    ):
        raise ValueError("Unsupported growth experiment contract")
    for name in ["num_boost_round", "early_stopping_rounds"]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("Require positive training/stopping round budgets")
    params = config["params"]
    if (
        params.get("objective") != "reg:squarederror"
        or params.get("eval_metric") != "mae"
        or params.get("tree_method") != "hist"
        or params.get("device") != "cpu"
    ):
        raise ValueError("Require the declared CPU regression/tree task")
    if type(params["nthread"]) is not int or params["nthread"] < 1:
        raise ValueError("Require a positive CPU thread limit")
    if not config["candidates"] or not config["learning_curve_groups"]:
        raise ValueError("Declare candidates and learning-curve group counts")
    for candidate in config["candidates"]:
        if (
            set(candidate) != {"max_depth", "eta"}
            or type(candidate["max_depth"]) is not int
            or candidate["max_depth"] < 1
            or isinstance(candidate["eta"], bool)
            or not 0 < candidate["eta"] <= 1
        ):
            raise ValueError("Candidate may vary only valid depth and learning rate")
    if any(type(n) is not int or n < 1 for n in config["learning_curve_groups"]):
        raise ValueError("Learning curves require positive group counts")


def feature_matrix(rows):
    """Native NaNs and explicit indicators; no imputation using another split."""
    values = []
    for row in rows:
        features, missing = row["features"], row["missing"]
        if set(features) != set(FEATURES) or set(missing) != set(FEATURES):
            raise ValueError("Growth input differs from the 14-feature whitelist")
        vector = []
        for name in FEATURES:
            value = features[name]
            if type(missing[name]) is not bool or missing[name] != (value is None):
                raise ValueError("Missing indicators must match unavailable inputs")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Available growth inputs must be finite numbers")
            vector.append(np.nan if value is None else value)
        values.append(vector + [int(missing[name]) for name in FEATURES])
    result = np.asarray(values, dtype=np.float32).reshape(-1, len(INPUTS))
    if np.isinf(result).any():
        raise ValueError("Growth input exceeds finite float32 range")
    return result


def validate_splits(rows):
    groups, families, ids = {}, {}, set()
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "test"} or row["sample_id"] in ids:
            raise ValueError("Unknown split or duplicate sample")
        ids.add(row["sample_id"])
        for values, key in [(groups, "group_id"), (families, "family_id")]:
            if values.setdefault(row[key], split) != split:
                raise ValueError("Related growth groups/families cross splits")
        target = row["labels"]["volume_gain"]
        if not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError("Finite volume-gain labels required")
    if set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("Require all three frozen splits")


def baseline_values(rows, mean_gain):
    recent, fallback = [], []
    for row in rows:
        rate = row["features"]["recent_volume_rate"]
        missing = rate is None
        recent.append(mean_gain if missing else rate * row["features"]["horizon"])
        fallback.append(missing)
    return dict(
        persistence=np.zeros(len(rows)),
        train_mean=np.full(len(rows), mean_gain),
        recent_rate=np.array(recent),
    ), fallback


def metrics(rows, predicted):
    if not rows:
        return dict(
            samples=0,
            groups=0,
            mae=None,
            rmse=None,
            signed_error=None,
            group_macro_mae=None,
        )
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != (len(rows),) or not np.isfinite(predicted).all():
        raise ValueError("Require one finite prediction per labeled row")
    errors = predicted - [r["labels"]["volume_gain"] for r in rows]
    by_group = defaultdict(list)
    for row, error in zip(rows, errors):
        by_group[row["group_id"]].append(abs(float(error)))
    return dict(
        samples=len(rows),
        groups=len(by_group),
        mae=float(abs(errors).mean()),
        rmse=float(np.sqrt(np.mean(errors**2))),
        signed_error=float(errors.mean()),
        group_macro_mae=float(np.mean([np.mean(v) for v in by_group.values()])),
    )


def summaries(rows, predictions):
    output = {}
    for method, values in predictions.items():
        cohorts = {}
        for name, indexes in [
            ("all", list(range(len(rows)))),
            ("startup", [i for i, r in enumerate(rows) if r["missing"]["volume"]]),
            ("observed", [i for i, r in enumerate(rows) if not r["missing"]["volume"]]),
        ]:
            cohorts[name] = metrics(
                [rows[i] for i in indexes], np.asarray(values)[indexes]
            )
        cohorts["per_recipe"] = {
            recipe: metrics(
                [r for r in rows if r["episode_id"].split("__", 1)[1] == recipe],
                [
                    v
                    for r, v in zip(rows, values)
                    if r["episode_id"].split("__", 1)[1] == recipe
                ],
            )
            for recipe in sorted({r["episode_id"].split("__", 1)[1] for r in rows})
        }
        output[method] = cohorts
    return output


def predict_saved(run, rows):
    """Reload a checked JSON model and apply only the recorded causal inputs."""
    import xgboost as xgb

    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if (
        manifest["schema"] != "crystalab_growth_xgboost_v1"
        or manifest["status"] != "complete"
        or manifest["input_columns"] != list(INPUTS)
    ):
        raise ValueError("Require a complete synthetic growth model")
    if file_hash(run / "model.json") != manifest["model_sha256"]:
        raise ValueError("Saved growth model identity changed")
    model = xgb.Booster()
    model.load_model(run / "model.json")
    if model.feature_names != list(INPUTS):
        raise ValueError("Serialized growth-model input schema changed")
    return model.predict(xgb.DMatrix(feature_matrix(rows), feature_names=list(INPUTS)))


def select_model(training, validation, config, output):
    """Only these two splits can affect model fitting or model selection."""
    import xgboost as xgb

    matrices = [
        xgb.DMatrix(
            feature_matrix(rows),
            label=[r["labels"]["volume_gain"] for r in rows],
            feature_names=list(INPUTS),
            nthread=config["params"]["nthread"],
        )
        for rows in [training, validation]
    ]
    candidates = []
    (output / "candidates").mkdir()
    for index, candidate in enumerate(config["candidates"]):
        history = {}
        model = xgb.train(
            dict(config["params"], **candidate),
            matrices[0],
            num_boost_round=config["num_boost_round"],
            evals=[(matrices[0], "train"), (matrices[1], "validation")],
            evals_result=history,
            verbose_eval=False,
            callbacks=[
                xgb.callback.EarlyStopping(
                    rounds=config["early_stopping_rounds"],
                    data_name="validation",
                    metric_name="mae",
                    save_best=True,
                )
            ],
        )
        relative = f"candidates/model_{index:02d}.json"
        model.save_model(output / relative)
        prediction = model.predict(matrices[1])
        score = metrics(validation, prediction)
        record = dict(
            index=index,
            params=candidate,
            validation_mae=score["mae"],
            validation_group_macro_mae=score["group_macro_mae"],
            boosted_rounds=model.num_boosted_rounds(),
            best_iteration=int(model.best_iteration),
            trained_rounds=len(history["train"]["mae"]),
            model=relative,
            model_sha256=file_hash(output / relative),
        )
        write_json(output / f"candidates/history_{index:02d}.json", history)
        candidates.append(record)
    selected = min(candidates, key=lambda r: (r["validation_mae"], r["index"]))
    shutil.copyfile(output / selected["model"], output / "model.json")
    selection = dict(
        candidates=candidates,
        selected=selected,
        selected_model_sha256=file_hash(output / "model.json"),
        selection_metric="validation_mae",
        test_used_for_selection=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output / "selection.json", selection)
    model = xgb.Booster()
    model.load_model(output / "model.json")
    return model, selection


def train(dataset, config_path, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    config = read_json(config_path)
    validate_config(config)
    import xgboost as xgb

    if xgb.__version__ != config["runtime_version"]:
        raise ValueError("XGBoost runtime differs from the frozen version")
    audit = audit_dataset(dataset)
    rows = read_rows(dataset / "samples.jsonl")
    validate_splits(rows)
    feature_matrix(rows)
    source = read_json(dataset / "manifest.json")
    if source["identity"]["feature_columns"] != list(FEATURES):
        raise ValueError("Source feature contract changed")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "config.json")
    hashes = {}
    for relative in [
        "src/models/growth_xgboost.py",
        "scripts/train_xgboost.py",
        "configs/xgboost_growth.json",
    ]:
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        hashes[relative] = file_hash(dest)
    manifest = dict(
        schema=config["schema"],
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_identity_hash=source["identity_hash"],
        dataset_manifest_sha256=file_hash(dataset / "manifest.json"),
        samples_sha256=file_hash(dataset / "samples.jsonl"),
        config_sha256=file_hash(config_path),
        input_columns=list(INPUTS),
        source_sha256=hashes,
        scope="Synthetic volume gain in illustrative model units; no quality target, real KDP validation or control promotion",
    )
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        splits = {
            name: [r for r in rows if r["split"] == name]
            for name in ["train", "validation", "test"]
        }
        training, validation = splits["train"], splits["validation"]
        mean_gain = float(np.mean([r["labels"]["volume_gain"] for r in training]))
        model, selection = select_model(training, validation, config, output)
        frozen = file_hash(output / "selection.json")
        # Fixed-parameter learning curves use training families and validation only.
        groups = sorted(
            {r["group_id"] for r in training},
            key=lambda value: hashlib.sha256(
                (config["learning_curve_seed"] + value).encode()
            ).hexdigest(),
        )
        learning = []
        (output / "learning_curve").mkdir()
        dvalidation = xgb.DMatrix(
            feature_matrix(validation), feature_names=list(INPUTS)
        )
        for count in sorted(
            set(min(n, len(groups)) for n in config["learning_curve_groups"])
        ):
            selected_rows = [r for r in training if r["group_id"] in groups[:count]]
            if count == len(groups):
                fitted = model
            else:
                dtrain = xgb.DMatrix(
                    feature_matrix(selected_rows),
                    label=[r["labels"]["volume_gain"] for r in selected_rows],
                    feature_names=list(INPUTS),
                )
                fitted = xgb.train(
                    dict(config["params"], **selection["selected"]["params"]),
                    dtrain,
                    num_boost_round=selection["selected"]["boosted_rounds"],
                    verbose_eval=False,
                )
            curve_model = f"learning_curve/groups_{count:02d}.json"
            fitted.save_model(output / curve_model)
            learning.append(
                dict(
                    training_groups=count,
                    training_samples=len(selected_rows),
                    group_ids=groups[:count],
                    training_sample_ids=[r["sample_id"] for r in selected_rows],
                    model=curve_model,
                    model_sha256=file_hash(output / curve_model),
                    validation=metrics(validation, fitted.predict(dvalidation)),
                )
            )
        write_json(output / "learning_curve.json", learning)
        if (
            file_hash(output / "selection.json") != frozen
            or file_hash(output / "model.json") != selection["selected_model_sha256"]
        ):
            raise ValueError("Selected model changed before test scoring")
        write_json(
            output / "test_release.json",
            dict(
                selection_sha256=frozen,
                model_sha256=selection["selected_model_sha256"],
                released_at_utc=datetime.now(timezone.utc).isoformat(),
                test_scoring_passes=1,
            ),
        )
        predictions, reports = [], {}
        for split, subset in splits.items():
            values, fallback = baseline_values(subset, mean_gain)
            values["xgboost"] = model.predict(
                xgb.DMatrix(feature_matrix(subset), feature_names=list(INPUTS))
            )
            reports[split] = summaries(subset, values)
            for i, row in enumerate(subset):
                predictions.append(
                    dict(
                        sample_id=row["sample_id"],
                        group_id=row["group_id"],
                        family_id=row["family_id"],
                        episode_id=row["episode_id"],
                        split=split,
                        startup=row["missing"]["volume"],
                        target_volume_gain=row["labels"]["volume_gain"],
                        recent_rate_fallback=fallback[i],
                        predictions={
                            key: float(value[i]) for key, value in values.items()
                        },
                    )
                )
        (output / "predictions.jsonl").write_text(
            "".join(
                json.dumps(r, sort_keys=True, allow_nan=False) + "\n"
                for r in predictions
            )
        )
        gain = model.get_score(importance_type="total_gain")
        write_json(
            output / "feature_importance.json",
            {name: float(gain.get(name, 0.0)) for name in INPUTS},
        )
        write_json(
            output / "fit_inputs.json",
            dict(
                train_sample_ids=[r["sample_id"] for r in training],
                validation_sample_ids=[r["sample_id"] for r in validation],
                train_groups=sorted({r["group_id"] for r in training}),
                validation_groups=sorted({r["group_id"] for r in validation}),
                test_groups=sorted({r["group_id"] for r in splits["test"]}),
                input_columns=list(INPUTS),
                native_missing_values=True,
                imputation=None,
                mean_training_gain=mean_gain,
            ),
        )
        report = dict(
            status="complete",
            samples=len(rows),
            samples_by_split=dict(Counter(r["split"] for r in rows)),
            groups_by_split={
                name: len({r["group_id"] for r in values})
                for name, values in splits.items()
            },
            metrics=reports,
            recent_rate_fallback_counts=dict(
                Counter(r["split"] for r in predictions if r["recent_rate_fallback"])
            ),
            selected=selection["selected"],
            test_selection_frozen=True,
            quality_target_available=False,
            physical_model_promoted=False,
            units="model_length^3 volume gain",
        )
        write_json(output / "report.json", report)
        write_json(
            output / "environment.json",
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                packages={
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "xgboost", "matplotlib"]
                },
            ),
        )
        plot_training(output, predictions, reports, learning, selection)
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            model_sha256=file_hash(output / "model.json"),
            dataset_audit=audit,
            files={
                str(p.relative_to(output)): file_hash(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(output / "manifest.json", manifest)
        return report
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        raise


def plot_training(output, predictions, reports, learning, selection):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    history = read_json(
        output / f"candidates/history_{selection['selected']['index']:02d}.json"
    )
    for split in ["train", "validation"]:
        axes[0, 0].plot(history[split]["mae"], label=split)
    axes[0, 0].set(
        title="Selected candidate: boosting history",
        xlabel="Boosting iteration",
        ylabel="MAE (model volume)",
    )
    axes[0, 0].legend()
    selected = [r for r in predictions if r["split"] == "test"]
    for startup, label in [(True, "startup"), (False, "observed")]:
        subset = [r for r in selected if r["startup"] == startup]
        axes[0, 1].scatter(
            [r["target_volume_gain"] for r in subset],
            [r["predictions"]["xgboost"] for r in subset],
            s=12,
            alpha=0.65,
            label=label,
        )
    limits = axes[0, 1].get_xlim() + axes[0, 1].get_ylim()
    axes[0, 1].plot([min(limits), max(limits)], [min(limits), max(limits)], "k--", lw=1)
    axes[0, 1].set(
        title="Frozen test predictions",
        xlabel="True volume gain",
        ylabel="Predicted volume gain",
    )
    axes[0, 1].legend()
    for offset, cohort in [(-0.2, "startup"), (0.2, "observed")]:
        axes[1, 0].bar(
            np.arange(len(METHODS)) + offset,
            [reports["test"][method][cohort]["mae"] for method in METHODS],
            width=0.38,
            label=cohort,
        )
    axes[1, 0].set_xticks(np.arange(len(METHODS)), METHODS, rotation=12)
    axes[1, 0].set(
        title="Test errors with startup separated", ylabel="MAE (model volume)"
    )
    axes[1, 0].legend()
    axes[1, 1].plot(
        [r["training_groups"] for r in learning],
        [r["validation"]["mae"] for r in learning],
        "o-",
    )
    axes[1, 1].set(
        title="Training-group learning curve",
        xlabel="Training groups",
        ylabel="Validation MAE",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Synthetic growth benchmark · illustrative units, no physical KDP accuracy claim"
    )
    fig.savefig(output / "training_report.png", dpi=150)
    fig.savefig(output / "training_report.svg")
    plt.close(fig)
