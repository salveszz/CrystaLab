# FILE PURPOSE: Fit a small training-prefix-only observable-gain diagnostic.
# This source-specific pixel proxy is not a physical response model or controller.
"""Transparent ridge fits, time-support partitions and observable errors."""

import math

import numpy as np

FEATURES = (
    "current_contour_area_px2",
    "current_sqrt_area_px",
    "trailing_sqrt_area_rate_px_per_h",
    "reported_infusion_rate_ml_per_h",
    "trailing_reported_infusion_mean_ml_per_h",
)


def input_matrix(inputs, features):
    if (
        not features
        or len(set(features)) != len(features)
        or not set(features) <= set(FEATURES)
    ):
        raise ValueError("Require a nonempty subset of the fixed observable inputs")
    if not inputs or any(set(row) != set(FEATURES) for row in inputs):
        raise ValueError("Inputs differ from the complete five-field whitelist")
    values = np.array([[row[key] for key in features] for row in inputs], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Observable inputs must be finite")
    return values


def fit_ridge(inputs, gains, features, alpha):
    x = input_matrix(inputs, features)
    y = np.asarray(gains, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all() or len(x) < 2:
        raise ValueError("Require finite matched training gains")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Require a positive finite fixed ridge penalty")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.diag([0.0] + [alpha] * len(features))
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return dict(
        schema="crystalab_observable_gain_ridge_v1",
        features=list(features),
        mean=mean.tolist(),
        scale=scale.tolist(),
        intercept=float(weights[0]),
        coefficients=weights[1:].tolist(),
        alpha=float(alpha),
        training_samples=len(x),
        target_units="sqrt-contour-area pixels",
        model_promoted=False,
        physical_command=None,
    )


def predict_gains(model, inputs):
    if model["schema"] != "crystalab_observable_gain_ridge_v1":
        raise ValueError("Wrong observable model schema")
    x = input_matrix(inputs, model["features"])
    mean, scale, coefficients = [
        np.asarray(model[k], dtype=float) for k in ("mean", "scale", "coefficients")
    ]
    if (
        any(
            v.shape != (x.shape[1],) or not np.isfinite(v).all()
            for v in (mean, scale, coefficients)
        )
        or np.any(scale <= 0)
        or not math.isfinite(model["intercept"])
    ):
        raise ValueError("Invalid serialized observable model")
    result = model["intercept"] + ((x - mean) / scale) @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite observable prediction")
    return result


def partition_samples(samples, observations, log_seconds, cohorts):
    """Check full time spans and actual source IDs, not just nominal cutoff gaps."""
    if not samples or len({r["sample_id"] for r in samples}) != len(samples):
        raise ValueError("Require nonempty unique source candidates")
    if any(b["cutoff_s"] <= a["cutoff_s"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Source candidates must be ordered")
    lookup = {r["sample_id"]: r["filename_elapsed_s"] for r in observations}
    if len(lookup) != len(observations):
        raise ValueError("Source observation IDs must be unique")
    names = [c["name"] for c in cohorts]
    if len(set(names)) != len(names) or "purged" in names or names[0] != "train_prefix":
        raise ValueError("Require unique ordered cohorts beginning with train_prefix")
    for i, cohort in enumerate(cohorts):
        if cohort["minimum_cutoff_s"] > cohort["maximum_cutoff_s"] or (
            i and cohort["minimum_cutoff_s"] <= cohorts[i - 1]["maximum_cutoff_s"]
        ):
            raise ValueError("Cohort cutoff intervals overlap or reverse")
    grouped = {name: [] for name in names + ["purged"]}
    image_sets, log_sets = (
        {name: set() for name in names},
        {name: set() for name in names},
    )
    spans = []
    for row in samples:
        input_matrix([row["inputs"]], FEATURES)
        cutoff = row["cutoff_s"]
        source_images = row["input_image_sample_ids"]
        if (
            not source_images
            or any(lookup[key] > cutoff for key in source_images)
            or row["sample_id"] not in source_images
            or row["maximum_input_time_s"] > cutoff
            or row["last_input_availability_bound_s"] > cutoff
        ):
            raise ValueError("Future or unavailable input in source candidate")
        if (
            lookup[row["target_sample_id"]] != row["target_time_s"]
            or row["target_time_s"] <= cutoff
        ):
            raise ValueError("Require a strictly future source target")
        first, last = row["input_log_row_range"]
        if (
            not 0 <= first <= last < len(log_seconds) - 1
            or log_seconds[last + 1] > cutoff
        ):
            raise ValueError("Input log availability is inconsistent")
        name = next(
            (
                c["name"]
                for c in cohorts
                if c["minimum_cutoff_s"] <= cutoff <= c["maximum_cutoff_s"]
            ),
            "purged",
        )
        support_start = float(
            min([lookup[key] for key in source_images] + [log_seconds[first]])
        )
        enriched = dict(
            row,
            cohort=name,
            support_start_s=support_start,
            support_end_s=float(row["target_time_s"]),
        )
        grouped[name].append(enriched)
        spans.append(
            {
                k: enriched[k]
                for k in ("sample_id", "cohort", "support_start_s", "support_end_s")
            }
        )
        if name != "purged":
            image_sets[name].update(source_images + [row["target_sample_id"]])
            log_sets[name].update(range(first, last + 1))
    for name in names:
        if not grouped[name]:
            raise ValueError("Frozen cohort has no source candidates")
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if (
                image_sets[name] & image_sets[other]
                or log_sets[name] & log_sets[other]
                or max(r["support_end_s"] for r in grouped[name])
                >= min(r["support_start_s"] for r in grouped[other])
            ):
                raise ValueError("Cohorts share source image/log/time supports")
    return grouped, spans


def disjoint_subset(rows):
    selected, previous_end = [], -math.inf
    for row in rows:
        if row["support_start_s"] > previous_end:
            selected.append(row["sample_id"])
            previous_end = row["support_end_s"]
    return selected


def errors(truth, prediction):
    truth, prediction = (
        np.asarray(truth, dtype=float),
        np.asarray(prediction, dtype=float),
    )
    if (
        truth.ndim != 1
        or not len(truth)
        or prediction.shape != truth.shape
        or not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Require matched finite nonempty score vectors")
    residual = prediction - truth
    return dict(
        samples=len(truth),
        MAE_px=float(abs(residual).mean()),
        RMSE_px=float(np.sqrt(np.mean(residual**2))),
        signed_error_px=float(residual.mean()),
        p95_absolute_error_px=float(np.quantile(abs(residual), 0.95)),
    )
