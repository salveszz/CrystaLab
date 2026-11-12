# FILE PURPOSE: Assemble grouped causal area-forecast examples from audited optical replay data.
# Frozen test families are excluded. Native estimated masks supply inputs; original
# optical proxy masks supply separate labels, including explicit startup coverage.
"""Prepare and audit a bounded projected-area forecasting development cohort."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from models.area_forecast import AreaEstimate, AreaForecastConfig, AreaTrend, FEATURES
from vision.learned import LearnedObservationConfig
from vision.optical_sequence import export_optical_sequence, validate_sequence_codec
from vision.replay import replay
from vision.training_data import safe_path, save_json
from vision.video import file_sha256, VideoReader

ROOT = Path(__file__).resolve().parents[2]


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class ReplayAreas:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.manifest_hash = file_sha256(self.root / "manifest.json")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if (
            m.get("schema") != "crystalab_video_observation_v1"
            or m["status"] != "complete"
        ):
            raise ValueError("Completed native observation replay required")
        if m["source_info"].get("split") == "test":
            raise ValueError("Development area forecasts exclude final test")
        self.clock_id = "source_video:" + (
            m["input"].get("sha256") or self.manifest_hash
        )
        self.view_id = "fixed_view_assumption:" + m["run_id"]
        self.verify()

    def verify(self):
        if file_sha256(self.root / "manifest.json") != self.manifest_hash:
            raise ValueError("Observation manifest changed")
        for name, digest in self.manifest["files"].items():
            if file_sha256(safe_path(self.root, name)) != digest:
                raise ValueError("Observation artifact identity changed")

    def estimates(self):
        count = 0
        with (
            VideoReader(self.root / "overlay.avi") as reader,
            (self.root / "measurements.jsonl").open() as records,
        ):
            for line in records:
                row = json.loads(line)
                frame = next(reader)
                if frame.index != count:
                    raise ValueError("Observation overlay order differs")
                digest = None
                if row["mask_path"] is not None:
                    path = safe_path(self.root, row["mask_path"])
                    mask = cv2.imread(str(path), 0)
                    if (
                        mask is None
                        or mask.shape != frame.image.shape[:2]
                        or not set(np.unique(mask)) <= {0, 255}
                    ):
                        raise ValueError("Invalid native observation mask")
                    if np.count_nonzero(mask) != row["projected_area_px"]:
                        raise ValueError("Projected area differs from its mask")
                    digest = file_sha256(path)
                estimate = AreaEstimate(
                    run_id=self.manifest["run_id"],
                    clock_id=self.clock_id,
                    view_id=self.view_id,
                    frame_index=row["frame_index"],
                    observed_at_s=row["video_time_s"],
                    received_at_s=row["video_time_s"],
                    image_shape=frame.image.shape[:2],
                    area_px2=row["projected_area_px"],
                    valid=row["valid"],
                    requires_inspection=row.get(
                        "requires_inspection", not row["valid"]
                    ),
                    evidence_kind=self.manifest["source_info"]["evidence_kind"],
                    mask_sha256=digest,
                )
                yield estimate, row
                count += 1
            if (
                next(reader, None) is not None
                or count != self.manifest["processed_frames"]
            ):
                raise ValueError("Observation frame count differs")


def make_samples(source, config, optical, original_rows, output):
    reader = ReplayAreas(source)
    estimates = list(reader.estimates())
    trend = AreaTrend(config)
    samples = []
    family = reader.manifest["source_info"]["family_id"]
    split = reader.manifest["source_info"]["split"]
    for estimate, record in estimates:
        state = trend.update(estimate)
        candidates = [
            (a, r)
            for a, r in estimates
            if math.isclose(
                a.observed_at_s,
                estimate.observed_at_s + config.horizon_source_s,
                abs_tol=1e-9,
                rel_tol=0,
            )
        ]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError("Forecast horizon has ambiguous source alignment")
        target, target_record = candidates[0]
        current_row = original_rows[record["source_frame"]["source_sample_id"]]
        future_row = original_rows[target_record["source_frame"]["source_sample_id"]]
        if any(
            row["family_id"] != family or row["split"] != split
            for row in (current_row, future_row)
        ):
            raise ValueError("Forecast label crosses a source family/split")
        refs = {}
        labels = {}
        available = True
        for label, row in [("current", current_row), ("future", future_row)]:
            seed = [obj for obj in row["objects"] if obj["class_id"] == 0]
            if len(seed) != 1 or not seed[0]["label_eligible"]:
                available = False
                break
            original = safe_path(optical, seed[0]["mask_path"])
            mask = cv2.imread(str(original), 0)
            if (
                mask.shape != estimate.image_shape
                or not set(np.unique(mask)) <= {0, 255}
                or not np.any(mask)
            ):
                raise ValueError("Invalid original optical seed proxy")
            relative = f"labels/{row['sample_id'].replace(':', '_')}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copyfile(original, path)
            elif file_sha256(path) != file_sha256(original):
                raise ValueError("Label path collision")
            refs[label] = dict(
                mask_path=relative,
                mask_sha256=file_sha256(original),
                original_mask_path=seed[0]["mask_path"],
                source_sample_id=row["sample_id"],
            )
            labels[label + "_proxy_area_px2"] = int(np.count_nonzero(mask))
        if available:
            labels["proxy_gain_px2"] = (
                labels["future_proxy_area_px2"] - labels["current_proxy_area_px2"]
            )
            labels["relative_proxy_gain"] = (
                labels["proxy_gain_px2"] / labels["current_proxy_area_px2"]
            )
        sample = dict(
            sample_id=f"{current_row['sample_id']}->{future_row['sample_id']}",
            family_id=family,
            split=split,
            source_replay=str(reader.root.relative_to(output)),
            source_replay_manifest_sha256=reader.manifest_hash,
            cut_frame_index=estimate.frame_index,
            target_frame_index=target.frame_index,
            cut_time_source_s=estimate.observed_at_s,
            target_time_source_s=target.observed_at_s,
            estimate=asdict(estimate),
            trend=state,
            labels=labels if available else None,
            label_refs=refs if available else {},
            label_semantics="original optical emission-proxy seed masks; not exact empirical visibility",
        )
        samples.append(sample)
    reader.verify()
    return samples


def prepare_cohort(
    optical, model_run, observer_path, forecast_path, output, *, sequence_codec="MJPG"
):
    from models.observation import YOLOObserver
    from rendering.optical import audit_optical

    sequence_codec = validate_sequence_codec(sequence_codec)
    optical, model_run, observer_path, forecast_path, output = [
        Path(p).resolve()
        for p in (optical, model_run, observer_path, forecast_path, output)
    ]
    audit_optical(optical)
    original = json.loads((optical / "manifest.json").read_text())
    original_hash = file_sha256(optical / "manifest.json")
    original_rows = {r["sample_id"]: r for r in json_rows(optical / "samples.jsonl")}
    scenes = sorted(
        {
            r["scene_id"]
            for r in original_rows.values()
            if r["split"] in {"train", "validation"}
        }
    )
    observer = LearnedObservationConfig(**json.loads(observer_path.read_text()))
    config = AreaForecastConfig(**json.loads(forecast_path.read_text()))
    observer_hash, forecast_hash = (
        file_sha256(observer_path),
        file_sha256(forecast_path),
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(observer_path, output / "observer_config.json")
    shutil.copyfile(forecast_path, output / "forecast_config.json")
    shutil.copyfile(optical / "manifest.json", output / "optical_manifest.json")
    shutil.copyfile(optical / "samples.jsonl", output / "optical_samples.jsonl")
    if (
        file_sha256(output / "optical_samples.jsonl")
        != original["artifact_sha256"]["samples.jsonl"]
    ):
        raise ValueError("Optical sample identity changed during preparation")
    source_hashes = {}
    for path in [
        Path(__file__),
        ROOT / "src/models/area_forecast.py",
        ROOT / "scripts/prepare_area_forecasts.py",
        ROOT / "src/vision/optical_sequence.py",
    ]:
        relative = path.relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_hashes[str(relative)] = file_sha256(target)
    manifest = dict(
        schema="crystalab_area_forecast_cohort_v1",
        status="running",
        source_dataset=str(optical),
        source_manifest_sha256=original_hash,
        observer_config_sha256=observer_hash,
        forecast_config_sha256=forecast_hash,
        source_sha256=source_hashes,
        model_run=str(model_run),
        sequence_codec=sequence_codec,
        scenes=scenes,
        scope="Synthetic development area forecast examples; no test scoring, physical clock or action-response model",
        label_semantics="optical emission proxy",
        final_test_scored=False,
    )
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    cases = []
    samples = []
    try:
        for scene in scenes:
            sequence = output / "sequences" / scene
            export_optical_sequence(
                optical, scene, sequence, display_fps=4.0, codec=sequence_codec
            )
            folder = output / "observations" / scene

            def factory(path):
                return YOLOObserver(observer, model_run, path / "settings")

            replay(
                sequence / "sequence.avi",
                observer,
                folder,
                run_id=scene + ":area_forecast",
                source_info=json.loads((sequence / "source_info.json").read_text()),
                observer_factory=factory,
                frame_map=sequence / "frame_map.jsonl",
            )
            run = json.loads((folder / "manifest.json").read_text())
            batch = make_samples(folder, config, optical, original_rows, output)
            samples.extend(batch)
            cases.append(
                dict(
                    scene_id=scene,
                    family_id=run["source_info"]["family_id"],
                    split=run["source_info"]["split"],
                    replay_path=str(folder.relative_to(output)),
                    replay_manifest_sha256=file_sha256(folder / "manifest.json"),
                    weights_sha256=run["observer"]["weights_sha256"],
                    frames=run["processed_frames"],
                    samples=len(batch),
                    rate_ready=sum(s["trend"]["history_ready"] for s in batch),
                )
            )
            print(
                json.dumps(
                    dict(
                        scene=scene,
                        completed=len(cases),
                        total=len(scenes),
                        rate_ready=cases[-1]["rate_ready"],
                    )
                ),
                flush=True,
            )
        (output / "samples.jsonl").write_text(
            "".join(json.dumps(s, allow_nan=False) + "\n" for s in samples)
        )
        if (
            file_sha256(optical / "manifest.json") != original_hash
            or file_sha256(observer_path) != observer_hash
            or file_sha256(forecast_path) != forecast_hash
        ):
            raise ValueError("Forecast preparation inputs changed")
        for relative, digest in source_hashes.items():
            if file_sha256(ROOT / relative) != digest:
                raise ValueError("Forecast preparation producer changed")
        manifest.update(
            status="complete",
            cases=cases,
            samples=len(samples),
            elapsed_seconds=time.perf_counter() - started,
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p != output / "manifest.json"
            },
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed",
            completed_cases=cases,
            error=f"{type(error).__name__}: {error}",
        )
        save_json(output / "manifest.json", manifest)
        raise


def audit_cohort(path):
    root = Path(path).resolve()
    manifest_hash = file_sha256(root / "manifest.json")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("schema") != "crystalab_area_forecast_cohort_v1"
        or m["status"] != "complete"
    ):
        raise ValueError("Completed area-forecast cohort required")
    for name, digest in m["files"].items():
        if file_sha256(safe_path(root, name)) != digest:
            raise ValueError("Forecast cohort artifact identity changed")
    config = AreaForecastConfig(
        **json.loads((root / "forecast_config.json").read_text())
    )
    original = json.loads((root / "optical_manifest.json").read_text())
    if (
        file_sha256(root / "optical_samples.jsonl")
        != original["artifact_sha256"]["samples.jsonl"]
    ):
        raise ValueError("Original optical sample identity differs")
    original_rows = {
        row["sample_id"]: row for row in json_rows(root / "optical_samples.jsonl")
    }
    if file_sha256(root / "optical_manifest.json") != m["source_manifest_sha256"]:
        raise ValueError("Original optical context differs")
    samples = json_rows(root / "samples.jsonl")
    groups = {}
    identifiers = set()
    runs = {}
    replay_hashes = {}
    expected_samples = set()
    for case in m["cases"]:
        if (
            case["split"] not in {"train", "validation"}
            or groups.setdefault(case["family_id"], case["split"]) != case["split"]
        ):
            raise ValueError("Forecast families cross splits or include final test")
        folder = safe_path(root, case["replay_path"])
        reader = ReplayAreas(folder)
        if (
            reader.manifest["source_info"]["family_id"] != case["family_id"]
            or reader.manifest["source_info"]["split"] != case["split"]
        ):
            raise ValueError("Forecast case differs from its source grouping")
        if reader.manifest_hash != case["replay_manifest_sha256"]:
            raise ValueError("Forecast source replay identity differs")
        if case["replay_path"] in runs:
            raise ValueError("Duplicate forecast source replay")
        runs[case["replay_path"]] = {
            row["frame_index"]: (estimate, row) for estimate, row in reader.estimates()
        }
        replay_hashes[case["replay_path"]] = reader.manifest_hash
        records = list(runs[case["replay_path"]].values())
        for estimate, row in records:
            for future, future_row in records:
                if math.isclose(
                    future.observed_at_s - estimate.observed_at_s,
                    config.horizon_source_s,
                    abs_tol=1e-9,
                    rel_tol=0,
                ):
                    expected_samples.add(
                        row["source_frame"]["source_sample_id"]
                        + "->"
                        + future_row["source_frame"]["source_sample_id"]
                    )
    for sample in samples:
        if sample["sample_id"] in identifiers:
            raise ValueError("Duplicate forecast sample ID")
        identifiers.add(sample["sample_id"])
        if groups.get(sample["family_id"]) != sample["split"]:
            raise ValueError("Forecast sample split differs")
        observations = runs[sample["source_replay"]]
        estimate, record = observations[sample["cut_frame_index"]]
        target, target_record = observations[sample["target_frame_index"]]
        if (
            sample["source_replay_manifest_sha256"]
            != replay_hashes[sample["source_replay"]]
            or sample["cut_time_source_s"] != estimate.observed_at_s
            or sample["target_time_source_s"] != target.observed_at_s
        ):
            raise ValueError("Forecast sample source identity/time differs")
        source_ids = dict(
            current=record["source_frame"]["source_sample_id"],
            future=target_record["source_frame"]["source_sample_id"],
        )
        for source_id in source_ids.values():
            source_row = original_rows[source_id]
            if (
                source_row["family_id"] != sample["family_id"]
                or source_row["split"] != sample["split"]
            ):
                raise ValueError(
                    "Forecast input/label source crosses families or splits"
                )
        if not math.isclose(
            target.observed_at_s - estimate.observed_at_s,
            config.horizon_source_s,
            abs_tol=1e-9,
        ):
            raise ValueError("Forecast horizon/source clock differs")
        f = sample["trend"]["features"]
        if (f is not None) != (estimate.valid and not estimate.requires_inspection):
            raise ValueError("Unusable current estimate entered forecast features")
        if f is not None:
            if (
                set(f) != set(FEATURES)
                or f["current_area_px2"] != estimate.area_px2
                or f["frame_pixels"]
                != estimate.image_shape[0] * estimate.image_shape[1]
                or f["horizon_source_s"] != config.horizon_source_s
            ):
                raise ValueError("Forecast causal inputs differ from native estimates")
            prior = [
                e
                for e, _ in observations.values()
                if e.frame_index < estimate.frame_index
            ]
            expected_ready = bool(
                prior
                and prior[-1].valid
                and not prior[-1].requires_inspection
                and estimate.observed_at_s - prior[-1].observed_at_s
                <= config.maximum_gap_source_s
            )
            if sample["trend"]["history_ready"] != expected_ready:
                raise ValueError(
                    "Forecast history readiness differs from source coverage"
                )
            if expected_ready:
                previous = prior[-1]
                duration = estimate.observed_at_s - previous.observed_at_s
                if (
                    not previous.valid
                    or previous.requires_inspection
                    or duration > config.maximum_gap_source_s
                ):
                    raise ValueError("Unusable history entered forecast")
                if f["history_duration_source_s"] != duration:
                    raise ValueError("Forecast history duration differs")
                if not math.isclose(
                    f["recent_rate_px2_per_source_s"],
                    (estimate.area_px2 - previous.area_px2) / duration,
                    abs_tol=1e-9,
                ):
                    raise ValueError("Forecast rate does not reconstruct")
            elif (
                f["recent_rate_px2_per_source_s"] is not None
                or f["history_duration_source_s"] is not None
            ):
                raise ValueError("Startup history was fabricated")
        elif sample["trend"]["history_ready"]:
            raise ValueError("Unusable estimate cannot have ready history")
        if sample["labels"] is not None:
            values = {}
            for label, ref in sample["label_refs"].items():
                source_row = original_rows[source_ids[label]]
                seed = next(
                    obj for obj in source_row["objects"] if obj["class_id"] == 0
                )
                if (
                    ref["source_sample_id"] != source_row["sample_id"]
                    or ref["original_mask_path"] != seed["mask_path"]
                    or not seed["label_eligible"]
                ):
                    raise ValueError(
                        "Forecast label does not match its original source sample"
                    )
                path = safe_path(root, ref["mask_path"])
                if (
                    file_sha256(path) != ref["mask_sha256"]
                    or ref["mask_sha256"]
                    != original["artifact_sha256"][ref["original_mask_path"]]
                ):
                    raise ValueError("Original proxy label identity differs")
                mask = cv2.imread(str(path), 0)
                if (
                    mask.shape != estimate.image_shape
                    or not set(np.unique(mask)) <= {0, 255}
                    or not np.any(mask)
                ):
                    raise ValueError("Invalid original proxy label")
                values[label] = int(np.count_nonzero(mask))
            labels = sample["labels"]
            if (
                values["current"] != labels["current_proxy_area_px2"]
                or values["future"] != labels["future_proxy_area_px2"]
                or values["future"] - values["current"] != labels["proxy_gain_px2"]
                or not math.isclose(
                    labels["relative_proxy_gain"],
                    (values["future"] - values["current"]) / values["current"],
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Forecast proxy labels do not reconstruct")
    if len(samples) != m["samples"] or identifiers != expected_samples:
        raise ValueError("Forecast sample count differs")
    if file_sha256(root / "manifest.json") != manifest_hash:
        raise ValueError("Forecast cohort changed during audit")
    for name, digest in m["files"].items():
        if file_sha256(safe_path(root, name)) != digest:
            raise ValueError("Forecast cohort artifact changed during audit")
    return dict(
        status="pass",
        samples=len(samples),
        families=len(groups),
        families_by_split={
            split: sum(v == split for v in groups.values())
            for split in ["train", "validation"]
        },
        rate_ready_by_split={
            split: sum(
                s["split"] == split and s["trend"]["history_ready"] for s in samples
            )
            for split in ["train", "validation"]
        },
        final_test_scored=False,
        manifest_sha256=file_sha256(root / "manifest.json"),
    )
