# FILE PURPOSE: Preserve causal adaptive forecasts as immutable operator replay context.
"""Optional source-bound forecast display; numerical actions still need explicit reviews."""

import copy
import json
from pathlib import Path
import shutil

from controllers.supervision import value_sha256
from models.adaptive_area import AdaptiveAreaModel
from models.adaptive_area_run import load_reference, replay_adaptive
from vision.area_forecast_data import ReplayAreas
from vision.training_data import safe_path, save_json
from vision.video import file_sha256


def state_summary(state):
    if state is None:
        return None
    return dict(
        state_sha256=state["state_sha256"],
        observed_at_s=state["estimate"]["observed_at_s"],
        received_at_s=state["estimate"]["received_at_s"],
        source_frame_index=state["estimate"]["frame_index"],
        current_area_px2=state["estimate"]["area_px2"],
        forecast=state["forecast"],
        **state["parameter_after"],
    )


def prepare_forecast_context(source, reference_path, output, *, base_pause, step):
    output = Path(output).resolve()
    reference_path = Path(reference_path).resolve()
    reference, config = load_reference(reference_path)
    reader = ReplayAreas(source.manifest["source_replay"])
    if reader.manifest_hash != source.manifest["source_replay_manifest_sha256"]:
        raise ValueError("Forecast context requires the exact candidate observations")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference_path, output / "reference")
    replay_adaptive(reader.root, output / "reference", output / "replay")
    saved = [
        json.loads(v) for v in (output / "replay/states.jsonl").read_text().splitlines()
    ]
    estimates = list(reader.estimates())
    model = AdaptiveAreaModel(config, reference)
    index, latest, contexts = 0, None, []
    for packet, _ in source.packets():
        now = packet.observed_at_s
        while index < len(estimates) and estimates[index][0].received_at_s <= now:
            latest = model.update(estimates[index][0])
            if value_sha256(latest) != value_sha256(saved[index]):
                raise ValueError(
                    "Displayed state differs from the causal forecast replay"
                )
            index += 1
        context = dict(
            source_frame_index=packet.frame_index,
            source_time_s=now,
            source_manifest_sha256=source.manifest_sha256,
            source_replay_manifest_sha256=reader.manifest_hash,
            reference_manifest_sha256=file_sha256(output / "reference/manifest.json"),
            state=state_summary(latest),
            recommendation=model.recommend(
                now, base_pause_model_s=base_pause, model_step_s=step
            ),
        )
        context["context_sha256"] = value_sha256(context)
        contexts.append(context)
    save_json(output / "contexts.json", contexts)
    producer = output / "source_snapshot/src/controllers/forecast_context.py"
    producer.parent.mkdir(parents=True)
    shutil.copyfile(__file__, producer)
    source.verify()
    reader.verify()
    save_json(
        output / "manifest.json",
        dict(
            schema="crystalab_operator_forecast_context_v1",
            status="complete",
            source_manifest_sha256=source.manifest_sha256,
            source_replay_manifest_sha256=reader.manifest_hash,
            reference_manifest_sha256=file_sha256(output / "reference/manifest.json"),
            base_pause_model_s=base_pause,
            model_step_s=step,
            frames=len(contexts),
            source_sha256={
                "src/controllers/forecast_context.py": file_sha256(producer)
            },
            scope="Observed-mask forecast diagnostic; no calibrated confidence, action-response utility or hardware authorization",
            calibration_note=(
                "Real-source forecasts use an uncalibrated reference and are unscored shadow output."
                if source.manifest["source_info"]["evidence_kind"] == "real"
                else "Synthetic/fixture forecast diagnostic; adaptation did not beat recent rate in the frozen study."
            ),
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )
    return ForecastContext(output, source)


class ForecastContext:
    def __init__(self, root, source):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.manifest_hash = file_sha256(self.root / "manifest.json")
        if (
            self.manifest.get("schema") != "crystalab_operator_forecast_context_v1"
            or self.manifest.get("status") != "complete"
            or self.manifest["source_manifest_sha256"] != source.manifest_sha256
            or self.manifest["source_replay_manifest_sha256"]
            != source.manifest["source_replay_manifest_sha256"]
        ):
            raise ValueError("Forecast context belongs to different source evidence")
        self.verify()
        self.contexts = {
            v["source_frame_index"]: v
            for v in json.loads((self.root / "contexts.json").read_text())
        }
        if len(self.contexts) != self.manifest["frames"]:
            raise ValueError("Forecast context contains duplicate frames")

    def verify(self):
        if file_sha256(self.root / "manifest.json") != self.manifest_hash:
            raise ValueError("Forecast context manifest changed")
        for path, digest in self.manifest["files"].items():
            if file_sha256(safe_path(self.root, path)) != digest:
                raise ValueError("Forecast context artifact changed")

    def frame(self, frame_index):
        return copy.deepcopy(self.contexts[frame_index])

    def validate_reviews(self, reviews):
        for review in reviews:
            context = self.frame(review["source_frame_index"])
            if review.get("forecast_context") != context:
                raise ValueError(
                    "Review must preserve the exact displayed forecast context"
                )
            basis = review.get("pause_basis", "manual")
            if basis not in {"manual", "adaptive_suggestion"}:
                raise ValueError("Unknown operator pause basis")
            if basis == "adaptive_suggestion":
                rec = context["recommendation"]
                if (
                    review["action"] != "approve_pause"
                    or not rec["eligible"]
                    or review["pause_duration_model_s"] != rec["pause_model_s"]
                ):
                    raise ValueError(
                        "An adaptive suggestion must match its eligible displayed duration"
                    )
