# FILE PURPOSE: Serve synchronized evidence, process replay and explicit operator decisions locally.
# Source artifacts remain immutable; receipts and counterfactual runs are separate.
"""Loopback operator console for shadow review and bounded simulated pause/resume."""

from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import threading
from urllib.parse import urlparse

import cv2

from controllers.audit import audit_control_benchmark
from controllers.operator_demo import run_operator_demo, validate_reviews
from controllers.shadow import CandidateEvidence, run_shadow
from controllers.supervised_demo import validate_demo
from controllers.supervision import value_sha256
from vision.training_data import safe_path
from vision.video import VideoReader, file_sha256
from .review import ReviewConflict, ReviewHandler, atomic_json

STATIC = Path(__file__).resolve().parent / "static"
read = lambda p: json.loads(Path(p).read_text())
rows = lambda p: [json.loads(v) for v in Path(p).read_text().splitlines()]


def prepare_operator_session(analysis, policy, output, *, config=None, qa_mode=False):
    source = CandidateEvidence(analysis)
    kind = source.manifest["source_info"]["evidence_kind"]
    if qa_mode and kind != "fixture":
        raise ValueError("Automated review QA requires generated fixture pixels")
    if config is not None:
        validate_demo(read(config), source)
    output, policy = Path(output).resolve(), Path(policy).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema="crystalab_operator_session_v1",
        status="preparing",
        source_analysis=str(source.root),
        source_manifest_sha256=source.manifest_sha256,
        source_info=source.manifest["source_info"],
        run_id=source.manifest["run_id"],
        clock_id=source.clock_id,
        reviewer_kind="automated_fixture_qa" if qa_mode else "human_self_attestation",
        simulation_available=config is not None,
        physical_hardware_contacted=False,
    )
    atomic_json(output / "manifest.json", manifest)
    try:
        shutil.copyfile(policy, output / "policy.json")
        if config is not None:
            shutil.copyfile(config, output / "simulation_config.json")
        shadow = run_shadow(source.root, output / "policy.json", output / "shadow")
        (output / "frames").mkdir()
        (output / "reviews").mkdir()
        (output / "runs").mkdir()
        replay_root = Path(source.manifest["source_replay"]).resolve()
        measurement_path = replay_root / "measurements.jsonl"
        if (
            file_sha256(measurement_path)
            != source.parent["files"]["measurements.jsonl"]
        ):
            raise ValueError("Native observation measurement identity changed")
        measurements = {r["frame_index"]: r for r in rows(measurement_path)}
        frames = []
        decisions = rows(output / "shadow/decisions.jsonl")
        with VideoReader(output / "shadow/overlay.avi") as video:
            for index, (packet, original) in enumerate(source.packets()):
                frame = next(video)
                path = f"frames/{index:06d}.png"
                if not cv2.imwrite(str(output / path), frame.image):
                    raise OSError("Cannot save replay preview")
                measurement = measurements[packet.frame_index]
                frames.append(
                    dict(
                        ordinal=index,
                        frame_index=packet.frame_index,
                        time_s=packet.observed_at_s,
                        source_frame=original.get("source_frame"),
                        experiment_time_s=original["experiment_time_s"],
                        seed_area_px2=measurement.get("projected_area_px"),
                        seed_valid=measurement["valid"],
                        inspection_required=measurement.get(
                            "requires_inspection", not measurement["valid"]
                        ),
                        observable=decisions[index]["observable"],
                        issues=decisions[index]["reasons"],
                        visible_track_ids=list(packet.visible_track_ids),
                        image_path=path,
                    )
                )
            if next(video, None) is not None:
                raise ValueError("Unmatched operator preview frames")
        atomic_json(output / "frames.json", frames)
        baseline = None
        if config is not None:
            run_operator_demo(
                source.root,
                output / "policy.json",
                output / "simulation_config.json",
                [],
                output / "runs/initial",
            )
            audit_control_benchmark(output / "runs/initial")
            baseline = "runs/initial"
        history = dict(
            entries=[],
            latest_run=baseline,
            latest_run_sha256=None
            if baseline is None
            else file_sha256(output / baseline / "manifest.json"),
        )
        atomic_json(output / "history.json", history)
        source.verify()
        manifest.update(
            status="ready",
            frames=len(frames),
            policy_sha256=file_sha256(output / "policy.json"),
            files={
                str(p.relative_to(output)): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
                and p != output / "manifest.json"
                and p != output / "history.json"
            },
        )
        atomic_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "manifest.json", manifest)
        raise


class OperatorStore:
    def __init__(self, session, *, read_only=False):
        self.root = Path(session).resolve()
        self.lock = threading.RLock()
        self.read_only = read_only
        self.manifest = read(self.root / "manifest.json")
        if (
            self.manifest.get("schema") != "crystalab_operator_session_v1"
            or self.manifest["status"] != "ready"
        ):
            raise ValueError("Open a prepared operator session")
        self.manifest_hash = file_sha256(self.root / "manifest.json")
        self.source = CandidateEvidence(self.manifest["source_analysis"])
        if self.source.manifest_sha256 != self.manifest["source_manifest_sha256"]:
            raise ValueError("Session source changed")
        for path, digest in self.manifest["files"].items():
            if file_sha256(safe_path(self.root, path)) != digest:
                raise ValueError("Operator session artifact changed")
        self.frames = read(self.root / "frames.json")
        self.proposals = rows(self.root / "shadow/proposals.jsonl")
        self.config = (
            read(self.root / "simulation_config.json")
            if self.manifest["simulation_available"]
            else None
        )
        self.mapping = (
            validate_demo(self.config, self.source) if self.config is not None else None
        )
        self.history()

    def history(self):
        if file_sha256(self.root / "manifest.json") != self.manifest_hash:
            raise ReviewConflict("The operator session changed. Restart the server.")
        for relative in (
            "policy.json",
            "frames.json",
            "shadow/proposals.jsonl",
            *(("simulation_config.json",) if self.config is not None else ()),
        ):
            if file_sha256(self.root / relative) != self.manifest["files"][relative]:
                raise ReviewConflict("The saved operator input changed")
        history = read(self.root / "history.json")
        reviews = []
        for entry in history["entries"]:
            path = safe_path(self.root, entry["review_path"])
            if file_sha256(path) != entry["review_file_sha256"]:
                raise ReviewConflict("A saved review changed")
            review = read(path)
            if (
                value_sha256(review) != entry["review_sha256"]
                or review["reviewer_kind"] != self.manifest["reviewer_kind"]
            ):
                raise ReviewConflict("Review receipt identity differs")
            if (
                entry.get("result_path") is not None
                and file_sha256(
                    safe_path(self.root, entry["result_path"]) / "manifest.json"
                )
                != entry["result_manifest_sha256"]
            ):
                raise ReviewConflict("A preserved review result changed")
            reviews.append(review)
        if len(list((self.root / "reviews").glob("*.json"))) != len(reviews):
            raise ReviewConflict(
                "An incomplete review needs recovery; saved receipts were preserved"
            )
        validate_reviews(reviews, self.source, mapping=self.mapping, config=self.config)
        for review in reviews:
            proposal = next(
                (
                    p
                    for p in self.proposals
                    if p["proposal"]["proposal_id"] == review["proposal_id"]
                ),
                None,
            )
            if (
                proposal is None
                or proposal["proposal_sha256"] != review["proposal_sha256"]
                or proposal["proposal"]["proposed_at_s"] > review["source_time_s"]
            ):
                raise ReviewConflict(
                    "Saved review refers to unobserved or changed evidence"
                )
            frame = next(
                f
                for f in self.frames
                if f["frame_index"] == review["source_frame_index"]
            )
            evidence = "shadow/" + proposal["artifact_evidence"]["image_path"]
            if (
                review["operator_session_manifest_sha256"] != self.manifest_hash
                or review["displayed_frame_sha256"]
                != self.manifest["files"][frame["image_path"]]
                or review["proposal_evidence_image_sha256"]
                != self.manifest["files"][evidence]
            ):
                raise ReviewConflict(
                    "Saved review belongs to different displayed pixels"
                )
        if history["latest_run"] is not None:
            if (
                file_sha256(
                    safe_path(self.root, history["latest_run"]) / "manifest.json"
                )
                != history["latest_run_sha256"]
            ):
                raise ReviewConflict("The simulated result changed")
        return history, reviews

    def process(self, history):
        if history["latest_run"] is None:
            return None
        path = safe_path(self.root, history["latest_run"])
        manifest = read(path / "manifest.json")
        if manifest["status"] != "complete":
            raise ReviewConflict("Simulation is incomplete")
        by_method = {}
        for run in manifest["runs"]:
            p = safe_path(path, run["records_path"])
            if file_sha256(p) != manifest["files"][run["records_path"]]:
                raise ReviewConflict("Process records changed")
            by_method[run["method"]] = rows(p)
        return dict(manifest=manifest, records=by_method)

    def summary(self):
        with self.lock:
            history, reviews = self.history()
            process = self.process(history)
            return dict(
                run_id=self.manifest["run_id"],
                session_path=str(self.root),
                source_info=self.manifest["source_info"],
                clock_id=self.manifest["clock_id"],
                source_manifest_sha256=self.source.manifest_sha256,
                operator_session_manifest_sha256=self.manifest_hash,
                revision=value_sha256(history),
                read_only=self.read_only,
                reviewer_kind=self.manifest["reviewer_kind"],
                frames=self.frames,
                pause_default_model_s=None
                if self.config is None
                else self.config["pause_model_duration_s"],
                proposals=[
                    dict(**p, evidence_url=f"/api/evidence/{i}")
                    for i, p in enumerate(self.proposals)
                ],
                reviews=reviews,
                simulation_available=self.mapping is not None,
                mapping=None if self.mapping is None else self.config["clock_mapping"],
                runs=[] if process is None else process["manifest"]["runs"],
                chart=None
                if process is None
                else {
                    m: [
                        dict(
                            time_s=r["time_s"],
                            temperature_K=r["observation"]["value"]
                            if r["observation"]["valid"]
                            else None,
                            target_K=r["setpoint_K"],
                            nominal_K=r.get(
                                "nominal_recipe_setpoint_K", r["setpoint_K"]
                            ),
                            applied_K=r["plant_truth"]["applied_bath_K"],
                        )
                        for r in rr
                    ]
                    for m, rr in process["records"].items()
                },
            )

    def frame(self, index):
        with self.lock:
            if type(index) is not int or not 0 <= index < len(self.frames):
                raise ValueError("Unknown replay frame")
            history, reviews = self.history()
            frame = self.frames[index]
            t = frame["time_s"]
            monotonic = not reviews or t >= reviews[-1]["source_time_s"]
            process = self.process(history)
            row = None
            active = None
            if process is not None:
                now = self.mapping.to_target(t)
                row = next(
                    (
                        r
                        for r in process["records"]["rule_supervised"]
                        if abs(r["time_s"] - now) < 1e-9
                    ),
                    None,
                )
                active = next(
                    (
                        a
                        for a in process["manifest"]["executions"]
                        if a["model_decision_time_s"] <= now < a["pause_until_model_s"]
                    ),
                    None,
                )
            current = []
            for i, entry in enumerate(self.proposals):
                p = entry["proposal"]
                if p["proposed_at_s"] > t:
                    continue
                dispositions = [
                    r
                    for r in reviews
                    if r["proposal_id"] == p["proposal_id"]
                    and r["action"] in {"approve_pause", "decline"}
                ]
                pending = not dispositions and t < p["expires_at_s"]
                fresh = (
                    row is not None and row["supervision"]["process_feedback_usable"]
                )
                budget = bool(
                    process
                    and sum(
                        a["requested_pause_duration_model_s"]
                        for a in process["manifest"]["executions"]
                    )
                    + self.config["pause_model_duration_s"]
                    <= self.config["maximum_total_pause_model_s"]
                )
                can_approve = bool(
                    self.mapping
                    and budget
                    and p["kind"] == "ramp_review"
                    and pending
                    and frame["observable"]["crystal_candidate"]
                    and p["track_id"] in frame["visible_track_ids"]
                    and fresh
                    and active is None
                )
                current.append(
                    dict(
                        index=i,
                        proposal_id=p["proposal_id"],
                        status=dispositions[-1]["action"]
                        if dispositions
                        else "pending_review"
                        if pending
                        else "expired",
                        can_approve=can_approve and monotonic and not self.read_only,
                        can_review=monotonic and not self.read_only,
                    )
                )
            return dict(
                frame=frame,
                proposals=current,
                process=row,
                active_pause=active,
                can_resume=bool(
                    active
                    and active["model_decision_time_s"] < now
                    and row["supervision"]["process_feedback_usable"]
                    and monotonic
                    and not self.read_only
                ),
                revision=value_sha256(history),
            )

    def image(self, index):
        if type(index) is not int or not 0 <= index < len(self.frames):
            raise ValueError("Unknown replay frame")
        self.history()
        path = self.frames[index]["image_path"]
        if file_sha256(self.root / path) != self.manifest["files"][path]:
            raise ReviewConflict("Replay preview changed")
        return (self.root / path).read_bytes()

    def evidence(self, index):
        if type(index) is not int or not 0 <= index < len(self.proposals):
            raise ValueError("Unknown proposal evidence")
        self.history()
        relative = "shadow/" + self.proposals[index]["artifact_evidence"]["image_path"]
        if file_sha256(self.root / relative) != self.manifest["files"][relative]:
            raise ReviewConflict("Proposal evidence changed")
        return (self.root / relative).read_bytes()

    def review(self, payload):
        with self.lock:
            if self.read_only:
                raise PermissionError("This operator preview is read-only")
            history, reviews = self.history()
            if payload.get("revision") != value_sha256(history):
                raise ReviewConflict("Another review was saved. Reload the session.")
            index = payload["frame_ordinal"]
            if type(index) is not int or not 0 <= index < len(self.frames):
                raise ValueError("Select an observed replay frame")
            if payload.get("acknowledged") is not True:
                raise ValueError("Acknowledge that you reviewed this replay frame")
            frame = self.frames[index]
            entry = next(
                (
                    p
                    for p in self.proposals
                    if p["proposal"]["proposal_id"] == payload.get("proposal_id")
                ),
                None,
            )
            if (
                entry is None
                or entry["proposal_sha256"] != payload.get("proposal_sha256")
                or entry["proposal"]["proposed_at_s"] > frame["time_s"]
            ):
                raise ValueError("Review must bind an already visible exact proposal")
            review = dict(
                schema="crystalab_operator_replay_review_v1",
                review_id=f"{self.manifest['run_id']}:operator_review:{len(reviews) + 1:06d}",
                reviewer=payload.get("reviewer"),
                reviewer_kind=self.manifest["reviewer_kind"],
                comment=payload.get("comment", ""),
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                acknowledged=True,
                run_id=self.manifest["run_id"],
                clock_id=self.manifest["clock_id"],
                source_manifest_sha256=self.source.manifest_sha256,
                operator_session_manifest_sha256=self.manifest_hash,
                displayed_frame_sha256=self.manifest["files"][frame["image_path"]],
                proposal_evidence_image_sha256=self.manifest["files"][
                    "shadow/" + entry["artifact_evidence"]["image_path"]
                ],
                evidence_kind=self.manifest["source_info"]["evidence_kind"],
                source_frame_index=frame["frame_index"],
                source_time_s=frame["time_s"],
                proposal_id=payload["proposal_id"],
                proposal_sha256=payload["proposal_sha256"],
                action=payload["action"],
                pause_duration_model_s=payload.get("pause_duration_model_s"),
                pause_review_id=payload.get("pause_review_id"),
                physical_hardware_authorized=False,
                previous_review_sha256=None
                if not reviews
                else value_sha256(reviews[-1]),
            )
            validate_reviews(
                [*reviews, review],
                self.source,
                mapping=self.mapping,
                config=self.config,
            )
            if review["action"] in {"approve_pause", "decline"} and any(
                r["proposal_id"] == review["proposal_id"]
                and r["action"] in {"approve_pause", "decline"}
                for r in reviews
            ):
                raise ReviewConflict("This proposal already has a disposition")
            self.source.verify()
            for relative, digest in self.manifest["files"].items():
                if file_sha256(safe_path(self.root, relative)) != digest:
                    raise ReviewConflict("Original operator evidence changed")
            run_path = None
            if self.mapping is not None:
                run_path = f"runs/review_{len(reviews) + 1:06d}_{secrets.token_hex(4)}"
                run_operator_demo(
                    self.source.root,
                    self.root / "policy.json",
                    self.root / "simulation_config.json",
                    [*reviews, review],
                    self.root / run_path,
                )
                audit_control_benchmark(self.root / run_path)
            relative = f"reviews/{len(reviews) + 1:06d}.json"
            with (self.root / relative).open("x") as stream:
                stream.write(json.dumps(review, indent=2, allow_nan=False) + "\n")
            history["entries"].append(
                dict(
                    review_path=relative,
                    review_file_sha256=file_sha256(self.root / relative),
                    review_sha256=value_sha256(review),
                    result_path=run_path,
                    result_manifest_sha256=None
                    if run_path is None
                    else file_sha256(self.root / run_path / "manifest.json"),
                )
            )
            if run_path is not None:
                history.update(
                    latest_run=run_path,
                    latest_run_sha256=file_sha256(
                        self.root / run_path / "manifest.json"
                    ),
                )
            atomic_json(self.root / "history.json", history)
            return self.summary()


class OperatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session, *, port=8766, read_only=False):
        self.store = OperatorStore(session, read_only=read_only)
        self.session_token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), OperatorHandler)


class OperatorHandler(ReviewHandler):
    def do_GET(self):
        try:
            self.trusted_request()
            path = urlparse(self.path).path
            if path == "/api/session":
                self.json_response(
                    200,
                    dict(
                        **self.server.store.summary(),
                        session_token=self.server.session_token,
                    ),
                )
                return
            for prefix, method, mime in (
                ("/api/frame/", self.server.store.frame, None),
                ("/api/image/", self.server.store.image, "image/png"),
                ("/api/evidence/", self.server.store.evidence, "image/png"),
            ):
                if path.startswith(prefix):
                    index = int(path[len(prefix) :])
                    value = method(index)
                    if mime:
                        self.send_data(200, value, mime)
                    else:
                        self.json_response(200, value)
                    return
            assets = {
                "/": ("operator.html", "text/html; charset=utf-8"),
                "/assets/operator.js": (
                    "operator.js",
                    "text/javascript; charset=utf-8",
                ),
                "/assets/operator.css": ("operator.css", "text/css; charset=utf-8"),
            }
            if path not in assets:
                self.json_response(404, dict(error="Page not found"))
                return
            name, mime = assets[path]
            self.send_data(200, (STATIC / name).read_bytes(), mime)
        except PermissionError as e:
            self.json_response(403, dict(error=str(e)))
        except ReviewConflict as e:
            self.json_response(409, dict(error=str(e)))
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self.json_response(422, dict(error=str(e)))
        except OSError:
            self.json_response(500, dict(error="Operator evidence could not be read"))

    def do_POST(self):
        try:
            self.trusted_request(write=True)
            length = int(self.headers.get("Content-Length", "0"))
            if (
                not 0 < length <= 20_000
                or self.headers.get("Content-Type", "").split(";")[0]
                != "application/json"
            ):
                raise ValueError("Send a bounded JSON review")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Send a review object")
            if urlparse(self.path).path != "/api/review":
                self.json_response(404, dict(error="Unknown operator action"))
                return
            self.json_response(200, self.server.store.review(payload))
        except PermissionError as e:
            self.json_response(403, dict(error=str(e)))
        except (ReviewConflict, FileExistsError) as e:
            self.json_response(409, dict(error=str(e)))
        except (ValueError, KeyError, TypeError, IndexError) as e:
            self.json_response(422, dict(error=str(e)))
        except OSError:
            self.json_response(500, dict(error="The replay review could not be saved"))
