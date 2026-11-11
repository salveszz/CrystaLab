# FILE PURPOSE: Serve a local polygon editor with explicit exact-byte human review and export.
# Drafts may be uncertain; review receipts remain immutable. Revision copies retain
# prior evidence and unchanged reviews. This module never records a review on read.
"""Repository-local annotation review UI and a loopback-only HTTP adapter."""

import copy
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
from urllib.parse import urlparse

from vision.annotations import (
    CLASSES,
    _validated_annotation,
    export_reviewed,
    record_review,
)
from vision.video import file_sha256

STATIC = Path(__file__).resolve().parent / "static"


class ReviewConflict(RuntimeError):
    pass


def member_path(root, relative):
    path = Path(relative)
    resolved = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root):
        raise ValueError("Bundle artifact path is invalid")
    return resolved


def atomic_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    handle, name = tempfile.mkstemp(prefix=".review-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


class ReviewStore:
    def __init__(self, bundle, *, read_only=False):
        self.lock = threading.RLock()
        self.read_only = read_only
        self.activate(bundle)

    def activate(self, bundle):
        root = Path(bundle).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != "crystalab_annotation_bundle_v1"
            or manifest["status"] != "ready_for_review"
            or manifest["classes"] != CLASSES
        ):
            raise ValueError("Open a prepared annotation bundle")
        if not manifest["samples"] or any(
            s["source_record"]["split"] != "train" for s in manifest["samples"]
        ):
            raise ValueError("The pilot review interface opens training bundles only")
        for sample in manifest["samples"]:
            for key in ("image_path", "annotation_path", "review_path"):
                member_path(root, sample[key])
        self.root, self.manifest = root, manifest
        self.manifest_hash = file_sha256(root / "manifest.json")
        self.bundle_id = hashlib.sha256(
            (str(root) + self.manifest_hash).encode()
        ).hexdigest()

    def guard(self, bundle_id=None, *, write=False):
        if write and self.read_only:
            raise PermissionError("This is a read-only preview.")
        if file_sha256(self.root / "manifest.json") != self.manifest_hash:
            raise ReviewConflict(
                "The bundle changed on disk. Restart the review server to reload it."
            )
        if bundle_id is not None and bundle_id != self.bundle_id:
            raise ReviewConflict(
                "A different review version is open. Reload this page before saving."
            )

    def sample(self, index):
        if type(index) is not int or not 0 <= index < len(self.manifest["samples"]):
            raise ValueError("Unknown image")
        return self.manifest["samples"][index]

    def document(self, index):
        sample = self.sample(index)
        path = member_path(self.root, sample["annotation_path"])
        document = json.loads(path.read_text())
        _validated_annotation(
            self.root, sample, annotation=document, allow_uncertain=True
        )
        return document, file_sha256(path)

    def status(self, index, document=None, revision=None):
        sample = self.sample(index)
        if document is None:
            document, revision = self.document(index)
        path = member_path(self.root, sample["review_path"])
        if not path.exists():
            return (
                "proposed" if revision == sample["initial_proposal_sha256"] else "draft"
            ), None
        receipt = json.loads(path.read_text())
        valid = bool(
            receipt.get("reviewer")
            and receipt.get("acknowledgment")
            and receipt.get("sample_id") == sample["source_record"]["sample_id"]
            and receipt.get("image_sha256") == sample["image_sha256"]
            and receipt.get("annotation_sha256") == revision
        )
        try:
            _validated_annotation(self.root, sample, annotation=document)
        except ValueError:
            valid = False
        if not document["shapes"] and receipt.get("empty_reviewed") is not True:
            valid = False
        return ("reviewed" if valid else "stale_review"), receipt

    def summary(self):
        with self.lock:
            self.guard()
            samples = []
            for index, sample in enumerate(self.manifest["samples"]):
                state, receipt = self.status(index)
                source = sample["source_record"]
                samples.append(
                    dict(
                        index=index,
                        name=Path(sample["annotation_path"]).stem,
                        source_sample_id=source["sample_id"],
                        frame_index=source["frame_index"],
                        status=state,
                        reviewer=None if receipt is None else receipt.get("reviewer"),
                    )
                )
            return dict(
                bundle_id=self.bundle_id,
                bundle_name=self.root.name,
                bundle_path=str(self.root),
                read_only=self.read_only,
                classes=CLASSES,
                material=self.manifest["samples"][0]["source_record"].get(
                    "material",
                    self.manifest["source_protocol"].get(
                        "material", "Unknown material"
                    ),
                ),
                run_id=self.manifest["run_id"],
                samples=samples,
                reviewed=sum(s["status"] == "reviewed" for s in samples),
                stale_reviews=sum(s["status"] == "stale_review" for s in samples),
            )

    def get(self, index):
        with self.lock:
            self.guard()
            document, revision = self.document(index)
            status, receipt = self.status(index, document, revision)
            return dict(
                bundle_id=self.bundle_id,
                index=index,
                annotation=document,
                revision=revision,
                status=status,
                review=receipt,
                source=self.sample(index)["source_record"],
                image_url=f"/api/image/{index}",
                read_only=self.read_only,
            )

    def image(self, index):
        with self.lock:
            self.guard()
            sample = self.sample(index)
            path = member_path(self.root, sample["image_path"])
            if file_sha256(path) != sample["image_sha256"]:
                raise ValueError("The original image changed.")
            return path.read_bytes()

    def editable(self, index, payload):
        self.guard(payload.get("bundle_id"), write=True)
        if payload.get("bundle_id") != self.bundle_id:
            raise ReviewConflict("Reload the current review version before saving.")
        document, revision = self.document(index)
        sample = self.sample(index)
        if payload.get("revision") != revision:
            raise ReviewConflict(
                "This draft changed in another window. Reload before saving."
            )
        if member_path(self.root, sample["review_path"]).exists():
            raise ReviewConflict(
                "This image already has a review. Use Revise selected to keep the earlier review intact."
            )
        return document, sample

    def save_draft(self, index, payload):
        with self.lock:
            document, sample = self.editable(index, payload)
            if type(payload.get("image_uncertain")) is not bool:
                raise ValueError("Choose whether the image is uncertain.")
            shapes = payload.get("shapes")
            if not isinstance(shapes, list) or len(shapes) > 128:
                raise ValueError("Provide a bounded list of object outlines.")
            cleaned = []
            for shape in shapes:
                if not isinstance(shape, dict) or shape.get("label") not in CLASSES:
                    raise ValueError("Choose a supported object label.")
                points = shape.get("points")
                if not isinstance(points, list) or not 3 <= len(points) <= 2000:
                    raise ValueError("Each outline needs at least three points.")
                if any(
                    not isinstance(p, list)
                    or len(p) != 2
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) for v in p
                    )
                    for p in points
                ):
                    raise ValueError(
                        "Outline coordinates must be finite pixel positions."
                    )
                flags = shape.get("flags", {})
                if not isinstance(flags, dict):
                    raise ValueError("Outline flags must be an object.")
                uncertain = flags.get("uncertain", False)
                if type(uncertain) is not bool:
                    raise ValueError("Outline uncertainty must be explicit.")
                cleaned.append(
                    dict(
                        label=shape["label"],
                        points=copy.deepcopy(points),
                        group_id=shape.get("group_id"),
                        shape_type="polygon",
                        flags=dict(proposed=False, uncertain=uncertain),
                        description=str(shape.get("description", ""))[:2000],
                    )
                )
            note = payload.get("note", "")
            if not isinstance(note, str) or len(note) > 10000:
                raise ValueError("Notes must be text under 10,000 characters.")
            document["shapes"] = cleaned
            document["flags"] = dict(
                document.get("flags", {}),
                reviewed=False,
                uncertain=payload["image_uncertain"],
            )
            document["review_note"] = note
            _validated_annotation(
                self.root, sample, annotation=document, allow_uncertain=True
            )
            atomic_json(member_path(self.root, sample["annotation_path"]), document)
            return self.get(index)

    def review(self, index, payload):
        with self.lock:
            document, sample = self.editable(index, payload)
            reviewer = payload.get("reviewer")
            if (
                not isinstance(reviewer, str)
                or not reviewer.strip()
                or len(reviewer) > 200
            ):
                raise ValueError("Enter the name of the person reviewing this image.")
            if payload.get("ack_image_reviewed") is not True:
                raise ValueError(
                    "Confirm that you checked this image and its outlines."
                )
            if type(payload.get("empty_reviewed")) is not bool:
                raise ValueError("An empty image needs its own explicit check.")
            record_review(
                self.root,
                Path(sample["annotation_path"]).name,
                reviewer=reviewer.strip(),
                acknowledge_image_reviewed=True,
                empty_reviewed=payload["empty_reviewed"],
                note=document.get("review_note", ""),
            )
            return self.get(index)

    def unique_output(self, suffix):
        for number in range(1, 10001):
            candidate = self.root.parent / f"{self.root.name}_{suffix}_{number:03d}"
            if not candidate.exists():
                return candidate
        raise ValueError(
            "Choose a new bundle location before creating another version."
        )

    def revise(self, index, payload):
        with self.lock:
            self.guard(payload.get("bundle_id"), write=True)
            if payload.get("bundle_id") != self.bundle_id:
                raise ReviewConflict("Reload this review version first.")
            document, revision = self.document(index)
            if payload.get("revision") != revision:
                raise ReviewConflict("The selected draft changed. Reload it first.")
            old_root = self.root
            manifest = copy.deepcopy(self.manifest)
            destination = self.unique_output("revision")
            destination.mkdir()
            try:
                for i, sample in enumerate(manifest["samples"]):
                    self.document(i)
                    for key in ("image_path", "annotation_path"):
                        target = member_path(destination, sample[key])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(member_path(old_root, sample[key]), target)
                    state, _ = self.status(i)
                    review = member_path(old_root, sample["review_path"])
                    target = member_path(destination, sample["review_path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if i != index and state == "reviewed":
                        shutil.copyfile(review, target)
                manifest["revision_parent"] = dict(
                    path=str(old_root),
                    manifest_sha256=self.manifest_hash,
                    selected_sample_id=self.sample(index)["source_record"]["sample_id"],
                    selected_annotation_sha256=revision,
                )
                atomic_json(destination / "manifest.json", manifest)
                self.activate(destination)
                return dict(summary=self.summary(), sample=self.get(index))
            except Exception:
                # A partial revision stays visible; the original bundle is never removed.
                self.activate(old_root)
                raise

    def export(self, payload):
        with self.lock:
            self.guard(payload.get("bundle_id"), write=True)
            if payload.get("bundle_id") != self.bundle_id:
                raise ReviewConflict("Reload this review version first.")
            destination = self.unique_output("reviewed")
            report = export_reviewed(self.root, destination)
            return dict(**report, output=str(destination))


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bundle, *, port=8765, read_only=False):
        self.store = ReviewStore(bundle, read_only=read_only)
        self.session_token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_data(self, status, data, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, status, value):
        self.send_data(status, json.dumps(value, allow_nan=False).encode())

    def trusted_request(self, *, write=False):
        authorities = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if self.headers.get("Host") not in authorities:
            raise PermissionError("Use the printed local review URL.")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {
            "http://" + host for host in authorities
        }:
            raise PermissionError("Review requests must come from this local page.")
        if write and not hmac.compare_digest(
            self.headers.get("X-CrystaLab-Session", ""), self.server.session_token
        ):
            raise PermissionError("Reload the review page before saving.")

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
            if path.startswith("/api/sample/"):
                self.json_response(
                    200, self.server.store.get(int(path.rsplit("/", 1)[-1]))
                )
                return
            if path.startswith("/api/image/"):
                self.send_data(
                    200,
                    self.server.store.image(int(path.rsplit("/", 1)[-1])),
                    "image/png",
                )
                return
            assets = {
                "/": ("review.html", "text/html; charset=utf-8"),
                "/assets/review.js": ("review.js", "text/javascript; charset=utf-8"),
                "/assets/review.css": ("review.css", "text/css; charset=utf-8"),
            }
            if path not in assets:
                self.json_response(404, dict(error="Page not found"))
                return
            name, mime = assets[path]
            self.send_data(200, (STATIC / name).read_bytes(), mime)
        except PermissionError as error:
            self.json_response(403, dict(error=str(error)))
        except ReviewConflict as error:
            self.json_response(409, dict(error=str(error)))
        except (ValueError, KeyError, TypeError) as error:
            self.json_response(422, dict(error=str(error)))
        except OSError:
            self.json_response(500, dict(error="The review files could not be read."))

    def do_POST(self):
        try:
            self.trusted_request(write=True)
            length = int(self.headers.get("Content-Length", "0"))
            if (
                not 0 < length <= 2_000_000
                or self.headers.get("Content-Type", "").split(";")[0]
                != "application/json"
            ):
                raise ValueError("Send a bounded JSON review request.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Invalid review request.")
            path = urlparse(self.path).path
            parts = path.strip("/").split("/")
            if path == "/api/export":
                value = self.server.store.export(payload)
            elif len(parts) == 4 and parts[:2] == ["api", "sample"]:
                index = int(parts[2])
                action = parts[3]
                method = {
                    "draft": self.server.store.save_draft,
                    "review": self.server.store.review,
                    "revise": self.server.store.revise,
                }.get(action)
                if method is None:
                    self.json_response(404, dict(error="Unknown review action"))
                    return
                value = method(index, payload)
            else:
                self.json_response(404, dict(error="Unknown review action"))
                return
            self.json_response(200, value)
        except PermissionError as error:
            self.json_response(403, dict(error=str(error)))
        except ReviewConflict as error:
            self.json_response(409, dict(error=str(error)))
        except FileExistsError as error:
            self.json_response(409, dict(error=str(error)))
        except (ValueError, KeyError, TypeError) as error:
            self.json_response(422, dict(error=str(error)))
        except OSError:
            self.json_response(500, dict(error="The review files could not be saved."))
