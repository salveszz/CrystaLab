# FILE PURPOSE: Audit physical/measurement source metadata and explicitly listed
# public files. This standalone tool preserves access failures and file hashes;
# it does not generate datasets, fit models or change simulation parameters.

"""Inspect physical-data sources, optionally caching bounded public downloads."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def _describe_payload(data, kind):
    prefix = data.lstrip()[:100].lower()
    if not data or prefix.startswith((b"<!doctype html", b"<html")):
        raise ValueError("Empty file or HTML page returned instead of data")
    details = {}
    if kind == "pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("Response is not a PDF")
    elif kind == "mov":
        if data[4:8] not in (b"ftyp", b"mdat", b"moov", b"wide"):
            raise ValueError("Response has no recognized QuickTime container header")
    elif kind == "csv":
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        if len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError("CSV has no tabular observations")
        details = {"columns": rows[0], "data_rows": len(rows) - 1}
    elif kind == "text":
        details["text_preview"] = data.decode("utf-8-sig")[:300]
    else:
        raise ValueError(f"Unsupported research asset format: {kind}")
    return details


def audit_sources(registry, cache, *, fetch=False):
    """Audit cached assets; network is used only when fetch=True."""
    cache = Path(cache)
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_requested": fetch,
        "sources": [],
    }
    for source in registry["sources"]:
        result = {
            "source_id": source["source_id"],
            "data_status": source["data_status"],
            "reason": source["reason"],
            "assets": [],
        }
        for asset in source.get("assets", []):
            entry = {"name": asset["name"], "url": asset["url"]}
            try:
                if (
                    Path(asset["name"]).name != asset["name"]
                    or Path(source["source_id"]).name != source["source_id"]
                ):
                    raise ValueError("Asset identifiers must be plain filenames")
                path = cache / source["source_id"] / asset["name"]
                if path.exists():
                    data = path.read_bytes()
                    entry["status"] = "cached"
                elif fetch:
                    request = urllib.request.Request(
                        asset["url"],
                        headers={"User-Agent": "CrystaLab research-data audit"},
                    )
                    with urllib.request.urlopen(request, timeout=15) as response:
                        entry["resolved_url"] = response.url
                        data = response.read(asset["max_bytes"] + 1)
                    entry["status"] = "downloaded"
                else:
                    entry["status"] = "not_cached"
                    result["assets"].append(entry)
                    continue
                if len(data) > asset["max_bytes"]:
                    raise ValueError("Asset exceeds the declared download size limit")
                entry.update(_describe_payload(data, asset["kind"]))
                digest = hashlib.sha256(data).hexdigest()
                if asset.get("sha256") is not None and asset["sha256"] != digest:
                    raise ValueError("Asset SHA256 changed; source review is required")
                entry.update(sha256=digest, bytes=len(data))
                if entry["status"] == "downloaded":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(data)
            except (OSError, ValueError, UnicodeError) as error:
                entry.update(
                    status="unavailable_or_invalid",
                    error=f"{type(error).__name__}: {error}",
                )
            result["assets"].append(entry)
        report["sources"].append(result)
    report["verified_kdp_trajectory_sources"] = [
        s["source_id"]
        for s in registry["sources"]
        if s["data_status"] == "verified_measured_trajectories"
    ]
    report["measurement_readiness"] = (
        "no_verified_KDP_growth_trajectories"
        if not report["verified_kdp_trajectory_sources"]
        else "review_units_conditions_and_measurement_uncertainty"
    )
    return report


def read_registry(path):
    registry = json.loads(Path(path).read_text())
    if (
        not isinstance(registry, dict)
        or type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or not isinstance(registry.get("sources"), list)
    ):
        raise ValueError("Use source registry schema v1")
    ids = [s["source_id"] for s in registry["sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs/data_sources.json"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Attempt listed public downloads; never log in or purchase access",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; retain the previous audit")
    report = audit_sources(read_registry(args.registry), args.cache, fetch=args.fetch)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
