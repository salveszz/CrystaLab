# FILE PURPOSE: Stream version-pinned public CrystalCV archives with bounded RAM.
# License metadata comes from preserved snapshots; downloads never execute code.
# Existing archives and partial attempts are not overwritten.
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def acquire(protocol_path, cache, output, *, fetch=False):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "crystalab_crystalcv_protocol_v1" or not protocol.get(
        "archives"
    ):
        raise ValueError("Require a populated, version-pinned CrystalCV protocol")
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for asset in protocol["archives"]:
        record = dict(
            filename=asset["filename"], url=asset["download_url"], status="unverified"
        )
        started = time.monotonic()
        try:
            for key in ("filename", "metadata_filename"):
                if Path(asset[key]).name != asset[key] or "\\" in asset[key]:
                    raise ValueError("Source files must use plain filenames")
            metadata_path = cache / asset["metadata_filename"]
            if not metadata_path.exists():
                snapshot = ROOT / asset["metadata_snapshot"]
                if hashes(snapshot)[0] != asset["metadata_sha256"]:
                    raise ValueError("Versioned license snapshot changed")
                with metadata_path.open("xb") as target, snapshot.open("rb") as source:
                    shutil.copyfileobj(source, target)
            if hashes(metadata_path)[0] != asset["metadata_sha256"]:
                raise ValueError("Cached metadata changed")
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata["metadata"]["license"]["id"] != protocol["license_id"]
                or protocol["license_id"] != "cc-by-4.0"
            ):
                raise ValueError(
                    "Versioned metadata must supply the declared CC BY 4.0 license"
                )
            published = next(
                v for v in metadata["files"] if v["key"] == asset["filename"]
            )
            if (
                published["size"] != asset["bytes"]
                or published["checksum"] != "md5:" + asset["md5"]
            ):
                raise ValueError("Protocol differs from published file identity")
            path = cache / asset["filename"]
            if path.exists():
                record["status"] = "cached"
            elif not fetch:
                record["status"] = "not_cached"
                records.append(record)
                continue
            else:
                partial = path.with_suffix(path.suffix + ".part")
                if partial.exists():
                    raise FileExistsError(
                        "Preserved partial download exists; use a fresh cache for a new attempt"
                    )
                request = urllib.request.Request(
                    asset["download_url"],
                    headers={"User-Agent": "CrystaLab public-data acquisition"},
                )
                received = 0
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    partial.open("xb") as stream,
                ):
                    record.update(
                        http_status=response.status,
                        content_type=response.headers.get("Content-Type"),
                        resolved_url=response.url,
                    )
                    while block := response.read(
                        min(4 * 1024 * 1024, asset["bytes"] - received + 1)
                    ):
                        received += len(block)
                        if received > asset["bytes"]:
                            raise ValueError("Download exceeded pinned byte count")
                        stream.write(block)
                if received != asset["bytes"] or hashes(partial) != (
                    asset["sha256"],
                    asset["md5"],
                ):
                    raise ValueError("Downloaded archive differs from pinned source")
                partial.rename(path)
                record["status"] = "downloaded"
            if path.stat().st_size != asset["bytes"] or hashes(path) != (
                asset["sha256"],
                asset["md5"],
            ):
                raise ValueError("Cached archive differs from pinned source")
            record.update(
                verified=True,
                bytes=path.stat().st_size,
                sha256=asset["sha256"],
                md5=asset["md5"],
            )
        except Exception as error:
            record.update(
                status="failed",
                verified=False,
                error=f"{type(error).__name__}: {error}",
            )
        record["elapsed_s"] = time.monotonic() - started
        records.append(record)
    report = dict(
        source_id=protocol["source_id"],
        fetch_requested=fetch,
        assets=records,
        all_verified=all(r.get("verified", False) for r in records),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/crystalcv_vision_import.json"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    report = acquire(args.protocol, args.cache, args.output, fetch=args.fetch)
    print(json.dumps(report, indent=2))
    if not report["all_verified"]:
        raise SystemExit(2)
