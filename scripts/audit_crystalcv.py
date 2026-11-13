# FILE PURPOSE: Independently reconstruct acquired CrystalCV pixels and measurements.
# Pillow decoding, original ZIP/CSV bytes and external protocol establish coverage;
# this script imports no producer and never treats boxes/tracks as pixel masks.
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def timestamp(member):
    stem = Path(member).stem
    if stem.endswith("-1"):
        stem = stem[:-2]
    parts = stem.split(" ")
    try:
        if (
            len(parts) != 3
            or not parts[0].endswith("h")
            or not parts[1].endswith("m")
            or not parts[2].endswith("s")
        ):
            return None
        h, m, s = [int(p[:-1]) for p in parts]
        if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
            return None
        return h * 3600 + m * 60 + s
    except ValueError:
        return None


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def audit(index, protocol_path, output, *, shard_index=0, shard_count=1):
    import zipfile

    if output.exists():
        raise FileExistsError(output)
    if not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard_index < positive shard_count")
    manifest = json.loads((index / "manifest.json").read_text())
    protocol = json.loads(protocol_path.read_text())
    assert (
        manifest["schema"] == "crystalab_crystalcv_index_v1"
        and manifest["status"] == "complete"
    )
    assert manifest["protocol"] == protocol and manifest["protocol_sha256"] == sha(
        protocol_path
    )
    assert sha(index / "protocol.json") == sha(protocol_path)
    assert sha(index / "importer_snapshot.py") == manifest["source_sha256"]
    for relative, expected in manifest["file_sha256"].items():
        assert sha(index / relative) == expected, relative
    stored = {
        kind: rows(index / f"{kind}.jsonl")
        for kind in ("members", "images", "tables", "manual_measurements")
    }
    lookups = {
        kind: {(r["archive_id"], r["member"]): r for r in stored[kind]}
        for kind in ("members", "images", "tables")
    }
    for kind in lookups:
        assert len(lookups[kind]) == len(stored[kind]), kind
    counts, formats, roles, decoded_duplicates = Counter(), Counter(), Counter(), {}
    image_suffixes = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    cache = Path(manifest["cache_path"])
    image_number, selected_ids = 0, []
    for asset in protocol["archives"]:
        metadata_path = cache / asset["metadata_filename"]
        assert sha(metadata_path) == asset["metadata_sha256"]
        metadata = json.loads(metadata_path.read_text())
        assert (
            metadata["metadata"]["license"]["id"]
            == protocol["license_id"]
            == "cc-by-4.0"
        )
        published = next(v for v in metadata["files"] if v["key"] == asset["filename"])
        assert (
            published["size"] == asset["bytes"]
            and published["checksum"] == "md5:" + asset["md5"]
        )
        path = cache / asset["filename"]
        assert path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"]
        with zipfile.ZipFile(path) as archive:
            members = [v for v in archive.infolist() if not v.is_dir()]
            assert len({v.filename for v in members}) == len(members)
            for entry in members:
                key = (asset["archive_id"], entry.filename)
                payload = archive.read(entry)
                record = lookups["members"][key]
                assert record["bytes"] == len(payload) == entry.file_size
                assert record["sha256"] == hashlib.sha256(payload).hexdigest()
                counts["members"] += 1
                if entry.filename.lower().endswith(image_suffixes):
                    record = lookups["images"][key]
                    image_format, pixel_hash = (
                        record["decoded_format"],
                        record["native_BGR_sha256"],
                    )
                    if image_number % shard_count == shard_index:
                        with Image.open(io.BytesIO(payload)) as im:
                            image_format = im.format
                            # Match explicit displayed orientation; no resize/crop.
                            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                        pixel_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
                        assert record["decoded_format"] == image_format
                        assert (
                            record["image_height"],
                            record["image_width"],
                            3,
                        ) == bgr.shape
                        selected_ids.append(record["sample_id"])
                        if len(selected_ids) % 1000 == 0:
                            print(
                                json.dumps(
                                    dict(
                                        shard_index=shard_index,
                                        verified_native_images=len(selected_ids),
                                    )
                                ),
                                flush=True,
                            )
                    image_number += 1
                    assert (
                        record["native_BGR_sha256"]
                        == record["pixel_family_id"]
                        == pixel_hash
                    )
                    assert record["filename_elapsed_s"] == timestamp(entry.filename)
                    role = (
                        "published_annotated_display"
                        if "/Manual Verification Annotated/" in entry.filename
                        else "automatic_annotated_display"
                        if "annotated_frames" in PurePosixPath(entry.filename).parts
                        else "source_growth_image"
                        if any(
                            entry.filename.startswith(p)
                            for p in asset["raw_image_path_prefixes"]
                        )
                        else "source_reference_image"
                    )
                    assert record["role"] == role
                    assert record["source_id"] == protocol["source_id"]
                    assert (
                        record["sample_id"] == f"{asset['archive_id']}:{entry.filename}"
                    )
                    assert record["source_folder"] == str(
                        PurePosixPath(entry.filename).parent
                    )
                    assert (
                        record["sequence_id"]
                        == f"{asset['archive_id']}:{record['source_folder']}"
                    )
                    expected_material = next(
                        (
                            r["material"]
                            for r in asset["material_rules"]
                            if r["path_component"]
                            in PurePosixPath(entry.filename).parts
                        ),
                        None,
                    )
                    assert record["material"] == expected_material
                    assert (
                        record["dependency_family_id"]
                        == protocol["dependency_family_id"]
                        and record["split"] == "development"
                    )
                    for field in (
                        "segmentation_labels",
                        "event_labels",
                        "sensor_values",
                        "applied_commands",
                        "quality_labels",
                        "first_hour_membership",
                        "growth_run_start",
                    ):
                        assert record[field] is None
                    assert record["local_human_review"] is False
                    decoded_duplicates.setdefault(pixel_hash, []).append(
                        record["sample_id"]
                    )
                    formats[image_format] += 1
                    roles[role] += 1
                    counts["images"] += 1
                if entry.filename.lower().endswith(".csv"):
                    source = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
                    source_rows = list(source)
                    record = lookups["tables"][key]
                    assert (
                        record["rows"] == len(source_rows)
                        and record["columns"] == source.fieldnames
                    )
                    assert record["pixel_segmentation_ground_truth"] is False
                    counts["tables"] += 1
    # Every indexed member/image/table must have an original, not only a valid hash.
    for kind in ("members", "images", "tables"):
        assert counts[kind] == len(stored[kind]) == manifest["counts"][kind]
    expected_duplicates = {k: v for k, v in decoded_duplicates.items() if len(v) > 1}
    assert (
        json.loads((index / "duplicate_pixels.json").read_text()) == expected_duplicates
    )
    assert manifest["formats"] == dict(formats) and manifest["roles"] == dict(roles)
    assert manifest["source_folders"] == dict(
        Counter(r["sequence_id"] for r in stored["images"])
    )
    manual_asset = next(
        a
        for a in protocol["archives"]
        if a["archive_id"] == protocol["manual_archive_id"]
    )
    image_rows = [
        r for r in stored["images"] if r["archive_id"] == manual_asset["archive_id"]
    ]
    thermal = sorted(
        (
            r
            for r in image_rows
            if r["source_folder"] == protocol["thermal_image_directory"]
        ),
        key=lambda r: r["filename_elapsed_s"],
    )
    by_name = {
        Path(r["member"]).name: r
        for r in image_rows
        if r["source_folder"] == protocol["manual_image_directory"]
    }
    with zipfile.ZipFile(cache / manual_asset["filename"]) as archive:
        source_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(protocol["manual_table_member"]).decode("utf-8-sig")
                )
            )
        )
    assert (
        len(source_rows)
        == len(stored["manual_measurements"])
        == manifest["counts"]["manual_measurements"]
    )
    maximum_clock_difference = 0
    rectangle_errors, reported_error_roundoff = [], []
    for source, record in zip(source_rows, stored["manual_measurements"]):
        frame = int(source["Frame"])
        image = thermal[frame]
        name = Path(image["member"]).stem + "-1.jpg"
        annotation = by_name[name]
        assert source == record["original_values"]
        assert (
            record["source_frame_index"] == frame
            and record["raw_sample_id"] == image["sample_id"]
        )
        assert record["raw_native_BGR_sha256"] == image["native_BGR_sha256"]
        assert record["annotated_sample_id"] == annotation["sample_id"]
        assert record["manual_rectangle_area_px2"] == float(source["Manual Box FIJI"])
        assert record["source_reported_rectangle_area_px2"] == float(source["Rect"])
        assert (
            record["filename_elapsed_s"]
            == image["filename_elapsed_s"]
            == annotation["filename_elapsed_s"]
        )
        assert (
            record["nominal_tracking_elapsed_s"]
            == frame * protocol["thermal_nominal_frame_interval_s"]
        )
        assert (
            record["segmentation_polygon"] is None
            and record["local_human_review"] is False
        )
        manual_area, automatic_rectangle = (
            float(source["Manual Box FIJI"]),
            float(source["Rect"]),
        )
        rectangle_errors.append(abs(automatic_rectangle / manual_area - 1))
        reported_error_roundoff.append(
            abs(
                100 * (automatic_rectangle / manual_area - 1) - float(source["% Error"])
            )
        )
        maximum_clock_difference = max(
            maximum_clock_difference,
            abs(record["filename_elapsed_s"] - record["nominal_tracking_elapsed_s"]),
        )
    assert (
        json.loads((index / "reported_actions.json").read_text())
        == protocol["reported_actions"]
    )
    result = dict(
        passed=True,
        full_pixel_audit=shard_count == 1,
        shard_index=shard_index,
        shard_count=shard_count,
        independently_decoded_images=len(selected_ids),
        selected_image_ids_sha256=hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode()
        ).hexdigest(),
        auditor_sha256=sha(Path(__file__)),
        protocol_sha256=sha(protocol_path),
        index_manifest_sha256=sha(index / "manifest.json"),
        counts=dict(counts, manual_measurements=len(source_rows)),
        formats=dict(formats),
        roles=dict(roles),
        duplicate_pixel_groups=len(expected_duplicates),
        maximum_manual_filename_vs_nominal_clock_difference_s=maximum_clock_difference,
        published_automatic_rectangle_MAPE=float(np.mean(rectangle_errors)),
        published_percent_error_column_max_roundoff=float(max(reported_error_roundoff)),
        local_human_reviews_added=0,
        pitch_segmentation_labels_added=0,
        interpretation="All source bytes/tables/metadata checked; native pixels checked for this explicit shard. Full coverage requires every shard or shard_count=1. No KDP accuracy, action benefit or mask-label acceptance.",
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


def merge(index, protocol_path, inputs, output):
    """Accept complete, disjoint native-image coverage bound to identical inputs/code."""
    import zipfile

    if output.exists():
        raise FileExistsError(output)
    reports = [json.loads(path.read_text()) for path in inputs]
    if not reports:
        raise ValueError("Require audit shards")
    count = reports[0]["shard_count"]
    assert len(reports) == count and {r["shard_index"] for r in reports} == set(
        range(count)
    )
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((index / "manifest.json").read_text())
    identifiers = []
    for asset in protocol["archives"]:
        path = Path(manifest["cache_path"]) / asset["filename"]
        assert sha(path) == asset["sha256"]
        with zipfile.ZipFile(path) as archive:
            identifiers.extend(
                f"{asset['archive_id']}:{v.filename}"
                for v in archive.infolist()
                if not v.is_dir()
                and v.filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".tif", ".tiff")
                )
            )
    assert len(identifiers) == len(set(identifiers)) == manifest["counts"]["images"]
    for report in reports:
        assert report["passed"] and report["shard_count"] == count
        assert report["protocol_sha256"] == sha(protocol_path) and report[
            "index_manifest_sha256"
        ] == sha(index / "manifest.json")
        assert report["auditor_sha256"] == sha(Path(__file__))
        selected = identifiers[report["shard_index"] :: count]
        assert len(selected) == report["independently_decoded_images"]
        assert (
            hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()
            == report["selected_image_ids_sha256"]
        )
        assert (
            report["counts"] == reports[0]["counts"]
            and report["formats"] == reports[0]["formats"]
            and report["roles"] == reports[0]["roles"]
        )
    assert sum(r["independently_decoded_images"] for r in reports) == len(identifiers)
    result = dict(
        passed=True,
        full_pixel_audit=True,
        independently_decoded_images=len(identifiers),
        shard_count=count,
        protocol_sha256=sha(protocol_path),
        index_manifest_sha256=sha(index / "manifest.json"),
        auditor_sha256=sha(Path(__file__)),
        counts=reports[0]["counts"],
        formats=reports[0]["formats"],
        roles=reports[0]["roles"],
        duplicate_pixel_groups=reports[0]["duplicate_pixel_groups"],
        maximum_manual_filename_vs_nominal_clock_difference_s=reports[0][
            "maximum_manual_filename_vs_nominal_clock_difference_s"
        ],
        published_automatic_rectangle_MAPE=reports[0][
            "published_automatic_rectangle_MAPE"
        ],
        shards=[dict(path=str(p), sha256=sha(p)) for p in inputs],
        interpretation="Every original image independently decoded once across source-bound disjoint shards; no KDP accuracy, action benefit or accepted mask labels.",
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        merge(args.index, args.protocol, args.merge_inputs, args.output)
    else:
        audit(
            args.index,
            args.protocol,
            args.output,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
