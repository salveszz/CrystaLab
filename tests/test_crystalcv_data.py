# FILE PURPOSE: Test source integrity, native TIFF/JPEG semantics and published
# rectangle/clock binding on small generated fixtures; these are not real labels.
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vision.crystalcv_data import filename_time, import_data, load_image
from audit_crystalcv import audit, merge
from acquire_crystalcv import acquire


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, unsafe=False, repeated_manual=False, derivatives=False):
    cache = tmp_path / "cache"
    cache.mkdir()
    payload = io.BytesIO()
    pixels = np.zeros((40, 60, 3), dtype=np.uint8)
    pixels[15:25, 20:30] = [180, 20, 10]
    Image.fromarray(pixels).save(payload, format="JPEG")
    archive = cache / "source.zip"
    times = ["00h 00m 00s", "00h 10m 02s", "00h 20m 01s"]
    raw = "CrystalCV/Source_datasets/2_ThermalCycle"
    annotation = "CrystalCV/Figures/Manual Verification Annotated"
    rows = [
        ["Frame", "CCV Reported", "Rect", "Manual Box FIJI", "Error", "% Error"],
        [0, 80, 90, 100, -10, -10],
        [0 if repeated_manual else 2, 85, 95, 105, -10, -9.523809524],
    ]
    table = io.StringIO()
    csv.writer(table).writerows(rows)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in times:
            z.writestr(f"{raw}/{name}.jpg", payload.getvalue())
        # Like the public payload: TIFF bytes under an annotation .jpg name.
        tiff = io.BytesIO()
        Image.fromarray(pixels).save(tiff, format="TIFF")
        for index in (0, 2):
            z.writestr(f"{annotation}/{times[index]}-1.jpg", tiff.getvalue())
        z.writestr("CrystalCV/Figures/SI_ACC_TEST.csv", table.getvalue())
        z.writestr(
            "CrystalCV/Pre_Processed/all_tracked/tracking.csv",
            "frame,time,object_id,area\n0,0,1,80\n1,10,1,82\n",
        )
        if unsafe:
            z.writestr("../escape.jpg", payload.getvalue())
        if derivatives:
            z.writestr(f"{raw}/output/annotated_frames/0.jpg", payload.getvalue())
            z.writestr(f"{raw}/process.csv", "Time (h),Infusion Rate (mL/h)\n0,2.5\n")
    metadata = dict(
        metadata=dict(license=dict(id="cc-by-4.0")),
        files=[
            dict(
                key=archive.name,
                size=archive.stat().st_size,
                checksum="md5:" + hashlib.md5(archive.read_bytes()).hexdigest(),
            )
        ],
    )
    metadata_path = cache / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    protocol = dict(
        schema="crystalab_crystalcv_protocol_v1",
        source_id="FIXTURE",
        dependency_family_id="fixture_study",
        split="development",
        license_id="cc-by-4.0",
        pitch_class_mapping=None,
        attribution="Generated fixture, not actual author data",
        maximum_member_bytes=1_000_000,
        maximum_expanded_bytes=10_000_000,
        maximum_image_pixels=10_000,
        archives=[
            dict(
                archive_id="software",
                filename=archive.name,
                bytes=archive.stat().st_size,
                sha256=sha(archive),
                md5=hashlib.md5(archive.read_bytes()).hexdigest(),
                metadata_filename=metadata_path.name,
                metadata_sha256=sha(metadata_path),
                raw_image_path_prefixes=[raw + "/"],
                table_roles={
                    f"{raw}/process.csv": "reported_control_and_vision_derived_series"
                },
                material_rules=[
                    dict(path_component="2_ThermalCycle", material="MAPbBr3")
                ],
            )
        ],
        manual_archive_id="software",
        manual_table_member="CrystalCV/Figures/SI_ACC_TEST.csv",
        thermal_image_directory=raw,
        manual_image_directory=annotation,
        author_scale_px2_per_mm2=79.26,
        thermal_nominal_frame_interval_s=600,
        reported_actions=[
            dict(
                evidence_kind="fixture", sensor_measurements=None, physical_command=None
            )
        ],
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return cache, protocol_path, protocol


def test_import_binds_source_manual_rectangle_values_without_inventing_masks_or_clocks(
    tmp_path,
):
    cache, protocol_path, _ = fixture(tmp_path)
    output = tmp_path / "index"
    result = import_data(cache, protocol_path, output)
    assert result["status"] == "complete"
    assert result["counts"] == dict(
        members=7, images=5, tables=2, manual_measurements=2
    )
    assert result["formats"] == dict(JPEG=3, TIFF=2)
    assert result["roles"] == dict(source_growth_image=3, published_annotated_display=2)
    assert result["duplicate_pixel_groups"] == 2
    rows = [
        json.loads(line)
        for line in (output / "manual_measurements.jsonl").read_text().splitlines()
    ]
    assert rows[1]["filename_elapsed_s"] == 1201
    assert rows[1]["nominal_tracking_elapsed_s"] == 1200
    assert rows[0]["manual_rectangle_area_px2"] == 100
    assert rows[0]["segmentation_polygon"] is None
    images = [
        json.loads(line) for line in (output / "images.jsonl").read_text().splitlines()
    ]
    for row in images:
        image, checked = load_image(output, row["sample_id"])
        assert image.shape == (40, 60, 3)
        assert checked == row
        for key in (
            "segmentation_labels",
            "event_labels",
            "sensor_values",
            "applied_commands",
            "first_hour_membership",
            "growth_run_start",
        ):
            assert row[key] is None
    with pytest.raises(FileExistsError):
        import_data(cache, protocol_path, output)


@pytest.mark.parametrize(
    "change", ["source_bytes", "license", "pixels", "mapping", "split"]
)
def test_source_integrity_license_and_scope_gates(tmp_path, change):
    cache, protocol_path, protocol = fixture(tmp_path)
    if change == "source_bytes":
        with (cache / "source.zip").open("ab") as stream:
            stream.write(b"changed")
    elif change == "license":
        path = cache / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["metadata"]["license"]["id"] = "unknown"
        path.write_text(json.dumps(metadata))
        protocol["archives"][0]["metadata_sha256"] = sha(path)
    elif change == "pixels":
        protocol["maximum_image_pixels"] = 10
    elif change == "mapping":
        protocol["pitch_class_mapping"] = {"crystal": "seed"}
    elif change == "split":
        protocol["split"] = "test"
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        import_data(cache, protocol_path, tmp_path / "index")
    assert (
        json.loads((tmp_path / "index/manifest.json").read_text())["status"] == "failed"
    )


@pytest.mark.parametrize("unsafe,repeated_manual", [(True, False), (False, True)])
def test_unsafe_paths_and_ambiguous_manual_binding_are_rejected(
    tmp_path, unsafe, repeated_manual
):
    cache, protocol_path, _ = fixture(
        tmp_path, unsafe=unsafe, repeated_manual=repeated_manual
    )
    with pytest.raises(ValueError):
        import_data(cache, protocol_path, tmp_path / "index")


def test_rehashed_pixel_forgery_is_rejected_against_source_bytes(tmp_path):
    cache, protocol_path, _ = fixture(tmp_path)
    output = tmp_path / "index"
    manifest = import_data(cache, protocol_path, output)
    path = output / "images.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["native_BGR_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["file_sha256"][path.name] = sha(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pixels"):
        load_image(output, rows[0]["sample_id"])


def test_independent_pixel_and_manual_reconstruction_rejects_rehashed_measurement_edit(
    tmp_path, capsys
):
    cache, protocol_path, _ = fixture(tmp_path)
    index = tmp_path / "index"
    manifest = import_data(cache, protocol_path, index)
    result = audit(index, protocol_path, tmp_path / "audit.json")
    assert (
        result["passed"]
        and result["maximum_manual_filename_vs_nominal_clock_difference_s"] == 1
    )
    path = index / "manual_measurements.jsonl"
    data = [json.loads(line) for line in path.read_text().splitlines()]
    data[0]["manual_rectangle_area_px2"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    manifest["file_sha256"][path.name] = sha(path)
    (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit(index, protocol_path, tmp_path / "edited_audit.json")
    capsys.readouterr()


@pytest.mark.parametrize(
    "name,seconds",
    [
        ("00h 10m 02s.jpg", 602),
        ("27h 00m 01s-1.jpg", 97201),
        ("100h 59m 59s.jpg", 363599),
        ("bad.jpg", None),
        ("00h 90m 00s.jpg", None),
    ],
)
def test_filename_clock_preserves_seconds_and_refuses_invalid_fields(name, seconds):
    assert filename_time(name) == seconds


def test_streaming_acquisition_checks_cached_bytes_without_network_or_overwrite(
    tmp_path,
):
    cache, path, protocol = fixture(tmp_path)
    protocol["archives"][0]["download_url"] = "https://example.invalid/source.zip"
    path.write_text(json.dumps(protocol))
    report = acquire(path, cache, tmp_path / "acquisition.json")
    assert report["all_verified"] and report["assets"][0]["status"] == "cached"
    with (cache / "source.zip").open("ab") as stream:
        stream.write(b"modified")
    report = acquire(path, cache, tmp_path / "changed.json")
    assert not report["all_verified"] and report["assets"][0]["status"] == "failed"
    assert (cache / "source.zip").read_bytes().endswith(b"modified")


def test_streaming_acquisition_refuses_oversized_response_without_publishing_it(
    tmp_path, monkeypatch
):
    cache, path, protocol = fixture(tmp_path)
    protocol["archives"][0]["download_url"] = "https://example.invalid/source.zip"
    path.write_text(json.dumps(protocol))
    original = cache / "source.zip"
    too_much = original.read_bytes() + b"extra"
    original.unlink()

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/zip"}
        url = "https://example.invalid/source.zip"

    monkeypatch.setattr(
        "acquire_crystalcv.urllib.request.urlopen", lambda *a, **kw: Response(too_much)
    )
    report = acquire(path, cache, tmp_path / "oversize.json", fetch=True)
    assert not report["all_verified"] and "exceeded" in report["assets"][0]["error"]
    assert not original.exists() and original.with_suffix(".zip.part").exists()


def test_streaming_acquisition_preserves_existing_partial_attempt(tmp_path):
    cache, path, protocol = fixture(tmp_path)
    protocol["archives"][0]["download_url"] = "https://example.invalid/source.zip"
    path.write_text(json.dumps(protocol))
    (cache / "source.zip").unlink()
    partial = cache / "source.zip.part"
    partial.write_bytes(b"earlier partial attempt")
    report = acquire(path, cache, tmp_path / "partial.json", fetch=True)
    assert (
        not report["all_verified"]
        and partial.read_bytes() == b"earlier partial attempt"
    )


def test_automatic_annotation_pixels_and_reported_controls_keep_their_source_roles(
    tmp_path, capsys
):
    cache, protocol_path, _ = fixture(tmp_path, derivatives=True)
    index = tmp_path / "index"
    result = import_data(cache, protocol_path, index)
    assert result["roles"]["automatic_annotated_display"] == 1
    tables = [
        json.loads(line) for line in (index / "tables.jsonl").read_text().splitlines()
    ]
    process = next(row for row in tables if row["member"].endswith("process.csv"))
    assert process["evidence_kind"] == "reported_control_and_vision_derived_series"
    assert process["pixel_segmentation_ground_truth"] is False
    assert audit(index, protocol_path, tmp_path / "audit.json")["passed"]
    capsys.readouterr()


def test_sharded_audit_requires_complete_disjoint_source_bound_pixel_coverage(
    tmp_path, capsys
):
    cache, protocol, _ = fixture(tmp_path)
    index = tmp_path / "index"
    import_data(cache, protocol, index)
    paths = [tmp_path / f"shard{i}.json" for i in range(2)]
    reports = [
        audit(index, protocol, path, shard_index=i, shard_count=2)
        for i, path in enumerate(paths)
    ]
    assert [r["independently_decoded_images"] for r in reports] == [3, 2]
    assert all(not r["full_pixel_audit"] for r in reports)
    assert merge(index, protocol, paths, tmp_path / "complete.json")["full_pixel_audit"]
    with pytest.raises(AssertionError):
        merge(index, protocol, paths[:1], tmp_path / "missing.json")
    with pytest.raises(AssertionError):
        merge(index, protocol, [paths[0], paths[0]], tmp_path / "duplicated.json")
    altered = reports[0]
    altered["selected_image_ids_sha256"] = "0" * 64
    paths[0].write_text(json.dumps(altered))
    with pytest.raises(AssertionError):
        merge(index, protocol, paths, tmp_path / "forged.json")
    capsys.readouterr()
