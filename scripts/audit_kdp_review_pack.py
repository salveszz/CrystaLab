# FILE PURPOSE: Independently verify the KDP frame index and its unreviewed native review pack.
# Source registry, external protocol, decoded pixels and container clocks are
# checked without importing the new index/bundle producer.
"""No inference, annotation review or accepted training labels are generated."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text())


def audit(index,bundle,protocol_path,cross_decoder,output):
    if output.exists():
        raise FileExistsError("Choose an unused audit output")
    protocol=read(protocol_path)
    source=next(s for s in read(ROOT/"configs/vision_sources.json")["sources"] if s["source_id"]==protocol["source_id"])
    assert source["license"]==protocol["license"] and source["url"]==protocol["source_url"]
    assets={a["name"]:a for a in source["assets"]}
    manifest=read(index/"manifest.json")
    assert manifest["status"]=="complete" and manifest["protocol"]==protocol
    assert manifest["protocol_sha256"]==sha(protocol_path)==sha(index/"protocol.json")
    assert manifest["source_sha256"]==sha(index/"source_snapshot/real_video.py")
    assert sha(index/"frames.jsonl")==manifest["index_sha256"]
    records=[json.loads(line) for line in (index/"frames.jsonl").read_text().splitlines()]
    cache=Path(manifest["cache_path"])
    assert sha(cache/protocol["caption_asset"])==protocol["caption_sha256"]==assets[protocol["caption_asset"]]["sha256"]
    assert len(records)==sum(c["frames"] for c in protocol["clips"])==manifest["frames"]
    assert len({r["sample_id"] for r in records})==len(records)
    by_id={r["sample_id"]:r for r in records}
    count=0
    clips=[]
    for clip in protocol["clips"]:
        path=cache/clip["file"]
        assert sha(path)==clip["sha256"]==assets[clip["file"]]["sha256"]
        assert clip["published_playback_speedup"]==assets[clip["file"]]["published_playback_speedup"]
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=time_base:frame=pts","-of","json",str(path)],timeout=60))
        timestamps=[f["pts"] for f in probe["frames"]]
        assert len(timestamps)==clip["frames"] and all(b>a for a,b in zip(timestamps,timestamps[1:]))
        time_base=Fraction(probe["streams"][0]["time_base"])
        rows=[r for r in records if r["run_id"]==clip["clip_id"]]
        assert [r["frame_index"] for r in rows]==list(range(clip["frames"]))
        capture=cv2.VideoCapture(str(path))
        assert capture.isOpened()
        try:
            for row,tick in zip(rows,timestamps):
                ok,pixels=capture.read()
                assert ok and pixels.shape==(clip["height"],clip["width"],3)
                assert hashlib.sha256(pixels.tobytes()).hexdigest()==row["decoded_bgr_sha256"]
                assert row["presentation_timestamp_ticks"]==tick
                assert Fraction(row["time_base"])==time_base
                relative=(tick-timestamps[0])*time_base
                assert math.isclose(row["display_elapsed_s"],float(relative),rel_tol=0,abs_tol=1e-12)
                assert math.isclose(row["published_speedup_relative_elapsed_s"],float(relative*clip["published_playback_speedup"]),rel_tol=0,abs_tol=1e-12)
                assert row["source_presentation_time_s"]==float(tick*time_base)
                assert row["original_video_sha256"]==clip["sha256"] and row["original_video_file"]==clip["file"]
                assert row["phase_context"]==clip["phase_context"] and row["caption_id"]==clip["caption_id"]
                assert row["family_id"]==protocol["dependency_family_id"] and row["split"]=="train"
                assert row["license"]==source["license"] and row["material"]==protocol["material"]
                for key in ("acquisition_time_s","growth_run_start_s","first_hour_membership","pixel_scale_m_per_px","pixel_labels","event_labels","process_measurements","actuator_commands","optical_quality"):
                    assert row[key] is None,key
                count+=1
            assert not capture.read()[0]
        finally:
            capture.release()
        clips.append(dict(clip_id=clip["clip_id"],frames=len(rows),display_end_s=rows[-1]["display_elapsed_s"],
            published_speedup_relative_end_s=rows[-1]["published_speedup_relative_elapsed_s"],phase_context=clip["phase_context"]))
    pack=read(bundle/"manifest.json")
    assert pack["status"]=="ready_for_review" and pack["source_protocol"]==protocol
    assert pack["source_index_sha256"]==manifest["index_sha256"]
    assert [(s["source_record"]["run_id"],s["source_record"]["frame_index"]) for s in pack["samples"]]==[(c["clip_id"],i) for c in protocol["clips"] for i in c["annotation_frames"]]
    assert not list((bundle/"reviews").glob("*.json"))
    drafts=0
    for sample in pack["samples"]:
        record=sample["source_record"]
        assert record==by_id[record["sample_id"]]
        image=cv2.imread(str(bundle/sample["image_path"]))
        assert image.shape==(record["image_height"],record["image_width"],3)
        assert hashlib.sha256(image.tobytes()).hexdigest()==record["decoded_bgr_sha256"]
        assert sha(bundle/sample["image_path"])==sample["image_sha256"]
        assert sha(bundle/sample["annotation_path"])==sample["initial_proposal_sha256"]
        annotation=read(bundle/sample["annotation_path"])
        assert annotation["flags"]["uncertain"] and not annotation["flags"]["reviewed"]
        assert all(s["flags"]["uncertain"] and s["flags"]["proposed"] for s in annotation["shapes"])
        drafts+=bool(annotation["shapes"])
    # Execute the second decoder here, rather than trusting an external claim.
    cross_checks=[]
    for clip in protocol["clips"]:
        selected=clip["annotation_frames"]
        expression="+".join(f"eq(n\\,{i})" for i in selected)
        payload=subprocess.check_output(["ffmpeg","-v","error","-nostdin","-threads","1",
            "-i",str(cache/clip["file"]),"-vf","select="+expression,"-fps_mode","passthrough",
            "-f","rawvideo","-pix_fmt","bgr24","pipe:1"],timeout=60)
        size=clip["width"]*clip["height"]*3
        assert len(payload)==size*len(selected)
        for j,frame_index in enumerate(selected):
            sample=next(s for s in pack["samples"] if s["source_record"]["run_id"]==clip["clip_id"] and s["source_record"]["frame_index"]==frame_index)
            decoded=np.frombuffer(payload[j*size:(j+1)*size],np.uint8).reshape(clip["height"],clip["width"],3)
            image=cv2.imread(str(bundle/sample["image_path"]))
            difference=np.abs(decoded.astype(np.int16)-image.astype(np.int16))
            cross_checks.append(dict(clip_id=clip["clip_id"],frame_index=frame_index,
                maximum_channel_difference=int(difference.max()),mean_channel_difference=float(difference.mean()),
                ffmpeg_decoded_bgr_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                indexed_bgr_sha256=sample["source_record"]["decoded_bgr_sha256"]))
    if cross_decoder.exists():
        cross=read(cross_decoder)
        assert cross["checks"]==cross_checks
    else:
        cross=dict(checks=cross_checks,ffmpeg=subprocess.check_output(["ffmpeg","-version"],text=True).splitlines()[0],
                   scope="Independent CLI decoder versus native annotation PNGs; no label or physical timing validation")
        cross_decoder.write_text(json.dumps(cross,indent=2)+"\n")
    assert len(cross["checks"])==len(pack["samples"])
    for row in cross["checks"]:
        sample=next(s for s in pack["samples"] if s["source_record"]["run_id"]==row["clip_id"] and s["source_record"]["frame_index"]==row["frame_index"])
        assert row["maximum_channel_difference"]==0 and row["mean_channel_difference"]==0
        assert row["ffmpeg_decoded_bgr_sha256"]==row["indexed_bgr_sha256"]==sample["source_record"]["decoded_bgr_sha256"]
    result=dict(status="passed",frames_independently_checked=count,clips=clips,native_annotation_images=len(pack["samples"]),
        uncertain_polygon_drafts=drafts,unknown_empty_sheets=len(pack["samples"])-drafts,actual_reviews=0,
        independent_cli_decoder_images=len(cross["checks"]),dependency_families=1,accepted_pixel_or_event_labels=0,
        protocol_sha256=sha(protocol_path),index_manifest_sha256=sha(index/"manifest.json"),bundle_manifest_sha256=sha(bundle/"manifest.json"),
        auditor_sha256=sha(__file__),limits="All frame pixels and PTS reconstructed independently from original media with OpenCV/ffprobe; six selected PNGs also match an independent FFmpeg CLI decode. No physical acquisition clock, empirical mask accuracy, annotation review or control outcome verified.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed",frames=count,native_annotation_images=len(pack["samples"]),reviews=0)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/kdp_video_import.json")
    parser.add_argument("--cross-decoder",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.index.resolve(),args.bundle.resolve(),args.protocol.resolve(),args.cross_decoder.resolve(),args.output.resolve())
