# FILE PURPOSE: Index native public-video frames and prepare unreviewed annotation bundles.
# File/decoded-pixel hashes, presentation clocks, phase captions and whole-study
# grouping remain explicit. No human receipt or negative/event label is generated.
"""Conservative real-video development import with lazy native frame extraction."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import cv2
import numpy as np

from .annotations import CLASSES, _validated_annotation
from .training_data import safe_path, save_json
from .video import file_sha256

SCHEMA="crystalab_real_video_index_v1"


def validate_protocol(protocol):
    if protocol.get("schema")!="crystalab_real_video_import_v1" or protocol.get("split")!="train":
        raise ValueError("Declare a development-only real-video import")
    for key in ("source_id","material","source_url","license","attribution","dependency_family_id","split_semantics","clock_semantics","label_scope"):
        if not isinstance(protocol.get(key),str) or not protocol[key].strip():
            raise ValueError(f"Missing real-video provenance: {key}")
    clips=protocol.get("clips")
    if not isinstance(clips,list) or not clips:
        raise ValueError("Declare source clips")
    ids,files,hashes=set(),set(),set()
    for clip in clips:
        if clip["clip_id"] in ids or not clip["clip_id"].replace("_","").isalnum():
            raise ValueError("Unsafe or duplicate clip ID")
        ids.add(clip["clip_id"])
        if Path(clip["file"]).name!=clip["file"] or len(clip["sha256"])!=64:
            raise ValueError("Declare a safe source file and SHA256")
        if clip["file"] in files or clip["sha256"] in hashes:
            raise ValueError("Duplicate clip files or payloads cannot become new source observations")
        files.add(clip["file"])
        hashes.add(clip["sha256"])
        for key in ("frames","width","height","published_playback_speedup"):
            if type(clip[key]) is not int or clip[key]<1:
                raise ValueError(f"Invalid source clip {key}")
        frames=clip["annotation_frames"]
        if not frames or any(type(i) is not int or not 0<=i<clip["frames"] for i in frames) or len(set(frames))!=len(frames):
            raise ValueError("Invalid annotation frame selection")
        method=clip["proposal"]["method"]
        if method not in {"manual_outline_needed","dark_component_uncertain_draft"}:
            raise ValueError("Unknown unreviewed proposal method")
        if method=="dark_component_uncertain_draft":
            proposal=clip["proposal"]
            roi=proposal["search_roi_xyxy_fraction"]
            if len(roi)!=4 or not all(type(v) in (int,float) and np.isfinite(v) for v in roi) or not 0<=roi[0]<roi[2]<=1 or not 0<=roi[1]<roi[3]<=1:
                raise ValueError("Invalid proposal search region")
            if type(proposal["gray_below"]) is not int or not 0<proposal["gray_below"]<255 or type(proposal["minimum_component_pixels"]) is not int or proposal["minimum_component_pixels"]<3 or not 0<proposal["polygon_epsilon_px"]<=10:
                raise ValueError("Invalid proposal sampling settings")


def _timing(path,clip,ffprobe):
    command=[str(ffprobe),"-v","error","-select_streams","v:0","-show_entries",
        "stream=time_base,width,height,codec_name,avg_frame_rate:frame=pts,best_effort_timestamp","-of","json",str(path)]
    result=json.loads(subprocess.check_output(command,timeout=60))
    if len(result.get("streams",[]))!=1 or len(result.get("frames",[]))!=clip["frames"]:
        raise ValueError("Source stream/frame count differs from protocol")
    stream=result["streams"][0]
    if (stream["width"],stream["height"])!=(clip["width"],clip["height"]):
        raise ValueError("Source dimensions differ from protocol")
    time_base=Fraction(stream["time_base"])
    ticks=[frame["best_effort_timestamp"] for frame in result["frames"]]
    if time_base<=0 or any(type(v) is not int for v in ticks) or any(b<=a for a,b in zip(ticks,ticks[1:])):
        raise ValueError("Frame presentation timestamps must increase strictly")
    return stream,time_base,ticks


def scan_videos(cache,protocol,*,ffprobe="ffprobe",selected=None):
    """Decode sequentially; never infer acquisition clocks from frame numbers."""
    validate_protocol(protocol)
    if file_sha256(safe_path(cache,protocol["caption_asset"]))!=protocol["caption_sha256"]:
        raise ValueError("Source supplementary caption bytes changed")
    records=[]
    decoded={}
    for clip in protocol["clips"]:
        path=safe_path(cache,clip["file"])
        if file_sha256(path)!=clip["sha256"]:
            raise ValueError("Original video bytes differ from pinned source")
        stream,time_base,ticks=_timing(path,clip,ffprobe)
        capture=cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Cannot decode original source clip")
        try:
            for index,tick in enumerate(ticks):
                ok,image=capture.read()
                if not ok or image.shape!=(clip["height"],clip["width"],3) or image.dtype!=np.uint8:
                    raise ValueError("Decoded source pixels differ from declared frame contract")
                elapsed=float((tick-ticks[0])*time_base)
                sample_id=f"{protocol['source_id']}:{clip['clip_id']}:{index:06d}"
                records.append(dict(sample_id=sample_id,source_id=protocol["source_id"],
                    run_id=clip["clip_id"],family_id=protocol["dependency_family_id"],split="train",
                    split_semantics=protocol["split_semantics"],material=protocol["material"],evidence_kind="real",
                    source_url=protocol["source_url"],license=protocol["license"],attribution=protocol["attribution"],
                    original_video_file=clip["file"],original_video_sha256=clip["sha256"],frame_index=index,
                    image_width=clip["width"],image_height=clip["height"],
                    decoded_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    presentation_timestamp_ticks=tick,time_base=stream["time_base"],
                    source_presentation_time_s=float(tick*time_base),display_elapsed_s=elapsed,
                    published_playback_speedup=clip["published_playback_speedup"],
                    published_speedup_relative_elapsed_s=elapsed*clip["published_playback_speedup"],
                    acquisition_time_s=None,growth_run_start_s=None,first_hour_membership=None,
                    pixel_scale_m_per_px=None,caption_id=clip["caption_id"],caption_page=protocol["caption_page"],
                    phase_context=clip["phase_context"],review_status="unreviewed_real_video",
                    pixel_labels=None,event_labels=None,process_measurements=None,actuator_commands=None,optical_quality=None))
                if selected is not None and (clip["clip_id"],index) in selected:
                    decoded[sample_id]=image.copy()
            if capture.read()[0]:
                raise ValueError("Decoded video has undeclared additional frames")
        finally:
            capture.release()
    return records,decoded


def import_videos(cache,protocol_path,output,*,ffprobe="ffprobe"):
    cache,output=Path(cache).resolve(),Path(output).resolve()
    protocol=json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    output.mkdir(parents=True,exist_ok=False)
    (output/"source_snapshot").mkdir()
    shutil.copyfile(__file__,output/"source_snapshot/real_video.py")
    shutil.copyfile(protocol_path,output/"protocol.json")
    manifest=dict(schema=SCHEMA,status="preparing",cache_path=str(cache),protocol=protocol,
                  protocol_sha256=file_sha256(protocol_path))
    save_json(output/"manifest.json",manifest)
    try:
        records,_=scan_videos(cache,protocol,ffprobe=ffprobe)
        (output/"frames.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in records))
        if file_sha256(protocol_path)!=manifest["protocol_sha256"]:
            raise ValueError("Frozen protocol changed during import")
        manifest.update(status="complete",frames=len(records),index_sha256=file_sha256(output/"frames.jsonl"),
            environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
                ffprobe=subprocess.check_output([str(ffprobe),"-version"],text=True).splitlines()[0]),
            source_sha256=file_sha256(output/"source_snapshot/real_video.py"),scope="Native real-video development index; no labels accepted, no independent evaluation or physical control claim")
        save_json(output/"manifest.json",manifest)
        return dict(status="complete",frames=len(records),clips=len(protocol["clips"]))
    except Exception as error:
        manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
        save_json(output/"manifest.json",manifest)
        raise


def audit_video_index(index,*,protocol_path,ffprobe="ffprobe",selected=None):
    index=Path(index).resolve()
    manifest=json.loads((index/"manifest.json").read_text())
    if manifest.get("schema")!=SCHEMA or manifest.get("status")!="complete" or file_sha256(index/"frames.jsonl")!=manifest["index_sha256"]:
        raise ValueError("Incomplete or changed real-video index")
    expected=json.loads(Path(protocol_path).read_text())
    if manifest["protocol"]!=expected or manifest["protocol_sha256"]!=file_sha256(protocol_path):
        raise ValueError("Source protocol differs from independently supplied frozen file")
    if file_sha256(index/"protocol.json")!=manifest["protocol_sha256"] or file_sha256(index/"source_snapshot/real_video.py")!=manifest["source_sha256"]:
        raise ValueError("Import protocol or producer snapshot bytes changed")
    actual,pixels=scan_videos(Path(manifest["cache_path"]),manifest["protocol"],ffprobe=ffprobe,selected=selected)
    stored=[json.loads(v) for v in (index/"frames.jsonl").read_text().splitlines()]
    if stored!=actual or len(stored)!=manifest["frames"]:
        raise ValueError("Real-video index differs from source pixels, clocks or provenance")
    return manifest,stored,pixels


def _draft(image,proposal):
    if proposal["method"]=="manual_outline_needed":
        return [],"manual_outline_needed"
    height,width=image.shape[:2]
    x0,y0,x1,y1=np.rint(np.array(proposal["search_roi_xyxy_fraction"])*[width,height,width,height]).astype(int)
    mask=np.zeros((height,width),np.uint8)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask[y0:y1,x0:x1]=(gray[y0:y1,x0:x1]<proposal["gray_below"]).astype(np.uint8)*255
    count,components,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=proposal["minimum_component_pixels"]]
    if not candidates:
        return [],"no_supported_component"
    choice=max(candidates,key=lambda i:stats[i,cv2.CC_STAT_AREA])
    selected=(components==choice).astype(np.uint8)*255
    contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour=cv2.approxPolyDP(max(contours,key=cv2.contourArea),proposal["polygon_epsilon_px"],True).reshape(-1,2)
    x,y,w,h=cv2.boundingRect(selected)
    if (x==x0 and x0>0) or (y==y0 and y0>0) or (x+w==x1 and x1<width) or (y+h==y1 and y1<height):
        return [],"component_crosses_proposal_search_window"
    if len(contour)<3:
        return [],"degenerate_component_requires_manual_outline"
    return [dict(label="seed",points=contour.astype(float).tolist(),group_id=None,shape_type="polygon",
        flags=dict(proposed=True,uncertain=True,truncated=bool(x==0 or y==0 or x+w==width or y+h==height)),
        description=proposal["note"])],"uncertain_dark_component"


def prepare_video_bundle(index,output,*,protocol_path,ffprobe="ffprobe"):
    index,output=Path(index).resolve(),Path(output).resolve()
    expected=json.loads(Path(protocol_path).read_text())
    validate_protocol(expected)
    wanted={(clip["clip_id"],i) for clip in expected["clips"] for i in clip["annotation_frames"]}
    manifest,rows,pixels=audit_video_index(index,protocol_path=protocol_path,ffprobe=ffprobe,selected=wanted)
    protocol=manifest["protocol"]
    output.mkdir(parents=True,exist_ok=False)
    for folder in ("images","annotations","reviews"):
        (output/folder).mkdir()
    bundle=dict(schema="crystalab_annotation_bundle_v1",status="preparing",run_id=protocol["source_id"]+"_development",
        classes=CLASSES,source_index_sha256=manifest["index_sha256"],source_index_path=str(index),
        source_protocol=protocol,baseline_config=None,samples=[])
    save_json(output/"manifest.json",bundle)
    try:
        by_clip={clip["clip_id"]:clip for clip in protocol["clips"]}
        for row in rows:
            if (row["run_id"],row["frame_index"]) not in wanted:
                continue
            name=f"{row['run_id']}_{row['frame_index']:06d}"
            image_path=f"images/{name}.png";annotation_path=f"annotations/{name}.json"
            image=pixels[row["sample_id"]]
            if not cv2.imwrite(str(output/image_path),image):
                raise OSError("Cannot preserve native annotation pixels")
            proposal=by_clip[row["run_id"]]["proposal"]
            shapes,reason=_draft(image,proposal)
            document=dict(version="5.0.0",flags=dict(reviewed=False,uncertain=True),shapes=shapes,
                imagePath=f"../{image_path}",imageData=None,imageHeight=row["image_height"],imageWidth=row["image_width"],
                description="Unreviewed KDP development frame. Check instance identity, clipping, phase context and all visible boundaries. Empty draft is unknown, not a negative label.")
            sample=dict(source_record=row,image_path=image_path,image_sha256=file_sha256(output/image_path),
                annotation_path=annotation_path,proposal_reason=reason,proposal_method=proposal,
                review_path=f"reviews/{name}.json")
            try:
                _validated_annotation(output,sample,annotation=document,allow_uncertain=True)
            except ValueError:
                document["shapes"]=[]
                sample["proposal_reason"]="complex_outline_requires_manual_annotation"
            save_json(output/annotation_path,document)
            sample["initial_proposal_sha256"]=file_sha256(output/annotation_path)
            bundle["samples"].append(sample)
        bundle["status"]="ready_for_review"
        save_json(output/"manifest.json",bundle)
        return dict(status=bundle["status"],samples=len(bundle["samples"]),reviewed=0,
                    polygon_drafts=sum(bool(json.loads((output/s["annotation_path"]).read_text())["shapes"]) for s in bundle["samples"]))
    except Exception as error:
        bundle.update(status="failed",error=f"{type(error).__name__}: {error}")
        save_json(output/"manifest.json",bundle)
        raise
