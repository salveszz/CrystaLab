# FILE PURPOSE: Measure training-target scale and score fixed memorization diagnostics.
# Last checkpoints and selected training rows are explicit. No reserved images or
# holdout accuracy are used to decide whether a full resolution comparison runs.
"""Training-only inclusion learnability diagnostic with archived native masks."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.segmentation import load_run_metadata, local_yolo
from models.segmentation_eval import match_instances
from vision.training_data import audit_segmentation, polygon_masks, save_json, safe_path
from vision.video import file_sha256


def prepare(protocol_path, output):
    protocol=json.loads(protocol_path.read_text())
    dataset=ROOT/protocol["dataset"]
    assert file_sha256(dataset/"manifest.json")==protocol["dataset_manifest_sha256"]
    dataset_audit=audit_segmentation(dataset)
    source=json.loads((dataset/"manifest.json").read_text())
    train=[r for r in source["records"] if r["split"]=="train"]
    selected=train[:protocol["stage_1"]["overfit_samples"]]
    output.mkdir(exist_ok=False)
    target_rows=[]
    for row in train:
        image=cv2.imread(str(dataset/row["image_path"]))
        for i,target in enumerate(polygon_masks(dataset/row["label_path"],image.shape[:2])):
            x,y,w,h=cv2.boundingRect(target["mask"])
            target_rows.append(dict(sample_id=row["sample_id"],family_key=row["family_key"],
                selected_for_memorization=row in selected,class_id=target["class_id"],object_index=i,
                area_px=int(np.count_nonzero(target["mask"])),bbox_px=[x,y,w,h],
                image_shape=list(image.shape[:2]),
                approximate_stride4_mask_cells=float(np.count_nonzero(target["mask"])/16)))
    configs=[]
    stage=protocol["stage_1"]
    for size in stage["image_sizes"]:
        config=dict(schema="crystalab_yolov8_seg_train_v1",purpose="overfit_check",
                    description="Fixed training-only inclusion memorization; no generalization claim",
                    epochs=stage["epochs"],imgsz=size,batch=stage["batch"],threads=2,
                    seed=stage["seed"],device="cpu",learning_rate=0.001,
                    mask_ratio=1,overlap_mask=False,overfit_samples=stage["overfit_samples"],
                    warmup_epochs=stage["warmup_epochs"],lrf=stage["lrf"])
        path=output/f"overfit_{size}.json"
        save_json(path,config)
        configs.append(dict(path=str(path.relative_to(ROOT)),sha256=file_sha256(path)))
    save_json(output/"preparation.json",dict(schema="crystalab_inclusion_resolution_preparation_v1",
        protocol_sha256=file_sha256(protocol_path),dataset_audit=dataset_audit,
        dataset_manifest_sha256=file_sha256(dataset/"manifest.json"),selected_sample_ids=[r["sample_id"] for r in selected],
        target_rows=target_rows,configs=configs,preparer_sha256=file_sha256(__file__),
        scope="All target statistics are training-only; selected rows are reused for memorization scoring, not held-out evaluation"))


def score(protocol_path, preparation, run, output):
    protocol=json.loads(protocol_path.read_text())
    prepared=json.loads((preparation/"preparation.json").read_text())
    assert prepared["protocol_sha256"]==file_sha256(protocol_path)
    dataset=ROOT/protocol["dataset"]
    audit_segmentation(dataset)
    assert file_sha256(dataset/"manifest.json")==prepared["dataset_manifest_sha256"]
    manifest,_=load_run_metadata(run)
    assert manifest["config"]["purpose"]=="overfit_check"
    assert manifest["overfit_sample_ids"]==prepared["selected_sample_ids"]
    weights=safe_path(run,manifest["last_weights"])
    assert file_sha256(weights)==manifest["last_weights_sha256"]
    source=json.loads((dataset/"manifest.json").read_text())
    ids=set(prepared["selected_sample_ids"])
    rows=[r for r in source["records"] if r["sample_id"] in ids]
    assert len(rows)==len(ids) and all(r["split"]=="train" for r in rows)
    output.mkdir(exist_ok=False)
    (output/"masks").mkdir()
    YOLO,versions=local_yolo(output/"settings",manifest["runtime"])
    import torch
    torch.set_num_threads(2)
    model=YOLO(str(weights),task="segment")
    assert [model.names[i] for i in range(3)]==source["classes"]
    diagnostics=[]
    totals={i:dict(tp=0,fp=0,fn=0) for i in range(3)}
    for index,row in enumerate(rows):
        image=cv2.imread(str(dataset/row["image_path"]))
        truth=polygon_masks(dataset/row["label_path"],image.shape[:2])
        result=model.predict(image,device="cpu",imgsz=manifest["config"]["imgsz"],
                             conf=.25,iou=.7,max_det=100,retina_masks=True,verbose=False,save=False)[0]
        predictions=[]
        if result.masks is not None:
            masks=result.masks.data.cpu().numpy()
            assert masks.shape[1:]==image.shape[:2]
            for j,(mask,cls,confidence) in enumerate(zip(masks,result.boxes.cls.cpu().numpy(),result.boxes.conf.cpu().numpy())):
                binary=(mask>.5).astype(np.uint8)*255
                path=f"masks/{index:03d}_{j:03d}.png"
                assert cv2.imwrite(str(output/path),binary)
                predictions.append(dict(class_id=int(cls),confidence=float(confidence),mask=binary,mask_path=path,mask_sha256=file_sha256(output/path)))
        matches=match_instances(truth,predictions,.5)
        for cls in range(3):
            tp=sum(truth[m["truth_index"]]["class_id"]==cls for m in matches)
            totals[cls]["tp"]+=tp
            totals[cls]["fn"]+=sum(t["class_id"]==cls for t in truth)-tp
            totals[cls]["fp"]+=sum(p["class_id"]==cls for p in predictions)-tp
        diagnostics.append(dict(sample_id=row["sample_id"],image_path=row["image_path"],
            label_path=row["label_path"],image_sha256=file_sha256(dataset/row["image_path"]),
            label_sha256=file_sha256(dataset/row["label_path"]),matches=matches,
            predictions=[{k:v for k,v in p.items() if k!="mask"} for p in predictions]))
    save_json(output/"report.json",dict(status="complete",selected_sample_ids=prepared["selected_sample_ids"],
        run=str(run.relative_to(ROOT)),run_manifest_sha256=file_sha256(run/"manifest.json"),
        weights_sha256=file_sha256(weights),checkpoint="last",imgsz=manifest["config"]["imgsz"],
        totals=totals,diagnostics=diagnostics,packages=versions,
        protocol_sha256=file_sha256(protocol_path),scorer_sha256=file_sha256(__file__),
        scope="Memorization on selected training images; not validation or real accuracy"))
    print(json.dumps(dict(imgsz=manifest["config"]["imgsz"],totals=totals)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/inclusion_resolution_experiment.json")
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--preparation",type=Path)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.protocol.resolve(),args.output.resolve())
    else:
        score(args.protocol.resolve(),args.preparation.resolve(),args.run.resolve(),args.output.resolve())
