# FILE PURPOSE: Plot the frozen inclusion experiment without changing labels or model settings.
# Display every validation inclusion example and the full per-class error counts.
"""Standalone study figures; validation visual inspection is after frozen fitting."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vision.training_data import polygon_masks
from vision.video import file_sha256


def plot(protocol_path, baseline, candidate, output):
    output.mkdir(exist_ok=False)
    protocol=json.loads(protocol_path.read_text())
    dataset=ROOT/protocol["dataset"]
    source=json.loads((dataset/"manifest.json").read_text())
    reports=[json.loads((p/"manifest.json").read_text()) for p in (baseline,candidate)]
    classes=source["classes"]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    for axis,cls in zip(axes,classes):
        for i,(report,color,label) in enumerate(zip(reports,("#315c8b","#bc622a"),("256 input","512 input"))):
            data=report["per_class"][cls]
            axis.bar(np.arange(3)+(i-.5)*.35,[data[k] for k in ("tp","fp","fn")],width=.33,color=color,label=label)
            for j,key in enumerate(("tp","fp","fn")):
                axis.annotate(str(data[key]),(j+(i-.5)*.35,data[key]),xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
        axis.set(xticks=range(3),xticklabels=("Matched","False positive","Missed"),title=cls.replace("_"," "),ylabel="Instances")
        axis.margins(y=.17)
        axis.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("44 synthetic validation images · same native masks and thresholds")
    fig.savefig(output/"class_counts.png",dpi=170)
    fig.savefig(output/"class_counts.pdf")
    plt.close(fig)

    predictions={r["sample_id"]:r for r in map(json.loads,(candidate/"predictions.jsonl").read_text().splitlines())}
    rows=[]
    for row in source["records"]:
        if row["split"]!="validation":
            continue
        image=cv2.imread(str(dataset/row["image_path"]))
        targets=[t for t in polygon_masks(dataset/row["label_path"],image.shape[:2]) if t["class_id"]==2]
        if targets:
            rows.append((row,image,targets))
    assert len(rows)==6,"This figure must show all six pre-existing validation inclusion images"
    fig,axes=plt.subplots(2,3,figsize=(9,6.8),layout="constrained")
    for axis,(row,image,targets) in zip(axes.flat,rows):
        # Fixed central crop is for readable display only; scoring uses full images.
        axis.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        for target in targets:
            axis.contour(target["mask"]>0,levels=[.5],colors=["#00cfff"],linewidths=1)
        predicted=[p for p in predictions[row["sample_id"]]["predictions"] if p["class_id"]==2]
        for p in predicted:
            mask=cv2.imread(str(candidate/p["mask_path"]),0)
            axis.contour(mask>0,levels=[.5],colors=["#fb36c0"],linewidths=.8)
        axis.set(xlim=(75,185),ylim=(185,75),title=f"{row['sample_id']} · {len(predicted)} predicted",xticks=[],yticks=[])
    fig.suptitle("All six inclusion validation examples · cyan target / magenta 512 prediction\nFixed display crop; all counts and scoring use complete images",fontsize=11)
    fig.savefig(output/"validation_inclusions.png",dpi=180)
    fig.savefig(output/"validation_inclusions.pdf")
    plt.close(fig)
    inputs=[protocol_path,baseline/"manifest.json",candidate/"manifest.json",candidate/"predictions.jsonl"]
    manifest=dict(plotter_sha256=file_sha256(__file__),
        input_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in inputs},
        output_sha256={p.name:file_sha256(p) for p in output.iterdir()},
        scope="Frozen synthetic validation comparison. Display crop is not an inference crop or post-result model change. No real KDP accuracy claim.")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/inclusion_resolution_experiment.json")
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plot(args.protocol.resolve(),args.baseline.resolve(),args.candidate.resolve(),args.output.resolve())
