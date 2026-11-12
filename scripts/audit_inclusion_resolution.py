# FILE PURPOSE: Reconstruct inclusion-study image scores from native label/prediction bytes.
# A separate augmenting-path matcher checks maximum same-class match counts.
# This auditor does not call the producer's polygon or instance-matching helpers.
"""Independent mask/denominator audit for memorization and grouped evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth_masks(path, shape):
    result=[]
    for line in path.read_text().splitlines():
        values=line.split()
        cls=int(values[0])
        points=np.array(list(map(float,values[1:]))).reshape(-1,2)
        vertices=np.rint(points*np.array(shape[::-1])).astype(np.int32)
        mask=np.zeros(shape,np.uint8)
        cv2.fillPoly(mask,[vertices],255)
        result.append((cls,mask))
    return result


def maximum_count(edges):
    """Maximum bipartite cardinality, independent of the producer's assignment."""
    owners={}
    def augment(i,seen):
        for j in np.flatnonzero(edges[i]):
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):
                owners[j]=i
                return True
        return False
    return sum(augment(i,set()) for i in range(len(edges)))


def audit(evaluation, output, protocol_path):
    if output.exists():
        raise FileExistsError("Choose a fresh audit output")
    protocol=json.loads(protocol_path.read_text())
    dataset=ROOT/protocol["dataset"]
    assert sha(dataset/"manifest.json")==protocol["dataset_manifest_sha256"]
    source=json.loads((dataset/"manifest.json").read_text())
    by_id={r["sample_id"]:r for r in source["records"]}
    if (evaluation/"report.json").exists():
        report=json.loads((evaluation/"report.json").read_text())
        rows=report["diagnostics"]
        assert report["checkpoint"]=="last"
        run=ROOT/report["run"]
        training=json.loads((run/"manifest.json").read_text())
        assert sha(run/"manifest.json")==report["run_manifest_sha256"]
        assert sha(run/training["last_weights"])==report["weights_sha256"]==training["last_weights_sha256"]
        assert training["config"]["purpose"]=="overfit_check"
        expected=[r["sample_id"] for r in source["records"] if r["split"]=="train"][:protocol["stage_1"]["overfit_samples"]]
        assert report["selected_sample_ids"]==training["overfit_sample_ids"]==expected
        assert report["protocol_sha256"]==sha(protocol_path)
        phase="training_memorization"
    else:
        report=json.loads((evaluation/"manifest.json").read_text())
        rows=[json.loads(line) for line in (evaluation/"predictions.jsonl").read_text().splitlines()]
        assert report["status"]=="complete" and report["split"]=="validation"
        assert report["confidence"]==.25 and report["iou_gate"]==.5
        assert report["dataset_audit"]["manifest_sha256"]==protocol["dataset_manifest_sha256"]
        expected=[r["sample_id"] for r in source["records"] if r["split"]=="validation"]
        for name,digest in report["artifact_sha256"].items():
            assert sha(evaluation/name)==digest,name
        phase="grouped_synthetic_validation"
    assert [r["sample_id"] for r in rows]==expected
    totals={i:dict(tp=0,fp=0,fn=0,iou_sum=0.,dice_sum=0.) for i in range(3)}
    families={}
    hashes={}
    for row in rows:
        origin=by_id[row["sample_id"]]
        assert origin["split"]==("train" if phase=="training_memorization" else "validation")
        image=cv2.imread(str(dataset/origin["image_path"]))
        truth=truth_masks(dataset/origin["label_path"],image.shape[:2])
        predictions=[]
        for p in row["predictions"]:
            path=evaluation/p["mask_path"]
            mask=cv2.imread(str(path),0)
            assert mask is not None and mask.shape==image.shape[:2]
            assert set(np.unique(mask))<={0,255}
            assert 0<=p["class_id"]<3 and p["confidence"]>=.25
            if "mask_sha256" in p:
                assert sha(path)==p["mask_sha256"]
            hashes[str(path.relative_to(ROOT))]=sha(path)
            predictions.append((p["class_id"],mask))
        ious=np.zeros((len(truth),len(predictions)))
        dices=np.zeros_like(ious)
        for i,(tc,tm) in enumerate(truth):
            for j,(pc,pm) in enumerate(predictions):
                if tc!=pc:
                    continue
                intersection=np.count_nonzero((tm>0)&(pm>0))
                union=np.count_nonzero((tm>0)|(pm>0))
                ious[i,j]=intersection/union if union else 0
                size=np.count_nonzero(tm)+np.count_nonzero(pm)
                dices[i,j]=2*intersection/size if size else 0
        pairs=[(m["truth_index"],m["prediction_index"]) for m in row["matches"]]
        assert len({i for i,j in pairs})==len(pairs)==len({j for i,j in pairs})
        assert len(pairs)==maximum_count(ious>=.5)
        family=families.setdefault(origin["family_key"],{i:dict(tp=0,fp=0,fn=0) for i in range(3)})
        for match in row["matches"]:
            i,j=match["truth_index"],match["prediction_index"]
            assert ious[i,j]>=.5 and truth[i][0]==predictions[j][0]
            assert match["iou"]==ious[i,j] and match["dice"]==dices[i,j]
            totals[truth[i][0]]["iou_sum"]+=ious[i,j]
            totals[truth[i][0]]["dice_sum"]+=dices[i,j]
        for cls in range(3):
            tp=sum(truth[i][0]==cls for i,j in pairs)
            numbers=dict(tp=tp,fp=sum(pc==cls for pc,pm in predictions)-tp,fn=sum(tc==cls for tc,tm in truth)-tp)
            for key,value in numbers.items():
                totals[cls][key]+=value
                family[cls][key]+=value
    for cls in range(3):
        stored=report["totals"][str(cls)] if phase=="training_memorization" else report["per_class"][source["classes"][cls]]
        for key in ("tp","fp","fn"):
            assert stored[key]==totals[cls][key]
        if phase!="training_memorization":
            for key in ("iou_sum","dice_sum"):
                assert np.isclose(stored[key],totals[cls][key],rtol=0,atol=1e-12)
    result=dict(status="passed",phase=phase,images=len(rows),families=families,totals=totals,
        protocol_sha256=sha(protocol_path),auditor_sha256=sha(__file__),prediction_mask_sha256=hashes,
        limits="Independent native-mask IoUs, maximum match counts, sample membership and aggregates. Does not rerun neural inference or validate physical labels.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="passed",images=len(rows),totals=totals)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/inclusion_resolution_experiment.json")
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.evaluation.resolve(),args.output.resolve(),args.protocol.resolve())
