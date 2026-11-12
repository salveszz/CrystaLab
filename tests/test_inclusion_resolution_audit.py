# FILE PURPOSE: Challenge the independent mask auditor with self-consistent false claims.
"""Hashes and aggregate counts cannot substitute for geometric match checks."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("inclusion_audit",ROOT/"scripts/audit_inclusion_resolution.py")
audit_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class InclusionAuditTests(unittest.TestCase):
    def test_assignment_finds_augmenting_path_instead_of_losing_a_match(self):
        # Greedy first-to-first matching loses the second target; a full match exists.
        edges=np.array([[True,True],[True,False]])
        self.assertEqual(audit_module.maximum_count(edges),2)
        self.assertEqual(audit_module.maximum_count(np.zeros((2,0),bool)),0)

    def test_rehashed_false_mask_and_retained_perfect_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            old_root=audit_module.ROOT
            audit_module.ROOT=root
            try:
                source=root/"source"
                evaluation=root/"evaluation"
                source.mkdir()
                (evaluation/"masks").mkdir(parents=True)
                cv2.imwrite(str(source/"image.png"),np.zeros((64,64,3),np.uint8))
                (source/"label.txt").write_text("0 .125 .125 .375 .125 .375 .375 .125 .375\n")
                data=dict(classes=["seed","secondary_crystal","inclusion_candidate"],records=[
                    dict(sample_id="source:0",family_key="family:0",split="validation",image_path="image.png",label_path="label.txt")])
                (source/"manifest.json").write_text(json.dumps(data))
                protocol=root/"protocol.json"
                protocol.write_text(json.dumps(dict(dataset="source",dataset_manifest_sha256=audit_module.sha(source/"manifest.json"))))
                mask=np.zeros((64,64),np.uint8)
                mask[8:25,8:25]=255
                path=evaluation/"masks/prediction.png"
                cv2.imwrite(str(path),mask)
                prediction=dict(sample_id="source:0",predictions=[dict(class_id=0,confidence=.9,mask_path="masks/prediction.png")],
                    matches=[dict(truth_index=0,prediction_index=0,iou=1.,dice=1.)])
                (evaluation/"predictions.jsonl").write_text(json.dumps(prediction)+"\n")
                report=dict(status="complete",split="validation",confidence=.25,iou_gate=.5,
                    dataset_audit=dict(manifest_sha256=audit_module.sha(source/"manifest.json")),
                    per_class={name:dict(tp=int(i==0),fp=0,fn=0,iou_sum=float(i==0),dice_sum=float(i==0)) for i,name in enumerate(data["classes"])},
                    artifact_sha256={"masks/prediction.png":audit_module.sha(path)})
                (evaluation/"manifest.json").write_text(json.dumps(report))
                audit_module.audit(evaluation,root/"accepted.json",protocol)
                # Move the prediction away; recompute its recorded hash while
                # falsely retaining the original perfect match and aggregates.
                cv2.imwrite(str(path),np.roll(mask,30,axis=1))
                report["artifact_sha256"]["masks/prediction.png"]=audit_module.sha(path)
                (evaluation/"manifest.json").write_text(json.dumps(report))
                with self.assertRaises(AssertionError):
                    audit_module.audit(evaluation,root/"rejected.json",protocol)
                self.assertFalse((root/"rejected.json").exists())
            finally:
                audit_module.ROOT=old_root
