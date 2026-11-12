# FILE PURPOSE: Verify native-video provenance, clock separation and guarded draft preparation.
# All generated pixels and metadata below are artificial test fixtures, not real labels.
"""Rehashed records must still match original pixels and the external frozen protocol."""

import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from dashboard.review import ReviewStore
from vision.annotations import export_reviewed,preview_bundle
from vision.real_video import import_videos,audit_video_index,prepare_video_bundle,scan_videos,validate_protocol
from vision.video import file_sha256


class RealVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cache=self.root/"cache"
        self.cache.mkdir()
        (self.cache/"caption.txt").write_text("AUTOMATED FIXTURE ONLY")
        path=self.cache/"fixture.avi"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"FFV1"),10,(64,64))
        self.assertTrue(writer.isOpened())
        self.images=[]
        for i in range(4):
            image=np.full((64,64,3),220,np.uint8)
            image[8:48,20+i:42+i]=15
            writer.write(image)
            self.images.append(image)
        writer.release()
        self.protocol=copy.deepcopy(json.loads((ROOT/"configs/kdp_video_import.json").read_text()))
        self.protocol.update(source_id="KDP-FIXTURE-ONLY",material="artificial_fixture",
            source_url="https://example.invalid/fixture",license="TEST FIXTURE ONLY",
            attribution="AUTOMATED FIXTURE ONLY; not experimental data",dependency_family_id="fixture-family",
            caption_asset="caption.txt",caption_sha256=file_sha256(self.cache/"caption.txt"))
        clip=self.protocol["clips"][0]
        clip.update(clip_id="FIXTURE",file="fixture.avi",sha256=file_sha256(path),frames=4,width=64,height=64,
                    phase_context="Artificial fixture; no actual KDP",annotation_frames=[0,3])
        clip["proposal"]["search_roi_xyxy_fraction"]=[0,0,1,1]
        self.protocol["clips"]=[clip]
        self.protocol_path=self.root/"protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.index=self.root/"index"
        import_videos(self.cache,self.protocol_path,self.index)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_pixels_and_uncertain_drafts_cannot_become_accepted_truth(self):
        bundle=self.root/"bundle"
        result=prepare_video_bundle(self.index,bundle,protocol_path=self.protocol_path)
        self.assertEqual(result["samples"],2)
        self.assertEqual(result["reviewed"],0)
        store=ReviewStore(bundle,read_only=True)
        self.assertEqual(store.summary()["reviewed"],0)
        manifest=json.loads((bundle/"manifest.json").read_text())
        for i,sample in enumerate(manifest["samples"]):
            pixels=cv2.imread(str(bundle/sample["image_path"]))
            np.testing.assert_array_equal(pixels,self.images[(0,3)[i]])
            document=store.get(i)["annotation"]
            self.assertTrue(document["flags"]["uncertain"])
            self.assertTrue(all(s["flags"]["uncertain"] for s in document["shapes"]))
            self.assertIsNone(sample["source_record"]["event_labels"])
        with self.assertRaisesRegex(ValueError,"No explicitly reviewed"):
            export_reviewed(bundle,self.root/"export")
        self.assertEqual(list((bundle/"reviews").iterdir()),[])
        before={str(p.relative_to(bundle)):file_sha256(p) for p in bundle.rglob("*") if p.is_file()}
        preview=preview_bundle(bundle,self.root/"previews")
        self.assertEqual(preview["reviewed"],0)
        self.assertEqual(preview["previews"],2)
        after={str(p.relative_to(bundle)):file_sha256(p) for p in bundle.rglob("*") if p.is_file()}
        self.assertEqual(before,after)

    def test_rehashed_clock_or_pixel_record_is_reconstructed_from_source(self):
        path=self.index/"frames.jsonl"
        original=path.read_text()
        metadata_path=self.index/"manifest.json"
        original_metadata=metadata_path.read_text()
        for field,value in (("display_elapsed_s",99),("decoded_bgr_sha256","0"*64)):
            with self.subTest(field=field):
                records=[json.loads(line) for line in original.splitlines()]
                records[0][field]=value
                path.write_text("".join(json.dumps(r)+"\n" for r in records))
                metadata=json.loads(original_metadata)
                metadata["index_sha256"]=file_sha256(path)
                metadata_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError,"differs from source"):
                    audit_video_index(self.index,protocol_path=self.protocol_path)
        path.write_text(original)
        metadata_path.write_text(original_metadata)

    def test_self_consistent_license_edit_cannot_replace_external_source_protocol(self):
        metadata_path=self.index/"manifest.json"
        metadata=json.loads(metadata_path.read_text())
        metadata["protocol"]["license"]="invented permission"
        snapshot=self.index/"protocol.json"
        snapshot.write_text(json.dumps(metadata["protocol"]))
        metadata["protocol_sha256"]=file_sha256(snapshot)
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,"independently supplied"):
            audit_video_index(self.index,protocol_path=self.protocol_path)

    def test_presentation_ticks_are_not_replaced_by_nominal_fps_or_acquisition_time(self):
        timing=({"time_base":"1/1000"},Fraction(1,1000),[1000,1100,1450,1900])
        with patch("vision.real_video._timing",return_value=timing):
            records,_=scan_videos(self.cache,self.protocol)
        self.assertEqual([r["display_elapsed_s"] for r in records],[0,.1,.45,.9])
        self.assertEqual(records[2]["published_speedup_relative_elapsed_s"],.45*14)
        self.assertTrue(all(r["acquisition_time_s"] is None and r["growth_run_start_s"] is None and r["first_hour_membership"] is None for r in records))

    def test_duplicate_payloads_and_changed_original_bytes_are_refused(self):
        duplicate=copy.deepcopy(self.protocol)
        extra=copy.deepcopy(duplicate["clips"][0])
        extra["clip_id"]="OTHER"
        duplicate["clips"].append(extra)
        with self.assertRaisesRegex(ValueError,"Duplicate clip"):
            validate_protocol(duplicate)
        with (self.cache/"fixture.avi").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"Original video bytes"):
            audit_video_index(self.index,protocol_path=self.protocol_path)
