# FILE PURPOSE: Import/audit public real-video frames or prepare a native annotation pack.
"""Local files only; original clips remain pinned and no labels are auto-approved."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from vision.real_video import import_videos,audit_video_index,prepare_video_bundle


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/kdp_video_import.json")
    parser.add_argument("--cache",type=Path,default=ROOT/"data/external_vision/KDP-2026")
    parser.add_argument("--index",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--ffprobe",default="ffprobe")
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--bundle",action="store_true")
    args=parser.parse_args()
    if args.audit:
        manifest,rows,_=audit_video_index(args.index,protocol_path=args.protocol,ffprobe=args.ffprobe)
        result=dict(status="passed",frames=len(rows),index_sha256=manifest["index_sha256"])
    elif args.bundle:
        result=prepare_video_bundle(args.index,args.output,protocol_path=args.protocol,ffprobe=args.ffprobe)
    else:
        result=import_videos(args.cache,args.protocol,args.output,ffprobe=args.ffprobe)
    print(json.dumps(result,indent=2))
