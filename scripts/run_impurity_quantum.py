# FILE PURPOSE: Execute the declared KDP-related molecular/quantum stages locally.
"""Use the separate pinned quantum environment; no remote service is contacted."""

import argparse
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from models.impurity_quantum import classical_case,vqe_case


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["classical","vqe"])
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/kdp_impurity_quantum.json")
    parser.add_argument("--case")
    parser.add_argument("--classical",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=classical_case(args.protocol,args.case,args.output) if args.stage=="classical" else vqe_case(args.classical,args.output,protocol_path=args.protocol)
    # Print the already validated JSON record, including converted NumPy scalars.
    print((args.output / "manifest.json").read_text(), end="")
