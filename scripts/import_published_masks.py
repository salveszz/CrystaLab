# FILE PURPOSE: Import/audit published CHiMP masks or preview their training images locally.
"""Keep published protein-crystal annotation semantics separate from pitch labels."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision.published_masks import (
    import_published_masks,
    audit_published_masks,
    preview_published_masks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--preview",
        type=Path,
        help="Completed source index; write a separate training-only preview",
    )
    args = parser.parse_args()
    if args.audit and args.preview:
        parser.error("Choose audit or preview")
    if args.audit:
        result = audit_published_masks(args.output)
    elif args.preview:
        result = preview_published_masks(args.preview, args.output)
    else:
        if None in (args.archive, args.metadata, args.protocol):
            parser.error("Import needs archive, metadata and protocol")
        result = import_published_masks(
            args.archive,
            args.metadata,
            json.loads(args.protocol.read_text()),
            args.output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
