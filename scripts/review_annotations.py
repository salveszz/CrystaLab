# FILE PURPOSE: Launch the repository-local polygon review interface on loopback.
# Reviews require per-image human acknowledgment; a read-only preview never writes labels.
"""Inspect, edit and explicitly review a prepared annotation bundle locally."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dashboard.review import ReviewServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    server = ReviewServer(args.bundle, port=args.port, read_only=args.read_only)
    print(f"CrystaLab review: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Bundle: {server.store.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
