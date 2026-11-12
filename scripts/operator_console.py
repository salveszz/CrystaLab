# FILE PURPOSE: Prepare or open the local evidence/process/operator replay console.
"""Review shadow evidence or run bounded operator-controlled software simulations."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dashboard.operator import OperatorServer, prepare_operator_session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--analysis", type=Path)
    p.add_argument(
        "--policy", type=Path, default=Path("configs/event_rule_policy.json")
    )
    p.add_argument("--simulation-config", type=Path)
    p.add_argument("--adaptive-reference", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--read-only", action="store_true")
    p.add_argument("--port", type=int, default=8766)
    a = p.parse_args()
    if not 0 <= a.port <= 65535:
        p.error("Port must be between 0 and 65535")
    if a.analysis is not None:
        prepare_operator_session(
            a.analysis,
            a.policy,
            a.session,
            config=a.simulation_config,
            adaptive_reference=a.adaptive_reference,
        )
    elif a.simulation_config is not None or a.adaptive_reference is not None:
        p.error(
            "Simulation configuration and adaptive reference are set when preparing a new session"
        )
    if a.prepare_only:
        return
    server = OperatorServer(a.session, port=a.port, read_only=a.read_only)
    print(f"CrystaLab operator: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Session: {server.store.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
