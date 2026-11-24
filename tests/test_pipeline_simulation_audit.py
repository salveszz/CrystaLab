# FILE PURPOSE: Require actual completed simulation outputs, including fractional
# final timesteps, and reject truncated history masquerading as a full run.
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_pipeline import audit_simulation


@pytest.fixture(scope="module")
def simulation(tmp_path_factory):
    root = tmp_path_factory.mktemp("simulation-audit")
    config = root / "config.json"
    config.write_text(json.dumps(dict(nx=5, ny=5, nz=5, duration=2.5, dt=0.7, seed=42)))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_simulation.py"),
            "--config",
            str(config),
            "--headless",
            "--output",
            str(root / "run"),
        ],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            MPLCONFIGDIR=str(root / "matplotlib"),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root / "run"


def test_complete_fractional_duration_simulation_has_real_fields_and_frames(simulation):
    result = audit_simulation(simulation, 2.5)
    assert (
        result["passed"] and result["history_rows"] == 5 and result["final_time"] == 2.5
    )
    assert result["gif_frames"] == 5


def test_truncated_history_cannot_be_reported_as_a_complete_simulation(
    simulation, tmp_path
):
    import shutil

    target = tmp_path / "shortened"
    shutil.copytree(simulation, target)
    rows = (target / "measurements.csv").read_text().splitlines()
    (target / "measurements.csv").write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(AssertionError):
        audit_simulation(target, 2.5)
