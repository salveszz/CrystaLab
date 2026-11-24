# FILE PURPOSE: Execute a complete fresh simulation/dataset/XGBoost pipeline and
# retain every numerical output, model, log and audit in a commit-ready artifact.
"""Reproducible full-run orchestration; never overwrite existing artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(4 * 1024**2):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(config_path, output):
    output = Path(output).resolve()
    config = json.loads(config_path.read_text())
    if config["schema"] != "crystalab_full_pipeline_v1":
        raise ValueError("Unsupported full pipeline configuration")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    snapshots = {}
    for path in [
        Path(__file__),
        config_path,
        *(
            ROOT / config[k]
            for k in ["simulation_config", "dataset_config", "training_config"]
        ),
        ROOT / "pyproject.toml",
    ]:
        relative = path.resolve().relative_to(ROOT)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshots[str(relative)] = sha(target)
    save(output / "protocol.json", config)
    manifest = dict(
        schema=config["schema"],
        status="running",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        config=config,
        source_sha256=snapshots,
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        stages={},
    )
    save(output / "manifest.json", manifest)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLCONFIGDIR="/tmp/crystalab-full-pipeline-matplotlib",
    )
    processes, streams = {}, {}
    started = time.monotonic()

    def launch(name, arguments):
        stream = (output / "logs" / (name + ".log")).open("wb")
        streams[name] = stream
        command = [sys.executable, *map(str, arguments)]
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        processes[name] = process
        manifest["stages"][name] = dict(
            status="running",
            command=command,
            pid=process.pid,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(output / "manifest.json", manifest)

    def finish(name):
        code = processes[name].wait()
        streams[name].close()
        manifest["stages"][name].update(
            status="complete" if code == 0 else "failed", return_code=code
        )
        save(output / "manifest.json", manifest)
        if code:
            raise RuntimeError(f"{name} failed; inspect logs/{name}.log")

    try:
        launch(
            "dataset",
            [
                "scripts/generate_dataset.py",
                "--config",
                config["dataset_config"],
                "--output",
                output / "dataset",
                "--jobs",
                config["dataset_jobs"],
            ],
        )
        launch(
            "simulation",
            [
                "scripts/run_simulation.py",
                "--config",
                config["simulation_config"],
                "--headless",
                "--output",
                output / "simulation",
            ],
        )
        while any(p.poll() is None for p in processes.values()):
            complete = len(list((output / "dataset/episodes").glob("*/episode.json")))
            progress = dict(
                elapsed_seconds=round(time.monotonic() - started, 1),
                completed_episodes=complete,
                expected_episodes=config["expected_episodes"],
                live_stages=[name for name, p in processes.items() if p.poll() is None],
            )
            save(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            if any(p.poll() not in (None, 0) for p in processes.values()):
                break
            time.sleep(30)
        for name in ["dataset", "simulation"]:
            finish(name)
        launch(
            "training",
            [
                "scripts/train_xgboost.py",
                "--dataset",
                output / "dataset",
                "--config",
                config["training_config"],
                "--output",
                output / "training",
            ],
        )
        finish("training")
        launch(
            "audit",
            [
                "scripts/audit_pipeline.py",
                "--run",
                output,
                "--output",
                output / "audit.json",
            ],
        )
        finish("audit")
        manifest.update(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        save(output / "manifest.json", manifest)
        files = {
            str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "files.json"
        }
        save(output / "files.json", files)
        print(
            json.dumps(
                dict(
                    status="complete",
                    output=str(output),
                    files=len(files),
                    bytes=sum(v["bytes"] for v in files.values()),
                ),
                indent=2,
            )
        )
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(output / "manifest.json", manifest)
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/full_pipeline.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.config, a.output)
