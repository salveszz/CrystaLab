# FILE PURPOSE: Generate and independently audit synthetic growth datasets, without ML.
# Whole scenario families receive fixed splits before any temporal samples exist.
# Raw episode events, complete configurations, final fields and source fingerprints
# remain separate from a strict causal feature whitelist and exact future labels.
# Existing output is never overwritten; resume verifies identity and completed files.
"""Reproducible dataset storage, causal sample construction and integrity audits."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np

from .kdp_env import (
    MODEL_MODE,
    OBSERVABLES,
    UNITS,
    RecordingConfig,
    expand_schedule,
    make_observations,
    record_episode,
    tick,
)
from simulation.tank import SimulationConfig
from simulation.crystal import Crystal

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "crystalab_growth_dataset_v1"
FEATURES = (
    *OBSERVABLES,
    "volume_lag",
    "recent_volume_rate",
    "observation_age",
    "temperature_command",
    "agitation_command",
    "horizon",
)
SPLITS = {"train", "validation", "test"}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Replace a small metadata file atomically; raw episode streams append separately."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def read_rows(path):
    return [
        json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        for line in Path(path).read_text().splitlines()
    ]


def source_identity():
    """Hash actual working sources, not just HEAD; capture numerical execution versions."""
    paths = [
        *sorted((ROOT / "src/simulation").glob("*.py")),
        *sorted((ROOT / "src/environment").glob("*.py")),
        ROOT / "scripts/generate_dataset.py",
        ROOT / "scripts/audit_dataset.py",
        ROOT / "scripts/check_dataset.py",
        ROOT / "pyproject.toml",
    ]
    return dict(
        sources={str(p.relative_to(ROOT)): file_hash(p) for p in paths},
        python=platform.python_version(),
        packages={n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
        threads={
            n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
    )


def snapshot_sources(source, output):
    """Preserve actual uncommitted contents for replay, detecting concurrent edits."""
    for name, fingerprint in source["sources"].items():
        destination = Path(output) / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if file_hash(destination) != fingerprint:
            raise ValueError(
                "Source changed while snapshotting; restart with stable sources"
            )


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value
    ):
        raise ValueError(f"Unsafe or invalid identifier {value!r}")
    return value


def build_cases(protocol):
    """Expand explicit scenario/recipe design; protect shared physics and families."""
    if set(protocol) != {"simulation", "recording", "groups", "recipes", "description"}:
        raise ValueError(
            "Protocol needs simulation, recording, groups, recipes, description"
        )
    base = SimulationConfig(**protocol["simulation"])
    recording = RecordingConfig(**protocol["recording"])
    if not protocol["groups"] or not protocol["recipes"]:
        raise ValueError("Groups and recipes cannot be empty")
    names, recipes = set(), []
    for recipe in protocol["recipes"]:
        if set(recipe) != {"id", "schedule"}:
            raise ValueError("Recipe needs exactly id and schedule")
        name = identifier(recipe["id"])
        if name in names:
            raise ValueError("Duplicate recipe ID")
        names.add(name)
        recipes.append(recipe)
    cases, names, families, physics_splits = [], set(), {}, {}
    for group in protocol["groups"]:
        if set(group) != {"id", "family_id", "split", "overrides"}:
            raise ValueError("Group needs exactly id, family_id, split, overrides")
        name, family = identifier(group["id"]), identifier(group["family_id"])
        split = group["split"]
        if name in names or split not in SPLITS:
            raise ValueError("Duplicate group ID or invalid split")
        names.add(name)
        if family in families and families[family] != split:
            raise ValueError("Related family cannot cross splits")
        families[family] = split
        config = SimulationConfig(**(asdict(base) | group["overrides"]))
        # Seeds cannot manufacture independent physical scenarios across splits.
        physics = asdict(config)
        physics.pop("seed")
        signature = digest(physics)
        if signature in physics_splits and physics_splits[signature] != split:
            raise ValueError(
                "Identical physical scenarios cannot cross splits by changing seed"
            )
        physics_splits[signature] = split
        for recipe in recipes:
            expand_schedule(config, recording, recipe["schedule"])
            # Paired actions share observation errors within a group; unrelated
            # groups get reproducible independent streams, separate from physics.
            sensor_settings = asdict(recording)
            sensor_settings["observation_seed"] = int(
                digest([recording.observation_seed, name])[:16], 16
            )
            case = dict(
                id=name + "__" + recipe["id"],
                group_id=name,
                family_id=family,
                split=split,
                recipe_id=recipe["id"],
                simulation=asdict(config),
                recording=sensor_settings,
                schedule=recipe["schedule"],
            )
            case["identity"] = digest(case)
            if any(previous["id"] == case["id"] for previous in cases):
                raise ValueError("Ambiguous combined episode ID; rename group/recipe")
            cases.append(case)
    if {c["split"] for c in cases} != SPLITS:
        raise ValueError("Declare nonempty train, validation and test scenario groups")
    return cases


def make_samples(case, truth, observations, controls):
    """Join exact future truth only as labels; held commands never cross the horizon."""
    cfg, recording = (
        SimulationConfig(**case["simulation"]),
        RecordingConfig(**case["recording"]),
    )
    truth_at = {tick(r["time"]): r for r in truth}
    observed_at = {tick(r["time"]): r for r in observations}
    samples, omitted = [], 0
    for command in controls:
        t = tick(command["time"])
        end = t + tick(recording.forecast_horizon)
        if end > tick(cfg.duration):
            omitted += 1
            continue
        if (
            end not in truth_at
            or t not in observed_at
            or end > tick(command["applies_until"])
        ):
            raise ValueError("Missing exact forecast endpoint or action hold")
        observation = observed_at[t]
        features = {
            k: observation[k]
            for k in (*OBSERVABLES, "volume_lag", "recent_volume_rate")
        }
        features.update(
            observation_age=observation["age"],
            temperature_command=command["temperature"],
            agitation_command=command["agitation"],
            horizon=recording.forecast_horizon,
        )
        gain = truth_at[end]["volume"] - truth_at[t]["volume"]
        samples.append(
            dict(
                sample_id=case["id"] + ":" + str(float(t)),
                episode_id=case["id"],
                group_id=case["group_id"],
                family_id=case["family_id"],
                split=case["split"],
                decision_time=float(t),
                sensor_time=observation["sensor_time"],
                label_end_time=float(end),
                features=features,
                missing={key: value is None for key, value in features.items()},
                labels=dict(
                    volume_gain=gain,
                    mean_volume_growth_rate=gain / recording.forecast_horizon,
                    mass_gain=gain * cfg.crystal_density,
                    quality=None,
                ),
            )
        )
    return samples, omitted


def _episode(task):
    """One worker owns one episode; failed/interrupted attempts retain raw evidence."""
    case, root = task
    folder = Path(root) / "episodes" / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "episode.json").exists():
        previous = read_json(folder / "episode.json")
        if previous.get("status") == "complete":
            verify_episode(folder, case)
            return previous
    attempt = 1
    while (folder / f"attempt_{attempt}").exists():
        attempt += 1
    destination = folder / f"attempt_{attempt}"
    destination.mkdir()
    metadata = dict(case=case, status="running", attempt=attempt)
    write_json(folder / "episode.json", metadata)
    started = time.perf_counter()
    try:
        with (destination / "events.jsonl").open("x") as stream:

            def append(event):
                stream.write(canonical(event) + "\n")
                stream.flush()

            result = record_episode(
                SimulationConfig(**case["simulation"]),
                RecordingConfig(**case["recording"]),
                case["schedule"],
                on_event=append,
            )
            os.fsync(stream.fileno())
        np.savez_compressed(destination / "final_fields.npz", **result.final_fields)
        metadata.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - started,
            diagnostics=result.diagnostics,
            files={p.name: file_hash(p) for p in sorted(destination.iterdir())},
        )
        write_json(folder / "episode.json", metadata)
        verify_episode(folder, case)
        return metadata
    except Exception as exc:
        metadata.update(
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json(folder / "episode.json", metadata)
        write_json(destination / "failure.json", metadata)
        raise


def verify_episode(folder, case):
    """Read hashes, raw clocks/actions/observations, invariants and independent final arrays."""
    metadata = read_json(folder / "episode.json")
    if metadata["status"] != "complete" or metadata["case"] != case:
        raise ValueError("Episode is incomplete or has a different identity")
    destination = folder / f"attempt_{metadata['attempt']}"
    if set(metadata["files"]) != {"events.jsonl", "final_fields.npz"}:
        raise ValueError("Unexpected episode artifact set")
    for name, value in metadata["files"].items():
        if file_hash(destination / name) != value:
            raise ValueError(f"Artifact hash mismatch: {case['id']}/{name}")
    events = read_rows(destination / "events.jsonl")
    if any(r.get("kind") not in {"truth", "observation", "control"} for r in events):
        raise ValueError("Unexpected raw event kind")

    def rows(kind):
        return [
            {k: v for k, v in r.items() if k != "kind"}
            for r in events
            if r["kind"] == kind
        ]

    truth, observations, controls = rows("truth"), rows("observation"), rows("control")
    cfg, rec = (
        SimulationConfig(**case["simulation"]),
        RecordingConfig(**case["recording"]),
    )
    expected_controls = expand_schedule(cfg, rec, case["schedule"])
    for row in expected_controls:
        row.update(
            applies_until=min(row["time"] + rec.decision_interval, cfg.duration),
            rotation=row["agitation"] / 2,
            circulation=row["agitation"] / 2,
            actuator_delay=0.0,
        )
    if controls != expected_controls:
        raise ValueError("Applied controls differ from declared schedule")
    times = {tick(0), tick(cfg.duration), *(tick(r["time"]) for r in controls)}
    t = tick(rec.sample_interval)
    while t < tick(cfg.duration):
        times.add(t)
        t += tick(rec.sample_interval)
    if [r["time"] for r in truth] != [float(t) for t in sorted(times)]:
        raise ValueError("Raw truth clocks do not match exact declared events")
    if observations != make_observations(truth, rec):
        raise ValueError(
            "Observation values, missingness or causal source times differ"
        )
    initial = cfg.make_crystal()
    for row in truth:
        if (
            np.asarray(row["face_normals"]).shape != initial.normals.shape
            or not np.allclose(row["face_normals"], initial.normals, atol=1e-12, rtol=0)
            or len(row["face_offsets"]) != len(initial.offsets)
            or len(row["face_areas"]) != len(initial.offsets)
        ):
            raise ValueError("Invalid facet identities in truth")
    if any(b["volume"] < a["volume"] - 1e-12 for a, b in zip(truth, truth[1:])):
        raise ValueError("Outward-growth truth lost volume")
    diag = metadata["diagnostics"]
    for field, limit in (
        ("max_mass_relative_error", rec.mass_relative_tolerance),
        ("max_carrier_relative_error", rec.carrier_relative_tolerance),
        ("max_continuity_cell_error", rec.continuity_cell_tolerance),
    ):
        if not 0 <= diag[field] <= limit:
            raise ValueError(f"Invalid recorded invariant: {field}")
    if not diag["exact_final_time"] or diag["final_time"] != cfg.duration:
        raise ValueError("Episode did not finish at exact duration")
    for row in truth:
        if not all(
            np.isfinite(np.asarray(value, dtype=float)).all() for value in row.values()
        ):
            raise ValueError("Nonfinite truth")
        if (
            abs(row["total_mass"] - diag["initial_total_mass"])
            > rec.mass_relative_tolerance * diag["initial_total_mass"]
        ):
            raise ValueError("Raw truth violates mass conservation")
        if (
            abs(row["liquid_volume"] - diag["initial_carrier_volume"])
            > rec.carrier_relative_tolerance * diag["initial_carrier_volume"]
        ):
            raise ValueError("Raw truth violates carrier conservation")
        if (
            row["volume"] < 0
            or abs(row["crystal_mass"] - row["volume"] * cfg.crystal_density) > 1e-10
        ):
            raise ValueError("Invalid crystal truth")
    with np.load(destination / "final_fields.npz", allow_pickle=False) as fields:
        if not all(np.isfinite(fields[k]).all() for k in fields.files):
            raise ValueError("Nonfinite final fields")
        c, v, cells = (
            fields["concentration"],
            fields["liquid_volumes"],
            fields["cell_volumes"],
        )
        if (
            c.shape != v.shape
            or cells.shape != v.shape
            or np.any(v < 0)
            or np.any(v > cells + 1e-10)
        ):
            raise ValueError("Invalid final partial-cell volumes")
        expected_shape = (cfg.nz, cfg.ny, cfg.nx)
        if c.shape != expected_shape or fields["temperature"].shape != expected_shape:
            raise ValueError("Final field grid shape differs from configuration")
        edges = [fields[key] for key in ("x_edges", "y_edges", "z_edges")]
        if any(np.any(np.diff(edge) <= 0) for edge in edges):
            raise ValueError("Final grid edges must increase")
        geometric_cells = (
            np.diff(edges[2])[:, None, None]
            * np.diff(edges[1])[None, :, None]
            * np.diff(edges[0])[None, None, :]
        )
        if not np.allclose(cells, geometric_cells, rtol=1e-13, atol=0):
            raise ValueError("Cell volumes differ from edge geometry")
        crystal = Crystal(fields["normals"], fields["offsets"])
        # Half-space reconstruction can reorder an identical vertex point set.
        vertices = fields["vertices"]
        distances = np.linalg.norm(
            crystal.vertices[:, None] - vertices[None, :], axis=-1
        )
        vertex_error = max(distances.min(axis=0).max(), distances.min(axis=1).max())
        if (
            abs(crystal.volume - truth[-1]["volume"]) > 1e-10
            or not np.allclose(
                fields["offsets"][:-6], truth[-1]["face_offsets"], atol=1e-12, rtol=0
            )
            or len(crystal.vertices) != len(vertices)
            or vertex_error > 1e-10
        ):
            raise ValueError("Final geometry differs from raw truth")
        if c.min() < -rec.negative_concentration_tolerance:
            raise ValueError("Negative final concentration")
        if (
            abs(float(np.sum(c * v)) - truth[-1]["dissolved_mass"])
            > rec.mass_relative_tolerance * diag["initial_total_mass"]
        ):
            raise ValueError("Final fields do not reproduce recorded dissolved mass")
        if (
            abs(float(v.sum()) - diag["initial_carrier_volume"])
            > rec.carrier_relative_tolerance * diag["initial_carrier_volume"]
        ):
            raise ValueError("Final fields violate carrier conservation")
    samples, omitted = make_samples(case, truth, observations, controls)
    return dict(
        metadata=metadata,
        truth=truth,
        controls=controls,
        observations=observations,
        samples=samples,
        omitted=omitted,
    )


def generate(protocol, output, *, jobs=1, resume=False):
    """Create an auditable dataset; resuming never silently reuses changed sources/settings."""
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    cases = build_cases(protocol)
    output = Path(output)
    identity = dict(
        schema=SCHEMA,
        mode=MODEL_MODE,
        protocol=protocol,
        cases=cases,
        numerical_environment=source_identity(),
        units=UNITS,
        feature_columns=list(FEATURES),
    )
    if resume:
        manifest = read_json(output / "manifest.json")
        if manifest["identity"] != identity:
            raise ValueError(
                "Resume requires identical protocol, sources and numerical environment"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_sources(identity["numerical_environment"], output)
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        manifest = dict(
            identity=identity,
            identity_hash=digest(identity),
            git_head=git.stdout.strip(),
            status="running",
            executions=[],
        )
    manifest["status"] = "running"
    manifest["executions"].append(dict(jobs=jobs, resumed=resume))
    write_json(output / "manifest.json", manifest)
    tasks = [(case, str(output)) for case in cases]
    try:
        if jobs == 1:
            for task in tasks:
                _episode(task)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                list(executor.map(_episode, tasks))
        all_samples = []
        for case in cases:
            all_samples.extend(
                verify_episode(output / "episodes" / case["id"], case)["samples"]
            )
        with (output / "samples.jsonl").open("w") as stream:
            for row in all_samples:
                stream.write(canonical(row) + "\n")
        manifest.update(
            status="complete", samples_hash=file_hash(output / "samples.jsonl")
        )
        write_json(output / "manifest.json", manifest)
        report = audit_dataset(output)
        write_json(output / "audit.json", report)
        return report
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise


def audit_dataset(output):
    """Reconstruct every sample from raw records, ignoring any previous green audit."""
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    identity = manifest["identity"]
    if manifest["status"] != "complete" or manifest["identity_hash"] != digest(
        identity
    ):
        raise ValueError("Incomplete dataset or corrupted manifest identity")
    if (
        identity["schema"] != SCHEMA
        or identity["mode"] != MODEL_MODE
        or identity["units"] != UNITS
    ):
        raise ValueError("Unsupported schema, simulator mode or unit contract")
    for name, fingerprint in identity["numerical_environment"]["sources"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source snapshot path")
        if file_hash(output / "source_snapshot" / relative) != fingerprint:
            raise ValueError(f"Source snapshot mismatch: {name}")
    cases = build_cases(identity["protocol"])
    if cases != identity["cases"] or identity["feature_columns"] != list(FEATURES):
        raise ValueError("Case design or feature whitelist changed")
    expected, omitted, counts, seconds, max_errors = [], 0, {}, [], {}
    initial_by_group = {}
    for case in cases:
        result = verify_episode(output / "episodes" / case["id"], case)
        expected.extend(result["samples"])
        omitted += result["omitted"]
        counts[case["split"]] = counts.get(case["split"], 0) + len(result["samples"])
        seconds.append(result["metadata"]["elapsed_seconds"])
        initial = result["truth"][0]
        if initial_by_group.setdefault(case["group_id"], initial) != initial:
            raise ValueError(
                "Paired action recipes do not share an identical initial state"
            )
        for field in (
            "max_mass_relative_error",
            "max_carrier_relative_error",
            "max_continuity_cell_error",
        ):
            max_errors[field] = max(
                max_errors.get(field, 0), result["metadata"]["diagnostics"][field]
            )
    actual = read_rows(output / "samples.jsonl")
    if (
        actual != expected
        or file_hash(output / "samples.jsonl") != manifest["samples_hash"]
    ):
        raise ValueError("Samples/labels differ from causal raw-record reconstruction")
    if any(counts.get(split, 0) == 0 for split in SPLITS):
        raise ValueError("Every declared split needs at least one usable sample")
    return dict(
        status="pass",
        identity_hash=manifest["identity_hash"],
        episodes=len(cases),
        groups=len(initial_by_group),
        samples=len(actual),
        samples_by_split=counts,
        omitted_incomplete_horizons=omitted,
        invariant_maxima=max_errors,
        summed_episode_seconds=sum(seconds),
        mean_episode_seconds=float(np.mean(seconds)),
        estimated_1000_episode_worker_seconds=1000 * float(np.mean(seconds)),
        scope="Synthetic growth only; no ML, quality calibration or physical KDP validation",
    )
