# HOW THIS FILE WORKS
# Run the same coupled growth/displacement experiment at different grids and dt.
# First vary only spatial resolution; then vary only dt on the finest grid.
# Convert one fixed physical sampling thickness into boundary_layer_cells for
# each grid. Disable concentration noise and random defect events in every run.
# Save the protocol and tolerances before running; save each completed case so a
# long study can resume without mixing different settings or solver versions.
# Compare adjacent levels and whole trajectories at common simulation times.
# The finest case is a numerical reference, not experimental truth. A failed
# refinement gate is a result to report, not a reason to retune its tolerance.
# Final archives include liquid centroids to inspect reconstructed facet sampling.
# Optional process workers run independent cases; the worker count is fingerprinted.
# Source identity includes the shared mesh and its input-validation dependencies.

"""Measure grid/time sensitivity of the existing 3D simulator, without AI."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.engine import Simulation
from simulation.tank import SimulationConfig


def _digest(value):
    """Hash exact settings, independent of JSON key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def solver_fingerprint():
    """Record the actual core sources, including any uncommitted solver changes."""
    return {
        name: hashlib.sha256((ROOT / "src/simulation" / name).read_bytes()).hexdigest()
        for name in (
            "crystal.py",
            "tank.py",
            "physics.py",
            "engine.py",
            "mesh.py",
            "units.py",
        )
    }


def build_cases(base, protocol):
    """Construct controlled spatial/time sweeps without changing physical inputs."""
    grids, timesteps = protocol["grids"], protocol["timesteps"]
    if len(grids) < 3 or len(timesteps) < 3:
        raise ValueError("Use at least three grid levels and three timestep levels")
    for grid in grids:
        if len(grid) != 3 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in grid
        ):
            raise ValueError("Each grid needs three integer counts >= 3, ordered x/y/z")
    if any(
        not all(b > a for a, b in zip(coarse, fine))
        for coarse, fine in zip(grids, grids[1:])
    ):
        raise ValueError("Each successive grid must refine all three axes")
    if any(
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not math.isfinite(dt)
        or dt <= 0
        for dt in timesteps
    ):
        raise ValueError("Timesteps must be positive finite numbers")
    if any(fine >= coarse for coarse, fine in zip(timesteps, timesteps[1:])):
        raise ValueError("Timesteps must be ordered from coarse to fine")
    for key in (
        "duration",
        "boundary_layer_thickness",
        "relative_change_tolerance",
        "mass_relative_tolerance",
        "carrier_relative_tolerance",
        "continuity_cell_tolerance",
    ):
        value = protocol[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be positive and finite")
    tolerance = protocol["negative_concentration_tolerance"]
    if (
        not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "negative_concentration_tolerance must be finite and nonnegative"
        )

    cases, spatial, temporal = {}, [], []

    def add(grid, dt):
        key = f"grid_{grid[0]}x{grid[1]}x{grid[2]}_dt_{dt:g}"
        settings = dict(base)
        settings.update(
            nx=grid[0],
            ny=grid[1],
            nz=grid[2],
            dt=float(dt),
            duration=float(protocol["duration"]),
            concentration_noise=0.0,
            defect_rate=0.0,
        )
        spacing = [
            settings[axis] / count
            for axis, count in zip(("width", "depth", "height"), grid)
        ]
        # The engine multiplies this value by its largest nominal mean spacing,
        # also on focused grids. Keep the sampling depth fixed in physical units.
        settings["boundary_layer_cells"] = protocol["boundary_layer_thickness"] / max(
            spacing
        )
        resolved = asdict(SimulationConfig(**settings))
        cases[key] = dict(id=key, config=resolved)
        return key

    for grid in grids:
        spatial.append(add(grid, timesteps[0]))
    for dt in timesteps:
        temporal.append(add(grids[-1], dt))
    return list(cases.values()), spatial, temporal


def run_case(case, output, signature):
    """Run one complete experiment; diagnostics use real liquid volumes."""
    directory = output / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sim = Simulation(SimulationConfig(**case["config"]))
    initial = sim.measurements()
    initial_offsets = sim.crystal.offsets[:-6].copy()  # Exclude fixed wall planes.
    initial_mass, initial_carrier = sim.total_mass, sim.tank.liquid_volume
    history, faces = [initial], [initial_offsets.tolist()]
    max_mass, max_carrier, max_continuity = 0.0, 0.0, 0.0
    minimum = float(sim.tank.concentration.min())
    next_progress = max(1.0, sim.config.duration / 10.0)
    progress_interval = next_progress
    while sim.time < sim.config.duration:
        row = sim.step()
        history.append(row)
        faces.append(sim.crystal.offsets[:-6].tolist())
        max_mass = max(max_mass, abs(row["total_mass"] - initial_mass))
        max_carrier = max(max_carrier, abs(row["liquid_volume"] - initial_carrier))
        max_continuity = max(max_continuity, row["continuity_error"])
        minimum = min(minimum, float(sim.tank.concentration.min()))
        if sim.time >= next_progress or sim.time == sim.config.duration:
            print(
                f"  {case['id']}: t={sim.time:g}/{sim.config.duration:g}, volume={row['volume']:.8f}",
                flush=True,
            )
            next_progress += progress_interval
    elapsed = time.perf_counter() - started
    final = history[-1]
    result = dict(
        id=case["id"],
        signature=signature,
        config=case["config"],
        initial=initial,
        final=final,
        history=history,
        face_offsets=faces,
        initial_face_offsets=initial_offsets.tolist(),
        final_dimensions=sim.crystal.dimensions.tolist(),
        active_faces=[
            i for i, face in enumerate(sim.crystal.faces[:-6]) if len(face) >= 3
        ],
        final_face_rates=sim.face_rates[:-6].tolist(),
        runtime_seconds=elapsed,
        physical_sampling_thickness=sim.config.boundary_layer_cells
        * max(
            sim.config.width / sim.config.nx,
            sim.config.depth / sim.config.ny,
            sim.config.height / sim.config.nz,
        ),
        max_mass_error=max_mass,
        max_carrier_error=max_carrier,
        max_continuity_error=max_continuity,
        max_mass_relative_error=max_mass / max(abs(initial_mass), 1e-30),
        max_carrier_relative_error=max_carrier / max(initial_carrier, 1e-30),
        max_continuity_cell_error=max_continuity / sim.tank.cell_volume,
        min_concentration=minimum,
        exact_final_time=sim.time == sim.config.duration,
        diffusion_diagnostics=dict(sim.tank.diffusion_diagnostics),
        remap_diagnostics=dict(sim.tank.remap_diagnostics),
    )
    with (directory / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        directory / "final_fields.npz",
        concentration=sim.tank.concentration,
        temperature=sim.tank.temperature,
        liquid_volumes=sim.tank.liquid_volumes,
        liquid_centroids=sim.tank.liquid_centroids,
        cell_volumes=sim.tank.cell_volumes,
        x_edges=sim.tank.x_edges,
        y_edges=sim.tank.y_edges,
        z_edges=sim.tank.z_edges,
        open_face_centroids_z=sim.tank.geometry.opening_centroids[0],
        open_face_centroids_y=sim.tank.geometry.opening_centroids[1],
        open_face_centroids_x=sim.tank.geometry.opening_centroids[2],
        solid_fraction=sim.tank.geometry.solid_fraction,
        velocity=sim.tank.displacement_velocity,
        normals=sim.crystal.normals,
        offsets=sim.crystal.offsets,
        vertices=sim.crystal.vertices,
        x=sim.tank.x,
        y=sim.tank.y,
        z=sim.tank.z,
    )
    (directory / "case.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def compare_cases(coarse, fine):
    """Compare final outcomes and trajectories; initial geometry is identical.

    Normalize growth/level changes by the growth or rise, not by their much
    larger initial baseline. Reference values are from the finer run, not truth.
    Linear interpolation only aligns saved histories; no run state is modified.
    """

    def gain(case, metric):
        return case["final"][metric] - case["initial"][metric]

    def relative(a, b):
        return abs(a - b) / max(abs(b), 1e-12)

    times = np.asarray([row["time"] for row in fine["history"]])
    coarse_times = np.asarray([row["time"] for row in coarse["history"]])
    shared_times = np.unique(np.r_[coarse_times, times])
    # The final-time check keeps extrapolation out of the comparison entirely.
    if coarse_times[-1] != times[-1]:
        raise ValueError("Convergence cases must end at the same physical time")
    volume_c = np.interp(
        shared_times, coarse_times, [row["volume"] for row in coarse["history"]]
    )
    volume_f = np.interp(
        shared_times, times, [row["volume"] for row in fine["history"]]
    )
    offsets_c, offsets_f = (
        np.asarray(coarse["face_offsets"]),
        np.asarray(fine["face_offsets"]),
    )
    face_error = max(
        float(
            np.max(
                abs(
                    np.interp(shared_times, coarse_times, offsets_c[:, i])
                    - np.interp(shared_times, times, offsets_f[:, i])
                )
            )
        )
        for i in range(offsets_f.shape[1])
    )
    reference_advance = float(np.max(abs(offsets_f[-1] - fine["initial_face_offsets"])))
    return dict(
        coarse=coarse["id"],
        fine=fine["id"],
        volume_gain_relative_change=relative(
            gain(coarse, "volume"), gain(fine, "volume")
        ),
        volume_trajectory_relative_change=float(np.max(abs(volume_c - volume_f)))
        / max(abs(gain(fine, "volume")), 1e-12),
        surface_area_relative_change=relative(
            coarse["final"]["surface_area"], fine["final"]["surface_area"]
        ),
        level_rise_relative_change=relative(
            gain(coarse, "liquid_level"), gain(fine, "liquid_level")
        ),
        liquid_level_absolute_change=abs(
            coarse["final"]["liquid_level"] - fine["final"]["liquid_level"]
        ),
        concentration_relative_change=relative(
            coarse["final"]["concentration"], fine["final"]["concentration"]
        ),
        face_advance_relative_change=face_error / max(reference_advance, 1e-12),
        active_faces_match=coarse["active_faces"] == fine["active_faces"],
    )


def assess(results, spatial, temporal, protocol):
    """Use the predeclared gate; do not fit tolerances to measured outcomes."""
    by_id = {case["id"]: case for case in results}
    threshold = protocol["relative_change_tolerance"]
    metrics = (
        "volume_gain_relative_change",
        "volume_trajectory_relative_change",
        "surface_area_relative_change",
        "level_rise_relative_change",
        "concentration_relative_change",
        "face_advance_relative_change",
    )
    comparisons = {}
    for name, ids in (("spatial", spatial), ("temporal", temporal)):
        rows = [compare_cases(by_id[a], by_id[b]) for a, b in zip(ids, ids[1:])]
        for row in rows:
            row["within_tolerance"] = (
                all(row[key] <= threshold for key in metrics)
                and row["active_faces_match"]
            )
        comparisons[name] = rows
    conservation = {}
    for case in results:
        conservation[case["id"]] = dict(
            mass=case["max_mass_relative_error"] <= protocol["mass_relative_tolerance"],
            carrier=case["max_carrier_relative_error"]
            <= protocol["carrier_relative_tolerance"],
            continuity=case["max_continuity_cell_error"]
            <= protocol["continuity_cell_tolerance"],
            concentration=case["min_concentration"]
            >= -protocol["negative_concentration_tolerance"],
            final_time=case["exact_final_time"],
        )
    conservation_pass = all(all(checks.values()) for checks in conservation.values())
    # The gate applies to the finest adjacent levels. Coarser disagreements remain
    # visible; they do not get averaged away or silently excluded from the report.
    spatial_pass = comparisons["spatial"][-1]["within_tolerance"]
    temporal_pass = comparisons["temporal"][-1]["within_tolerance"]
    return dict(
        comparisons=comparisons,
        conservation_checks=conservation,
        conservation_pass=conservation_pass,
        spatial_pass=spatial_pass,
        temporal_pass=temporal_pass,
        overall_pass=conservation_pass and spatial_pass and temporal_pass,
    )


def write_report(output, results, assessment, protocol):
    """Export a human-readable report and machine-readable assessment."""
    (output / "report.json").write_text(
        json.dumps(
            dict(protocol=protocol, results=results, **assessment),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    lines = [
        "# Coupled growth and displacement convergence",
        "",
        "Numerical refinement study only; this does not validate real KDP kinetics or quality.",
        "",
        f"Duration: {protocol['duration']:g}. Fixed physical sampling thickness: {protocol['boundary_layer_thickness']:g}.",
        f"Predeclared refinement tolerance: {protocol['relative_change_tolerance']:.1%}.",
        "",
        "| Grid / dt | Final volume | Level | Mean liquid concentration | Max relative mass error | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in results:
        final = case["final"]
        lines.append(
            f"| {case['id']} | {final['volume']:.8f} | {final['liquid_level']:.8f} | "
            f"{final['concentration']:.8f} | {case['max_mass_relative_error']:.3e} | {case['runtime_seconds']:.2f} |"
        )
    for name in ("spatial", "temporal"):
        lines += [
            "",
            f"## {name.capitalize()} refinement",
            "",
            "| Comparison | Growth change | Trajectory change | Face advance change | Level rise change | Within declared gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in assessment["comparisons"][name]:
            lines.append(
                f"| {row['coarse']} → {row['fine']} | {row['volume_gain_relative_change']:.2%} | "
                f"{row['volume_trajectory_relative_change']:.2%} | {row['face_advance_relative_change']:.2%} | "
                f"{row['level_rise_relative_change']:.2%} | {'PASS' if row['within_tolerance'] else 'FAIL'} |"
            )
    lines += [
        "",
        "## Assessment",
        "",
        f"- Conservation/positivity/final time: {'PASS' if assessment['conservation_pass'] else 'FAIL'}.",
        f"- Finest spatial pair: {'PASS' if assessment['spatial_pass'] else 'FAIL'}.",
        f"- Finest timestep pair: {'PASS' if assessment['temporal_pass'] else 'FAIL'}.",
        f"- Combined numerical gate: {'PASS' if assessment['overall_pass'] else 'FAIL'}.",
        "",
        "The finest run is a reference, not an exact solution. No convergence order is inferred from these few levels.",
        "Final fields are saved per case; this gate compares geometric/growth outcomes, not registered full-field norms.",
        "A failed gate must be investigated before claiming grid-independent coupled predictions. Tolerances were not adjusted after measurement.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_study(base, protocol, output, resume=False, jobs=1):
    """Execute controlled cases, with resumable source/config-checked artifacts."""
    output = Path(output)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("Study jobs must be a positive integer")
    cases, spatial, temporal = build_cases(base, protocol)
    sources = solver_fingerprint()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    environment = dict(
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name) for name in ("numpy", "scipy")
        },
        threads={
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
    )
    identity = _digest(
        dict(
            protocol=protocol,
            cases=cases,
            sources=sources,
            runner=runner_hash,
            environment=environment,
            workers=jobs,
        )
    )
    manifest_path = output / "study.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                "Study already exists; use --resume or a new output directory"
            )
        if json.loads(manifest_path.read_text())["identity"] != identity:
            raise ValueError(
                "Cannot resume a different protocol, runner or solver source"
            )
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = dict(
        identity=identity,
        protocol=protocol,
        cases=cases,
        spatial=spatial,
        temporal=temporal,
        solver_sources=sources,
        runner_sha256=runner_hash,
        git_commit=commit,
        environment=environment,
        workers=jobs,
        initial_conditions="Same analytic profiles, zero concentration noise and zero defect event rate",
    )
    # Save the experiment design before the first simulation or result is observed.
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    completed, pending = {}, []
    for index, case in enumerate(cases, 1):
        signature = _digest(
            dict(config=case["config"], sources=sources, identity=identity)
        )
        cached = output / case["id"] / "case.json"
        print(f"Case {index}/{len(cases)}: {case['id']}", flush=True)
        if resume and cached.exists():
            result = json.loads(cached.read_text())
            if result["signature"] != signature:
                raise ValueError(
                    "Cached case does not match its resolved settings/source"
                )
            print("  reusing matching completed case", flush=True)
            completed[case["id"]] = result
        else:
            pending.append((case, signature))
    if jobs == 1:
        for case, signature in pending:
            completed[case["id"]] = run_case(case, output, signature)
    else:
        # Each process owns its RNG/state and writes only its own case directory.
        # Restore protocol order after collection; scheduling cannot change metrics.
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(run_case, case, output, signature): case["id"]
                for case, signature in pending
            }
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
    results = [completed[case["id"]] for case in cases]
    assessment = assess(results, spatial, temporal, protocol)
    write_report(output, results, assessment, protocol)
    print(
        f"ASSESSMENT: {'PASS' if assessment['overall_pass'] else 'FAIL'}; report: {output / 'report.md'}",
        flush=True,
    )
    return dict(results=results, **assessment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config", type=Path, default=ROOT / "configs/convergence.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/convergence")
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional smoke-test horizon; recorded in protocol",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent case processes; keep the same count when resuming",
    )
    args = parser.parse_args()
    protocol = json.loads(args.study_config.read_text())
    base_path = args.study_config.resolve().parent / protocol["simulation_config"]
    base = json.loads(base_path.read_text())
    if args.duration is not None:
        protocol["duration"] = args.duration
    run_study(base, protocol, args.output, args.resume, args.jobs)


if __name__ == "__main__":
    main()
