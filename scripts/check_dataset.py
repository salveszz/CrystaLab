# FILE PURPOSE: Measure resolution sensitivity and action responses for a declared
# synthetic-data protocol before using its labels. Raw runs, thresholds, source
# snapshots and failed gates remain inspectable. No fitting or model scoring.
"""Numerical/action audit of the synthetic growth data mode."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from environment.dataset import (  # noqa: E402
    _episode,
    build_cases,
    canonical,
    digest,
    read_json,
    source_identity,
    snapshot_sources,
    verify_episode,
    write_json,
)
from simulation.tank import SimulationConfig


def study(protocol, settings, output, jobs=1, resume=False):
    """Compare every selected paired recipe at three grids and three timesteps."""
    if set(settings) != {
        "groups",
        "recipes",
        "grids",
        "timesteps",
        "boundary_layer_thickness",
        "relative_tolerance",
    }:
        raise ValueError("Invalid sensitivity settings")
    grids, timesteps = settings["grids"], settings["timesteps"]
    if (
        len(grids) != 3
        or len(timesteps) != 3
        or any(
            len(g) != 3
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in g)
            for g in grids
        )
        or any(any(b <= a for a, b in zip(g1, g2)) for g1, g2 in zip(grids, grids[1:]))
        or any(not np.isfinite(dt) or dt <= 0 for dt in timesteps)
        or any(b >= a for a, b in zip(timesteps, timesteps[1:]))
        or not 0 < settings["relative_tolerance"] < 1
        or not np.isfinite(settings["boundary_layer_thickness"])
        or settings["boundary_layer_thickness"] <= 0
    ):
        raise ValueError(
            "Declare three increasing grids, decreasing positive dt, fixed layer and tolerance"
        )
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be positive integer")
    base_cases = build_cases(protocol)
    for case in base_cases:
        cfg = case["simulation"]
        if (
            [cfg[n] for n in ("nx", "ny", "nz")] != grids[-1]
            or cfg["dt"] != timesteps[0]
            or cfg["concentration_noise"] != 0
            or cfg["defect_rate"] != 0
            or not np.isclose(
                cfg["boundary_layer_cells"]
                * max(
                    cfg["width"] / cfg["nx"],
                    cfg["depth"] / cfg["ny"],
                    cfg["height"] / cfg["nz"],
                ),
                settings["boundary_layer_thickness"],
                rtol=1e-13,
                atol=0,
            )
        ):
            raise ValueError(
                "Pilot must match finest grid/coarsest dt, fixed sampling layer and deterministic physics"
            )
    selected = [
        c
        for c in base_cases
        if c["group_id"] in settings["groups"] and c["recipe_id"] in settings["recipes"]
    ]
    if len(selected) != len(settings["groups"]) * len(settings["recipes"]):
        raise ValueError("Sensitivity group/recipe selection not found or duplicated")
    output = Path(output)
    identity = dict(protocol=protocol, settings=settings, source=source_identity())
    if resume:
        if read_json(output / "study.json")["identity"] != identity:
            raise ValueError(
                "Study resume requires identical sources, environment and settings"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_sources(identity["source"], output)
    write_json(
        output / "study.json",
        dict(identity=identity, identity_hash=digest(identity), jobs=jobs),
    )
    cases, mapping = [], {}
    for base in selected:
        for kind, values in (("grid", grids), ("dt", timesteps)):
            for i, value in enumerate(values):
                # Finest spatial / coarsest time is deliberately one shared run.
                key = (
                    f"{base['id']}__grid_2"
                    if kind == "dt" and i == 0
                    else f"{base['id']}__{kind}_{i}"
                )
                mapping[base["id"], kind, i] = key
                if any(c["id"] == key for c in cases):
                    continue
                case = deepcopy(base)
                cfg = dict(case["simulation"])
                cfg.update(
                    zip(("nx", "ny", "nz"), value if kind == "grid" else grids[-1])
                )
                cfg["dt"] = timesteps[0] if kind == "grid" else value
                cfg["boundary_layer_cells"] = settings[
                    "boundary_layer_thickness"
                ] / max(
                    cfg["width"] / cfg["nx"],
                    cfg["depth"] / cfg["ny"],
                    cfg["height"] / cfg["nz"],
                )
                case.update(id=key, simulation=asdict(SimulationConfig(**cfg)))
                case["identity"] = digest(
                    {k: v for k, v in case.items() if k != "identity"}
                )
                cases.append(case)
    tasks = [(c, str(output)) for c in cases]
    if jobs == 1:
        for task in tasks:
            _episode(task)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            list(executor.map(_episode, tasks))
    results = {c["id"]: verify_episode(output / "episodes" / c["id"], c) for c in cases}
    comparisons = []
    for base in selected:
        for kind in ("grid", "dt"):
            for a, b in ((0, 1), (1, 2)):
                coarse, fine = (results[mapping[base["id"], kind, i]] for i in (a, b))
                left = np.array([s["labels"]["volume_gain"] for s in coarse["samples"]])
                right = np.array([s["labels"]["volume_gain"] for s in fine["samples"]])
                # Compare each actual forecast increment; never hide early-time errors
                # by normalizing against the large initial crystal volume.
                label_error = float(
                    np.max(abs(left - right) / np.maximum(abs(right), 1e-12))
                )
                lc, lf = coarse["truth"], fine["truth"]
                dc = np.array(lc[-1]["face_offsets"]) - lc[0]["face_offsets"]
                df = np.array(lf[-1]["face_offsets"]) - lf[0]["face_offsets"]
                face_error = float(
                    np.max(abs(dc - df)) / max(float(np.max(abs(df))), 1e-12)
                )
                comparisons.append(
                    dict(
                        base=base["id"],
                        kind=kind,
                        pair=[a, b],
                        max_relative_label_error=label_error,
                        face_advance_error=face_error,
                        pass_gate=max(label_error, face_error)
                        <= settings["relative_tolerance"],
                    )
                )
    actions = []
    for group in settings["groups"]:
        selected_group = [b for b in selected if b["group_id"] == group]
        reference = next(
            (b for b in selected_group if b["recipe_id"] == "baseline"), None
        )
        if reference is None:
            raise ValueError("Action audit needs a baseline recipe")
        ref = results[mapping[reference["id"], "grid", 2]]
        for base in selected_group:
            result = results[mapping[base["id"], "grid", 2]]
            delta = np.array(
                [s["labels"]["volume_gain"] for s in result["samples"]]
            ) - np.array([s["labels"]["volume_gain"] for s in ref["samples"]])
            error, time_error = 0.0, 0.0
            for item in (base, reference):
                medium, fine = (results[mapping[item["id"], "grid", i]] for i in (1, 2))
                error += max(
                    abs(a["labels"]["volume_gain"] - b["labels"]["volume_gain"])
                    for a, b in zip(medium["samples"], fine["samples"])
                )
                finer_time = results[mapping[item["id"], "dt", 1]]
                time_error += max(
                    abs(a["labels"]["volume_gain"] - b["labels"]["volume_gain"])
                    for a, b in zip(fine["samples"], finer_time["samples"])
                )
            actions.append(
                dict(
                    group=group,
                    recipe=base["recipe_id"],
                    label_differences_from_baseline=delta.tolist(),
                    max_pairwise_grid_difference_estimate=error,
                    max_pairwise_time_difference_estimate=time_error,
                    effects_exceed_summed_resolution_estimate=bool(
                        np.all(abs(delta) > error + time_error)
                    )
                    if base != reference
                    else None,
                    interpretation="Illustrative model response, not a physical optimum or calibrated actuator",
                )
            )
    report = dict(
        identity_hash=digest(identity),
        status="pass"
        if all(
            c["pass_gate"]
            for c in comparisons
            if c["pair"] == [1, 2] or c["kind"] == "dt"
        )
        else "fail",
        comparisons=comparisons,
        actions=actions,
        summed_episode_seconds=sum(
            r["metadata"]["elapsed_seconds"] for r in results.values()
        ),
        cases=len(cases),
        settings=settings,
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_growth.json")
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("configs/synthetic_growth_verification.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = study(
        read_json(args.config),
        read_json(args.settings),
        args.output,
        args.jobs,
        args.resume,
    )
    print(canonical(report))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
