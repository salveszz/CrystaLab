"""Archive fresh-process CPU/hybrid-GPU flow runs against a frozen protocol."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_worker(protocol, backend, n, output):
    import numpy as np
    from simulation.flow import IncompressibleFlow
    from simulation.linear import using_product_backend
    from simulation.solute import FlowSoluteSimulation
    from simulation.units import FluidPropertiesSI
    from simulation.vulkan_spmv import VulkanSpMV
    from check_flow import abc_velocity, abc_pressure

    output.mkdir(exist_ok=False)
    reference = json.loads((ROOT / protocol["fluid_reference"]).read_text())
    properties = FluidPropertiesSI(**reference["fluid"])
    started = time.perf_counter()
    gpu = VulkanSpMV(expected_device_name=protocol["expected_gpu_name"],
                     max_buffer_bytes=protocol["gpu_buffer_budget_bytes"]) if backend == "gpu" else None
    try:
        flow = IncompressibleFlow([protocol["length_m"]]*3, [n]*3, properties,
                                  periodic_xyz=(True,)*3)
        L, speed = protocol["length_m"], protocol["coefficient_speed_m_per_s"]
        nu = properties.kinematic_viscosity_m2_per_s
        flow.set_velocity([abc_velocity(flow.face_positions(i), 0, L, speed, nu)[..., i]
                           for i in range(3)])
        flow.set_pressure(abc_pressure(flow, 0, L, speed))
        k = 2*np.pi/L
        # Exact cell averages of the initial product of cosines.
        average = np.prod([np.sinc(k*w/(2*np.pi))[0] for w in flow.grid.widths])
        initial = protocol["concentration_mean_kg_per_m3"] + protocol["concentration_amplitude_kg_per_m3"] * average * np.prod(np.cos(k*flow.cell_positions()), axis=-1)
        model = FlowSoluteSimulation(flow, initial)
        initial_mass = model.solute.mass_kg
        initial_mean = initial_mass / float(flow.cell_volumes.sum())
        initialization_s = time.perf_counter() - started
        step_started = time.perf_counter()
        with using_product_backend(gpu) if gpu is not None else nullcontext():
            for _ in range(protocol["steps"]):
                model.step(protocol["dt_s"])
        stepping_s = time.perf_counter() - step_started
        cold_total_s = time.perf_counter() - started
        # Post-run references, diagnostics and serialization are excluded for both.
        error, scale = 0., 0.
        fields = dict(dynamic_pressure_Pa=flow.pressure,
                      concentration_kg_per_m3=model.solute.concentration,
                      initial_concentration_kg_per_m3=initial,
                      cell_volumes_m3=flow.cell_volumes)
        for i, c in enumerate(flow.components):
            exact = abc_velocity(flow.face_positions(i), flow.time_s, L, speed, nu)[..., i]
            ix = tuple(c.cells.T)
            error += float(np.sum(c.volumes*(flow.velocity[i][ix]-exact[ix])**2))
            scale += float(np.sum(c.volumes*exact[ix]**2))
            fields[f"velocity_{i}_m_per_s"] = flow.velocity[i]
        pressure_ref = abc_pressure(flow, flow.time_s, L, speed)
        concentration = model.solute.concentration
        indicator = np.maximum(concentration-initial_mean, 0)/initial_mean
        fields["positive_concentration_excess_indicator"] = indicator
        np.savez_compressed(output / "fields.npz", **fields)
        metrics = dict(relative_velocity_reference_error=float(np.sqrt(error/scale)),
                       relative_pressure_reference_error=float(np.sqrt(np.sum(flow.cell_volumes*(flow.pressure-pressure_ref)**2)/np.sum(flow.cell_volumes*pressure_ref**2))),
                       relative_solute_mass_drift=abs(model.solute.mass_kg-initial_mass)/initial_mass,
                       minimum_kg_per_m3=float(concentration.min()), maximum_kg_per_m3=float(concentration.max()),
                       initial_minimum_kg_per_m3=float(initial.min()), initial_maximum_kg_per_m3=float(initial.max()),
                       initial_volume_mean_kg_per_m3=initial_mean,
                       **model.measurements())
        criteria = protocol["criteria"]
        checks = {
            "velocity_reference": metrics["relative_velocity_reference_error"] <= criteria["max_relative_velocity_reference_error"],
            "pressure_reference": metrics["relative_pressure_reference_error"] <= criteria["max_relative_pressure_reference_error"],
            "divergence": metrics["scaled_divergence"] <= criteria["max_scaled_divergence"],
            "mass": metrics["relative_solute_mass_drift"] <= criteria["max_relative_solute_mass_drift"],
            "bounds": metrics["minimum_kg_per_m3"] >= metrics["initial_minimum_kg_per_m3"]-criteria["concentration_bound_slack_kg_per_m3"] and metrics["maximum_kg_per_m3"] <= metrics["initial_maximum_kg_per_m3"]+criteria["concentration_bound_slack_kg_per_m3"],
            "actual_gpu_work": gpu is None or (gpu.stats["solves"] > 0 and gpu.stats["dispatches"] > 0),
        }
        report = dict(backend=backend, n=n, time_s=flow.time_s,
                      initialization_s=initialization_s, stepping_s=stepping_s, cold_total_s=cold_total_s,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      fields_sha256=digest(output/"fields.npz"), metrics=metrics,
                      checks={k: bool(value) for k,value in checks.items()},
                      device=gpu.info if gpu is not None else None,
                      gpu_stats=dict(gpu.stats) if gpu is not None else None,
                      gpu_setup_wall_s=gpu.setup_wall_s if gpu is not None else None)
        write(output / "report.json", report)
        if not all(checks.values()):
            raise ArithmeticError(f"Frozen accuracy checks failed: {checks}")
    finally:
        if gpu is not None:
            gpu.close()


def probe(protocol, output):
    import numpy as np
    from scipy.sparse import diags
    from simulation.vulkan_spmv import VulkanSpMV
    from simulation.linear import using_product_backend, solve_spd

    rng = np.random.default_rng(20261112)
    checks = []
    with VulkanSpMV(expected_device_name=protocol["expected_gpu_name"]) as gpu:
        for n in (1, 127, 128, 129, 4097):
            a = diags([-np.ones(n-1),4*np.ones(n),-np.ones(n-1)], [-1,0,1], shape=(n,n)).tocsr()
            x = rng.normal(size=n)
            reference, actual = a@x, gpu.matvec(a, x)
            scaled = float(np.max(abs(actual-reference))/max(np.max(abs(reference)),1))
            checks.append(dict(n=n, scaled_error=scaled))
            if scaled > protocol["criteria"]["max_spmv_scaled_error"]:
                raise ArithmeticError("GPU sparse product verification failed")
        with using_product_backend(gpu):
            answer = solve_spd(a, a@x)
        relative = float(np.linalg.norm(answer-x)/np.linalg.norm(x))
        if relative > 1e-12:
            raise ArithmeticError("GPU CG solution failed independent known-solution check")
        before = gpu.stats["matrices"]
        a.data *= 1.5
        changed = gpu.matvec(a, x)
        if gpu.stats["matrices"] != before+1 or not np.allclose(changed, a@x, rtol=5e-14, atol=5e-14):
            raise ArithmeticError("Mutated matrix reused a stale GPU cache")
        for invalid in (np.ones(3), np.full(4097, np.nan), x.astype(complex)):
            try:
                gpu.matvec(a, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid vector was accepted")
        write(output, dict(device=gpu.info, checks=checks, known_solution_relative_error=relative,
                           stats=gpu.stats, setup_wall_s=gpu.setup_wall_s))


def run_study(protocol_path, output):
    import numpy as np

    protocol = json.loads(protocol_path.read_text())
    output.mkdir(exist_ok=False)
    write(output/"protocol.json", protocol)
    sources = [*sorted((ROOT/"src/simulation").glob("*.py")), Path(__file__),
               ROOT/"scripts/check_flow.py", ROOT/protocol["fluid_reference"], protocol_path,
               ROOT/"pyproject.toml"]
    cpu_models = [line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    manifest = dict(status="running", protocol_sha256=digest(protocol_path),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    environment=dict(python=sys.version, platform=platform.platform(),
                        cpu=cpu_models[0] if cpu_models else platform.processor(),
                        packages={p: importlib.metadata.version(p) for p in ("numpy","scipy","vulkan","cffi")}),
                    timing_scope="Host perf_counter around construction+initialization+stepping; separately stepping. Process startup/imports, diagnostics, archive writes and teardown excluded equally. Each run is a fresh process. GPU submission/fence and x/y copies are inside stepping.",
                    memory_scope="Linux ru_maxrss is process high-water RSS, not incremental solver RAM. GPU allocated buffer bytes include Vulkan alignment; driver/pipeline memory is not measured. Integrated GPU shared RAM may overlap RSS. No dedicated VRAM claim.",
                    runs=[])
    write(output/"manifest.json", manifest)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    def child(arguments, log):
        with log.open("w") as handle:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), "--protocol",str(protocol_path), *arguments],
                                  cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    try:
        if child(["--probe", "--output",str(output/"gpu_probe.json")], output/"gpu_probe.log"):
            raise ArithmeticError("GPU probe failed; inspect preserved log")
        for n in protocol["counts"]:
            order = [("warmup",0,b) for b in ("cpu","gpu")]
            for repeat in range(protocol["repeats"]):
                order += [("measured",repeat,b) for b in (("cpu","gpu") if repeat%2==0 else ("gpu","cpu"))]
            for phase, repeat, backend in order:
                name = f"n{n}_{phase}_{repeat}_{backend}"
                run_output = output/name
                code = child(["--worker",backend,"--n",str(n),"--output",str(run_output)], output/f"{name}.log")
                manifest["runs"].append(dict(name=name, n=n, phase=phase, repeat=repeat, backend=backend, exit_code=code))
                write(output/"manifest.json",manifest)
                print(f"{name}: exit {code}",flush=True)
                if code:
                    raise ArithmeticError(f"Worker {name} failed; inspect retained output")
        comparisons, timings = [], []
        for n in protocol["counts"]:
            rows = [r for r in manifest["runs"] if r["n"]==n and r["phase"]=="measured"]
            timing = dict(n=n)
            for backend in ("cpu","gpu"):
                reports = [json.loads((output/r["name"]/"report.json").read_text()) for r in rows if r["backend"]==backend]
                timing[backend] = {key: [r[key] for r in reports] for key in ("cold_total_s","stepping_s","initialization_s","process_peak_rss_kib")}
            for key in ("cold_total_s","stepping_s"):
                timing[f"{key}_cpu_over_gpu_median"] = float(np.median(timing["cpu"][key])/np.median(timing["gpu"][key]))
            timings.append(timing)
            for phase, repeat in [("warmup",0), *[("measured",r) for r in range(protocol["repeats"])]]:
                base = f"n{n}_{phase}_{repeat}"
                with np.load(output/f"{base}_cpu"/"fields.npz") as cpu, np.load(output/f"{base}_gpu"/"fields.npz") as gpu:
                    velocity_difference = np.sqrt(sum(np.sum((cpu[f"velocity_{i}_m_per_s"]-gpu[f"velocity_{i}_m_per_s"])**2) for i in range(3))/sum(np.sum(cpu[f"velocity_{i}_m_per_s"]**2) for i in range(3)))
                    pressure_difference = np.linalg.norm(cpu["dynamic_pressure_Pa"]-gpu["dynamic_pressure_Pa"])/np.linalg.norm(cpu["dynamic_pressure_Pa"])
                    concentration_difference = np.max(abs(cpu["concentration_kg_per_m3"]-gpu["concentration_kg_per_m3"]))
                    indicator_difference = np.max(abs(cpu["positive_concentration_excess_indicator"]-gpu["positive_concentration_excess_indicator"]))
                c=protocol["criteria"]
                passed = velocity_difference <= c["max_cpu_gpu_relative_velocity_difference"] and pressure_difference <= c["max_cpu_gpu_relative_pressure_difference"] and concentration_difference <= c["max_cpu_gpu_concentration_difference_kg_per_m3"] and indicator_difference <= c["max_cpu_gpu_indicator_difference"]
                comparisons.append(dict(n=n,phase=phase,repeat=repeat,relative_velocity_difference=float(velocity_difference),relative_pressure_difference=float(pressure_difference),concentration_difference_kg_per_m3=float(concentration_difference),indicator_difference=float(indicator_difference),passed=bool(passed)))
        write(output/"summary.json",dict(timings=timings,comparisons=comparisons,all_comparisons_pass=all(r["passed"] for r in comparisons)))
        if not all(r["passed"] for r in comparisons):
            raise ArithmeticError("CPU/GPU comparison failed frozen tolerance")
        manifest["status"]="complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write(output/"manifest.json",manifest)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=ROOT/"configs/vulkan_flow_experiment.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",choices=("cpu","gpu"))
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--n",type=int)
    args=parser.parse_args()
    if args.worker:
        run_worker(json.loads(args.protocol.read_text()),args.worker,args.n,args.output)
    elif args.probe:
        probe(json.loads(args.protocol.read_text()),args.output)
    else:
        run_study(args.protocol.resolve(),args.output.resolve())
