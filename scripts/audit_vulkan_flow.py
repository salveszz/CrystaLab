"""Independently reconstruct references, invariants and timings from saved fields.

Does not import the flow solver, benchmark producer or GPU backend. It does not
rerun inference/compute or independently measure device activity or wall time.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(study, output):
    if output.exists():
        raise FileExistsError("Choose a fresh audit output")
    manifest = json.loads((study/"manifest.json").read_text())
    protocol = json.loads((study/"protocol.json").read_text())
    summary = json.loads((study/"summary.json").read_text())
    assert manifest["status"] == "complete"
    for path, expected in manifest["source_sha256"].items():
        assert sha(ROOT/path) == expected, path
    assert sha(ROOT/"configs/vulkan_flow_experiment.json") == manifest["protocol_sha256"]
    fluid = json.loads((ROOT/protocol["fluid_reference"]).read_text())["fluid"]
    c = protocol["criteria"]
    rows, hashes, raw = [], {}, {}
    expected_order = []
    for n in protocol["counts"]:
        expected_order += [f"n{n}_warmup_0_{b}" for b in ("cpu","gpu")]
        for r in range(protocol["repeats"]):
            expected_order += [f"n{n}_measured_{r}_{b}" for b in (("cpu","gpu") if r%2==0 else ("gpu","cpu"))]
    assert [r["name"] for r in manifest["runs"]] == expected_order
    for run in manifest["runs"]:
        assert run["exit_code"] == 0
        name, n = run["name"], run["n"]
        report = json.loads((study/name/"report.json").read_text())
        fields_path = study/name/"fields.npz"
        assert sha(fields_path) == report["fields_sha256"]
        hashes[f"{name}/fields.npz"] = sha(fields_path)
        hashes[f"{name}/report.json"] = sha(study/name/"report.json")
        with np.load(fields_path) as arrays:
            fields = {k: arrays[k].copy() for k in arrays.files}
        raw[name] = fields
        L, h = protocol["length_m"], protocol["length_m"]/n
        t = protocol["steps"]*protocol["dt_s"]
        assert report["time_s"] == t
        amplitude = protocol["coefficient_speed_m_per_s"]*np.exp(-fluid["dynamic_viscosity_Pa_s"]/fluid["density_kg_per_m3"]*(2*np.pi/L)**2*t)
        cell = (np.arange(n)+0.5)*2*np.pi/n
        z, y, x = np.meshgrid(cell, cell, cell, indexing="ij")
        # Each component is constant along its own normal coordinate in ABC flow.
        exact = (amplitude*(np.sin(z)+np.cos(y)), amplitude*(np.sin(x)+np.cos(z)), amplitude*(np.sin(y)+np.cos(x)))
        active = []
        divergence = np.zeros((n,n,n))
        face_scale = 0.
        for i in range(3):
            field = fields[f"velocity_{i}_m_per_s"]
            axis = 2-i
            expected_shape = [n]*3
            expected_shape[axis] += 1
            assert field.shape == tuple(expected_shape) and np.isfinite(field).all()
            assert np.array_equal(np.take(field,0,axis=axis),np.take(field,-1,axis=axis))
            # One periodic end face is an alias, not another independent unknown.
            active.append(np.take(field,range(n),axis=axis))
            divergence += np.diff(field,axis=axis)/h
            face_scale = max(face_scale,float(np.max(abs(field)))/h)
        velocity_error = float(np.sqrt(sum(np.sum((a-e)**2) for a,e in zip(active,exact))/sum(np.sum(e**2) for e in exact)))
        pressure = fields["dynamic_pressure_Pa"]
        reference_pressure = -0.5*fluid["density_kg_per_m3"]*sum(e**2 for e in exact)
        reference_pressure -= reference_pressure.mean()
        pressure_error = float(np.linalg.norm(pressure-reference_pressure)/np.linalg.norm(reference_pressure))
        scaled_divergence = float(np.max(abs(divergence))/face_scale)
        initial = protocol["concentration_mean_kg_per_m3"] + protocol["concentration_amplitude_kg_per_m3"]*np.sinc(1/n)**3*np.cos(x)*np.cos(y)*np.cos(z)
        concentration = fields["concentration_kg_per_m3"]
        volumes = fields["cell_volumes_m3"]
        assert np.allclose(volumes,h**3,rtol=1e-14,atol=0)
        assert np.allclose(fields["initial_concentration_kg_per_m3"],initial,rtol=0,atol=1e-12)
        mean = np.sum(initial*volumes)/np.sum(volumes)
        indicator = np.maximum(concentration-mean,0)/mean
        assert np.allclose(fields["positive_concentration_excess_indicator"],indicator,rtol=0,atol=1e-14)
        mass_drift = float(abs(np.sum(concentration*volumes)-np.sum(initial*volumes))/np.sum(initial*volumes))
        assert velocity_error <= c["max_relative_velocity_reference_error"]
        assert pressure_error <= c["max_relative_pressure_reference_error"]
        assert scaled_divergence <= c["max_scaled_divergence"]
        assert mass_drift <= c["max_relative_solute_mass_drift"]
        assert concentration.min() >= initial.min()-c["concentration_bound_slack_kg_per_m3"]
        assert concentration.max() <= initial.max()+c["concentration_bound_slack_kg_per_m3"]
        assert np.isclose(velocity_error,report["metrics"]["relative_velocity_reference_error"],rtol=1e-11,atol=1e-14)
        assert np.isclose(pressure_error,report["metrics"]["relative_pressure_reference_error"],rtol=1e-11,atol=1e-14)
        assert np.isclose(scaled_divergence,report["metrics"]["scaled_divergence"],rtol=0,atol=1e-14)
        assert np.isclose(mass_drift,report["metrics"]["relative_solute_mass_drift"],rtol=0,atol=1e-14)
        assert all(report["checks"].values())
        assert report["cold_total_s"] >= report["initialization_s"]+report["stepping_s"]-1e-6
        if run["backend"] == "gpu":
            assert report["device"]["name"] == protocol["expected_gpu_name"]
            assert report["device"]["device_type"] in (1,2) and report["device"]["shader_float64"]
            stats=report["gpu_stats"]
            assert stats["dispatches"]>0 and stats["solves"]>0
            assert stats["downloaded_bytes"]>0 and stats["uploaded_bytes"]>0
            assert 0<stats["allocated_buffer_bytes"]<=protocol["gpu_buffer_budget_bytes"]
        rows.append(dict(name=name,velocity_reference_error=velocity_error,pressure_reference_error=pressure_error,
                         scaled_divergence=scaled_divergence,relative_mass_drift=mass_drift))
    comparisons=[]
    for row in summary["comparisons"]:
        base=f"n{row['n']}_{row['phase']}_{row['repeat']}"
        a,b=raw[base+"_cpu"],raw[base+"_gpu"]
        values=dict(
            relative_velocity_difference=float(np.sqrt(sum(np.sum((a[f"velocity_{i}_m_per_s"]-b[f"velocity_{i}_m_per_s"])**2) for i in range(3))/sum(np.sum(a[f"velocity_{i}_m_per_s"]**2) for i in range(3)))),
            relative_pressure_difference=float(np.linalg.norm(a["dynamic_pressure_Pa"]-b["dynamic_pressure_Pa"])/np.linalg.norm(a["dynamic_pressure_Pa"])),
            concentration_difference_kg_per_m3=float(np.max(abs(a["concentration_kg_per_m3"]-b["concentration_kg_per_m3"]))),
            indicator_difference=float(np.max(abs(a["positive_concentration_excess_indicator"]-b["positive_concentration_excess_indicator"]))))
        for key,value in values.items():
            assert value == row[key],key
        assert values["relative_velocity_difference"] <= c["max_cpu_gpu_relative_velocity_difference"]
        assert values["relative_pressure_difference"] <= c["max_cpu_gpu_relative_pressure_difference"]
        assert values["concentration_difference_kg_per_m3"] <= c["max_cpu_gpu_concentration_difference_kg_per_m3"]
        assert values["indicator_difference"] <= c["max_cpu_gpu_indicator_difference"]
        assert row["passed"]
        comparisons.append(dict(base=base,**values))
    assert len(comparisons)==len(protocol["counts"])*(protocol["repeats"]+1)
    assert {r["base"] for r in comparisons} == {f"n{n}_{phase}_{repeat}" for n in protocol["counts"] for phase,repeat in [("warmup",0),*[("measured",r) for r in range(protocol["repeats"])]]}
    timings=[]
    for timing in summary["timings"]:
        n=timing["n"]
        for backend in ("cpu","gpu"):
            reports=[json.loads((study/r["name"]/"report.json").read_text()) for r in manifest["runs"] if r["n"]==n and r["phase"]=="measured" and r["backend"]==backend]
            for key,values in timing[backend].items():
                assert values == [r[key] for r in reports]
        ratios={key:float(np.median(timing["cpu"][key])/np.median(timing["gpu"][key])) for key in ("cold_total_s","stepping_s")}
        for key,value in ratios.items():
            assert value==timing[f"{key}_cpu_over_gpu_median"]
        timings.append(dict(n=n,ratios_cpu_over_gpu=ratios))
    result=dict(status="passed",study=str(study.relative_to(ROOT)),auditor_sha256=sha(__file__),
                source_hashes_checked=len(manifest["source_sha256"]),runs_checked=len(rows),comparisons=comparisons,
                field_checks=rows,timings=timings,artifact_sha256=hashes,
                limits="Independent field/reference/map/mass/periodic/divergence/aggregate reconstruction. Timings and hardware dispatch counters are recorded observations, not independently remeasured. Heat/growth/physical-risk/quality validation is absent.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],runs_checked=len(rows),comparisons=len(comparisons))))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.study.resolve(),args.output.resolve())
