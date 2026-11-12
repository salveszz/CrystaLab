# FILE PURPOSE: Compare local VQE and classical solvers for declared KDP-related Al clusters.
# Fixed designed geometry, frozen inactive electrons and tiny active spaces are
# explicit approximations. This module never supplies growth labels or commands.
"""Optional molecular active-space experiment, isolated from the online system."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
VERSIONS={"qiskit":"2.6.2","qiskit-nature":"0.8.0","qiskit-algorithms":"0.4.0",
          "pyscf":"2.15.0","numpy":"2.5.0","scipy":"1.18.0"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    def scalar(item):
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(f"Unsupported record value: {type(item).__name__}")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=scalar)+"\n")
    temporary.replace(path)


def require_runtime():
    versions={p:importlib.metadata.version(p) for p in VERSIONS}
    if versions!=VERSIONS:
        raise ValueError(f"Quantum runtime differs from pinned experiment: {versions}")
    return versions


def make_geometry(protocol,case):
    """Construct the explicitly designed octahedral/bidentate starting scaffold."""
    g=protocol["geometry"]
    radius=g["al_oxygen_distance"]
    atoms=[("Al",np.zeros(3))]
    donors=[]
    if case["case_id"]=="al_hexaaqua_reference":
        waters=[radius*np.array(v,float) for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
    elif case["case_id"]=="al_dihydrogen_phosphate":
        po=g["phosphate_donor_PO_distance"]
        transverse=po*math.sqrt(2/3)
        x=math.sqrt(radius**2-transverse**2)
        phosphorus=np.array([x+po/math.sqrt(3),0,0])
        atoms.append(("P",phosphorus))
        for sign in (1,-1):
            donors.append(len(atoms))
            atoms.append(("O",np.array([x,sign*transverse,0])))
        for sign in (1,-1):
            axis=np.array([1/math.sqrt(3),0,sign*math.sqrt(2/3)])
            oxygen=phosphorus+g["phosphate_OH_PO_distance"]*axis
            atoms.append(("O",oxygen))
            angle=math.radians(180-g["phosphate_POH_angle_deg"])
            direction=math.cos(angle)*axis+math.sin(angle)*np.array([0,sign,0])
            atoms.append(("H",oxygen+g["OH_distance"]*direction))
        waters=[np.array([-x,transverse,0]),np.array([-x,-transverse,0]),np.array([0,0,radius]),np.array([0,0,-radius])]
    else:
        raise ValueError("Unknown predeclared cluster")
    for oxygen in waters:
        if case["case_id"]=="al_hexaaqua_reference":
            donors.append(len(atoms))
        atoms.append(("O",oxygen))
        axis=oxygen/np.linalg.norm(oxygen)
        auxiliary=np.array([0,0,1.]) if abs(axis[2])<.9 else np.array([0,1.,0])
        tangent=np.cross(axis,auxiliary)
        tangent/=np.linalg.norm(tangent)
        half=math.radians(g["water_HOH_angle_deg"]/2)
        for sign in (1,-1):
            direction=math.cos(half)*axis+sign*math.sin(half)*tangent
            atoms.append(("H",oxygen+g["OH_distance"]*direction))
    positions=np.array([p for element,p in atoms])
    distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    if not np.isfinite(positions).all() or distances.min()<g["minimum_any_atom_distance"]:
        raise ValueError("Designed geometry contains invalid/overlapping atoms")
    charges={"Al":13,"P":15,"O":8,"H":1}
    electrons=sum(charges[e] for e,p in atoms)-case["charge"]
    if electrons!=case["expected_electrons"] or electrons%2 or case["spin_2S"]!=0:
        raise ValueError("Cluster formula, charge and closed-shell electron count disagree")
    return atoms,donors


def _snapshot(output,protocol_path):
    (output/"source_snapshot").mkdir()
    shutil.copyfile(__file__,output/"source_snapshot/impurity_quantum.py")
    shutil.copyfile(protocol_path,output/"protocol.json")
    return dict(module_sha256=sha(__file__),protocol_sha256=sha(protocol_path))


def classical_case(protocol_path,case_id,output):
    from pyscf import gto,scf,mcscf,ao2mo,fci,lib
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.properties import AngularMomentum

    versions=require_runtime()
    protocol_path,output=Path(protocol_path).resolve(),Path(output).resolve()
    protocol=json.loads(protocol_path.read_text())
    case=next(c for c in protocol["cases"] if c["case_id"]==case_id)
    for source in protocol["sources"]:
        if sha(ROOT/source["local_path"])!=source["sha256"]:
            raise ValueError("Chemical method reference bytes changed")
    atoms,donors=make_geometry(protocol,case)
    output.mkdir(parents=True,exist_ok=False)
    identity=_snapshot(output,protocol_path)
    report=dict(status="running",case=case,identity=identity,versions=versions,
        geometry_source=protocol["geometry"]["source"],basis=protocol["electronic_structure"]["basis"],
        scope=protocol["purpose"])
    save(output/"manifest.json",report)
    started=time.perf_counter()
    try:
        settings=protocol["electronic_structure"]
        lib.num_threads(settings["threads"])
        xyz=str(len(atoms))+"\nFixed idealized cluster; not optimized or measured\n"+"".join(f"{e} {p[0]:.15g} {p[1]:.15g} {p[2]:.15g}\n" for e,p in atoms)
        (output/"geometry.xyz").write_text(xyz)
        save(output/"geometry.json",dict(elements=[e for e,p in atoms],positions_Angstrom=[p.tolist() for e,p in atoms],
            donor_oxygen_atom_indices=donors,charge=case["charge"],spin_2S=case["spin_2S"],designed_not_optimized=True))
        mol=gto.M(atom=[(e,p.tolist()) for e,p in atoms],unit="Angstrom",basis=settings["basis"],
            charge=case["charge"],spin=case["spin_2S"],cart=False,symmetry=False,
            max_memory=settings["max_memory_MB"],verbose=4,output=str(output/"scf.log"))
        if mol.nelectron!=case["expected_electrons"]:
            raise ValueError("Electronic driver changed the expected electron count")
        mean=scf.RHF(mol)
        mean.conv_tol=settings["scf_energy_tolerance_Ha"]
        mean.conv_tol_grad=settings["scf_gradient_tolerance"]
        mean.max_cycle=settings["scf_max_cycles"]
        mean.kernel()
        newton_used=False
        if not mean.converged:
            mean=mean.newton()
            mean.max_cycle=settings["newton_fallback_max_cycles"]
            mean.kernel()
            newton_used=True
        if not mean.converged:
            raise ArithmeticError("RHF failed its frozen convergence criteria")
        _,_,stable,_=mean.stability(return_status=True)
        if settings["internal_stability_required"] and not stable:
            raise ArithmeticError("Closed-shell reference is internally unstable")
        nuclear_gradient=mean.nuc_grad_method().kernel()
        scf_wall=time.perf_counter()-started

        overlap=mean.get_ovlp()
        values,vectors=np.linalg.eigh(overlap)
        if values.min()<=1e-10:
            raise ArithmeticError("AO basis is too linearly dependent for the declared selection")
        lowdin=(vectors*np.sqrt(values))@vectors.T@mean.mo_coeff
        if not np.allclose(lowdin.T@lowdin,np.eye(lowdin.shape[1]),rtol=0,atol=1e-8):
            raise ArithmeticError("MO/AO overlap normalization failed")
        active=protocol["active_space"]
        labels=mol.ao_labels(fmt=False)
        donor_ao=[i for i,label in enumerate(labels) if label[0] in donors and label[2] in active["donor_shells"]]
        metal_ao=[i for i,label in enumerate(labels) if label[0]==0 and label[2] in active["al_acceptor_shells"]]
        if not donor_ao or not metal_ao:
            raise ValueError("Declared chemical AO subspaces are absent in the basis")
        donor_weight=np.sum(lowdin[donor_ao]**2,axis=0)
        acceptor_weight=np.sum(lowdin[metal_ao]**2,axis=0)
        occupied=list(np.flatnonzero(mean.mo_occ>1))
        virtual=list(np.flatnonzero(mean.mo_occ==0))
        chosen_occupied=sorted(sorted(occupied,key=lambda i:(-donor_weight[i],i))[:2])
        chosen_virtual=sorted(sorted(virtual,key=lambda i:(-acceptor_weight[i],i))[:2])
        chosen=chosen_occupied+chosen_virtual
        if min(donor_weight[chosen_occupied])<active["minimum_selected_donor_weight"] or min(acceptor_weight[chosen_virtual])<active["minimum_selected_acceptor_weight"]:
            raise ArithmeticError("Chosen orbitals lack the predeclared chemical character")
        cas=mcscf.CASCI(mean,4,(2,2))
        mo=cas.sort_mo(chosen,base=0)
        h1,constant=cas.get_h1eff(mo)
        h2=ao2mo.restore(1,cas.get_h2eff(mo),4)
        solver=fci.direct_spin1.FCISolver(mol)
        solver.conv_tol=1e-12
        fci_energy,ci=solver.kernel(h1,h2,4,(2,2),ecore=0.)
        if not solver.converged:
            raise ArithmeticError("Independent active-space FCI did not converge")
        mapper=JordanWignerMapper()
        operator=mapper.map(ElectronicEnergy.from_raw_integrals(h1,h2).second_q_op()).simplify(atol=1e-12)
        matrix=operator.to_matrix()
        spin_operator=mapper.map(AngularMomentum(4).second_q_ops()["AngularMomentum"])
        sector=[i for i in range(256) if (i&15).bit_count()==2 and (i>>4).bit_count()==2]
        energies,eigenvectors=np.linalg.eigh(matrix[np.ix_(sector,sector)])
        reference=eigenvectors[:,0]
        spin_squared=float(np.vdot(reference,spin_operator.to_matrix()[np.ix_(sector,sector)]@reference).real)
        criteria=protocol["criteria"]
        hermiticity=float(np.max(abs(matrix-matrix.conj().T)))
        hf_index=(1<<0)|(1<<1)|(1<<4)|(1<<5)
        hf_reconstructed=float(matrix[hf_index,hf_index].real+constant)
        checks=dict(hermitian=hermiticity<=criteria["max_matrix_hermiticity_error"],
            fci_agreement=abs(float(energies[0])-fci_energy)<=criteria["max_qubit_vs_FCI_energy_difference_Ha"],
            hf_agreement=abs(hf_reconstructed-mean.e_tot)<=criteria["max_HF_determinant_vs_SCF_difference_Ha"],
            exact_singlet=abs(spin_squared)<=criteria["max_exact_spin_squared"])
        np.savez_compressed(output/"active_integrals.npz",h1=h1,h2_chemist=h2,constant_Ha=constant,
            sector_indices=np.array(sector),exact_sector_eigenvalues_Ha=energies,exact_sector_ground_state=reference,
            fci_ci_vector=ci,mo_coefficients=mean.mo_coeff,ao_overlap=overlap,mo_energies_Ha=mean.mo_energy,
            donor_weights=donor_weight,acceptor_weights=acceptor_weight,selected_mo_indices=np.array(chosen))
        save(output/"qubit_operator.json",dict(num_qubits=operator.num_qubits,
            terms=[dict(pauli=label,real=float(value.real),imag=float(value.imag)) for label,value in operator.to_list()],
            constant_Ha=float(constant),convention="Active electronic operator only. Add constant once for nuclear plus inactive-electron energy."))
        save(output/"orbital_selection.json",dict(rule=active["selection"],donor_atom_indices=donors,
            donor_ao_indices=donor_ao,metal_ao_indices=metal_ao,ao_labels=[list(label) for label in labels],
            selected_occupied_mo_indices=chosen_occupied,selected_virtual_mo_indices=chosen_virtual,
            frozen_occupied_mo_indices=[int(i) for i in occupied if i not in chosen_occupied],
            selected=[dict(mo_index=int(i),energy_Ha=float(mean.mo_energy[i]),donor_weight=float(donor_weight[i]),acceptor_weight=float(acceptor_weight[i])) for i in chosen]))
        report.update(status="complete" if all(checks.values()) else "failed",checks=checks,
            atoms=len(atoms),electrons=mol.nelectron,atomic_orbitals=mol.nao_nr(),
            scf_converged=bool(mean.converged),newton_used=newton_used,internally_stable=bool(stable),
            scf_energy_Ha=float(mean.e_tot),scf_wall_s=scf_wall,
            fixed_geometry_max_nuclear_force_Ha_per_Bohr=float(np.max(np.linalg.norm(nuclear_gradient,axis=1))),
            active_electrons=4,spatial_orbitals=4,qubits=operator.num_qubits,pauli_terms=len(operator),
            sector_dimension=len(sector),fci_active_energy_Ha=float(fci_energy),
            qubit_sector_active_energy_Ha=float(energies[0]),constant_Ha=float(constant),
            exact_total_energy_Ha=float(energies[0]+constant),exact_spin_squared=spin_squared,
            hf_reconstructed_Ha=hf_reconstructed,hermiticity_error=hermiticity,
            elapsed_s=time.perf_counter()-started,peak_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=dict(python=platform.python_version(),platform=platform.platform(),threads=lib.num_threads()))
        mol.stdout.flush()
        mol.stdout.close()
        report["files"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"}
        save(output/"manifest.json",report)
        if not all(checks.values()):
            raise ArithmeticError(f"Classical consistency checks failed: {checks}")
        return report
    except Exception as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_s=time.perf_counter()-started)
        save(output/"manifest.json",report)
        raise


def vqe_case(classical,output,*,protocol_path):
    from qiskit import qpy,transpile
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp,Statevector
    from qiskit_algorithms import VQE
    from qiskit_algorithms.gradients import ReverseEstimatorGradient
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.circuit.library import HartreeFock,UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.properties import AngularMomentum

    require_runtime()
    classical,output=Path(classical).resolve(),Path(output).resolve()
    reference=json.loads((classical/"manifest.json").read_text())
    protocol=json.loads((classical/"protocol.json").read_text())
    if reference["status"]!="complete":
        raise ValueError("VQE requires a completed, consistent classical Hamiltonian")
    if protocol!=json.loads(Path(protocol_path).read_text()) or sha(protocol_path)!=reference["identity"]["protocol_sha256"]:
        raise ValueError("Classical Hamiltonian differs from the external frozen experiment")
    if sha(classical/"source_snapshot/impurity_quantum.py")!=reference["identity"]["module_sha256"]:
        raise ValueError("Classical producer snapshot changed")
    for name,digest in reference["files"].items():
        if sha(classical/name)!=digest:
            raise ValueError("Classical Hamiltonian artifact changed")
    raw=json.loads((classical/"qubit_operator.json").read_text())
    operator=SparsePauliOp.from_list([(t["pauli"],complex(t["real"],t["imag"])) for t in raw["terms"]])
    mapper=JordanWignerMapper()
    initial=HartreeFock(4,(2,2),mapper)
    ansatz=UCCSD(4,(2,2),mapper,reps=1,preserve_spin=True,initial_state=initial)
    circuit=transpile(ansatz, basis_gates=["rx","ry","rz","h","s","sdg","x","cx"],optimization_level=0,
                      seed_transpiler=protocol["quantum"]["seed"])
    output.mkdir(parents=True,exist_ok=False)
    (output/"source_snapshot").mkdir()
    shutil.copyfile(__file__,output/"source_snapshot/impurity_quantum.py")
    with (output/"ansatz.qpy").open("wb") as handle:
        qpy.dump(circuit,handle)
    report=dict(status="running",classical_path=str(classical.relative_to(ROOT)),classical_manifest_sha256=sha(classical/"manifest.json"),
        source_sha256=sha(__file__),protocol_sha256=reference["identity"]["protocol_sha256"],
        qubits=circuit.num_qubits,parameters=circuit.num_parameters,depth=circuit.depth(),gate_counts=dict(circuit.count_ops()),
        trials=[],simulator="Local exact statevector; no quantum hardware, shots or provider")
    save(output/"manifest.json",report)
    started=time.perf_counter()
    rng=np.random.default_rng(protocol["quantum"]["seed"])
    spin_operator=mapper.map(AngularMomentum(4).second_q_ops()["AngularMomentum"])
    criteria=protocol["criteria"]
    try:
        for trial,std in enumerate((0.,.02,.05)):
            point=np.zeros(circuit.num_parameters) if std==0 else rng.normal(0,std,circuit.num_parameters)
            trace=output/f"trial_{trial}_evaluations.jsonl"
            trial_started=time.perf_counter()
            class RecordedEstimator(StatevectorEstimator):
                # Algorithms 0.4.0 reshapes callback parameters to (-1, 1),
                # exposing only the first scalar for this single-point call.
                # Capture the actual primitive input without changing its job.
                def run(self,pubs,*,precision=None):
                    pubs=list(pubs)
                    if len(pubs)!=1:
                        raise ValueError("Trace requires one estimator publication")
                    self.evaluated_parameters=np.asarray(pubs[0][2]).reshape(-1,circuit.num_parameters).copy()
                    if len(self.evaluated_parameters)!=1:
                        raise ValueError("Trace requires one parameter point per evaluation")
                    return super().run(pubs,precision=precision)
            estimator=RecordedEstimator(default_precision=0.)
            with trace.open("w") as handle:
                def callback(count,parameters,energy,metadata):
                    handle.write(json.dumps(dict(evaluation=int(count),
                        parameters=estimator.evaluated_parameters[0].tolist(),
                        sdk_callback_parameters=np.asarray(parameters).tolist(),
                        energy_Ha=float(energy)),allow_nan=False)+"\n")
                    handle.flush()
                    if count%10==0:
                        print(json.dumps(dict(trial=trial,evaluations=int(count),active_energy_Ha=float(energy))),flush=True)
                solver=VQE(estimator,circuit,
                    SLSQP(maxiter=protocol["quantum"]["maxiter"],ftol=protocol["quantum"]["ftol"]),
                    gradient=ReverseEstimatorGradient(),initial_point=point,callback=callback)
                result=solver.compute_minimum_eigenvalue(operator)
            final=np.asarray(result.optimal_point)
            state=Statevector.from_instruction(circuit.assign_parameters(final))
            energy=float(state.expectation_value(operator).real)
            probabilities=state.probabilities()
            alpha=sum(p*(i&15).bit_count() for i,p in enumerate(probabilities))
            beta=sum(p*(i>>4).bit_count() for i,p in enumerate(probabilities))
            leakage=sum(p for i,p in enumerate(probabilities) if (i&15).bit_count()!=2 or (i>>4).bit_count()!=2)
            spin_squared=float(state.expectation_value(spin_operator).real)
            gap=energy-reference["qubit_sector_active_energy_Ha"]
            valid=bool(np.isfinite(energy) and gap>=-criteria["max_variational_below_reference_Ha"]
                and abs(alpha-2)<=criteria["max_particle_number_error"] and abs(beta-2)<=criteria["max_particle_number_error"]
                and leakage<=criteria["max_particle_number_error"] and abs(spin_squared)<=criteria["max_VQE_spin_squared"])
            np.savez_compressed(output/f"trial_{trial}_state.npz",initial_parameters=point,optimal_parameters=final,statevector=state.data)
            trial_report=dict(trial=trial,initial_std=std,active_energy_Ha=energy,total_energy_Ha=energy+raw["constant_Ha"],
                gap_to_classical_Ha=gap,alpha_electrons=float(alpha),beta_electrons=float(beta),sector_leakage=float(leakage),
                spin_squared=spin_squared,valid=valid,target_gap_met=bool(valid and gap<=criteria["target_VQE_energy_gap_Ha"]),
                evaluations=int(result.cost_function_evals),optimizer_evaluations=int(result.optimizer_result.nfev),
                optimizer_iterations=int(result.optimizer_result.nit),elapsed_s=time.perf_counter()-trial_started)
            report["trials"].append(trial_report)
            save(output/"manifest.json",report)
            print(json.dumps(trial_report),flush=True)
        valid=[trial for trial in report["trials"] if trial["valid"]]
        best=min(valid,key=lambda row:row["active_energy_Ha"]) if valid else None
        report.update(status="complete",selected_trial=None if best is None else best["trial"],
            all_trials_valid=all(t["valid"] for t in report["trials"]),
            target_gap_met=bool(best and best["target_gap_met"]),elapsed_s=time.perf_counter()-started,
            peak_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report["files"]={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"}
        save(output/"manifest.json",report)
        return report
    except Exception as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}",elapsed_s=time.perf_counter()-started)
        save(output/"manifest.json",report)
        raise
