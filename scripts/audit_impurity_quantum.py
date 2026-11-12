# FILE PURPOSE: Independently reconstruct the declared impurity Hamiltonians and VQE records.
"""No producer import: explicit fermion signs and Pauli matrices check the mapping.

Run in the pinned quantum environment. The physical approximation is not validated
by agreement between electronic solvers on the same reduced Hamiltonian.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_ladder(bits, operations):
    """Apply rightmost-first (orbital, create) operations with explicit signs."""
    sign = 1
    for orbital, create in operations:
        occupied = bool(bits & (1 << orbital))
        if occupied == create:
            return None, 0
        sign *= -1 if (bits & ((1 << orbital) - 1)).bit_count() % 2 else 1
        bits ^= 1 << orbital
    return bits, sign


def fermion_matrix(h1, h2):
    """Chemists' (pq|rs); alpha block then beta block; no core constant."""
    n = h1.shape[0]
    dimension = 1 << (2 * n)
    terms = []
    for p, q in np.ndindex(n, n):
        for sigma in (0, n):
            terms.append((h1[p, q], [(q + sigma, False), (p + sigma, True)]))
    for p, q, r, s in np.ndindex(n, n, n, n):
        for sigma in (0, n):
            for tau in (0, n):
                terms.append((.5 * h2[p, q, r, s], [
                    (q + sigma, False), (s + tau, False),
                    (r + tau, True), (p + sigma, True)]))
    matrix = np.zeros((dimension, dimension), complex)
    for coefficient, operations in terms:
        if abs(coefficient) == 0:
            continue
        for column in range(dimension):
            row, sign = apply_ladder(column, operations)
            if sign:
                matrix[row, column] += coefficient * sign
    return matrix


def spin_matrix(n):
    dimension = 1 << (2 * n)
    plus = np.zeros((dimension, dimension))
    sz = np.zeros(dimension)
    for bits in range(dimension):
        sz[bits] = .5 * ((bits & ((1 << n) - 1)).bit_count() - (bits >> n).bit_count())
        for p in range(n):
            row, sign = apply_ladder(bits, [(p + n, False), (p, True)])
            if sign:
                plus[row, bits] += sign
    return np.diag(sz ** 2) + .5 * (plus @ plus.T + plus.T @ plus)


def pauli_matrix(raw):
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    matrix = np.zeros((2 ** raw["num_qubits"],) * 2, complex)
    for term in raw["terms"]:
        product = np.ones((1, 1))
        for letter in term["pauli"]:
            product = np.kron(product, paulis[letter])
        matrix += complex(term["real"], term["imag"]) * product
    return matrix


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(directory, report, source_key):
    for name, expected in report["files"].items():
        require(Path(name).name == name, "Artifact name escapes output directory")
        require(digest(directory / name) == expected, f"Changed artifact: {directory / name}")
    require(digest(directory / "source_snapshot/impurity_quantum.py") == source_key,
            "Producer snapshot differs from manifest")


def audit(classical, vqe, protocol_path):
    # Qiskit is used only to replay the stored circuit, never to construct the
    # independent Hamiltonian, particle counts, exact eigenvalue or spin matrix.
    from qiskit import qpy
    from qiskit.quantum_info import Statevector

    classical, vqe = Path(classical), Path(vqe)
    protocol = json.loads(Path(protocol_path).read_text())
    reference = json.loads((classical / "manifest.json").read_text())
    run = json.loads((vqe / "manifest.json").read_text())
    require(reference["status"] == run["status"] == "complete", "Incomplete calculation")
    require(digest(protocol_path) == reference["identity"]["protocol_sha256"] == run["protocol_sha256"],
            "External frozen protocol differs")
    require(digest(classical / "protocol.json") == digest(protocol_path), "Protocol snapshot differs")
    require(digest(classical / "manifest.json") == run["classical_manifest_sha256"], "Classical identity differs")
    verify_files(classical, reference, reference["identity"]["module_sha256"])
    verify_files(vqe, run, run["source_sha256"])
    for source in protocol["sources"]:
        require(digest(ROOT / source["local_path"]) == source["sha256"], "Chemical source changed")
    values = np.load(classical / "active_integrals.npz", allow_pickle=False)
    h1, h2 = values["h1"], values["h2_chemist"]
    require(h1.shape == (4, 4) and h2.shape == (4, 4, 4, 4), "Wrong active space")
    matrix = fermion_matrix(h1, h2)
    raw = json.loads((classical / "qubit_operator.json").read_text())
    mapped = pauli_matrix(raw)
    matrix_error = float(np.max(abs(matrix - mapped)))
    require(matrix_error < 1e-8, "Independent fermionic and Pauli matrices disagree")
    require(np.max(abs(matrix - matrix.conj().T)) <= protocol["criteria"]["max_matrix_hermiticity_error"],
            "Non-Hermitian Hamiltonian")
    sector = [i for i in range(256) if (i & 15).bit_count() == 2 and (i >> 4).bit_count() == 2]
    require(sector == values["sector_indices"].tolist(), "Wrong particle sector")
    exact, eigenvectors = np.linalg.eigh(matrix[np.ix_(sector, sector)])
    s2 = spin_matrix(4)
    ground = eigenvectors[:, 0]
    exact_s2 = float(np.vdot(ground, s2[np.ix_(sector, sector)] @ ground).real)
    criteria = protocol["criteria"]
    energy_error = max(abs(exact[0] - reference["fci_active_energy_Ha"]),
                       abs(exact[0] - reference["qubit_sector_active_energy_Ha"]),
                       float(np.max(abs(exact - values["exact_sector_eigenvalues_Ha"]))))
    require(energy_error <= criteria["max_qubit_vs_FCI_energy_difference_Ha"], "Classical reference disagrees")
    require(abs(exact_s2) <= criteria["max_exact_spin_squared"], "Exact reference is not singlet")
    constant = float(values["constant_Ha"])
    require(abs(constant - raw["constant_Ha"]) < 1e-10, "Core constant disagrees")
    hf_error = abs(matrix[51, 51].real + constant - reference["scf_energy_Ha"])
    require(hf_error <= criteria["max_HF_determinant_vs_SCF_difference_Ha"], "HF energy/core accounting disagrees")
    selection = json.loads((classical / "orbital_selection.json").read_text())
    overlap_values, overlap_vectors = np.linalg.eigh(values["ao_overlap"])
    lowdin = (overlap_vectors * np.sqrt(overlap_values)) @ overlap_vectors.T @ values["mo_coefficients"]
    geometry = json.loads((classical / "geometry.json").read_text())
    donors = geometry["donor_oxygen_atom_indices"]
    labels = selection["ao_labels"]
    donor_aos = [i for i, label in enumerate(labels) if label[0] in donors and
                 label[2] in protocol["active_space"]["donor_shells"]]
    metal_aos = [i for i, label in enumerate(labels) if label[0] == 0 and
                 label[2] in protocol["active_space"]["al_acceptor_shells"]]
    require(donor_aos == selection["donor_ao_indices"] and metal_aos == selection["metal_ao_indices"],
            "Chemical AO subspaces differ")
    donor = np.sum(lowdin[donor_aos] ** 2, axis=0)
    acceptor = np.sum(lowdin[metal_aos] ** 2, axis=0)
    occupied = reference["electrons"] // 2
    occupied_indices = sorted(sorted(range(occupied), key=lambda i: (-donor[i], i))[:2])
    virtual_indices = sorted(sorted(range(occupied, len(donor)), key=lambda i: (-acceptor[i], i))[:2])
    require(occupied_indices + virtual_indices == values["selected_mo_indices"].tolist(), "Chemical MO selection differs")
    require(np.allclose(donor, values["donor_weights"], rtol=0, atol=1e-10) and
            np.allclose(acceptor, values["acceptor_weights"], rtol=0, atol=1e-10), "Selection weights differ")
    with (vqe / "ansatz.qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    require(len(circuits) == 1, "Unexpected circuit count")
    circuit = circuits[0]
    require(circuit.num_qubits == run["qubits"] == 8 and circuit.num_parameters == run["parameters"],
            "Circuit dimensions differ")
    require(circuit.depth() == run["depth"] and dict(circuit.count_ops()) == run["gate_counts"],
            "Circuit resource record differs")
    require(len(run["trials"]) == 3, "Missing predeclared start")
    rng = np.random.default_rng(protocol["quantum"]["seed"])
    trials = []
    for trial, std in zip(run["trials"], (0., .02, .05)):
        index = len(trials)
        require(trial["trial"] == index and trial["initial_std"] == std, "Trial order differs")
        data = np.load(vqe / f"trial_{index}_state.npz", allow_pickle=False)
        initial = np.zeros(circuit.num_parameters) if std == 0 else rng.normal(0, std, circuit.num_parameters)
        require(np.array_equal(initial, data["initial_parameters"]), "Initial point differs from frozen seed")
        state = data["statevector"]
        replay = Statevector.from_instruction(circuit.assign_parameters(data["optimal_parameters"])).data
        state_error = float(np.max(abs(state - replay)))
        require(state_error < 1e-10 and abs(np.vdot(state, state) - 1) < 1e-10, "Circuit/state disagreement")
        energy = float(np.vdot(state, matrix @ state).real)
        gap = float(energy - exact[0])
        probabilities = abs(state) ** 2
        alpha = sum(p * (i & 15).bit_count() for i, p in enumerate(probabilities))
        beta = sum(p * (i >> 4).bit_count() for i, p in enumerate(probabilities))
        leakage = float(1 - probabilities[sector].sum())
        spin = float(np.vdot(state, s2 @ state).real)
        valid = bool(gap >= -criteria["max_variational_below_reference_Ha"] and
                     max(abs(alpha - 2), abs(beta - 2), abs(leakage)) <= criteria["max_particle_number_error"] and
                     abs(spin) <= criteria["max_VQE_spin_squared"])
        target = bool(valid and gap <= criteria["target_VQE_energy_gap_Ha"])
        require(valid == trial["valid"] and target == trial["target_gap_met"], "Trial acceptance differs")
        for key, value in [("active_energy_Ha", energy), ("total_energy_Ha", energy + constant),
                           ("gap_to_classical_Ha", gap), ("spin_squared", spin),
                           ("alpha_electrons", alpha), ("beta_electrons", beta), ("sector_leakage", leakage)]:
            require(abs(value - trial[key]) < 1e-8, f"Trial {index} {key} differs")
        trace = [json.loads(line) for line in (vqe / f"trial_{index}_evaluations.jsonl").read_text().splitlines()]
        require(len(trace) == trial["evaluations"] and [t["evaluation"] for t in trace] == list(range(1, len(trace) + 1)),
                "Incomplete evaluation trace")
        # Replay initial and last recorded evaluations, without invoking VQE.
        for record in (trace[0], trace[-1]):
            evaluated = Statevector.from_instruction(circuit.assign_parameters(record["parameters"])).data
            require(abs(np.vdot(evaluated, matrix @ evaluated).real - record["energy_Ha"]) < 1e-8,
                    "Stored evaluation energy differs")
        require(np.allclose(trace[0]["parameters"], initial, rtol=0, atol=1e-14), "First evaluation is not frozen start")
        trials.append(dict(trial=index, gap_Ha=gap, valid=valid, target_gap_met=target,
                           state_replay_max_error=state_error, spin_squared=spin, evaluations=len(trace)))
    eligible = [t for t in trials if t["valid"]]
    best = min(eligible, key=lambda t: t["gap_Ha"]) if eligible else None
    require(run["selected_trial"] == (None if best is None else best["trial"]), "Selected trial differs")
    require(run["target_gap_met"] == bool(best and best["target_gap_met"]), "Selected acceptance differs")
    return dict(status="passed", case=reference["case"]["case_id"],
                protocol_sha256=digest(protocol_path), classical_manifest_sha256=digest(classical / "manifest.json"),
                vqe_manifest_sha256=digest(vqe / "manifest.json"), independent_sector_dimension=len(sector),
                independent_fermion_vs_Pauli_max_error=matrix_error,
                independent_classical_energy_max_error_Ha=float(energy_error), HF_energy_error_Ha=float(hf_error),
                exact_spin_squared=exact_s2, trials=trials,
                interpretation="Record/solver consistency on a tiny frozen Hamiltonian; no physical chemistry validation or quantum speedup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--vqe", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/kdp_impurity_quantum.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output already exists")
    report = audit(args.classical, args.vqe, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
