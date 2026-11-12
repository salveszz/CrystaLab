# KDP-related Al precursor: local VQE/classical comparison

A first chemically motivated **electronic-solver diagnostic is complete** for
two hydrated Al(III) clusters. All three fixed VQE starts per case agree with the
classical solution of the same four-electron/four-orbital Hamiltonian within the
frozen 0.001 Ha target. Independent fermion-matrix reconstruction, particle/spin
checks and circuit replay pass. This establishes solver consistency at designed
geometries; surface adsorption, KDP defects and impurity tolerance remain open.

The Hartree–Fock baseline already meets that energy target in both tiny active
spaces. The experiment supplies neither a quantum advantage nor chemically
converged predictions. It runs entirely on a local CPU, outside vision, labels
and online control. The [frozen protocol](../configs/kdp_impurity_quantum.json)
and [checkpoint](QUANTUM_IMPURITY_CHECKPOINT.json) preserve settings, results,
source identities and failed attempts.

## Chemical scope

[Asakuma, Maeda and Fukui (2008), DOI 10.2477/jccj.H1918](https://www.jstage.jst.go.jp/article/jccj/7/1/7_H1918/_pdf)
discuss Al aquo/phosphate complexes in KDP growth solution. Section 2.2 and Figure
2 motivate a hexaaqua reference and a bidentate dihydrogen-phosphate complex with
four water ligands. Their topology motivates these cases; the coordinates and
energies here are not reproduced from the paper.
[Salter, Wierzbicki and Land's primary stepped-KDP study](https://www.osti.gov/servlets/purl/862209)
instead models Al/phosphate interaction at a KDP surface. It motivates further
surface work and cautions against a simple Al-for-K substitution. Our isolated
precursors do not reproduce its surface model.

| Case | Formula and charge | Electrons | Role |
|---|---|---:|---|
| `al_hexaaqua_reference` | [Al(H₂O)₆]³⁺ | 70 | Aqueous impurity-precursor reference; not pristine KDP |
| `al_dihydrogen_phosphate` | [Al(H₂O)₄(H₂PO₄)]²⁺ | 100 | Hydrated Al/phosphate precursor related to KDP solution |

`DESIGN-AL-PRECURSOR-001` specifies deterministic, fixed six-coordinate molecular
scaffolds: Al–O 1.90 Å, donor P–O 1.50 Å, protonated P–O 1.57 Å and O–H 0.96 Å.
The phosphate scaffold has two unprotonated donor oxygens and two terminal OH
groups. There is no geometry optimization, solvent, periodic lattice or embedding.
Maximum RHF nuclear-force magnitudes are **0.01565 and 0.10590 Ha/Bohr** respectively;
these geometries are not stationary structures. They do not establish dominant
solution species or equilibrium populations.

The species differ in atoms and charge. Their total energies must not be
subtracted as an adsorption or reaction energy. The formally balanced ligand
exchange would also need H₂PO₄⁻ and two water energies, chemically comparable
geometries, solvation and further thermochemical terms; those calculations have
not been performed.

## Frozen electronic and quantum methods

Both cases use gas-phase RHF with spherical **6-31G***, energy tolerance 1e-10 Ha,
SCF-gradient tolerance 1e-7, at most 100 cycles, two threads and a 1,024 MB PySCF
working-memory setting. Both converge without the predeclared Newton fallback
and pass internal RHF stability analysis. This is not an external/unrestricted
stability or basis-convergence study. There are 126/168 atomic orbitals.

The active space contains **four electrons in four spatial orbitals**. A frozen
Löwdin-weight rule selects two occupied orbitals with greatest donor-O valence-p
character and two virtual orbitals with greatest Al valence-s/p character. The
reference uses all six water oxygens; the phosphate case uses its two donor
oxygens. Selected zero-based canonical MO indices are **32, 34, 47, 54** and
**47, 48, 64, 71**. Minimum selected donor/acceptor weights exceed 0.77/0.57.
The remaining 33/48 occupied orbitals are doubly occupied and frozen; other
virtual orbitals are excluded. No active-space convergence is established.
Near-degenerate canonical orbitals can change under a fresh numerical solve, so
the actual coefficients, weights, indices and Hamiltonian are archived.

PySCF CASCI effective one-/two-electron integrals define each reduced Hamiltonian.
The nuclear plus inactive-electron constant is added exactly once when reporting
total energies. PySCF FCI is checked against the Jordan–Wigner matrix in the
**2-alpha/2-beta sector: 36 determinants**, rather than an unconstrained global
Fock-space minimum. The HF determinant reconstructs the full RHF energy.

Qiskit Nature UCCSD uses one repetition, spin-preserving excitations and a
Hartree–Fock initial state, without tapering: **8 qubits, 26 parameters, circuit
depth 519 and 420 CX gates** at the declared decomposition, before hardware routing.
There are 105/185 Pauli terms. Qiskit's exact `StatevectorEstimator` has zero
precision/no shots; reverse-mode gradients feed SLSQP with `maxiter=200` and
`ftol=1e-10`. The three fixed starts are zero and Gaussian σ=0.02/0.05, seed
20261112. Classical eigenvectors/energies do not initialize or stop optimization.
The SDK's [ground-state documentation](https://qiskit-community.github.io/qiskit-nature/tutorials/03_ground_state_solvers.html)
and [VQE API](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.VQE.html)
describe the solver interfaces used here.

## Actual results

Errors below refer only to the same frozen reduced Hamiltonian.

| Case | Start σ | VQE gap to exact (Ha) | Energy evaluations | SLSQP iterations |
|---|---:|---:|---:|---:|
| Hexaaqua | 0 | 2.46918e-9 | 9 | 5 |
| Hexaaqua | 0.02 | 2.50639e-9 | 19 | 12 |
| Hexaaqua | 0.05 | 2.47449e-9 | 22 | 13 |
| Phosphate | 0 | 7.99317e-9 | 9 | 5 |
| Phosphate | 0.02 | 8.00748e-9 | 20 | 14 |
| Phosphate | 0.05 | 7.99845e-9 | 21 | 15 |

All six states satisfy the frozen particle-number, sector-leakage, spin and
variational-bound checks. Maximum independently reconstructed VQE S² is below
4.3e-11. Start zero has the lowest valid energy for both cases.

| Quantity | Hexaaqua reference | Phosphate complex |
|---|---:|---:|
| Exact active electronic energy (Ha) | -5.946408371371455 | -5.535589938949068 |
| Nuclear + inactive constant (Ha) | -691.1953862783187 | -1181.5092929836728 |
| Exact total at this approximation (Ha) | -697.1417946496902 | -1187.0448829226218 |
| HF error against same-space exact (mHa) | 0.07578 | 0.09105 |
| Classical preparation elapsed (s) | 14.15 | 259.24 |
| Classical peak process RSS (KiB) | 469,168 | 873,892 |
| Three-start VQE elapsed (s) | 5.13 | 5.54 |
| VQE peak process RSS (KiB) | 220,648 | 226,544 |

The frozen target is **1 mHa**, larger than both HF errors. Tight VQE agreement
does not establish meaningful chemistry accuracy. Classical preparation includes
RHF, stability, nuclear gradients and integral construction; VQE consumes the
already built Hamiltonian. These stage times are not comparable solver-speedup
measurements. CPU diagonalization of a 36-dimensional matrix is the small exact
reference; no quantum hardware or advantage is demonstrated.

The [standalone PNG](../results/kdp_quantum_figures_v2/impurity_quantum.png) and
[SVG](../results/kdp_quantum_figures_v2/impurity_quantum.svg) show the designed
geometries, every energy trace, HF baselines and target. The second layout was
visually checked after increasing room around 3D axis labels. Generated files
remain local/Git-ignored; the plotting script and checkpoint are tracked.

## Independent audit and retained failures

`scripts/audit_impurity_quantum.py` does not import the producer. It explicitly
applies fermionic creation/annihilation signs to the stored chemists' integrals,
forms Pauli Kronecker matrices independently, diagonalizes the physical sector
and constructs S² from spin ladder operators. It reconstructs orbital-selection
weights, checks source/artifact hashes against the external protocol, replays all
six final circuits and the first/last recorded evaluations, and verifies fixed
initial points and the reported best trial. Full-matrix disagreement is at most
**1.54e-12**; classical-sector energy disagreement is at most **1.25e-14 Ha**;
the worst HF reconstruction error is **1.69e-11 Ha**. This checks the saved
Hamiltonian and records, not the adequacy of that Hamiltonian for real KDP.

All attempts remain available locally:

1. Both `classical_v1` attempts completed numerical work but failed while saving
   NumPy integer indices. The strict record writer was fixed and both cases were
   rerun into fresh `classical_v2` directories with identical frozen settings.
2. Both v2 classical manifests are complete and independently audited. Their old
   CLI formatter subsequently failed to print a NumPy boolean, giving process
   exit 1 after the valid record was written. The CLI now prints the saved JSON.
   Those terminal logs remain preserved; they are not reported as exit-0 runs.
3. The first hexaaqua VQE had valid final states but incomplete callback parameter
   traces. Installed Algorithms 0.4.0 reshapes the callback input to `(-1, 1)`.
   The first audit refused circuit replay. The estimator wrapper now captures the
   complete actual primitive input without changing calculations or SDK files.
   Both accepted VQE runs use `vqe_v2`; the earlier run/audit log remain retained.

The protocol, geometry, basis, orbital rule, solver settings and thresholds were
not relaxed after these failures. Classical producer snapshots precede the
logging-only VQE change; each stage records its exact source hash.

The full source-matched regression suite passes **429 tests / 309 subtests in
115.15 s**. The seven focused checks cover ligand stoichiometry/charge, molecular
bonds, strict records, fermion exchange signs, interaction counting and spin.
The two-case numerical/circuit audits were executed separately in the quantum
runtime. The main suite checks the lightweight geometry and audit helpers.

## Reproduction and next gate

Use a separate environment; the existing vision runtime is unchanged. The actual
runtime is Python 3.14.8 with the exact
[14 distribution pins](../configs/quantum-runtime-requirements.txt), including
Qiskit 2.6.2, Nature 0.8.0, Algorithms 0.4.0 and PySCF 2.15.0. The locally saved
install report and official versioned PyPI metadata agree on all 14 downloaded
wheel hashes. No cloud provider, remote QPU, account or paid service is used.

```bash
python3.14 -m venv .venv/quantum-runtime
.venv/quantum-runtime/bin/python -m pip install \
  --only-binary=:all: -r configs/quantum-runtime-requirements.txt
```

The two source PDFs must first exist at the paths/hashes in the frozen protocol.
Choose unused output paths; all producers refuse existing destinations. For each
case in that protocol:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/quantum-runtime/bin/python scripts/run_impurity_quantum.py classical \
  --case al_hexaaqua_reference --output results/al_reference_classical_new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/quantum-runtime/bin/python scripts/run_impurity_quantum.py vqe \
  --classical results/al_reference_classical_new --output results/al_reference_vqe_new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/quantum-runtime/bin/python scripts/audit_impurity_quantum.py \
  --classical results/al_reference_classical_new --vqe results/al_reference_vqe_new \
  --output results/al_reference_audit_new.json
```

Repeat with `--case al_dihydrogen_phosphate` and its own outputs. The plotting
script's current command reproduces the archived v2 paths:

```bash
.venv/bin/python scripts/plot_impurity_quantum.py --output results/al_figures_new
```

Before physical chemistry claims, define an appropriate KDP reference/surface,
speciation/solvation model, relaxed geometries, basis/active-space convergence and
balanced observable. Compare with relevant classical chemistry and independent
data. Fe/Cr, solid defects, adsorption energies, ppm thresholds and optical
quality remain uncalculated. This extension cannot close the core requirement
for reviewed real camera labels or synchronized action-response trajectories.
