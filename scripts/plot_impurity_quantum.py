# FILE PURPOSE: Plot fixed cluster geometries and every declared VQE start.
"""Use the main plotting environment after the separate quantum runs finish."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    figure = plt.figure(figsize=(12, 9), layout="constrained")
    cases = [("al_hexaaqua", "Hexaaqua Al(III) reference"),
             ("al_phosphate", "Hydrated Al–dihydrogen phosphate")]
    colors = {"Al": "#ce6a00", "P": "#7853a6", "O": "#b62e40", "H": "#708392"}
    for column, (stem, title) in enumerate(cases):
        classical = Path(f"results/kdp_quantum_{stem}_classical_v2")
        vqe = Path(f"results/kdp_quantum_{stem}_vqe_v2")
        reference = json.loads((classical / "manifest.json").read_text())
        run = json.loads((vqe / "manifest.json").read_text())
        geometry = json.loads((classical / "geometry.json").read_text())
        positions = np.asarray(geometry["positions_Angstrom"])
        elements = geometry["elements"]
        axis = figure.add_subplot(2, 2, column + 1, projection="3d")
        for element in dict.fromkeys(elements):
            points = positions[np.array(elements) == element]
            axis.scatter(*points.T, s=65 if element != "H" else 18, color=colors[element], label=element,
                         depthshade=False)
        # Draw a coordination schematic; distances are the declared design,
        # and these lines do not assert an optimized bond order.
        for i, (first, p) in enumerate(zip(elements, positions)):
            for j in range(i):
                second, q = elements[j], positions[j]
                distance = np.linalg.norm(p - q)
                if (({first, second} == {"O", "H"} and distance < 1.1) or
                    ({first, second} == {"P", "O"} and distance < 1.7) or
                    ({first, second} == {"Al", "O"} and distance < 2.0)):
                    axis.plot(*np.array([p, q]).T, color="#9aa1a8", linewidth=1)
        axis.set_title(title + f"\ncharge +{geometry['charge']}; 2S = 0", fontsize=11)
        axis.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
        axis.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 1), zoom=.85)
        axis.legend(loc="upper left", fontsize=8, ncol=2)
        axis = figure.add_subplot(2, 2, column + 3)
        exact = reference["qubit_sector_active_energy_Ha"]
        constant = reference["constant_Ha"]
        for trial in run["trials"]:
            trace = [json.loads(line) for line in (vqe / f"trial_{trial['trial']}_evaluations.jsonl").read_text().splitlines()]
            gap = np.array([row["energy_Ha"] - exact for row in trace]) * 1000
            axis.plot([row["evaluation"] for row in trace], np.maximum(gap, 1e-8),
                      marker=".", markersize=4, label=f"Start {trial['trial']}: σ={trial['initial_std']}")
        hf_gap = (reference["scf_energy_Ha"] - constant - exact) * 1000
        axis.axhline(hf_gap, color="#565656", linestyle=":", label=f"HF: {hf_gap:.4f} mHa")
        axis.axhline(1., color="#b54435", linestyle="--", label="Frozen 1 mHa target")
        axis.set(yscale="log", xlabel="Energy evaluations", ylabel="Gap to same-space exact energy (mHa)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Local 8-qubit method diagnostic at fixed designed geometries\n"
                    "4 electrons / 4 spatial orbitals • no solvent, KDP lattice or quantum hardware", fontsize=13)
    figure.savefig(output / "impurity_quantum.png", dpi=180)
    figure.savefig(output / "impurity_quantum.svg")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
