# KDP Crystal Growth Simulation & Synthetic Vision Dataset Specification

This document compiles the verified physical constants, crystallographic geometry, fluid transport properties, empirical growth kinetics, and optical parameters required to run the custom 3D KDP ($\text{KH}_2\text{PO}_4$) growth simulation engine and construct the Blender synthetic computer vision pipeline (YOLOv8 segmentation).

---

## 1. Primary Physical Constants & Sourced Parameters

| Parameter | Value | Units | Primary Usage / Source File |
| :--- | :--- | :--- | :--- |
| **Solid Crystal Density ($\rho_s$)** | $2338.0$ | $\text{kg/m}^3$ | Structural / CFD Mass Balance (`tank.py`, `cutcell.py`) |
| **Liquid Solution Density ($\rho_{\text{sol}}$)** | $1170.0$ | $\text{kg/m}^3$ | Solute Mass Fraction (`units.py`, `flow.py`) |
| **Kinematic Viscosity ($\nu$)** | $1.05 \times 10^{-6}$ | $\text{m}^2/\text{s}$ | Solution Shear & Navier-Stokes (`flow.py`) |
| **Dynamic Viscosity ($\mu$)** | $1.23 \times 10^{-3}$ | $\text{Pa}\cdot\text{s}$ | Saturated Solution Transport (`units.py`) |
| **Solute Diffusion Coefficient ($D_{KDP}$)** | $1.01 \times 10^{-9}$ | $\text{m}^2/\text{s}$ | Liquid Solute Diffusion (`solute.py`, `physics.py`) |
| **Molar Mass ($M_{KDP}$)** | $0.1360855$ | $\text{kg/mol}$ | Molar Conversions (`units.py` - NIST) |
| **Hydrodynamic Boundary Layer ($\delta$)** | $1.5 \times 10^{-4}$ | $\text{m}$ ($150 \text{ }\mu\text{m}$) | Sampling Prism Height (`tank.py`, `engine.py`) |

---

## 2. Structural & Crystallographic Specifications

Data parsed from standard room-temperature Crystallography Open Database (COD) entry **`1010458.cif`**:

* **Chemical Formula:** $\text{KH}_2\text{PO}_4$ (Hill notation: `H2 K O4 P`)
* **Space Group:** $I\overline{4}2d$ (Space Group #122, Tetragonal System)
* **Unit Cell Dimensions:**
  * $a = b = 7.4300 \text{ \AA}$
  * $c = 6.9700 \text{ \AA}$
  * $\alpha = \beta = \gamma = 90.00^\circ$
* **Unit Cell Volume ($V$):** $384.80 \text{ \AA}^3$
* **Baseline Geometric Aspect Ratio ($c/a$):** $\approx 0.9381$
* **Theoretical $\{101\}$ Pyramid Facet Angle ($\theta$):** $\arctan(a/c) \approx 46.83^\circ$ relative to the z-axis.

---

## 3. Empirical Facet Kinetic Equations & Rates

Growth kinetics follow power-law equations driven by dimensionless relative supersaturation $\sigma = (C - C_{\text{eq}})/C_{\text{eq}}$ (where $\sigma = 0.05$ represents $5\%$ supersaturation).

### A. Pyramid $\{101\}$ Facet Rate:
$$R_{101}(\sigma) = k_{101} \cdot \sigma^{n_{101}}$$
* **Prefactor ($k_{101}$):** $150.0 \text{ mm/day} \approx 1.736 \times 10^{-6} \text{ m/s}$
* **Power Exponent ($n_{101}$):** $1.2$ (Spiral step-propagation / 2D nucleation regime)

### B. Prism $\{100\}$ Facet Rate (With Impurity Dead-Zone):
$$R_{100}(\sigma) = k_{100} \cdot \max(0, \sigma - \sigma_d)^{n_{100}}$$
* **Prefactor ($k_{100}$):** $100.0 \text{ mm/day} \approx 1.157 \times 10^{-6} \text{ m/s}$
* **Dead-Zone Threshold ($\sigma_d$):** $0.015$ ($1.5\%$ relative supersaturation)
* **Power Exponent ($n_{100}$):** $1.5$ (Macrostep bunching regime)

### C. Operational Demonstration Bounds (Qin et al., 2020 & Literature):
* **Normal Growth Velocity (Defect-Free):** $12.0 - 14.0 \text{ mm/day}$ ($1.39 \times 10^{-7} \text{ to } 1.62 \times 10^{-7} \text{ m/s}$) at $30 \text{ rpm}$ platform rotation.
* **Inclusion / Defect Threshold:** Pushing growth rates to $\ge 16.0 \text{ mm/day}$ ($1.85 \times 10^{-7} \text{ m/s}$) or $\sigma \ge 0.07 - 0.08$ induces macrostep instability, spontaneous nucleation, and liquid veil inclusions.

---

## 4. Optical Properties for Blender Domain Randomization

Use these optical parameters in Blender’s Cycles shader engine to render synthetic training frames for YOLOv8 segmentation:

* **KDP Crystal Birefringence (Glass BSDF Shader):**
  * **Ordinary Axis Refractive Index ($n_o$):** $1.510$ (at $589 \text{ nm}$)
  * **Extraordinary Axis Refractive Index ($n_e$):** $1.468$ (at $589 \text{ nm}$)
* **Aqueous Solution Index ($n_{\text{sol}}$):** $1.370$ (Liquid tank volume shader)
* **Micro-Topography Bump Maps:**
  * $\{100\}$ Prism faces: Step-bunching streaks parallel to the z-axis (height $\approx 25 - 130 \text{ nm}$).
  * $\{101\}$ Pyramid caps: Triangular step-hillocks (height $\approx 6.0 - 6.9 \text{ nm}$).

---

## 5. Integration Workflow for Team Repositories

```
           [ COD Database ]                    [ Research Literature ]
          (1010458.cif file)               (Qin et al. / Zaitseva kinetics)
                  │                                        │
                  ▼                                        ▼
    Extract Geometry (c/a = 0.938)             Extract Kinetics & Fluid Properties
                  │                                        │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        [ Python Simulation Engine ]
                      (run simulation -> output JSON)
                                      │
                                      ▼
                     [ Blender Synthetic Rendering ]
                (Read JSON -> Render photorealistic frames)
                                      │
                                      ▼
                      [ YOLOv8 Segmentation Model ]
```

1. **Step 1:** Run `python engine.py` to generate the physics-grounded time-series output `crystal_growth_frames.json`.
2. **Step 2:** Execute the Blender procedural rendering script (`bpy`) to ingest `crystal_growth_frames.json`.
3. **Step 3:** Render domain-randomized synthetic images and automatic polygon bounding boxes (`.txt`) for YOLOv8 model training.