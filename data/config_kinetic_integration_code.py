"""
KDP CRYSTAL GROWTH SIMULATION & SYNTHETIC DATASET PIPELINE
==========================================================
Translates all parameters from DATASET_SPECIFICATION.md into code.
Configures:
 1. Fluid & Sourced Material Properties (FluidPropertiesSI)
 2. Power-Law Facet Kinetic Functions (R_101 & R_100)
 3. Simulation Engine Initialization (SimulationConfig & Simulation)
 4. Export Stream for Blender Rendering Pipeline (crystal_growth_frames.json)
"""

import json
import numpy as np

# Import from your existing codebase modules
from units import FluidPropertiesSI, growth_rate_m_per_s, kdp_solubility_mass_ratio
from tank import SimulationConfig, Tank
from engine import Simulation

# ==============================================================================
# 1. SOURCED PHYSICAL CONSTANTS & FLUID PROPERTIES
# ==============================================================================
# Sourced from NIST and KDP Literature (DATASET_SPECIFICATION.md §1)
SOLID_DENSITY_KG_M3 = 2338.0        # Solid KDP crystal density (rho_s)
SOLUTION_DENSITY_KG_M3 = 1170.0     # Saturated aqueous solution density (rho_sol)
DYNAMIC_VISCOSITY_PA_S = 1.23e-3    # Dynamic viscosity (mu) at 30°C
KINEMATIC_VISCOSITY_M2_S = 1.05e-6  # Kinematic viscosity (nu = mu / rho_sol)
SOLUTE_DIFFUSIVITY_M2_S = 1.01e-9   # Solute diffusion coefficient (D_KDP)
HYDRODYNAMIC_BOUNDARY_LAYER_M = 1.5e-4 # Boundary layer thickness delta (150 um)

# Instantiate physical properties container for flow.py
KDP_FLUID_PROPERTIES = FluidPropertiesSI(
    temperature_K=303.15,               # 30°C operating temperature
    molality_mol_per_kg_water=1.80,      # Equilibrium molality
    density_kg_per_m3=SOLUTION_DENSITY_KG_M3,
    dynamic_viscosity_Pa_s=DYNAMIC_VISCOSITY_PA_S,
    solute_diffusivity_m2_per_s=SOLUTE_DIFFUSIVITY_M2_S,
    source_id="DATASET_SPECIFICATION_V1"
)

# ==============================================================================
# 2. EMPIRICAL FACET KINETIC EQUATIONS
# ==============================================================================
# Implements power-law kinetics from DATASET_SPECIFICATION.md §3

def calculate_kdp_facet_rates(sigma, impurity_deadzone=0.015):
    """
    Calculates individual facet normal growth velocities in m/s 
    as a function of relative supersaturation sigma = (C - C_eq) / C_eq.
    
    Parameters:
      sigma (float): Dimensionless relative supersaturation (e.g., 0.05 for 5%)
      impurity_deadzone (float): Threshold below which {100} growth is halted (sigma_d)
      
    Returns:
      R_101 (float): Normal rate of {101} pyramid cap (m/s)
      R_100 (float): Normal rate of {100} prism wall (m/s)
    """
    if sigma <= 0.0:
        return 0.0, 0.0

    # --------------------------------------------------------------------------
    # Pyramid {101} Facet: R_101 = k_101 * (sigma ^ n_101)
    # k_101 = 150.0 mm/day, n_101 = 1.2
    # --------------------------------------------------------------------------
    k_101_m_per_s = growth_rate_m_per_s(150.0, "mm/day")
    n_101 = 1.2
    R_101 = k_101_m_per_s * (sigma ** n_101)

    # --------------------------------------------------------------------------
    # Prism {100} Facet: R_100 = k_100 * max(0, sigma - sigma_d) ^ n_100
    # k_100 = 100.0 mm/day, sigma_d = 0.015, n_100 = 1.5
    # --------------------------------------------------------------------------
    effective_sigma = max(0.0, sigma - impurity_deadzone)
    k_100_m_per_s = growth_rate_m_per_s(100.0, "mm/day")
    n_100 = 1.5
    R_100 = k_100_m_per_s * (effective_sigma ** n_100)

    return R_101, R_100


# ==============================================================================
# 3. CONSTRUCT SIMULATION CONFIGURATION
# ==============================================================================
def create_calibrated_config():
    """
    Builds a SimulationConfig instance populated with parameters 
    matching DATASET_SPECIFICATION.md.
    """
    # COD Entry 1010458 baseline seed geometry (mm scale)
    # Unit cell c/a aspect ratio ~ 0.9381
    seed_width_m = 0.06
    seed_depth_m = 0.06
    seed_height_m = seed_width_m * 0.9381  # Preserves physical c/a ratio

    config = SimulationConfig(
        # Tank Dimensions (x, y, z in meters)
        width=0.20,
        depth=0.20,
        height=0.25,
        nx=20, ny=20, nz=25,
        
        # Initial Liquid Headspace & Concentration
        initial_liquid_height=0.20,
        temperature=30.0,
        concentration=1.35,
        
        # Physical Seed Setup
        seed_width=seed_width_m,
        seed_depth=seed_depth_m,
        seed_height=seed_height_m,
        crystal_density=SOLID_DENSITY_KG_M3,
        
        # Transport & Kinetic Coefficients
        solute_diffusivity=SOLUTE_DIFFUSIVITY_M2_S,
        thermal_diffusivity=4.0e-2,
        growth_rate=growth_rate_m_per_s(150.0, "mm/day"), # Nominal rate prefactor
        impurity=0.015,                                   # Dead-zone threshold (sigma_d)
        
        # Sampling & Time Integration
        boundary_layer_cells=1.5, # 1.5 cells ~ 150 um boundary layer
        dt=1.0,                   # Step size (seconds)
        duration=120.0,           # Total run duration (seconds)
        seed=42
    )
    return config


# ==============================================================================
# 4. RUN SIMULATION & EXPORT BLENDER DATASET STREAM
# ==============================================================================
def run_and_export_dataset(output_filename="crystal_growth_frames.json"):
    """
    Runs the full simulation step-by-step, calculates local facet rates, 
    and exports time-series metrics into JSON for Blender rendering.
    """
    config = create_calibrated_config()
    sim = Simulation(config)
    
    print(f"Starting simulation run for duration {config.duration}s...")
    
    # Run the physical simulation engine
    sim.run()
    
    # Process history for Blender ingestion
    export_frames = []
    for step in sim.history:
        current_sigma = step["supersaturation"]
        
        # Compute exact R_101 and R_100 facet rates for current supersaturation
        r_101, r_100 = calculate_kdp_facet_rates(current_sigma, impurity_deadzone=config.impurity)
        
        # Check defect inclusion safety bound (from DATASET_SPECIFICATION.md §3.C)
        # R_101 >= 16 mm/day (1.85e-7 m/s) triggers micro-defect veil inclusions
        defect_flag = r_101 >= 1.85e-7 or current_sigma >= 0.07
        
        export_frames.append({
            "time_s": step["time"],
            "width_m": step["width"],
            "depth_m": step["depth"],
            "height_m": step["height"],
            "aspect_ratio_ca": step["height"] / max(step["width"], 1e-9),
            "liquid_level_m": step["liquid_level"],
            "supersaturation": current_sigma,
            "rate_101_m_per_s": r_101,
            "rate_100_m_per_s": r_100,
            "trigger_inclusion_defects": defect_flag,
            
            # Optical Render Metadata (§4)
            "optical_n_ordinary": 1.510,
            "optical_n_extraordinary": 1.468,
            "optical_n_solution": 1.370
        })
        
    with open(output_filename, "w") as f:
        json.dump(export_frames, f, indent=2)
        
    print(f"Success: Exported {len(export_frames)} keyframes to '{output_filename}'.")
    return export_frames

if __name__ == "__main__":
    run_and_export_dataset()