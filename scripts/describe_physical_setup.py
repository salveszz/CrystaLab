# HOW THIS FILE WORKS
# Read the separate SI reference definition and validate its physical inputs,
# geometry and stationary open-tank boundary assumptions before flow work.
# Convert water-based concentration to solution mass fraction and kg/m^3, then
# report viscosity and transport scales with source IDs. A supplied speed is
# a diagnostic scale only; it does not impose a flow boundary condition.
# Print the description to stdout, leaving results/ empty. No flow/growth solve
# runs here, and this profile cannot silently relabel the illustrative demo.

"""Inspect the sourced SI setup for the next stationary 3D flow benchmark."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.units import (
    FluidPropertiesSI,
    finite_number,
    kdp_relative_supersaturation,
    mass_ratio_to_mass_fraction,
    solution_concentration_kg_per_m3,
)


def describe_setup(settings, speed_m_per_s=0.0):
    """Validate a rest-case definition and derive quantities without solving it."""
    if (
        type(settings.get("schema_version")) is not int
        or settings["schema_version"] != 1
        or settings.get("unit_system") != "SI"
    ):
        raise ValueError("Use schema_version=1 and unit_system=SI")
    if settings.get("mode") != "stationary_isothermal_rest":
        raise ValueError("This initial definition supports stationary_isothermal_rest")
    fluid = FluidPropertiesSI(**settings["fluid"])
    if settings["temperature_K"] != fluid.temperature_K:
        raise ValueError("Temperature differs from the fluid property reference point")
    if settings["molality_mol_per_kg_water"] != fluid.molality_mol_per_kg_water:
        raise ValueError("Composition differs from the fluid property reference point")

    def vector(values, name, positive=False):
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{name} needs three x/y/z components")
        return [finite_number(v, name, positive=positive) for v in values]

    domain = settings["domain"]
    size = vector(domain["size_m"], "size_m", positive=True)
    level = finite_number(domain["liquid_height_m"], "liquid_height_m", positive=True)
    if level >= size[2]:
        raise ValueError("The open reference tank needs headspace")
    crystal = settings["crystal"]
    if crystal["shape"] != "box":
        raise ValueError("The first reference uses a fixed box crystal")
    dimensions = vector(crystal["size_m"], "crystal_size_m", positive=True)
    center = vector(crystal["center_m"], "crystal_center_m")
    for i in range(3):
        limit = level if i == 2 else size[i]
        if center[i] - dimensions[i] / 2 <= 0 or center[i] + dimensions[i] / 2 >= limit:
            raise ValueError("The fixed crystal must fit strictly inside the liquid")
    expected_boundaries = dict(
        walls="stationary_no_slip",
        crystal="stationary_no_slip_no_solute_flux",
        free_surface="flat_impermeable_zero_tangential_stress",
        pressure="dynamic_pressure_fluid_mean_zero",
        temperature="uniform_isothermal",
        solute="closed_no_flux",
        inlets_outlets="none",
    )
    if settings["boundaries"] != expected_boundaries:
        raise ValueError("Boundary assumptions differ from the verified definition")
    actuation = settings["actuation"]
    for name in ("rotation_rad_per_s", "circulation_m3_per_s"):
        if finite_number(actuation[name], name) != 0:
            raise ValueError("Rest-case actuation must be zero")
    vector(settings["gravity_m_per_s2"], "gravity_m_per_s2")
    finite_number(settings["duration_s"], "duration_s", positive=True)
    q = fluid.mass_ratio
    volume = size[0] * size[1] * level - dimensions[0] * dimensions[1] * dimensions[2]
    return dict(
        case_id=settings["case_id"],
        unit_system="SI",
        status="input_definition_only_no_flow_or_growth_prediction",
        sources=[fluid.source_id, "NIST-KDP", "KDP-2025", domain["source_id"]],
        temperature_K=fluid.temperature_K,
        solute_water_mass_ratio=q,
        solute_solution_mass_fraction=mass_ratio_to_mass_fraction(q),
        solute_solution_concentration_kg_per_m3=solution_concentration_kg_per_m3(
            q, fluid.density_kg_per_m3
        ),
        relative_supersaturation_fraction=kdp_relative_supersaturation(
            q, fluid.temperature_K
        ),
        supersaturation_status="derived_cross_source_estimate_not_a_VTT_measurement",
        liquid_volume_m3=volume,
        kinematic_viscosity_m2_per_s=fluid.kinematic_viscosity_m2_per_s,
        actual_forcing="rest",
        diagnostic_speed_m_per_s=speed_m_per_s,
        scale_length_m=size[0],
        transport_scales=fluid.transport_scales(size[0], speed_m_per_s),
        additional_data_needed=settings["additional_data_needed"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/physical_reference.json"
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=0.0,
        help="Diagnostic speed in m/s; does not drive the rest case",
    )
    args = parser.parse_args()
    try:
        description = describe_setup(
            json.loads(args.config.read_text()), args.speed_scale
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(description, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
