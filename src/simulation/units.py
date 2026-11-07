# HOW THIS FILE WORKS
# Define SI conversions and concentration bases for the physical-model work.
# A water-based solute mass ratio is not a solution mass fraction or kg/m^3.
# Convert between them explicitly, using solution density only where required.
# The KDP solubility relation is transcribed from KDP-2025 in SOURCES.md; its
# project use is restricted to the temperatures covered by that reference.
# FluidPropertiesSI computes transport scales from supplied physical properties.
# These helpers do not assign physical units to the legacy illustrative engine
# or claim that a Navier-Stokes solver or calibrated growth law exists yet.

"""SI units, concentration conversions and sourced KDP reference relations."""

from dataclasses import dataclass
import math
from numbers import Real


# NIST-KDP in SOURCES.md: 136.0855 g/mol converted to kg/mol.
KDP_MOLAR_MASS_KG_PER_MOL = 0.1360855


def finite_number(value, name, *, positive=False, nonnegative=False):
    """Reject booleans, nonfinite values and invalid physical scalar inputs."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def celsius_to_kelvin(value):
    result = finite_number(value, "temperature_C") + 273.15
    return finite_number(result, "temperature_K", nonnegative=True)


def kelvin_to_celsius(value):
    return finite_number(value, "temperature_K", nonnegative=True) - 273.15


def rpm_to_rad_per_s(value):
    """Signed rotation; the sign selects the direction around the chosen axis."""
    return finite_number(value, "rotation_rpm") * (2 * math.pi / 60)


def litres_per_minute_to_m3_per_s(value):
    return finite_number(value, "flow_L_per_min") * 1e-3 / 60


def growth_rate_m_per_s(value, unit):
    """Convert a published normal rate without changing its sign or statistic."""
    factors = {"m/s": 1.0, "nm/s": 1e-9, "um/h": 1e-6 / 3600, "mm/day": 1e-3 / 86400}
    if unit not in factors:
        raise ValueError(f"Unsupported normal-growth-rate unit: {unit}")
    return finite_number(value, "normal_growth_rate") * factors[unit]


def mass_ratio_to_mass_fraction(ratio):
    """q=kg solute/kg water -> w=kg solute/kg solution, for a binary solution."""
    ratio = finite_number(ratio, "solute_water_mass_ratio", nonnegative=True)
    return ratio / (1 + ratio)


def mass_fraction_to_mass_ratio(fraction):
    fraction = finite_number(fraction, "solute_mass_fraction", nonnegative=True)
    if fraction >= 1:
        raise ValueError("solute_mass_fraction must be less than one")
    return fraction / (1 - fraction)


def solution_concentration_kg_per_m3(ratio, solution_density_kg_per_m3):
    """C=rho_solution*w; this is per solution volume, not per carrier volume."""
    density = finite_number(
        solution_density_kg_per_m3, "solution_density", positive=True
    )
    return density * mass_ratio_to_mass_fraction(ratio)


def concentration_to_mass_ratio(concentration, solution_density_kg_per_m3):
    density = finite_number(
        solution_density_kg_per_m3, "solution_density", positive=True
    )
    concentration = finite_number(
        concentration, "solute_concentration", nonnegative=True
    )
    return mass_fraction_to_mass_ratio(concentration / density)


def kdp_solubility_mass_ratio(temperature_K):
    """KDP-2025: kg KDP/kg water, with Celsius in the published polynomial.

    24--34.8 C is the project evaluation window from the source's growth and
    saturation temperatures. It is not a claimed universal fit-validity range.
    The original 1967 measurements and coefficient uncertainty remain unaudited.
    """
    temperature_C = kelvin_to_celsius(temperature_K)
    if not 24.0 - 1e-10 <= temperature_C <= 34.8 + 1e-10:
        raise ValueError("KDP solubility reference is restricted to 24--34.8 C")
    return 0.17554 + 0.00102 * temperature_C + 0.0000743 * temperature_C**2


def kdp_relative_supersaturation(ratio, temperature_K):
    """Return (q-q_eq)/q_eq as a fraction; multiply by 100 only for percent."""
    ratio = finite_number(ratio, "solute_water_mass_ratio", nonnegative=True)
    return ratio / kdp_solubility_mass_ratio(temperature_K) - 1


@dataclass(frozen=True)
class FluidPropertiesSI:
    """One constant-property reference point, not a temperature/composition model."""

    temperature_K: float
    molality_mol_per_kg_water: float
    density_kg_per_m3: float
    dynamic_viscosity_Pa_s: float
    solute_diffusivity_m2_per_s: float
    source_id: str

    def __post_init__(self):
        for name in (
            "temperature_K",
            "density_kg_per_m3",
            "dynamic_viscosity_Pa_s",
            "solute_diffusivity_m2_per_s",
        ):
            finite_number(getattr(self, name), name, positive=True)
        finite_number(self.molality_mol_per_kg_water, "molality", nonnegative=True)
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Physical properties require a source_id")

    @property
    def mass_ratio(self):
        return self.molality_mol_per_kg_water * KDP_MOLAR_MASS_KG_PER_MOL

    @property
    def kinematic_viscosity_m2_per_s(self):
        return self.dynamic_viscosity_Pa_s / self.density_kg_per_m3

    def transport_scales(self, length_m, speed_m_per_s=0.0):
        """Re/Pe/Sc and diffusion time; a chosen speed is not a computed flow."""
        length = finite_number(length_m, "length_m", positive=True)
        speed = finite_number(speed_m_per_s, "speed_m_per_s", nonnegative=True)
        nu = self.kinematic_viscosity_m2_per_s
        D = self.solute_diffusivity_m2_per_s
        return dict(
            reynolds=speed * length / nu,
            solutal_peclet=speed * length / D,
            schmidt=nu / D,
            solute_diffusion_time_s=length**2 / D,
        )
