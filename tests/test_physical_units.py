# WHAT THESE TESTS CHECK
# Verify physical unit conversions and distinct concentration bases against mass
# identities and published data, rather than fitting the old illustrative demo.
# Reproduce the KDP-2025 supersaturations to their reported rounding precision.
# Check the sourced VTT property point, transport scales and the rest-case input
# contract; reject extrapolation, inconsistent composition and unsupported BCs.

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from simulation.units import (
    FluidPropertiesSI,
    celsius_to_kelvin,
    concentration_to_mass_ratio,
    growth_rate_m_per_s,
    kdp_relative_supersaturation,
    kdp_solubility_mass_ratio,
    kelvin_to_celsius,
    litres_per_minute_to_m3_per_s,
    mass_fraction_to_mass_ratio,
    mass_ratio_to_mass_fraction,
    rpm_to_rad_per_s,
    solution_concentration_kg_per_m3,
)
from describe_physical_setup import describe_setup


class PhysicalUnitsTests(unittest.TestCase):
    def setUp(self):
        self.setup = json.loads((ROOT / "configs/physical_reference.json").read_text())

    def test_temperature_and_actuation_units(self):
        self.assertAlmostEqual(celsius_to_kelvin(30), 303.15)
        self.assertAlmostEqual(kelvin_to_celsius(303.15), 30)
        self.assertAlmostEqual(rpm_to_rad_per_s(60), 2 * math.pi)
        self.assertAlmostEqual(rpm_to_rad_per_s(-60), -2 * math.pi)
        self.assertAlmostEqual(litres_per_minute_to_m3_per_s(60), 0.001)

    def test_published_rate_conversions_preserve_sign(self):
        self.assertAlmostEqual(
            growth_rate_m_per_s(13.56, "um/h"), 3.766666666666666e-9, delta=1e-23
        )
        self.assertAlmostEqual(growth_rate_m_per_s(39, "nm/s"), 3.9e-8, delta=1e-22)
        self.assertAlmostEqual(growth_rate_m_per_s(-86.4, "mm/day"), -1e-6, delta=1e-20)
        with self.assertRaises(ValueError):
            growth_rate_m_per_s(1, "unknown")

    def test_concentration_conversion_preserves_binary_solution_mass(self):
        # 0.25 kg solute per kg water is 0.20 kg solute per kg solution.
        q, rho, volume = 0.25, 1200.0, 0.002
        self.assertAlmostEqual(mass_ratio_to_mass_fraction(q), 0.2)
        C = solution_concentration_kg_per_m3(q, rho)
        self.assertAlmostEqual(C, 240)
        solute = C * volume
        water = (rho - C) * volume
        self.assertAlmostEqual(solute / water, q)
        self.assertAlmostEqual(concentration_to_mass_ratio(C, rho), q)
        self.assertAlmostEqual(mass_fraction_to_mass_ratio(0.2), q)

    def test_published_supersaturations_match_at_reported_precision(self):
        data = json.loads(
            (ROOT / "configs/benchmarks/kdp_maksimovic2025.json").read_text()
        )
        q = kdp_solubility_mass_ratio(celsius_to_kelvin(31))
        for row in data["table_1"]["rows"]:
            with self.subTest(temperature_C=row[0]):
                calculated = 100 * kdp_relative_supersaturation(
                    q, celsius_to_kelvin(row[0])
                )
                self.assertLessEqual(abs(calculated - row[1]), 0.05)

    def test_supersaturation_uses_water_mass_basis_and_can_be_negative(self):
        q = kdp_solubility_mass_ratio(303.15)
        self.assertAlmostEqual(kdp_relative_supersaturation(q, 303.15), 0)
        self.assertAlmostEqual(kdp_relative_supersaturation(0.9 * q, 303.15), -0.1)
        # A nonlinear conversion to solution mass fraction changes that ratio.
        self.assertNotAlmostEqual(
            mass_ratio_to_mass_fraction(0.9 * q) / mass_ratio_to_mass_fraction(q) - 1,
            -0.1,
        )

    def test_reference_does_not_silently_extrapolate_solubility(self):
        for temperature in (273.15, 310, 30, float("nan"), True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                kdp_solubility_mass_ratio(temperature)

    def test_invalid_physical_inputs_are_rejected(self):
        for fn, value in (
            (mass_fraction_to_mass_ratio, 1),
            (mass_fraction_to_mass_ratio, -0.1),
            (mass_ratio_to_mass_fraction, -1),
            (rpm_to_rad_per_s, True),
            (celsius_to_kelvin, -274),
            (litres_per_minute_to_m3_per_s, float("inf")),
        ):
            with (
                self.subTest(function=fn.__name__, value=value),
                self.assertRaises(ValueError),
            ):
                fn(value)
        with self.assertRaises(ValueError):
            concentration_to_mass_ratio(1200, 1200)
        for name in (
            "density_kg_per_m3",
            "dynamic_viscosity_Pa_s",
            "solute_diffusivity_m2_per_s",
        ):
            values = dict(self.setup["fluid"], **{name: 0})
            with self.subTest(name=name), self.assertRaises(ValueError):
                FluidPropertiesSI(**values)

    def test_vtt_reference_scales_and_dimensionless_groups(self):
        fluid = FluidPropertiesSI(**self.setup["fluid"])
        scales = fluid.transport_scales(0.01, 0.001)
        self.assertAlmostEqual(fluid.mass_ratio, 0.2993881)
        self.assertAlmostEqual(scales["reynolds"], 7.8)
        self.assertAlmostEqual(scales["solutal_peclet"], 17543.859649122805)
        self.assertAlmostEqual(
            scales["solutal_peclet"] / scales["reynolds"], scales["schmidt"]
        )
        self.assertAlmostEqual(scales["solute_diffusion_time_s"], 175438.59649122806)

    def test_rest_definition_is_si_and_fits_a_submerged_3d_crystal(self):
        result = describe_setup(self.setup)
        self.assertEqual(result["unit_system"], "SI")
        self.assertEqual(result["transport_scales"]["reynolds"], 0)
        self.assertAlmostEqual(result["liquid_volume_m3"], 7.99e-7)
        self.assertEqual(
            result["status"], "input_definition_only_no_flow_or_growth_prediction"
        )
        # A diagnostic scale must not turn on an actuator in the rest profile.
        diagnostic = describe_setup(self.setup, 0.001)
        self.assertEqual(diagnostic["actual_forcing"], "rest")
        self.assertAlmostEqual(diagnostic["transport_scales"]["reynolds"], 7.8)

    def test_inconsistent_reference_conditions_or_boundaries_are_rejected(self):
        for key, value in (
            ("temperature_K", 298.15),
            ("molality_mol_per_kg_water", 1.0),
            ("unit_system", "model"),
            ("schema_version", True),
        ):
            changed = dict(self.setup, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                describe_setup(changed)
        for section, key, value in (
            ("boundaries", "walls", "free_slip"),
            ("actuation", "rotation_rad_per_s", 1),
            ("domain", "liquid_height_m", 0.01),
            ("crystal", "center_m", [0.005, 0.005, 0.009]),
        ):
            changed = deepcopy(self.setup)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                describe_setup(changed)


if __name__ == "__main__":
    unittest.main()
