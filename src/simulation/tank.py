# HOW THIS FILE WORKS
# SimulationConfig holds and validates the tank, physics, and experiment settings.
# Tank divides a rectangle into ny rows and nx columns. Each grid cell stores
# temperature and dissolved-solute concentration; dx and dy are cell dimensions.
# Initial temperature can vary vertically. Initial concentration can include
# seeded random noise, so repeated runs with the same settings are reproducible.
# Dissolved mass is the sum of concentration times cell volume (unit depth).
# Liquid volume stays fixed: the crystal is an overlay and does not displace
# liquid or block transport. Units and default parameters are illustrative.

"""Tank configuration and spatial fields, in illustrative model units."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class SimulationConfig:
    # Tank dimensions and grid resolution determine the physical size of each cell.
    width: float = 10.0
    height: float = 8.0
    nx: int = 50
    ny: int = 40
    # Initial solution conditions; temperature also acts as the relaxation target.
    temperature: float = 25.0
    concentration: float = 1.35
    temperature_gradient: float = 0.0
    concentration_noise: float = 0.005
    # Coefficients of the illustrative linear concentration-versus-temperature curve.
    solubility_reference: float = 1.0
    solubility_slope: float = 0.02
    reference_temperature: float = 25.0
    # Centered rectangular seed dimensions and density for unit-depth crystal mass.
    seed_width: float = 0.6
    seed_height: float = 0.8
    crystal_density: float = 8.0
    # Independent kinetic coefficients for the four tank-aligned crystal faces.
    growth_left: float = 0.025
    growth_right: float = 0.025
    growth_bottom: float = 0.015
    growth_top: float = 0.015
    # Diffusivities smooth neighboring cells; relaxation drives the temperature target.
    solute_diffusivity: float = 0.02
    thermal_diffusivity: float = 0.04
    thermal_relaxation: float = 0.01
    # Simplified mixing/transport settings and optional synthetic defect parameters.
    rotation: float = 0.5
    circulation: float = 0.5
    impurity: float = 0.0
    defect_rate: float = 0.0
    # External output interval, total experiment duration and reproducible random seed.
    dt: float = 1.0
    duration: float = 120.0
    seed: int = 42

    def __post_init__(self):
        # Reject nonnumeric, Boolean and nonfinite settings before any array is created.
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        # These values are divisors or physical sizes, so zero is also invalid.
        for name in ("width", "height", "seed_width", "seed_height", "crystal_density",
                     "solubility_reference", "dt", "duration"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        # Counts and random seeds require integers, not merely integer-looking floats.
        for name in ("nx", "ny", "seed"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.nx < 3 or self.ny < 3 or self.seed < 0:
            raise ValueError("Grid dimensions must be >= 3 and seed >= 0")
        # Leave solution around the seed rather than starting with a wall-sized crystal.
        if self.seed_width >= self.width or self.seed_height >= self.height:
            raise ValueError("Seed must fit inside the tank")
        # Zero is meaningful for disabled effects; negative rates/amounts are invalid.
        for name in ("concentration", "concentration_noise", "solute_diffusivity",
                     "thermal_diffusivity", "thermal_relaxation", "rotation", "circulation",
                     "impurity", "defect_rate", "growth_left", "growth_right", "growth_bottom", "growth_top"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


class Tank:
    def __init__(self, config, rng):
        # Convert the number of rows/columns into cell spacing and unit-depth volume.
        self.dx, self.dy = config.width / config.nx, config.height / config.ny
        self.cell_volume = self.dx * self.dy  # unit depth, fixed dilute reservoir
        # Half-cell offsets locate values at cell centers, not at the tank walls.
        self.x = (np.arange(config.nx) + 0.5) * self.dx
        self.y = (np.arange(config.ny) + 0.5) * self.dy
        shape = (config.ny, config.nx)
        # Add a vertical gradient centered on the configured mean temperature.
        gradient = config.temperature_gradient * (self.y / config.height - 0.5)
        # Broadcast the row temperatures across columns; copy to obtain writable storage.
        self.temperature = np.broadcast_to(config.temperature + gradient[:, None], shape).copy()
        # Seeded Gaussian variation makes repeatable initial fields; clip below zero.
        self.concentration = np.maximum(0.0, config.concentration + rng.normal(
            0.0, config.concentration_noise, shape))

    @property
    def solute_mass(self):
        # Sum concentration times equal cell volume to obtain total dissolved solute.
        return float(self.concentration.sum() * self.cell_volume)
