# HOW THIS FILE WORKS
# Provides the numerical rules used by the engine, without plotting or AI.
# solubility(): estimates how much solute can remain dissolved at a temperature
# using a configurable linear curve, clamped above zero to avoid division by zero.
# supersaturation(): computes concentration / solubility - 1 for each grid cell.
# Positive values allow growth; zero means equilibrium; negative means undersaturation.
# diffuse(): smooths a field using neighboring cells and sealed, zero-flux walls.
# The engine chooses small enough internal timesteps to keep diffusion stable.
# These are illustrative equations and coefficients, not calibrated KDP physics.

"""Illustrative dimensionless physics; no calibrated KDP constants."""
import numpy as np


def solubility(temperature, reference=1.0, slope=0.02, reference_temperature=25.0):
    """Linear solubility curve with a positive floor."""
    return np.maximum(1e-9, reference + slope * (temperature - reference_temperature))


def supersaturation(concentration, temperature, reference=1.0, slope=0.02,
                    reference_temperature=25.0):
    equilibrium = solubility(temperature, reference, slope, reference_temperature)
    return concentration / equilibrium - 1.0


def diffuse(field, coefficient, dt, dx, dy):
    """Conservative explicit diffusion with sealed (zero-flux) walls.

    The caller must respect dt * D * (1/dx**2 + 1/dy**2) <= 0.5.
    """
    padded = np.pad(field, 1, mode="edge")
    laplacian = ((padded[1:-1, 2:] - 2 * field + padded[1:-1, :-2]) / dx**2
                 + (padded[2:, 1:-1] - 2 * field + padded[:-2, 1:-1]) / dy**2)
    return field + coefficient * dt * laplacian
