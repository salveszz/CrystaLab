# FILE PURPOSE: Carry KDP mass concentration on the fixed SI flow grid.
# Shared transport numerics live in scalar.py; this wrapper retains concentration
# bounds, kg mass accounting and the existing flow/solute API.
"""Passive solute and synchronized isothermal flow/mass transport."""

import numpy as np

from .flow import IncompressibleFlow
from .scalar import ConservativeScalarTransport
from .units import finite_number


class SoluteTransport(ConservativeScalarTransport):
    """KDP concentration [kg/m3 solution], distinct from carrier-volume labels."""

    def __init__(self, flow, concentration_kg_per_m3, *, diffusivity_m2_per_s=None):
        if not isinstance(flow, IncompressibleFlow):
            raise ValueError("Solute transport requires IncompressibleFlow")
        super().__init__(
            flow,
            concentration_kg_per_m3,
            diffusivity_m2_per_s=(
                flow.properties.solute_diffusivity_m2_per_s
                if diffusivity_m2_per_s is None
                else diffusivity_m2_per_s
            ),
        )
        self.initial_mass_kg = self.initial_integral

    @property
    def concentration(self):
        return self.values

    @concentration.setter
    def concentration(self, value):
        self.values = value

    @property
    def mass_kg(self):
        return self.volume_integral

    def _field(self, value):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Concentration must be real, in kg/m3 solution")
        field = np.array(raw, dtype=float, copy=True)
        if field.ndim == 0:
            field = np.full(self.flow.shape, float(field))
        if field.shape != self.flow.shape or not np.isfinite(field).all():
            raise ValueError("Concentration must be finite and match the liquid grid")
        liquid = field[self._fluid]
        if np.any(liquid < 0) or np.any(
            liquid >= self.flow.properties.density_kg_per_m3
        ):
            raise ValueError("Concentration must satisfy 0 <= C < solution density")
        field[~self._fluid] = 0.0
        return field

    def measurements(self):
        values = self.concentration[self._fluid]
        mass = self.mass_kg
        return dict(
            time_s=self.time_s,
            solute_mass_kg=mass,
            solute_mass_drift_kg=mass - self.initial_mass_kg,
            min_concentration_kg_per_m3=float(values.min()),
            max_concentration_kg_per_m3=float(values.max()),
            solute_substeps=self.substeps,
            limited_face_stages=self.limited_face_stages,
        )


class FlowSoluteSimulation:
    """Fixed-domain momentum/solute coupling with synchronized SI clocks."""

    def __init__(self, flow, concentration_kg_per_m3, *, diffusivity_m2_per_s=None):
        self.flow = flow
        self.solute = SoluteTransport(
            flow, concentration_kg_per_m3, diffusivity_m2_per_s=diffusivity_m2_per_s
        )

    def step(self, dt_s, acceleration=(0.0, 0.0, 0.0)):
        if self.flow.time_s != self.solute.time_s:
            raise ValueError("Advance the coupled flow and solute clocks together")
        dt = finite_number(dt_s, "dt_s", positive=True)
        self.solute._fluxes()
        old_flux = self.flow.volume_fluxes_m3_per_s()
        before = (self.flow.velocity, self.flow.pressure, self.flow.time_s)
        try:
            self.flow.step(dt, acceleration)
            mean_flux = [
                0.5 * (a + b)
                for a, b in zip(old_flux, self.flow.volume_fluxes_m3_per_s())
            ]
            self.solute.advance(dt, volume_fluxes_m3_per_s=mean_flux)
        except Exception:
            self.flow.velocity, self.flow.pressure, self.flow.time_s = before
            raise
        return self.measurements()

    def run_until(self, final_time_s, max_dt_s, acceleration=(0.0, 0.0, 0.0)):
        if self.flow.time_s != self.solute.time_s:
            raise ValueError("Advance the coupled flow and solute clocks together")
        final = finite_number(final_time_s, "final_time_s", nonnegative=True)
        maximum = finite_number(max_dt_s, "max_dt_s", positive=True)
        if final < self.flow.time_s:
            raise ValueError("Coupled transport cannot integrate backward in time")
        while self.flow.time_s < final:
            rate = sum(
                float(np.max(abs(u), initial=0)) / self.flow.spacing[c.axis]
                for u, c in zip(self.flow.velocity, self.flow.components)
            )
            dt = min(maximum, final - self.flow.time_s, 0.4 / rate if rate else maximum)
            self.step(dt, acceleration)
        return self.measurements()

    def measurements(self):
        return dict(
            **self.flow.measurements(),
            **{k: v for k, v in self.solute.measurements().items() if k != "time_s"},
        )
