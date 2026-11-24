# FILE PURPOSE: Couple fixed-grid SI momentum, sensible heat and dissolved mass.
# Constant-property Boussinesq buoyancy and external thermal walls are explicit.
# This numerical component supplies no crystal growth, latent heat or risk labels.
"""Heat/solute transport with small-density-perturbation buoyancy."""

from dataclasses import dataclass

import numpy as np

from .scalar import ConservativeScalarTransport
from .solute import SoluteTransport
from .units import finite_number


def _source(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Properties and boundaries require a source_id")


@dataclass(frozen=True)
class HeatPropertiesSI:
    heat_capacity_J_per_kg_K: float
    conductivity_W_per_m_K: float
    reference_temperature_K: float
    source_id: str

    def __post_init__(self):
        for name in (
            "heat_capacity_J_per_kg_K",
            "conductivity_W_per_m_K",
            "reference_temperature_K",
        ):
            finite_number(getattr(self, name), name, positive=True)
        _source(self.source_id)


@dataclass(frozen=True)
class ThermalBoundary:
    """Ambient temperature [K] and film h [W/m²/K]; None means Dirichlet.

    Ambient can be a function of simulation time [s]. h=0 means insulation.
    The half-cell conductive resistance remains in series with the film.
    """

    ambient_temperature_K: object
    heat_transfer_W_per_m2_K: float | None
    source_id: str

    def __post_init__(self):
        self.temperature(0.0)
        if self.heat_transfer_W_per_m2_K is not None:
            finite_number(
                self.heat_transfer_W_per_m2_K,
                "heat_transfer_W_per_m2_K",
                nonnegative=True,
            )
        _source(self.source_id)

    def temperature(self, time_s):
        value = self.ambient_temperature_K
        return finite_number(
            value(time_s) if callable(value) else value,
            "ambient_temperature_K",
            positive=True,
        )


class HeatTransport(ConservativeScalarTransport):
    """Kelvin field, constant rho/cp/k, adiabatic aligned solids and outer walls.

    Declared thermal walls replace insulation on outer liquid faces only. There
    is no solid conduction, crystallization heat or temperature-dependent rho/cp/k.
    """

    def __init__(self, flow, temperature_K, properties, *, boundaries=None):
        if not isinstance(properties, HeatPropertiesSI):
            raise ValueError("Heat transport requires HeatPropertiesSI")
        self.properties = properties
        self.volumetric_heat_capacity = (
            flow.properties.density_kg_per_m3 * properties.heat_capacity_J_per_kg_K
        )
        super().__init__(
            flow,
            temperature_K,
            diffusivity_m2_per_s=(
                properties.conductivity_W_per_m_K / self.volumetric_heat_capacity
            ),
        )
        self._walls = []
        boundaries = {} if boundaries is None else dict(boundaries)
        names = {
            f"{name}_{side}": (axis, side)
            for axis, name in enumerate("zyx")
            for side in ("min", "max")
        }
        if set(boundaries) - names.keys():
            raise ValueError("Unknown thermal wall; use x/y/z_min/max")
        coordinates = np.array(np.unravel_index(self._active, flow.shape))
        for name, boundary in boundaries.items():
            axis, side = names[name]
            if not isinstance(boundary, ThermalBoundary):
                raise ValueError("Thermal walls require ThermalBoundary")
            if flow.periodic[axis]:
                raise ValueError("A periodic direction cannot have thermal walls")
            h = boundary.heat_transfer_W_per_m2_K
            if h == 0:
                continue
            selected = np.flatnonzero(
                coordinates[axis] == (0 if side == "min" else flow.shape[axis] - 1)
            )
            width = self._widths[axis, selected]
            area = self._volumes[selected] / width
            resistance = width / (2 * properties.conductivity_W_per_m_K)
            if h is not None:
                resistance += 1 / h
            rate = (
                area
                / resistance
                / (self.volumetric_heat_capacity * self._volumes[selected])
            )
            self._walls.append((selected, rate, boundary))
        self.initial_energy_J = self.sensible_energy_J

    def _field(self, value):
        field = super()._field(value)
        if np.any(field[self._fluid] <= 0):
            raise ValueError("Liquid temperature must be positive Kelvin")
        return field

    @property
    def temperature(self):
        return self.values

    @property
    def sensible_energy_J(self):
        return float(
            self.volumetric_heat_capacity
            * np.dot(
                self.values[self._fluid] - self.properties.reference_temperature_K,
                self._volumes,
            )
        )

    def _boundary_state(self, time_s):
        if not self._walls:
            return None
        count = len(self._active)
        rate, source = np.zeros(count), np.zeros(count)
        lower, upper = np.full(count, np.inf), np.full(count, -np.inf)
        for selected, coefficient, boundary in self._walls:
            temperature = boundary.temperature(time_s)
            np.add.at(rate, selected, coefficient)
            np.add.at(source, selected, coefficient * temperature)
            np.minimum.at(lower, selected, temperature)
            np.maximum.at(upper, selected, temperature)
        return rate, source, lower, upper

    def measurements(self):
        values = self.temperature[self._fluid]
        energy = self.sensible_energy_J
        boundary_heat = self.volumetric_heat_capacity * self.boundary_integral
        return dict(
            time_s=self.time_s,
            min_temperature_K=float(values.min()),
            max_temperature_K=float(values.max()),
            sensible_energy_J=energy,
            boundary_heat_J=boundary_heat,
            heat_budget_residual_J=energy - self.initial_energy_J - boundary_heat,
            heat_substeps=self.substeps,
            heat_limited_face_stages=self.limited_face_stages,
        )


@dataclass(frozen=True)
class BoussinesqProperties:
    """rho'/rho0=-beta_T*(T-T0)+beta_C*(C-C0), C in kg/m³ solution.

    Only the buoyancy term uses this perturbation. The remaining incompressible
    equations use constant rho0. Gravity is a Cartesian acceleration [m/s²].
    """

    thermal_expansion_per_K: float
    solutal_expansion_m3_per_kg: float
    reference_temperature_K: float
    reference_concentration_kg_per_m3: float
    gravity_m_per_s2: tuple
    maximum_density_perturbation_fraction: float
    source_id: str

    def __post_init__(self):
        for name in (
            "thermal_expansion_per_K",
            "solutal_expansion_m3_per_kg",
            "reference_concentration_kg_per_m3",
        ):
            finite_number(getattr(self, name), name, nonnegative=True)
        finite_number(
            self.reference_temperature_K, "reference_temperature_K", positive=True
        )
        fraction = finite_number(
            self.maximum_density_perturbation_fraction,
            "maximum_density_perturbation_fraction",
            positive=True,
        )
        if fraction > 0.1:
            raise ValueError("Boussinesq perturbation budget must be <=0.1")
        if len(self.gravity_m_per_s2) != 3:
            raise ValueError("Gravity requires three Cartesian components")
        for value in self.gravity_m_per_s2:
            finite_number(value, "gravity_m_per_s2")
        _source(self.source_id)

    def density_fraction(self, temperature, concentration):
        thermal = self.thermal_expansion_per_K * (
            temperature - self.reference_temperature_K
        )
        solutal = self.solutal_expansion_m3_per_kg * (
            concentration - self.reference_concentration_kg_per_m3
        )
        if not np.isfinite(thermal).all() or not np.isfinite(solutal).all():
            raise ValueError("Buoyancy fields must be finite")
        if (
            np.max(abs(thermal) + abs(solutal), initial=0)
            > self.maximum_density_perturbation_fraction
        ):
            raise ValueError(
                "Density perturbation exceeds the declared Boussinesq budget"
            )
        return -thermal + solutal


class FlowHeatSoluteSimulation:
    """First-order explicit buoyancy coupling with bounded RK2 scalar steps.

    Mean old/new projected volume flux advects both scalars. Physical state rolls
    back together if any stage fails; reusable linear-system caches may persist.
    """

    def __init__(
        self,
        flow,
        temperature_K,
        concentration_kg_per_m3,
        heat_properties,
        buoyancy,
        *,
        thermal_boundaries=None,
    ):
        if not isinstance(buoyancy, BoussinesqProperties):
            raise ValueError("Coupling requires BoussinesqProperties")
        self.flow, self.buoyancy = flow, buoyancy
        self.heat = HeatTransport(
            flow, temperature_K, heat_properties, boundaries=thermal_boundaries
        )
        self.solute = SoluteTransport(flow, concentration_kg_per_m3)
        if (
            buoyancy.reference_concentration_kg_per_m3
            >= flow.properties.density_kg_per_m3
        ):
            raise ValueError("Reference concentration must be below solution density")
        self.buoyancy_acceleration()

    def buoyancy_acceleration(self):
        flow = self.flow
        # Validate mutable public fields before momentum consumes them.
        temperature = self.heat._field(self.heat.temperature)
        concentration = self.solute._field(self.solute.concentration)
        fraction = np.zeros(flow.shape)
        fraction[flow.fluid] = self.buoyancy.density_fraction(
            temperature[flow.fluid], concentration[flow.fluid]
        )
        result = []
        for index, component in enumerate(flow.components):
            hi = component.cells
            lo = hi.copy()
            lo[:, component.axis] -= 1
            if flow.periodic[component.axis]:
                lo[:, component.axis] %= flow.shape[component.axis]
            width = flow.grid.widths[component.axis]
            a, b = width[hi[:, component.axis]], width[lo[:, component.axis]]
            face = (a * fraction[tuple(lo.T)] + b * fraction[tuple(hi.T)]) / (a + b)
            acceleration = np.zeros((*component.mask.shape, 3))
            acceleration[tuple(hi.T)] = face[:, None] * np.asarray(
                self.buoyancy.gravity_m_per_s2
            )
            if flow.periodic[component.axis]:
                first, last = [slice(None)] * 4, [slice(None)] * 4
                first[component.axis], last[component.axis] = 0, -1
                acceleration[tuple(last)] = acceleration[tuple(first)]
            result.append(acceleration)
        return result

    def _clocks(self):
        if not (self.flow.time_s == self.heat.time_s == self.solute.time_s):
            raise ValueError("Advance coupled flow, heat and solute clocks together")

    def step(self, dt_s, acceleration=(0.0, 0.0, 0.0)):
        self._clocks()
        dt = finite_number(dt_s, "dt_s", positive=True)
        self.heat._fluxes()
        self.solute._fluxes()
        buoyancy = self.buoyancy_acceleration()

        def force(positions, time_s):
            index = next(
                i for i, field in enumerate(buoyancy) if field.shape == positions.shape
            )
            external = np.asarray(
                acceleration(positions, time_s)
                if callable(acceleration)
                else acceleration,
                dtype=float,
            )
            if (
                external.shape not in ((3,), positions.shape)
                or not np.isfinite(external).all()
            ):
                raise ValueError(
                    "External acceleration must be finite Cartesian vector or face field"
                )
            return buoyancy[index] + external

        old_flux = self.flow.volume_fluxes_m3_per_s()
        flow_state = self.flow.velocity, self.flow.pressure, self.flow.time_s
        scalars = (self.heat, self.solute)
        scalar_states = [
            (s.values, s.time_s, s.substeps, s.limited_face_stages, s.boundary_integral)
            for s in scalars
        ]
        try:
            self.flow.step(dt, force)
            mean_flux = [
                0.5 * (a + b)
                for a, b in zip(old_flux, self.flow.volume_fluxes_m3_per_s())
            ]
            for scalar in scalars:
                scalar.advance(dt, volume_fluxes_m3_per_s=mean_flux)
            # Heating or redistribution must not silently invalidate the next step.
            self.buoyancy_acceleration()
        except Exception:
            self.flow.velocity, self.flow.pressure, self.flow.time_s = flow_state
            for scalar, state in zip(scalars, scalar_states):
                (
                    scalar.values,
                    scalar.time_s,
                    scalar.substeps,
                    scalar.limited_face_stages,
                    scalar.boundary_integral,
                ) = state
            raise
        return self.measurements()

    def run_until(self, final_time_s, max_dt_s, acceleration=(0.0, 0.0, 0.0)):
        self._clocks()
        final = finite_number(final_time_s, "final_time_s", nonnegative=True)
        maximum = finite_number(max_dt_s, "max_dt_s", positive=True)
        if final < self.flow.time_s:
            raise ValueError("Coupled transport cannot integrate backward")
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
            **{
                k: v
                for s in (self.heat, self.solute)
                for k, v in s.measurements().items()
                if k != "time_s"
            },
        )
