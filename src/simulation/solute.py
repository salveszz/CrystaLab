# HOW THIS FILE WORKS
# Carry C [kg KDP/m3 solution] on the fixed SI flow grid. Each independent open
# face transfers one signed solute mass between its two neighboring liquid cells.
# Donor advection plus centered diffusion supplies a positive low-order update.
# Limited linear reconstruction supplies an advective correction; paired flux
# budgets bound it by local concentration extrema without clipping solved fields.
# SSP RK2 combines two such conservative Euler stages. Explicit diffusion and
# advection share an outgoing-rate timestep limit and automatic subcycling.
# FlowSoluteSimulation advances momentum and carries solute with the average of
# the old/new divergence-free face fluxes. Geometry/density/properties remain fixed;
# there is no heat, buoyancy, interfacial uptake or calibrated crystal growth here.

"""Conservative, bounded SI solute transport on a fixed staggered flow grid."""

import math

import numpy as np

from .flow import IncompressibleFlow
from .units import finite_number


class SoluteTransport:
    """Passive advection-diffusion using the flow's native volume fluxes.

    ``advance`` holds its supplied fluxes fixed and advances only the solute.
    Use FlowSoluteSimulation to advance momentum and solute together. Diffusivity
    defaults to the sourced fluid property; an explicit override is useful for
    numerical tests, including pure advection. Concentration is per solution
    volume, distinct from the illustrative growth engine's carrier-volume basis.
    """

    def __init__(self, flow, concentration_kg_per_m3, *, diffusivity_m2_per_s=None):
        if not isinstance(flow, IncompressibleFlow):
            raise ValueError("Solute transport requires IncompressibleFlow")
        self.flow = flow
        self.diffusivity_m2_per_s = finite_number(
            flow.properties.solute_diffusivity_m2_per_s
            if diffusivity_m2_per_s is None
            else diffusivity_m2_per_s,
            "diffusivity_m2_per_s",
            nonnegative=True,
        )
        self._fluid = flow.fluid.copy()
        self._spacing = flow.spacing.copy()
        self._active = np.flatnonzero(self._fluid)
        if not len(self._active):
            raise ValueError("Solute transport requires liquid cells")
        lookup = np.full(flow.shape, -1, dtype=int)
        lookup.flat[self._active] = np.arange(len(self._active))
        left, right, axes, conductance = [], [], [], []
        self._faces = []
        # Component.cells omits the duplicate periodic end face. Each connection
        # therefore appears exactly once, including connections across a seam.
        for component in flow.components:
            hi = component.cells.copy()
            lo = hi.copy()
            lo[:, component.axis] -= 1
            if flow.periodic[component.axis]:
                lo[:, component.axis] %= flow.shape[component.axis]
            left.append(lookup[tuple(lo.T)])
            right.append(lookup[tuple(hi.T)])
            axes.append(np.full(len(hi), component.axis))
            conductance.append(
                np.full(
                    len(hi),
                    self.diffusivity_m2_per_s
                    * flow.cell_volume
                    / flow.spacing[component.axis] ** 2,
                )
            )
            self._faces.append(tuple(component.cells.T))
        self._i, self._j = np.concatenate(left), np.concatenate(right)
        self._axes = np.concatenate(axes)
        self._conductance = np.concatenate(conductance)
        self._neighbors = np.full((3, 2, len(self._active)), -1, dtype=int)
        for axis in range(3):
            select = self._axes == axis
            i, j = self._i[select], self._j[select]
            self._neighbors[axis, 1, i] = j
            self._neighbors[axis, 0, j] = i
        self.concentration = self._field(concentration_kg_per_m3)
        self.time_s = flow.time_s
        self.initial_mass_kg = self.mass_kg
        self.substeps = 0
        self.limited_face_stages = 0

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

    @property
    def mass_kg(self):
        return float(np.sum(self.concentration[self._fluid]) * self.flow.cell_volume)

    def _fluxes(self, supplied=None):
        flow = self.flow
        if not np.array_equal(flow.fluid, self._fluid) or not np.array_equal(
            flow.spacing, self._spacing
        ):
            raise ValueError("Solute geometry must remain fixed")
        fields = flow.volume_fluxes_m3_per_s() if supplied is None else supplied
        if len(fields) != 3:
            raise ValueError("Supply three staggered volume fluxes in x/y/z order")
        checked = []
        for value, component in zip(fields, flow.components):
            if np.iscomplexobj(value):
                raise ValueError("Volume fluxes must be real")
            field = np.asarray(value, dtype=float)
            if field.shape != component.mask.shape or not np.isfinite(field).all():
                raise ValueError("Volume fluxes must be finite and match flow faces")
            checked.append(field)
        scale = max(float(abs(f).max(initial=0)) for f in checked)
        tolerance = 1e-12 * max(scale, np.finfo(float).tiny)
        for field, component in zip(checked, flow.components):
            if np.max(abs(field[~component.mask]), initial=0) > tolerance:
                raise ValueError("Volume flux crosses a blocked wall or solid face")
            if flow.periodic[component.axis] and not np.allclose(
                np.take(field, 0, axis=component.axis),
                np.take(field, -1, axis=component.axis),
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError("Periodic volume-flux end faces must agree")
        flux = np.concatenate([f[ix] for f, ix in zip(checked, self._faces)])
        divergence = self._balance(flux)
        if np.max(abs(divergence), initial=0) > 1e-9 * max(scale, np.finfo(float).tiny):
            raise ValueError("Solute advection requires divergence-free volume fluxes")
        return flux

    def _balance(self, transfer):
        """Net mass/volume entering each cell; opposite signs share one value."""
        return np.bincount(
            self._j, weights=transfer, minlength=len(self._active)
        ) - np.bincount(self._i, weights=transfer, minlength=len(self._active))

    def _euler(self, old, flux, dt):
        i, j, volume = self._i, self._j, self.flow.cell_volume
        donor = np.where(flux >= 0, old[i], old[j])
        low_flux = flux * donor + self._conductance * (old[i] - old[j])
        low = old + self._balance(dt * low_flux) / volume
        slopes = np.zeros((3, len(old)))
        lower, upper = old.copy(), old.copy()
        for axis, (left, right) in enumerate(self._neighbors):
            a, b = old[np.maximum(left, 0)], old[np.maximum(right, 0)]
            a, b = np.where(left >= 0, a, old), np.where(right >= 0, b, old)
            lower = np.minimum(lower, np.minimum(a, b))
            upper = np.maximum(upper, np.maximum(a, b))
            backward, forward = old - a, b - old
            slopes[axis] = np.where(
                backward * forward > 0,
                np.sign(backward)
                * np.minimum(
                    abs(0.5 * (backward + forward)),
                    np.minimum(2 * abs(backward), 2 * abs(forward)),
                ),
                0.0,
            )
        # Antidiffusive mass, positive left -> right. Diffusion is already in the
        # low-order update. Limit total incoming/outgoing corrections jointly in
        # all three directions, rather than limiting each face independently.
        correction = (
            0.5
            * dt
            * flux
            * np.where(flux >= 0, slopes[self._axes, i], -slopes[self._axes, j])
        )
        positive, negative = np.maximum(correction, 0), np.maximum(-correction, 0)
        incoming = np.bincount(j, positive, len(old)) + np.bincount(
            i, negative, len(old)
        )
        outgoing = np.bincount(i, positive, len(old)) + np.bincount(
            j, negative, len(old)
        )
        add, remove = np.ones(len(old)), np.ones(len(old))
        np.divide(
            np.maximum(upper - low, 0) * volume,
            incoming,
            out=add,
            where=incoming > 0,
        )
        np.divide(
            np.maximum(low - lower, 0) * volume,
            outgoing,
            out=remove,
            where=outgoing > 0,
        )
        add, remove = np.minimum(add, 1), np.minimum(remove, 1)
        limiter = np.where(
            correction >= 0,
            np.minimum(remove[i], add[j]),
            np.minimum(add[i], remove[j]),
        )
        answer = low + self._balance(limiter * correction) / volume
        limited = int(np.count_nonzero((limiter < 1) & (correction != 0)))
        return answer, limited

    def advance(self, dt_s, *, volume_fluxes_m3_per_s=None, max_substep_s=None):
        """Advance passive solute, subcycling both RK stages under the same CFL.

        A failed validation/update leaves the stored scalar and time unchanged.
        Supplied volume fluxes are checked, never projected or silently zeroed.
        """
        dt = finite_number(dt_s, "dt_s", positive=True)
        maximum = (
            dt
            if max_substep_s is None
            else finite_number(max_substep_s, "max_substep_s", positive=True)
        )
        flux = self._fluxes(volume_fluxes_m3_per_s)
        old = self._field(self.concentration).flat[self._active].copy()
        rate = (
            np.bincount(
                self._i,
                np.maximum(flux, 0) + self._conductance,
                len(old),
            )
            + np.bincount(
                self._j,
                np.maximum(-flux, 0) + self._conductance,
                len(old),
            )
        ).max(initial=0) / self.flow.cell_volume
        count = math.ceil(dt / min(maximum, 0.45 / rate if rate else maximum))
        sub_dt = dt / count
        initial_min, initial_max = old.min(), old.max()
        tolerance = 3e-12 * max(float(initial_max), 1.0)
        limited = 0
        for _ in range(count):
            stage, first = self._euler(old, flux, sub_dt)
            stage, second = self._euler(stage, flux, sub_dt)
            old = 0.5 * old + 0.5 * stage
            if (
                not np.isfinite(old).all()
                or old.min() < initial_min - tolerance
                or old.max() > initial_max + tolerance
            ):
                raise ArithmeticError(
                    "Solute transport violated its concentration bounds"
                )
            limited += first + second
        result = np.zeros(self.flow.shape)
        result.flat[self._active] = old
        self.concentration = result
        self.time_s += dt
        self.substeps += count
        self.limited_face_stages += limited
        return self.measurements()

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
