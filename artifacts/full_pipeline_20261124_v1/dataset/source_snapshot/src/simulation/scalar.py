# FILE PURPOSE: Share conservative bounded scalar transport with explicit field units in wrappers.
# Internal transfers use actual finite volumes, SSP RK2 and limited reconstruction.
# Solute and thermal wrappers retain separate physical state and budgets.
"""Fixed-grid conservative scalar operator, independent of growth kinetics."""

import math

import numpy as np

from .flow import IncompressibleFlow
from .units import finite_number


class ConservativeScalarTransport:
    """Shared finite-volume advection/diffusion; subclasses assign field units.

    The scalar's volume integral is conserved with closed/periodic boundaries.
    No concentration, heat-capacity or equation-of-state meaning is implicit.
    """

    def __init__(self, flow, initial_values, *, diffusivity_m2_per_s):
        if not isinstance(flow, IncompressibleFlow):
            raise ValueError("Scalar transport requires IncompressibleFlow")
        self.flow = flow
        self.diffusivity_m2_per_s = finite_number(
            diffusivity_m2_per_s,
            "diffusivity_m2_per_s",
            nonnegative=True,
        )
        self._fluid = flow.fluid.copy()
        self._spacing = flow.spacing.copy()
        self._edges = tuple(e.copy() for e in flow.grid.edges)
        self._active = np.flatnonzero(self._fluid)
        if not len(self._active):
            raise ValueError("Scalar transport requires liquid cells")
        lookup = np.full(flow.shape, -1, dtype=int)
        lookup.flat[self._active] = np.arange(len(self._active))
        self._volumes = flow.cell_volumes[self._fluid].copy()
        coordinates = np.array(np.unravel_index(self._active, flow.shape))
        self._widths = np.array([w[ix] for w, ix in zip(flow.grid.widths, coordinates)])
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
                self.diffusivity_m2_per_s
                * flow.grid.face_areas[component.axis][tuple(hi.T)]
                / flow.grid.pressure_distances[component.axis][hi[:, component.axis]]
            )
            self._faces.append(tuple(component.cells.T))
        self._i, self._j = np.concatenate(left), np.concatenate(right)
        self._axes = np.concatenate(axes)
        self._conductance = np.concatenate(conductance)
        self._neighbors = np.full((3, 2, len(self._active)), -1, dtype=int)
        self._distances = np.ones((3, 2, len(self._active)))
        for axis in range(3):
            select = self._axes == axis
            i, j = self._i[select], self._j[select]
            self._neighbors[axis, 1, i] = j
            self._neighbors[axis, 0, j] = i
            distance = 0.5 * (self._widths[axis, i] + self._widths[axis, j])
            self._distances[axis, 1, i] = distance
            self._distances[axis, 0, j] = distance
        self.values = self._field(initial_values)
        self.time_s = flow.time_s
        self.initial_integral = self.volume_integral
        self.substeps = 0
        self.limited_face_stages = 0
        self.boundary_integral = 0.0

    def _field(self, value):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Scalar values must be real")
        field = np.array(raw, dtype=float, copy=True)
        if field.ndim == 0:
            field = np.full(self.flow.shape, float(field))
        if field.shape != self.flow.shape or not np.isfinite(field).all():
            raise ValueError("Scalar values must be finite and match the liquid grid")
        field[~self._fluid] = 0.0
        return field

    @property
    def volume_integral(self):
        return float(np.sum(self.values[self._fluid] * self._volumes))

    def _fluxes(self, supplied=None):
        flow = self.flow
        if (
            not np.array_equal(flow.fluid, self._fluid)
            or not np.array_equal(flow.spacing, self._spacing)
            or any(
                not np.array_equal(e, previous)
                for e, previous in zip(flow.grid.edges, self._edges)
            )
        ):
            raise ValueError("Scalar geometry must remain fixed")
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
            raise ValueError("Scalar advection requires divergence-free volume fluxes")
        return flux

    def _balance(self, transfer):
        """Net mass/volume entering each cell; opposite signs share one value."""
        return np.bincount(
            self._j, weights=transfer, minlength=len(self._active)
        ) - np.bincount(self._i, weights=transfer, minlength=len(self._active))

    def _boundary_state(self, time_s):
        """Optional (rate, source, ambient_min, ambient_max) linear wall exchange.

        rate [1/s] is constant in time; source has scalar-units/s. Subclasses
        validate coefficients and ambient values. None means insulated walls.
        """
        return None

    def _euler(self, old, flux, dt, boundary=None):
        i, j, volume = self._i, self._j, self._volumes
        donor = np.where(flux >= 0, old[i], old[j])
        low_flux = flux * donor + self._conductance * (old[i] - old[j])
        low = old + self._balance(dt * low_flux) / volume
        slopes = np.zeros((3, len(old)))
        lower, upper = old.copy(), old.copy()
        if boundary is not None:
            rate, source, ambient_min, ambient_max = boundary
            low += dt * (source - rate * old)
            lower = np.minimum(lower, ambient_min)
            upper = np.maximum(upper, ambient_max)
        for axis, (left, right) in enumerate(self._neighbors):
            a, b = old[np.maximum(left, 0)], old[np.maximum(right, 0)]
            a, b = np.where(left >= 0, a, old), np.where(right >= 0, b, old)
            lower = np.minimum(lower, np.minimum(a, b))
            upper = np.maximum(upper, np.maximum(a, b))
            backward = (old - a) / self._distances[axis, 0]
            forward = (b - old) / self._distances[axis, 1]
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
        donor_offset = np.where(
            flux >= 0,
            0.5 * self._widths[self._axes, i] * slopes[self._axes, i],
            -0.5 * self._widths[self._axes, j] * slopes[self._axes, j],
        )
        reconstructed = np.clip(
            donor + donor_offset, np.minimum(old[i], old[j]), np.maximum(old[i], old[j])
        )
        correction = dt * flux * (reconstructed - donor)
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
        """Advance the scalar, subcycling both RK stages under the same CFL.

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
        old = self._field(self.values).flat[self._active].copy()
        rates = (
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
        ) / self._volumes
        boundary = self._boundary_state(self.time_s)
        if boundary is not None:
            rates += boundary[0]
        rate = rates.max(initial=0)
        count = math.ceil(dt / min(maximum, 0.45 / rate if rate else maximum))
        sub_dt = dt / count
        initial_min, initial_max = old.min(), old.max()
        tolerance = 3e-12 * max(abs(float(initial_min)), abs(float(initial_max)), 1.0)
        limited = 0
        boundary_integral = 0.0
        for index in range(count):
            t = self.time_s + index * sub_dt
            first_boundary = self._boundary_state(t)
            if first_boundary is None:
                stage, first = self._euler(old, flux, sub_dt)
                stage, second = self._euler(stage, flux, sub_dt)
            else:
                second_boundary = self._boundary_state(t + sub_dt)
                stage, first = self._euler(old, flux, sub_dt, first_boundary)
                change = np.dot(
                    self._volumes, first_boundary[1] - first_boundary[0] * old
                )
                change += np.dot(
                    self._volumes, second_boundary[1] - second_boundary[0] * stage
                )
                boundary_integral += 0.5 * sub_dt * change
                stage, second = self._euler(stage, flux, sub_dt, second_boundary)
                initial_min = min(
                    initial_min, first_boundary[2].min(), second_boundary[2].min()
                )
                initial_max = max(
                    initial_max, first_boundary[3].max(), second_boundary[3].max()
                )
                tolerance = 3e-12 * max(
                    abs(float(initial_min)), abs(float(initial_max)), 1.0
                )
            old = 0.5 * old + 0.5 * stage
            if (
                not np.isfinite(old).all()
                or old.min() < initial_min - tolerance
                or old.max() > initial_max + tolerance
            ):
                raise ArithmeticError("Scalar transport violated its bounds")
            limited += first + second
        result = np.zeros(self.flow.shape)
        result.flat[self._active] = old
        self.values = result
        self.time_s += dt
        self.substeps += count
        self.limited_face_stages += limited
        self.boundary_integral += boundary_integral
        return self.measurements()

    def measurements(self):
        values = self.values[self._fluid]
        return dict(
            time_s=self.time_s,
            volume_integral=self.volume_integral,
            minimum_value=float(values.min()),
            maximum_value=float(values.max()),
            scalar_substeps=self.substeps,
            limited_face_stages=self.limited_face_stages,
        )
