# HOW THIS FILE WORKS
# Solve constant-density, isothermal, incompressible 3D Navier-Stokes in SI units.
# Pressure is cell-centered; x/y/z velocities live on their normal grid faces.
# A bounded linear upwind momentum predictor includes nonlinear advection, physical
# acceleration, the old pressure gradient and implicit viscosity. Incremental
# pressure correction removes divergence without discarding established pressure.
# Closed faces enforce no penetration; reflected tangential ghosts impose no-slip
# walls/solid and a shear-free rigid lid. Periodic axes support analytic tests.
# This first verified momentum discretization uses uniform cells and a stationary
# grid-aligned solid. Canonical Crystal intersections reject partial cut cells;
# no sloped solid is silently rounded to voxels. Growth coupling is a later step.
# Pressure is a dynamic pressure in Pa with a mean-zero gauge, not an absolute
# free-surface stress condition. Sources/method limits are recorded in SOURCES.md.

"""Staggered-grid SI flow for fixed, grid-aligned three-dimensional domains."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, factorized

from .crystal import Crystal
from .units import FluidPropertiesSI, finite_number


def _solve(system, rhs, guess=None, direct=None):
    """Solve an SPD system, checking the actual residual instead of clipping it."""
    if not np.any(rhs):
        return np.zeros_like(rhs)
    if direct is not None:
        answer = direct(rhs)
        status = 0
    else:
        answer, status = cg(
            system,
            rhs,
            x0=guess,
            M=diags(1 / system.diagonal()),
            rtol=2e-13,
            atol=0.0,
            maxiter=4000,
        )
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    if (
        status != 0
        or not np.isfinite(answer).all()
        or np.linalg.norm(system @ answer - rhs) > 3e-12 * scale
    ):
        raise ArithmeticError("Flow linear solve failed its residual check")
    return answer


@dataclass
class _Component:
    axis: int
    mask: np.ndarray
    cells: np.ndarray
    lookup: np.ndarray
    neighbors: list
    ghosts: list
    laplacian: object


class IncompressibleFlow:
    """Uniform MAC grid with fixed solid and first-order projection stepping.

    User coordinates/counts/velocity components follow x/y/z. Scalar arrays index
    z/y/x. A periodic normal direction retains a duplicate end face for convenient
    divergence/export, but that alias never becomes a second independent unknown.
    """

    def __init__(
        self,
        lengths_m,
        counts_xyz,
        properties,
        *,
        periodic_xyz=(False, False, False),
        shear_free_top=True,
        solid_box_m=None,
    ):
        if len(lengths_m) != 3 or len(counts_xyz) != 3 or len(periodic_xyz) != 3:
            raise ValueError("Flow geometry needs three x/y/z dimensions")
        self.lengths = np.array(
            [finite_number(v, "length_m", positive=True) for v in lengths_m]
        )
        if any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 3
            for n in counts_xyz
        ):
            raise ValueError("Flow counts must be integers >=3 on all three axes")
        if any(not isinstance(value, bool) for value in periodic_xyz) or not isinstance(
            shear_free_top, bool
        ):
            raise ValueError("Periodic and shear-free flags must be booleans")
        if not isinstance(properties, FluidPropertiesSI):
            raise ValueError("Flow requires FluidPropertiesSI")
        self.properties = properties
        self.shape = tuple(int(n) for n in counts_xyz[::-1])
        self.periodic = tuple(periodic_xyz[::-1])
        self.spacing = self.lengths[::-1] / np.array(self.shape)
        self.cell_volume = float(np.prod(self.spacing))
        self.shear_free_top = shear_free_top
        self.fluid = np.ones(self.shape, dtype=bool)
        self.solid_vertices_m = np.empty((0, 3))
        if solid_box_m is not None:
            self._exclude_solid(solid_box_m)
        self.components = [self._component(axis) for axis in (2, 1, 0)]
        self.velocity = [np.zeros(c.mask.shape) for c in self.components]
        self.pressure = np.zeros(self.shape)
        self.time_s = 0.0
        self._viscous_cache = {}
        self._pressure_system()

    def _exclude_solid(self, bounds):
        bounds = np.asarray(bounds, dtype=float)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[0] <= 0)
            or np.any(bounds[1] >= self.lengths)
            or np.any(bounds[1] <= bounds[0])
        ):
            raise ValueError(
                "A fixed solid box must fit strictly inside the flow domain"
            )
        # Normalize geometry before calling the existing convex-solid code. Its
        # tolerances were verified at model scales, not arbitrary SI magnitudes.
        scale = float(self.lengths.max())
        box = Crystal.box(bounds.mean(axis=0) / scale, (bounds[1] - bounds[0]) / scale)
        self.solid_vertices_m = box.vertices * scale
        lower = np.floor(bounds[0] / self.spacing[::-1]).astype(int)
        upper = np.ceil(bounds[1] / self.spacing[::-1]).astype(int)
        for z in range(lower[2], upper[2]):
            for y in range(lower[1], upper[1]):
                for x in range(lower[0], upper[0]):
                    lo = np.array([x, y, z]) * self.spacing[::-1] / scale
                    hi = lo + self.spacing[::-1] / scale
                    fraction = box.clipped_volume(lo, hi) / (
                        self.cell_volume / scale**3
                    )
                    if 1e-8 < fraction < 1 - 1e-8:
                        raise ValueError(
                            "Momentum cut cells are not implemented; align the fixed solid with grid faces"
                        )
                    if fraction >= 1 - 1e-8:
                        self.fluid[z, y, x] = False

    def _component(self, axis):
        shape = list(self.shape)
        shape[axis] += 1
        mask = np.zeros(shape, dtype=bool)
        inside, low, high = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
        inside[axis], low[axis], high[axis] = (
            slice(1, -1),
            slice(None, -1),
            slice(1, None),
        )
        mask[tuple(inside)] = self.fluid[tuple(low)] & self.fluid[tuple(high)]
        if self.periodic[axis]:
            first, last = [slice(None)] * 3, [slice(None)] * 3
            first[axis], last[axis] = 0, -1
            edge = self.fluid[tuple(first)] & self.fluid[tuple(last)]
            mask[tuple(first)] = mask[tuple(last)] = edge
        independent = mask.copy()
        if self.periodic[axis]:
            end = [slice(None)] * 3
            end[axis] = -1
            independent[tuple(end)] = False
        cells = np.argwhere(independent)
        lookup = np.full(shape, -1, dtype=int)
        lookup[tuple(cells.T)] = np.arange(len(cells))
        if self.periodic[axis]:
            first, last = [slice(None)] * 3, [slice(None)] * 3
            first[axis], last[axis] = 0, -1
            lookup[tuple(last)] = lookup[tuple(first)]
        neighbors, ghosts = [], []
        rows, columns, values = [], [], []
        diagonal = np.zeros(len(cells))
        for direction_axis in range(3):
            ns, gs = [], []
            for sign in (-1, 1):
                other = cells.copy()
                other[:, direction_axis] += sign
                if self.periodic[direction_axis]:
                    other[:, direction_axis] %= self.shape[direction_axis]
                valid = np.all((other >= 0) & (other < np.array(shape)), axis=1)
                ids = np.full(len(cells), -1, dtype=int)
                ids[valid] = lookup[tuple(other[valid].T)]
                # Normal fixed faces lie one spacing away; tangential no-slip
                # walls lie halfway to the missing face and use reflected ghosts.
                ghost = np.full(len(cells), 0.0 if direction_axis == axis else -1.0)
                if (
                    self.shear_free_top
                    and direction_axis == 0
                    and axis != 0
                    and sign == 1
                    and not self.periodic[0]
                ):
                    ghost[other[:, 0] == self.shape[0]] = 1.0
                coefficient = 1 / self.spacing[direction_axis] ** 2
                present = ids >= 0
                diagonal += coefficient * np.where(present, 1.0, 1 - ghost)
                rows.extend(np.flatnonzero(present))
                columns.extend(ids[present])
                values.extend(np.full(present.sum(), -coefficient))
                ns.append(ids)
                gs.append(ghost)
            neighbors.append(ns)
            ghosts.append(gs)
        rows.extend(np.arange(len(cells)))
        columns.extend(np.arange(len(cells)))
        values.extend(diagonal)
        matrix = coo_matrix(
            (values, (rows, columns)), shape=(len(cells), len(cells))
        ).tocsr()
        return _Component(axis, mask, cells, lookup, neighbors, ghosts, matrix)

    def _pressure_system(self):
        cells = np.argwhere(self.fluid)
        lookup = np.full(self.shape, -1, dtype=int)
        lookup[tuple(cells.T)] = np.arange(len(cells))
        rows, columns, values = [], [], []
        diagonal = np.zeros(len(cells))
        for axis in range(3):
            for sign in (-1, 1):
                other = cells.copy()
                other[:, axis] += sign
                if self.periodic[axis]:
                    other[:, axis] %= self.shape[axis]
                valid = np.all((other >= 0) & (other < np.array(self.shape)), axis=1)
                ids = np.full(len(cells), -1, dtype=int)
                ids[valid] = lookup[tuple(other[valid].T)]
                present = ids >= 0
                weight = 1 / self.spacing[axis] ** 2
                diagonal[present] += weight
                rows.extend(np.flatnonzero(present))
                columns.extend(ids[present])
                values.extend(np.full(present.sum(), -weight))
        rows.extend(np.arange(len(cells)))
        columns.extend(np.arange(len(cells)))
        values.extend(diagonal)
        self._p_matrix = coo_matrix(
            (values, (rows, columns)), shape=(len(cells), len(cells))
        ).tocsr()
        count, labels = connected_components(self._p_matrix, directed=False)
        # One pressure gauge per disconnected fluid component. No pressure
        # Dirichlet condition is added to an already impermeable rigid lid.
        groups = [np.flatnonzero(labels == k) for k in range(count)]
        self._pressure_groups = []
        for nodes in groups:
            reduced = self._p_matrix[nodes[:-1]][:, nodes[:-1]].tocsc()
            direct = (
                factorized(reduced)
                if 0 < len(nodes) <= 2000 and len(nodes) > 1
                else None
            )
            self._pressure_groups.append((nodes, reduced, direct))
        self._p_cells = cells
        self._p_lookup = lookup

    def face_positions(self, component):
        axis = self.components[component].axis
        coordinates = [
            (np.arange(self.shape[d] + (d == axis)) + (0 if d == axis else 0.5))
            * self.spacing[d]
            for d in range(3)
        ]
        z, y, x = np.meshgrid(*coordinates, indexing="ij")
        return np.stack((x, y, z), axis=-1)

    def _enforce_boundaries(self, fields):
        if len(fields) != 3:
            raise ValueError("Supply all three velocity components")
        result = []
        for field, component in zip(fields, self.components):
            value = np.array(field, dtype=float, copy=True)
            if value.shape != component.mask.shape or not np.isfinite(value).all():
                raise ValueError(
                    "Velocity fields must be finite and match their staggered grids"
                )
            value[~component.mask] = 0.0
            if self.periodic[component.axis]:
                first, last = [slice(None)] * 3, [slice(None)] * 3
                first[component.axis], last[component.axis] = 0, -1
                value[tuple(last)] = value[tuple(first)]
            result.append(value)
        if len(result) != 3:
            raise ValueError("Supply all three velocity components")
        return result

    def set_velocity(self, fields, *, project=False):
        fields = self._enforce_boundaries(fields)
        if project:
            fields, self.pressure = self._project(fields, 1.0)
        self.velocity = fields

    def centered_velocity(self, fields=None):
        fields = self.velocity if fields is None else fields
        result = []
        for value, component in zip(fields, self.components):
            lower, upper = [slice(None)] * 3, [slice(None)] * 3
            lower[component.axis], upper[component.axis] = (
                slice(None, -1),
                slice(1, None),
            )
            result.append(0.5 * (value[tuple(lower)] + value[tuple(upper)]))
        return np.stack(result, axis=-1)

    def set_pressure(self, pressure_Pa):
        """Initialize dynamic pressure in Pa; choose a gauge in each fluid region."""
        pressure = np.array(pressure_Pa, dtype=float, copy=True)
        if pressure.shape != self.shape or not np.isfinite(pressure).all():
            raise ValueError("Pressure must be finite and match cell centers")
        values = pressure[self.fluid]
        for nodes, _system, _direct in self._pressure_groups:
            values[nodes] -= values[nodes].mean()
        pressure[self.fluid] = values
        pressure[~self.fluid] = 0
        self.pressure = pressure

    def _pressure_gradient(self, component):
        left, right = component.cells.copy(), component.cells.copy()
        left[:, component.axis] -= 1
        if self.periodic[component.axis]:
            left[:, component.axis] %= self.shape[component.axis]
        return (
            self.pressure[tuple(right.T)] - self.pressure[tuple(left.T)]
        ) / self.spacing[component.axis]

    def divergence(self, fields=None):
        fields = self.velocity if fields is None else fields
        return sum(
            np.diff(value, axis=c.axis) / self.spacing[c.axis]
            for value, c in zip(fields, self.components)
        )

    def _project(self, fields, dt):
        rhs = -self.divergence(fields)[self.fluid]
        velocity_scale = max(
            float(np.max(abs(u), initial=0)) / self.spacing[c.axis]
            for u, c in zip(fields, self.components)
        )
        phi = np.zeros(len(rhs))
        for nodes, system, direct in self._pressure_groups:
            local = rhs[nodes].copy()
            scale = max(
                float(np.linalg.norm(local)),
                velocity_scale * np.sqrt(len(nodes)),
                1e-30,
            )
            if abs(float(local.sum())) > 1e-10 * scale * max(1, np.sqrt(len(nodes))):
                raise ArithmeticError(
                    "Pressure projection has an incompatible net volume flux"
                )
            local -= local.mean()
            if len(nodes) > 1:
                phi[nodes[:-1]] = _solve(system, local[:-1], direct=direct)
            phi[nodes] -= phi[nodes].mean()
        corrected = []
        for value, component in zip(fields, self.components):
            cells = component.cells
            left, right = cells.copy(), cells.copy()
            left[:, component.axis] -= 1
            if self.periodic[component.axis]:
                left[:, component.axis] %= self.shape[component.axis]
            p_left = phi[self._p_lookup[tuple(left.T)]]
            p_right = phi[self._p_lookup[tuple(right.T)]]
            updated = value.copy()
            updated[tuple(cells.T)] -= (p_right - p_left) / self.spacing[component.axis]
            corrected.append(updated)
        corrected = self._enforce_boundaries(corrected)
        pressure = np.zeros(self.shape)
        pressure[self.fluid] = phi * self.properties.density_kg_per_m3 / dt
        return corrected, pressure

    def _advection(self, index):
        component = self.components[index]
        cells = component.cells
        value = self.velocity[index][tuple(cells.T)]
        left, right = cells.copy(), cells.copy()
        left[:, component.axis] -= 1
        if self.periodic[component.axis]:
            left[:, component.axis] %= self.shape[component.axis]
        result = np.zeros(len(cells))
        for axis in range(3):
            neighbors = []
            for side in range(2):
                ids = component.neighbors[axis][side]
                other = component.ghosts[axis][side] * value
                present = ids >= 0
                other[present] = value[ids[present]]
                neighbors.append(other)
            backward, forward = value - neighbors[0], neighbors[1] - value
            # Monotonized central slopes reduce donor-cell numerical diffusion
            # without creating new reconstruction extrema. They become zero at
            # an extremum; unavailable boundary slopes use the low-order base.
            slope = np.where(
                backward * forward > 0,
                np.sign(backward)
                * np.minimum(
                    abs(0.5 * (backward + forward)),
                    np.minimum(2 * abs(backward), 2 * abs(forward)),
                ),
                0.0,
            )
            fluxes = []
            for side in range(2):
                ids = component.neighbors[axis][side]
                present = ids >= 0
                other_slope = np.zeros_like(slope)
                other_slope[present] = slope[ids[present]]
                if axis == component.axis:
                    speed = 0.5 * (value + neighbors[side])
                else:
                    a, b = left.copy(), right.copy()
                    a[:, axis] = b[:, axis] = cells[:, axis] + side
                    transverse = self.velocity[2 - axis]
                    speed = 0.5 * (transverse[tuple(a.T)] + transverse[tuple(b.T)])
                if side == 1:
                    donor = np.where(
                        speed >= 0,
                        value + 0.5 * slope,
                        neighbors[side] - 0.5 * other_slope,
                    )
                else:
                    donor = np.where(
                        speed >= 0,
                        neighbors[side] + 0.5 * other_slope,
                        value - 0.5 * slope,
                    )
                fluxes.append(speed * donor)
            result += (fluxes[1] - fluxes[0]) / self.spacing[axis]
        return result

    def step(self, dt_s, acceleration=(0.0, 0.0, 0.0)):
        dt = finite_number(dt_s, "dt_s", positive=True)
        cfl = dt * sum(
            float(np.max(abs(u), initial=0)) / self.spacing[c.axis]
            for u, c in zip(self.velocity, self.components)
        )
        if cfl > 0.45 + 1e-12:
            raise ValueError(
                "Explicit momentum advection requires CFL <=0.45; reduce dt_s"
            )
        predictor = []
        for index, component in enumerate(self.components):
            cells = component.cells
            old = self.velocity[index][tuple(cells.T)]
            force = (
                acceleration(self.face_positions(index), self.time_s)
                if callable(acceleration)
                else np.asarray(acceleration)
            )
            force = np.asarray(force, dtype=float)
            if force.shape == (3,):
                force_value = np.full(len(cells), force[index])
            elif force.shape == (*component.mask.shape, 3):
                force_value = force[..., index][tuple(cells.T)]
            else:
                raise ValueError(
                    "Acceleration must be an SI x/y/z vector or face-position vector field"
                )
            if not np.isfinite(force_value).all():
                raise ValueError("Acceleration must be finite")
            rhs = old + dt * (
                force_value
                - self._advection(index)
                - self._pressure_gradient(component) / self.properties.density_kg_per_m3
            )
            key = (index, dt)
            if key not in self._viscous_cache:
                system = (
                    eye(len(cells), format="csc")
                    + dt
                    * self.properties.kinematic_viscosity_m2_per_s
                    * component.laplacian
                ).tocsc()
                direct = factorized(system) if 0 < len(cells) <= 2000 else None
                # Keep only the current timestep per component; adaptive stepping
                # must not retain an unbounded sequence of sparse factorizations.
                self._viscous_cache = {
                    k: v for k, v in self._viscous_cache.items() if k[0] != index
                }
                self._viscous_cache[key] = (system, direct)
            system, direct = self._viscous_cache[key]
            answer = _solve(system, rhs, old, direct)
            field = np.zeros(component.mask.shape)
            field[tuple(cells.T)] = answer
            predictor.append(field)
        predictor = self._enforce_boundaries(predictor)
        velocity, correction = self._project(predictor, dt)
        self.velocity = velocity
        self.set_pressure(self.pressure + correction)
        self.time_s += dt
        return self.measurements()

    def run_until(self, final_time_s, max_dt_s, acceleration=(0.0, 0.0, 0.0)):
        final = finite_number(final_time_s, "final_time_s", nonnegative=True)
        maximum = finite_number(max_dt_s, "max_dt_s", positive=True)
        if final < self.time_s:
            raise ValueError("Flow cannot integrate backward in time")
        while self.time_s < final:
            rate = sum(
                float(np.max(abs(u), initial=0)) / self.spacing[c.axis]
                for u, c in zip(self.velocity, self.components)
            )
            dt = min(maximum, final - self.time_s, 0.4 / rate if rate else maximum)
            self.step(dt, acceleration)
            if final - self.time_s <= 1e-13 * max(final, 1.0):
                self.time_s = final
        return self.measurements()

    def measurements(self):
        speed = np.linalg.norm(self.centered_velocity(), axis=-1)
        face_scale = max(
            (
                float(np.max(abs(u), initial=0)) / self.spacing[c.axis]
                for u, c in zip(self.velocity, self.components)
            ),
            default=0.0,
        )
        div = float(np.max(abs(self.divergence()[self.fluid]), initial=0))
        energy = (
            0.5
            * self.properties.density_kg_per_m3
            * self.cell_volume
            * sum(
                float(np.sum(u[tuple(c.cells.T)] ** 2))
                for u, c in zip(self.velocity, self.components)
            )
        )
        leakage = max(
            (
                float(np.max(abs(u[~c.mask]), initial=0))
                for u, c in zip(self.velocity, self.components)
            ),
            default=0.0,
        )
        return dict(
            time_s=self.time_s,
            max_speed_m_per_s=float(speed[self.fluid].max(initial=0)),
            max_divergence_per_s=div,
            scaled_divergence=div / max(face_scale, 1e-30),
            blocked_face_speed_m_per_s=leakage,
            kinetic_energy_J=energy,
            mean_dynamic_pressure_Pa=float(self.pressure[self.fluid].mean()),
        )
