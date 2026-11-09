# HOW THIS FILE WORKS
# Diffuse three liquid-volume mean velocity components on fixed cut polyhedra.
# A mimetic local flux metric uses actual volumes, face centroids and normals;
# shared face-average traces conserve viscous flux. Embedded and outer stationary
# walls have an exactly zero trace, including tangential velocity on oblique walls.
# Implicit Euler condenses cell means into a symmetric face system, retaining
# every positive liquid capacity. Diagnostics check flux and energy balances.
# This is a standalone constant-viscosity building block, not a coupled MAC
# Navier-Stokes step: pressure transfer, advection and moving walls remain pending.

"""Fixed cut-cell vector diffusion with stationary no-slip wall traces."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

from .cutcell import CutCellGeometry
from .linear import solve_spd
from .units import finite_number


@dataclass(frozen=True)
class ViscousState:
    """Volume means and auxiliary face means, both Cartesian x/y/z vectors."""

    cell_velocity_m_per_s: np.ndarray
    trace_velocity_m_per_s: np.ndarray


class CutCellViscosity:
    """Mimetic diffusion of constant-nu vector velocity on stationary geometry.

    All nonperiodic outer faces and embedded facets are Dirichlet. Optional
    prescribed OUTER face averages support manufactured tests; embedded walls
    always remain stationary. No free-surface stress condition is implemented.
    Cell means are distinct from the existing pressure solver's MAC velocities.
    """

    def __init__(self, geometry):
        if not isinstance(geometry, CutCellGeometry) or not geometry.fluid.any():
            raise ValueError("Viscosity requires nonempty CutCellGeometry")
        self.geometry = geometry
        self.cells = np.argwhere(geometry.fluid)
        self.volumes = geometry.volumes_m3[geometry.fluid]
        self.face_ids = []
        centers, boundaries, embedded = [], [], []
        grid = geometry.grid
        for axis, area in enumerate(geometry.open_areas_m2):
            ids = np.full(area.shape, -1, dtype=int)
            if grid.periodic[axis]:
                if not np.allclose(
                    area.take(0, axis), area.take(-1, axis), rtol=1e-11, atol=0
                ):
                    raise ValueError("Periodic end apertures must match")
                first = geometry.face_centroids_m[axis].take(0, axis)
                last = geometry.face_centroids_m[axis].take(-1, axis)
                transverse = [i for i in range(3) if i != 2 - axis]
                active = area.take(0, axis) > 0
                if not np.allclose(
                    first[active][:, transverse],
                    last[active][:, transverse],
                    rtol=1e-11,
                    atol=1e-13 * geometry.scale_m,
                ):
                    raise ValueError("Periodic end aperture centroids must match")
            for raw in np.argwhere(area > 0):
                face = tuple(raw)
                if grid.periodic[axis] and raw[axis] == grid.shape[axis]:
                    other = list(raw)
                    other[axis] = 0
                    ids[face] = ids[tuple(other)]
                    continue
                ids[face] = len(centers)
                centers.append(geometry.face_centroids_m[axis][face])
                boundaries.append(
                    not grid.periodic[axis] and raw[axis] in (0, grid.shape[axis])
                )
                embedded.append(False)
            self.face_ids.append(ids)
        self.wall_ids = np.arange(len(centers), len(centers) + len(geometry.wall_cells))
        centers.extend(geometry.wall_centroids_m)
        boundaries.extend([True] * len(self.wall_ids))
        embedded.extend([True] * len(self.wall_ids))
        self.face_centroids_m = np.array(centers)
        self.boundary = np.array(boundaries)
        self.embedded = np.array(embedded)
        self.boundary_ids = np.flatnonzero(self.boundary)
        self._free = np.flatnonzero(~self.boundary)
        walls = {}
        for wall, cell in enumerate(geometry.wall_cells):
            walls.setdefault(tuple(cell), []).append(wall)
        self._local = []
        self.max_relative_moment_defect = 0.0
        owners = [[] for _ in centers]
        for index, raw in enumerate(self.cells):
            cell = tuple(raw)
            ids, areas, positions, normals = [], [], [], []
            for axis in range(3):
                for side in (0, 1):
                    face = list(raw)
                    face[axis] += side
                    face = tuple(face)
                    area = geometry.open_areas_m2[axis][face]
                    if area <= 0:
                        continue
                    ids.append(self.face_ids[axis][face])
                    areas.append(area)
                    positions.append(geometry.face_centroids_m[axis][face])
                    normal = np.zeros(3)
                    normal[2 - axis] = 2 * side - 1
                    normals.append(normal)
            for wall in walls.get(cell, []):
                ids.append(self.wall_ids[wall])
                areas.append(geometry.wall_areas_m2[wall])
                positions.append(geometry.wall_centroids_m[wall])
                normals.append(geometry.wall_normals[wall])
            ids, areas, normals = np.array(ids), np.array(areas), np.array(normals)
            r = areas[:, None] * (np.array(positions) - geometry.centroids_m[cell])
            volume = self.volumes[index]
            defect = np.max(abs(r.T @ normals / volume - np.eye(3)))
            self.max_relative_moment_defect = max(
                self.max_relative_moment_defect, float(defect)
            )
            # Brezzi/Lipnikov/Simoncini (2005), eqs. 3.9-3.10, K=I.
            # QR supplies an orthogonal projector onto ker(N.T).
            q, _ = np.linalg.qr(normals, mode="complete")
            m0 = (r / np.sqrt(volume)) @ (r / np.sqrt(volume)).T
            metric = m0 + np.trace(m0) / 3 * (q[:, 3:] @ q[:, 3:].T)
            try:
                np.linalg.cholesky(metric)
                stiffness = areas[:, None] * np.linalg.solve(metric, np.diag(areas))
            except np.linalg.LinAlgError as error:
                raise ArithmeticError("Degenerate viscous flux metric") from error
            stiffness = (stiffness + stiffness.T) / 2
            b = stiffness.sum(axis=1)
            self._local.append((ids, stiffness, b, float(b.sum())))
            for face_id in ids:
                owners[face_id].append(index)
        rows, cols = [], []
        anchored = np.zeros(len(self.cells), dtype=bool)
        for face_id, cells in enumerate(owners):
            expected = 1 if self.boundary[face_id] else 2
            if len(cells) != expected:
                raise ValueError("Open viscous face has inconsistent liquid neighbors")
            if expected == 1:
                anchored[cells[0]] = True
            else:
                rows.extend(cells)
                cols.extend(cells[::-1])
        graph = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(self.cells),) * 2
        ).tocsr()
        count, labels = connected_components(graph, directed=False)
        self._steady_anchored = all(anchored[labels == i].any() for i in range(count))
        self._cache = None
        for array in (
            self.cells,
            self.volumes,
            *self.face_ids,
            self.wall_ids,
            self.face_centroids_m,
            self.boundary,
            self.embedded,
            self.boundary_ids,
            self._free,
        ):
            array.setflags(write=False)

    @staticmethod
    def _array(value, shape, name):
        raw = np.asarray(value)
        if (
            raw.dtype.kind not in "iuf"
            or raw.shape != shape
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                f"{name} requires finite numeric values with shape {shape}"
            )
        return raw.astype(float, copy=True)

    def _cells_array(self, value, name):
        shape = (*self.geometry.grid.shape, 3)
        result = self._array(value, shape, name)
        if np.any(result[~self.geometry.fluid]):
            raise ValueError(f"{name} must be zero in solid cells")
        return result[self.geometry.fluid]

    def _boundary_values(self, value):
        shape = (len(self.boundary_ids), 3)
        values = (
            np.zeros(shape)
            if value is None
            else self._array(value, shape, "Boundary velocity")
        )
        if np.any(values[self.embedded[self.boundary_ids]]):
            raise ValueError("Embedded walls must have zero velocity")
        full = np.zeros((len(self.boundary), 3))
        full[self.boundary_ids] = values
        return full

    def _system(self, ratio):
        if self._cache is not None and self._cache[0] == ratio:
            return self._cache[1:]
        rows, cols, data = [], [], []
        factors = []
        for volume, (ids, stiffness, b, total) in zip(self.volumes, self._local):
            denominator = total + (volume / ratio if ratio is not None else 0.0)
            factors.append(1 / denominator)
            local = stiffness - np.outer(b, b) / denominator
            rows.extend(np.repeat(ids, len(ids)))
            cols.extend(np.tile(ids, len(ids)))
            data.extend(local.ravel())
        matrix = coo_matrix(
            (data, (rows, cols)), shape=(len(self.boundary),) * 2
        ).tocsr()
        reduced = matrix[self._free][:, self._free].tocsc()
        direct = factorized(reduced) if 0 < len(self._free) <= 20000 else None
        factors = np.array(factors)
        self._cache = (ratio, matrix, reduced, direct, factors)
        return self._cache[1:]

    def _solve(self, acceleration, nu, boundary_velocity, previous=None, dt=None):
        nu = finite_number(nu, "kinematic viscosity", positive=True)
        boundary = self._boundary_values(boundary_velocity)
        force = self._cells_array(acceleration, "Acceleration")
        if dt is None:
            if not self._steady_anchored:
                raise ValueError(
                    "Steady diffusion needs a Dirichlet wall in every liquid region"
                )
            ratio = None
            load = self.volumes[:, None] * force / nu
        else:
            dt = finite_number(dt, "time step", positive=True)
            ratio = dt * nu
            old = self._cells_array(previous, "Previous velocity")
            load = self.volumes[:, None] * (old / ratio + force / nu)
        matrix, reduced, direct, factors = self._system(ratio)
        rhs = np.zeros_like(boundary)
        for i, (ids, _stiffness, b, _total) in enumerate(self._local):
            np.add.at(rhs, ids, b[:, None] * (load[i] * factors[i]))
        rhs -= matrix @ boundary
        trace = boundary.copy()
        for component in range(3):
            trace[self._free, component] = solve_spd(
                reduced, rhs[self._free, component], direct=direct
            )
        means = np.zeros((*self.geometry.grid.shape, 3))
        means[self.geometry.fluid] = np.array(
            [
                (load[i] + b @ trace[ids]) * factors[i]
                for i, (ids, _k, b, _s) in enumerate(self._local)
            ]
        )
        if not np.isfinite(means).all():
            raise ArithmeticError("Nonfinite viscous velocity")
        means.setflags(write=False)
        trace.setflags(write=False)
        return ViscousState(means, trace)

    def steady(
        self,
        acceleration_m_per_s2,
        kinematic_viscosity_m2_per_s,
        *,
        boundary_velocity_m_per_s=None,
    ):
        """Solve -nu Laplacian(u)=acceleration with specified outer face means."""
        return self._solve(
            acceleration_m_per_s2,
            kinematic_viscosity_m2_per_s,
            boundary_velocity_m_per_s,
        )

    def implicit(
        self,
        velocity_m_per_s,
        dt_s,
        kinematic_viscosity_m2_per_s,
        *,
        acceleration_m_per_s2=None,
    ):
        """One implicit Euler diffusion step with all walls stationary."""
        force = (
            np.zeros((*self.geometry.grid.shape, 3))
            if acceleration_m_per_s2 is None
            else acceleration_m_per_s2
        )
        return self._solve(
            force,
            kinematic_viscosity_m2_per_s,
            None,
            previous=velocity_m_per_s,
            dt=dt_s,
        )

    def normal_face_velocity(self, state):
        """Export normal traces in existing x/y/z MAC array shapes (no projection)."""
        result = []
        for axis in (2, 1, 0):
            ids = self.face_ids[axis]
            values = np.zeros(ids.shape)
            active = ids >= 0
            values[active] = state.trace_velocity_m_per_s[ids[active], 2 - axis]
            result.append(values)
        return tuple(result)

    def balances(self, state, nu, *, previous=None, dt_s=None, acceleration=None):
        """Integrated momentum residuals and kinetic-energy work identity (rho=1).

        Energy is the volume-weighted cell-mean quadrature, not exact unresolved
        subcell energy. Flux residuals have units m4/s2; powers have units m5/s3.
        """
        nu = finite_number(nu, "kinematic viscosity", positive=True)
        means = self._cells_array(state.cell_velocity_m_per_s, "Velocity")
        traces = self._array(
            state.trace_velocity_m_per_s, (len(self.boundary), 3), "Trace velocity"
        )
        force = (
            np.zeros_like(means)
            if acceleration is None
            else self._cells_array(acceleration, "Acceleration")
        )
        residual = -self.volumes[:, None] * force
        trace_flux = np.zeros_like(traces)
        dissipation = 0.0
        for i, (ids, k, _b, _s) in enumerate(self._local):
            difference = means[i] - traces[ids]
            flux = nu * (k @ difference)
            residual[i] += flux.sum(axis=0)
            np.add.at(trace_flux, ids, flux)
            dissipation += float(np.sum(difference * flux))
        energy = 0.5 * float(np.sum(self.volumes[:, None] * means**2))
        body_power = float(np.sum(self.volumes[:, None] * means * force))
        boundary_power = -float(
            np.sum(traces[self.boundary] * trace_flux[self.boundary])
        )
        work_residual = dissipation - body_power - boundary_power
        if previous is not None:
            dt_s = finite_number(dt_s, "time step", positive=True)
            old = self._cells_array(previous, "Previous velocity")
            residual += self.volumes[:, None] * (means - old) / dt_s
            previous_energy = 0.5 * float(np.sum(self.volumes[:, None] * old**2))
            change_energy = 0.5 * float(
                np.sum(self.volumes[:, None] * (means - old) ** 2)
            )
            work_residual += (energy - previous_energy + change_energy) / dt_s
        return dict(
            energy_m5_per_s2=energy,
            dissipation_m5_per_s3=dissipation,
            body_power_m5_per_s3=body_power,
            boundary_power_m5_per_s3=boundary_power,
            work_residual_m5_per_s3=work_residual,
            max_cell_momentum_residual_m4_per_s2=float(np.max(abs(residual))),
            max_interior_flux_residual_m4_per_s2=float(
                np.max(abs(trace_flux[self._free]), initial=0)
            ),
        )
