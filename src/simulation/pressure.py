# HOW THIS FILE WORKS
# Assemble one pressure graph from actual liquid capacities and open Cartesian
# apertures. Each shared face has one volume flux and pressure conductance A/d.
# Solve the volume-integrated Neumann system without dividing its rows by tiny
# cut-cell volumes, and impose a volume-mean gauge per connected liquid region.
# Compatible divergence and gradient operators project face velocities to zero
# net cell flux while decreasing their A*d weighted norm. Pressure samples stay
# at nominal Cartesian centers, separately from actual liquid-volume centroids.
# The canonical aligned-solid flow and standalone sloped-wall verification share
# this operator. It imposes impermeability; tangential no slip needs viscosity.

"""Compatible fixed-domain pressure projection on full or partial liquid cells."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized

from .linear import solve_spd
from .mesh import RectilinearGrid


@dataclass(frozen=True)
class PressureFaces:
    axis: int
    mask: np.ndarray
    cells: np.ndarray
    left: np.ndarray
    right: np.ndarray
    area_m2: np.ndarray
    distance_m: np.ndarray


class PressureProjection:
    """Zero-normal-flow projection; face fields follow x/y/z component order.

    Aperture arrays follow grid normal axes z/y/x. The positive A*d metric is
    a pressure-projection norm, not a measured cut-cell kinetic-energy integral.
    Two-point pressure differences form the first embedded-wall operator; no
    second-order boundary-gradient or complete cut-cell Navier-Stokes claim.
    """

    def __init__(self, grid, liquid_volumes_m3, open_areas_m2):
        if not isinstance(grid, RectilinearGrid):
            raise ValueError("Pressure projection requires RectilinearGrid")
        self.grid = grid
        self.volumes_m3 = self._array(
            liquid_volumes_m3, grid.shape, "liquid volumes"
        ).copy()
        if np.any(self.volumes_m3 < 0) or np.any(
            self.volumes_m3 > (1 + 1e-10) * grid.volumes
        ):
            raise ValueError("Liquid capacities must lie within their Cartesian cells")
        self.fluid = self.volumes_m3 > 0
        self.cells = np.argwhere(self.fluid)
        if not len(self.cells):
            raise ValueError("Pressure projection requires positive liquid volume")
        self._volumes = self.volumes_m3[self.fluid]
        if len(open_areas_m2) != 3:
            raise ValueError("Supply three z/y/x aperture arrays")
        self.areas_m2 = tuple(
            self._array(a, full.shape, "face apertures").copy()
            for a, full in zip(open_areas_m2, grid.face_areas)
        )
        for area, full in zip(self.areas_m2, grid.face_areas):
            if np.any(area < 0) or np.any(area > (1 + 1e-10) * full):
                raise ValueError("Open apertures must lie within their Cartesian faces")
        lookup = np.full(grid.shape, -1, dtype=int)
        lookup[tuple(self.cells.T)] = np.arange(len(self.cells))
        self.faces = []
        rows, columns, values = [], [], []
        diagonal = np.zeros(len(self.cells))
        for axis in (2, 1, 0):
            area = self.areas_m2[axis]
            mask = np.zeros(area.shape, dtype=bool)
            interior, lo, hi = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            interior[axis], lo[axis], hi[axis] = (
                slice(1, -1),
                slice(None, -1),
                slice(1, None),
            )
            mask[tuple(interior)] = (
                self.fluid[tuple(lo)]
                & self.fluid[tuple(hi)]
                & (area[tuple(interior)] > 0)
            )
            if grid.periodic[axis]:
                first, last = [slice(None)] * 3, [slice(None)] * 3
                first[axis], last[axis] = 0, -1
                if not np.allclose(
                    area[tuple(first)], area[tuple(last)], rtol=1e-11, atol=0
                ):
                    raise ValueError("Periodic end apertures must match")
                edge = (
                    self.fluid[tuple(first)]
                    & self.fluid[tuple(last)]
                    & (area[tuple(first)] > 0)
                )
                mask[tuple(first)] = mask[tuple(last)] = edge
            independent = mask.copy()
            if grid.periodic[axis]:
                end = [slice(None)] * 3
                end[axis] = -1
                independent[tuple(end)] = False
            cells = np.argwhere(independent)
            left = cells.copy()
            left[:, axis] -= 1
            if grid.periodic[axis]:
                left[:, axis] %= grid.shape[axis]
            i, j = lookup[tuple(left.T)], lookup[tuple(cells.T)]
            a = area[tuple(cells.T)]
            d = grid.pressure_distances[axis][cells[:, axis]]
            conductance = a / d
            np.add.at(diagonal, i, conductance)
            np.add.at(diagonal, j, conductance)
            rows.extend(np.r_[i, j])
            columns.extend(np.r_[j, i])
            values.extend(np.r_[-conductance, -conductance])
            self.faces.append(PressureFaces(axis, mask, cells, i, j, a, d))
        rows.extend(np.arange(len(self.cells)))
        columns.extend(np.arange(len(self.cells)))
        values.extend(diagonal)
        self.stiffness = coo_matrix(
            (values, (rows, columns)), shape=(len(self.cells), len(self.cells))
        ).tocsr()
        count, labels = connected_components(self.stiffness, directed=False)
        self.groups = []
        for index in range(count):
            nodes = np.flatnonzero(labels == index)
            reduced = self.stiffness[nodes[:-1]][:, nodes[:-1]].tocsc()
            direct = factorized(reduced) if 1 < len(nodes) <= 2000 else None
            self.groups.append((nodes, reduced, direct))
        for array in (
            self.volumes_m3,
            self.fluid,
            self.cells,
            self._volumes,
            *self.areas_m2,
        ):
            array.setflags(write=False)
        for face in self.faces:
            for array in (
                face.mask,
                face.cells,
                face.left,
                face.right,
                face.area_m2,
                face.distance_m,
            ):
                array.setflags(write=False)

    @staticmethod
    def _array(value, shape, name):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be real numbers")
        field = np.asarray(raw, dtype=float)
        if field.shape != shape or not np.isfinite(field).all():
            raise ValueError(f"{name} must be finite and have shape {shape}")
        return field

    def mean_zero(self, field):
        result = self._array(field, self.grid.shape, "pressure field").copy()
        values = result[self.fluid]
        for nodes, _system, _direct in self.groups:
            values[nodes] -= np.average(values[nodes], weights=self._volumes[nodes])
        result[self.fluid] = values
        result[~self.fluid] = 0
        return result

    def gradient_at_open_faces(self, field, component):
        values = self._array(field, self.grid.shape, "pressure field")[self.fluid]
        face = self.faces[component]
        return (values[face.right] - values[face.left]) / face.distance_m

    def gradient(self, field):
        result = []
        for index, face in enumerate(self.faces):
            value = np.zeros(face.mask.shape)
            value[tuple(face.cells.T)] = self.gradient_at_open_faces(field, index)
            self._periodic_alias(value, face.axis)
            result.append(value)
        return result

    def _periodic_alias(self, value, axis):
        if self.grid.periodic[axis]:
            first, last = [slice(None)] * 3, [slice(None)] * 3
            first[axis], last[axis] = 0, -1
            value[tuple(last)] = value[tuple(first)]

    def _velocities(self, fields):
        if len(fields) != 3:
            raise ValueError("Supply three staggered x/y/z velocity fields")
        checked = [
            self._array(value, face.mask.shape, "face velocity")
            for value, face in zip(fields, self.faces)
        ]
        scale = max(float(abs(value).max(initial=0)) for value in checked)
        tolerance = 1e-12 * max(scale, np.finfo(float).tiny)
        for value, face in zip(checked, self.faces):
            if np.max(abs(value[~face.mask]), initial=0) > tolerance:
                raise ValueError(
                    "Pressure projection requires zero blocked-face velocity"
                )
            if self.grid.periodic[face.axis] and not np.allclose(
                np.take(value, 0, axis=face.axis),
                np.take(value, -1, axis=face.axis),
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError("Periodic velocity end aliases must agree")
        return checked

    def volume_fluxes(self, velocities):
        fields = self._velocities(velocities)
        return [
            value * self.areas_m2[face.axis] for value, face in zip(fields, self.faces)
        ]

    def net_flux(self, velocities):
        fluxes = self.volume_fluxes(velocities)
        return sum(
            np.diff(value, axis=face.axis) for value, face in zip(fluxes, self.faces)
        )

    def divergence(self, velocities):
        return np.divide(
            self.net_flux(velocities),
            self.volumes_m3,
            out=np.zeros(self.grid.shape),
            where=self.fluid,
        )

    def _solve_rhs(self, rhs, scale):
        phi = np.zeros(len(self.cells))
        for nodes, system, direct in self.groups:
            local = rhs[nodes].copy()
            reference = max(
                float(np.linalg.norm(local)),
                scale * np.sqrt(len(nodes)),
                np.finfo(float).tiny,
            )
            if abs(float(local.sum())) > 1e-10 * reference * max(
                1, np.sqrt(len(nodes))
            ):
                raise ValueError(
                    "Incompatible net flux/source in a closed liquid region"
                )
            local -= local.sum() * self._volumes[nodes] / self._volumes[nodes].sum()
            if len(nodes) > 1:
                phi[nodes[:-1]] = solve_spd(system, local[:-1], direct=direct)
            phi[nodes] -= np.average(phi[nodes], weights=self._volumes[nodes])
        result = np.zeros(self.grid.shape)
        result[self.fluid] = phi
        return result

    def solve_poisson_source(self, source_per_s):
        """Solve -Laplacian(phi)=source with zero normal flux on every boundary.

        Source values are liquid-volume averages in 1/s; phi is in m2/s.
        Each connected component must have zero integrated source.
        """
        source = self._array(source_per_s, self.grid.shape, "Poisson source")
        rhs = source[self.fluid] * self._volumes
        return self._solve_rhs(rhs, float(abs(rhs).max(initial=0)))

    def project(self, velocities):
        fields = self._velocities(velocities)
        fluxes = self.volume_fluxes(fields)
        rhs = -self.net_flux(fields)[self.fluid]
        phi = self._solve_rhs(rhs, max(float(abs(q).max(initial=0)) for q in fluxes))
        corrected = []
        for component, (value, face) in enumerate(zip(fields, self.faces)):
            updated = value.copy()
            updated[tuple(face.cells.T)] -= self.gradient_at_open_faces(phi, component)
            self._periodic_alias(updated, face.axis)
            corrected.append(updated)
        return corrected, phi

    def projection_norm_squared(self, velocities):
        fields = self._velocities(velocities)
        return sum(
            float(
                np.sum(face.area_m2 * face.distance_m * value[tuple(face.cells.T)] ** 2)
            )
            for value, face in zip(fields, self.faces)
        )
