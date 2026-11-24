# HOW THIS FILE WORKS
# Define one rectilinear mesh from explicit Cartesian edges or uniform dimensions.
# Validate physical coordinates and expose actual primal cell widths, centers,
# volumes and staggered face areas. MAC pressure distances and momentum-control
# widths include the wraparound connection on periodic axes. These measures keep
# pressure, viscosity, momentum and scalar transfers consistent on unequal cells.
# The focused-axis generator is shared with the illustrative growth tank; it
# changes resolution, not units, domain size or crystal geometry.

"""Rectilinear finite-volume geometry shared by SI flow and the growth grid."""

from numbers import Integral

import numpy as np

from .units import finite_number


def focused_axis_edges(length, count, center, focus_width):
    """Fixed rectilinear mesh with a uniform central band and stretched tails."""
    length = finite_number(length, "length", positive=True)
    center = finite_number(center, "center", nonnegative=True)
    focus_width = finite_number(focus_width, "focus_width", nonnegative=True)
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 3:
        raise ValueError("Axis cell count must be an integer >=3")
    if center > length:
        raise ValueError("Focus center must lie inside its axis")
    if focus_width <= 0 or focus_width >= length:
        return np.linspace(0.0, length, count + 1)
    low = max(0.0, center - focus_width / 2)
    high = min(length, center + focus_width / 2)
    tails = low + length - high
    if tails <= 0:
        return np.linspace(0.0, length, count + 1)
    inner = max(1, min(count - 2, int(round(0.6 * count))))
    outside = count - inner
    left = int(round(outside * low / tails))
    left = (
        max(1, min(outside - 1, left))
        if low > 0 and high < length
        else (0 if low == 0 else outside)
    )
    right = outside - left
    step = (high - low) / inner

    def widths(span, n):
        if not n:
            return np.empty(0)
        if n == 1:
            return np.array([span])
        lo, hi = 1e-8, 4.0
        powers = np.arange(1, n + 1)
        for _ in range(64):
            ratio = (lo + hi) / 2
            if step * np.sum(ratio**powers) < span:
                lo = ratio
            else:
                hi = ratio
        values = step * ((lo + hi) / 2) ** powers
        return values * (span / values.sum())

    left_edges = low - np.r_[0.0, np.cumsum(widths(low, left))]
    right_edges = high + np.cumsum(widths(length - high, right))
    result = np.r_[
        left_edges[::-1][:-1], np.linspace(low, high, inner + 1), right_edges
    ]
    result[0], result[-1] = 0.0, length
    return result


class RectilinearGrid:
    """Metric arrays index z/y/x; constructor arguments and exports use x/y/z."""

    def __init__(self, lengths_xyz, counts_xyz, periodic_xyz, edges_xyz=None):
        if len(lengths_xyz) != 3 or len(counts_xyz) != 3 or len(periodic_xyz) != 3:
            raise ValueError("A rectilinear grid requires three x/y/z axes")
        self.lengths = np.array(
            [finite_number(v, "length", positive=True) for v in lengths_xyz]
        )
        if any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 3
            for n in counts_xyz
        ):
            raise ValueError("Grid counts must be integers >=3")
        if any(not isinstance(value, bool) for value in periodic_xyz):
            raise ValueError("Periodic flags must be booleans")
        self.shape = tuple(int(n) for n in counts_xyz[::-1])
        self.periodic = tuple(periodic_xyz[::-1])
        if edges_xyz is not None and len(edges_xyz) != 3:
            raise ValueError("Supply three x/y/z edge arrays")
        edges, widths, centers = [], [], []
        for axis, (length, count) in enumerate(zip(self.lengths, counts_xyz)):
            if edges_xyz is None:
                width = np.full(count, length / count)
                edge = np.arange(count + 1) * (length / count)
                edge[-1] = length
                center = (np.arange(count) + 0.5) * (length / count)
            else:
                raw = np.asarray(edges_xyz[axis])
                if raw.dtype.kind not in "iuf":
                    raise ValueError("Grid edges must be real numbers")
                edge = np.array(raw, dtype=float, copy=True)
                if (
                    edge.shape != (count + 1,)
                    or not np.isfinite(edge).all()
                    or edge[0] != 0
                    or edge[-1] != length
                    or np.any(np.diff(edge) <= 0)
                ):
                    raise ValueError(
                        "Grid edges must increase strictly from 0 to axis length"
                    )
                width = np.diff(edge)
                center = 0.5 * (edge[:-1] + edge[1:])
            edges.append(edge)
            widths.append(width)
            centers.append(center)
        self.edges = tuple(edges[::-1])
        self.widths = tuple(widths[::-1])
        self.centers = tuple(centers[::-1])
        self.volumes = self.product(self.widths)
        self.face_areas = []
        self.pressure_distances = []
        self.momentum_widths = []
        self.momentum_volumes = []
        for axis in range(3):
            width = self.widths[axis]
            distance = np.r_[
                width[0] / 2, 0.5 * (width[:-1] + width[1:]), width[-1] / 2
            ]
            if self.periodic[axis]:
                distance[0] = distance[-1] = 0.5 * (width[0] + width[-1])
            self.pressure_distances.append(distance)
            measures = list(self.widths)
            measures[axis] = distance
            self.momentum_widths.append(tuple(measures))
            self.momentum_volumes.append(self.product(measures))
            measures = list(self.widths)
            measures[axis] = np.ones(len(distance))
            self.face_areas.append(self.product(measures))
        for array in (
            *self.edges,
            *self.widths,
            *self.centers,
            self.volumes,
            *self.face_areas,
            *self.pressure_distances,
            *self.momentum_volumes,
        ):
            array.setflags(write=False)

    @staticmethod
    def product(axes):
        return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]

    def positions(self, normal=None):
        coordinates = list(self.centers)
        if normal is not None:
            coordinates[normal] = self.edges[normal]
        z, y, x = np.meshgrid(*coordinates, indexing="ij")
        return np.stack((x, y, z), axis=-1)
