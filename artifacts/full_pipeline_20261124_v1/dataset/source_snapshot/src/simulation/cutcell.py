# HOW THIS FILE WORKS
# Intersect one fixed convex crystal with a rectilinear SI grid using the
# canonical plane/edge/polygon geometry, normalized to a safe geometric scale.
# Store actual liquid volumes/centroids, open Cartesian face areas/centroids and
# per-cell pieces of the crystal wall, including their normals and first moments.
# Surface closure and first-moment identities independently audit each cell.
# Reject cells containing disconnected liquid pockets instead of connecting them
# through one pressure unknown. Preserve resolved small cells; only roundoff-sized
# capacities/apertures are snapped to zero. This is geometry, not a flow step.
# An optional bounded preprocessing pass bisects Cartesian axis intervals until
# split liquid pockets are represented by distinct cells; it records every split.

"""Fixed embedded-boundary geometry for conservative SI cut-cell operators."""

from itertools import product

import numpy as np

from .crystal import Crystal
from .mesh import RectilinearGrid


class SplitLiquidCellError(ValueError):
    """A single Cartesian cell contains multiple disconnected liquid regions."""

    def __init__(self, lower_m, upper_m):
        self.lower_m = np.array(lower_m, copy=True)
        self.upper_m = np.array(upper_m, copy=True)
        super().__init__(
            f"Split liquid cell at {self.lower_m.tolist()} .. {self.upper_m.tolist()} m: refine the mesh; disconnected pockets cannot share one pressure unknown"
        )


class CutCellGeometry:
    """Liquid is the Cartesian domain outside one convex, stationary solid.

    Volumes and scalar arrays index z/y/x. Face arrays follow z/y/x normal
    axes, matching RectilinearGrid. Cartesian vector components always follow
    x/y/z. Pressure operators may sample at nominal Cartesian cell centers;
    liquid centroids separately locate actual volume averages.
    """

    def __init__(self, grid, solid=None):
        if not isinstance(grid, RectilinearGrid):
            raise ValueError("Cut-cell geometry requires RectilinearGrid")
        if solid is not None and not isinstance(solid, Crystal):
            raise ValueError("The fixed solid must be a canonical Crystal")
        self.grid = grid
        self.scale_m = float(grid.lengths.max())
        self._solid = (
            None
            if solid is None
            else Crystal(solid.normals, solid.offsets / self.scale_m)
        )
        self.solid_vertices_m = (
            np.empty((0, 3))
            if self._solid is None
            else self._solid.vertices * self.scale_m
        )
        self.refinement_history = np.empty((0, 8))
        self.refinement_history.setflags(write=False)
        self._outside_pairs = {}
        self._domain = None
        self.volumes_m3 = grid.volumes.copy()
        self.centroids_m = grid.positions()
        self.solid_fractions = np.zeros(grid.shape)
        self.open_areas_m2 = [a.copy() for a in grid.face_areas]
        self.face_centroids_m = [grid.positions(axis) for axis in range(3)]
        patches = []
        if solid is not None:
            self._cells(patches)
            self._faces()
        self.fluid = self.volumes_m3 > 0
        self.wall_cells = np.array([p[0] for p in patches], dtype=int).reshape(-1, 3)
        self.wall_face_ids = np.array([p[1] for p in patches], dtype=int)
        self.wall_areas_m2 = np.array([p[2] for p in patches], dtype=float)
        self.wall_centroids_m = np.array([p[3] for p in patches], dtype=float).reshape(
            -1, 3
        )
        self.wall_normals = np.array([p[4] for p in patches], dtype=float).reshape(
            -1, 3
        )
        self.wall_area_vectors_m2 = np.zeros((*grid.shape, 3))
        self.wall_normal_moments_m3 = np.zeros((*grid.shape, 3, 3))
        if len(patches):
            ix = tuple(self.wall_cells.T)
            np.add.at(
                self.wall_area_vectors_m2,
                ix,
                self.wall_areas_m2[:, None] * self.wall_normals,
            )
            np.add.at(
                self.wall_normal_moments_m3,
                ix,
                self.wall_areas_m2[:, None, None]
                * self.wall_centroids_m[:, :, None]
                * self.wall_normals[:, None, :],
            )
        for array in (
            self.solid_vertices_m,
            self.volumes_m3,
            self.centroids_m,
            self.solid_fractions,
            *self.open_areas_m2,
            *self.face_centroids_m,
            self.fluid,
            self.wall_cells,
            self.wall_face_ids,
            self.wall_areas_m2,
            self.wall_centroids_m,
            self.wall_normals,
            self.wall_area_vectors_m2,
            self.wall_normal_moments_m3,
        ):
            array.setflags(write=False)

    @classmethod
    def resolved(cls, grid, solid=None, *, max_axis_refinements=48):
        """Resolve split liquid cells by bounded, static Cartesian bisection.

        Each split adds a global axis plane; this is fixed preprocessing, not
        moving adaptive refinement. The original grid and crystal are untouched.
        Failure at the declared limit remains explicit, never a pocket merge.
        History columns: axis_xyz, coordinate_m, lower_xyz_m, upper_xyz_m.
        """
        if (
            isinstance(max_axis_refinements, bool)
            or not isinstance(max_axis_refinements, int)
            or max_axis_refinements < 0
        ):
            raise ValueError("max_axis_refinements must be a nonnegative integer")
        current = grid
        history = []
        for iteration in range(max_axis_refinements + 1):
            try:
                result = cls(current, solid)
            except SplitLiquidCellError as error:
                if iteration == max_axis_refinements:
                    raise ValueError(
                        f"Split-cell refinement limit ({max_axis_refinements}) reached: {error}"
                    ) from error
                axis = int(np.argmax(error.upper_m - error.lower_m))
                coordinate = 0.5 * (error.lower_m[axis] + error.upper_m[axis])
                if not error.lower_m[axis] < coordinate < error.upper_m[axis]:
                    raise ArithmeticError(
                        "Split-cell refinement reached coordinate precision"
                    ) from error
                edges = [edge.copy() for edge in current.edges[::-1]]
                if np.any(edges[axis] == coordinate):
                    raise ArithmeticError(
                        "Split-cell refinement did not create a new edge"
                    ) from error
                edges[axis] = np.sort(np.r_[edges[axis], coordinate])
                history.append(np.r_[axis, coordinate, error.lower_m, error.upper_m])
                current = RectilinearGrid(
                    current.lengths,
                    [len(edge) - 1 for edge in edges],
                    current.periodic[::-1],
                    edges,
                )
            else:
                result.refinement_history = np.array(history, dtype=float).reshape(
                    -1, 8
                )
                result.refinement_history.setflags(write=False)
                return result

    def _candidates(self):
        bounds = self._solid.bounds * self.scale_m
        first = [
            max(0, np.searchsorted(e, lo, side="left") - 1)
            for e, lo in zip(self.grid.edges[::-1], bounds[0])
        ]
        last = [
            min(len(e) - 1, np.searchsorted(e, hi, side="right"))
            for e, hi in zip(self.grid.edges[::-1], bounds[1])
        ]
        return product(
            range(first[2], last[2]), range(first[1], last[1]), range(first[0], last[0])
        )

    def _cells(self, patches):
        solid, scale = self._solid, self.scale_m
        xyz_edges = self.grid.edges[::-1]
        for z, y, x in self._candidates():
            cell = (z, y, x)
            lower = np.array([e[i] for e, i in zip(xyz_edges, (x, y, z))]) / scale
            upper = np.array([e[i + 1] for e, i in zip(xyz_edges, (x, y, z))]) / scale
            midpoint, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
            signed = solid.normals @ midpoint - solid.offsets
            radius = abs(solid.normals) @ half
            if np.any(signed - radius > 1e-12):
                continue
            volume, moment = solid.clipped_moments(lower, upper)
            box_volume = self.grid.volumes[cell]
            liquid = box_volume - volume * scale**3
            if liquid < -1e-10 * box_volume or liquid > (1 + 1e-10) * box_volume:
                raise ArithmeticError("Invalid clipped liquid capacity")
            if abs(liquid) <= 1e-12 * box_volume:
                liquid = 0.0
            elif abs(liquid - box_volume) <= 1e-12 * box_volume:
                liquid = box_volume
            self.volumes_m3[cell] = liquid
            self.solid_fractions[cell] = 1 - liquid / box_volume
            if liquid == 0:
                self.centroids_m[cell] = 0
                continue
            self.centroids_m[cell] = (
                box_volume * midpoint * scale - moment * scale**4
            ) / liquid
            if 0 < self.solid_fractions[cell] < 1:
                self._check_connected(lower, upper)
            areas, moments = solid.clipped_face_moments(lower, upper)
            for face in np.flatnonzero(areas > 0):
                centroid = moments[face] / areas[face]
                normal = solid.normals[face]
                # A coincident axis-aligned facet belongs to the box toward
                # which the solid outward normal points: that is its liquid side.
                direction = int(np.argmax(abs(normal)))
                if np.count_nonzero(abs(normal) > 1e-13) == 1 and (
                    (
                        abs(centroid[direction] - lower[direction]) < 1e-13
                        and normal[direction] < 0
                    )
                    or (
                        abs(centroid[direction] - upper[direction]) < 1e-13
                        and normal[direction] > 0
                    )
                ):
                    continue
                patches.append(
                    (
                        cell,
                        face,
                        float(areas[face] * scale**2),
                        centroid * scale,
                        -normal,
                    )
                )

    def _check_connected(self, lower, upper):
        """Outside half-space pieces must form one connected liquid volume."""
        corners = np.array(list(product(*zip(lower, upper))))
        signed = corners @ self._solid.normals.T - self._solid.offsets
        faces = np.flatnonzero(signed.max(axis=0) > 1e-12)
        if len(faces) < 2:
            return
        parent = np.arange(len(faces))

        def root(k):
            while parent[k] != k:
                k = parent[k]
            return k

        for first in range(len(faces)):
            for second in range(first + 1, len(faces)):
                if root(first) == root(second):
                    continue
                a, b = int(faces[first]), int(faces[second])
                connected = np.any((signed[:, a] > 1e-12) & (signed[:, b] > 1e-12))
                if not connected:
                    key = (a, b)
                    if key not in self._outside_pairs:
                        if self._domain is None:
                            length = self.grid.lengths / self.scale_m
                            self._domain = Crystal.box(length / 2, length)
                        try:
                            piece = Crystal(
                                np.vstack(
                                    (self._domain.normals, -self._solid.normals[[a, b]])
                                ),
                                np.r_[
                                    self._domain.offsets, -self._solid.offsets[[a, b]]
                                ],
                            )
                        except ValueError:
                            piece = None
                        self._outside_pairs[key] = piece
                    piece = self._outside_pairs[key]
                    connected = piece is not None and piece.clipped_volume(
                        lower, upper
                    ) > 1e-12 * np.prod(upper - lower)
                if connected:
                    parent[root(first)] = root(second)
        if len({root(k) for k in range(len(faces))}) > 1:
            raise SplitLiquidCellError(lower * self.scale_m, upper * self.scale_m)

    def _faces(self):
        solid, scale = self._solid, self.scale_m
        xyz_edges = self.grid.edges[::-1]
        bounds = solid.bounds * scale
        for axis in range(3):
            normal = 2 - axis
            other = [d for d in range(3) if d != normal]
            for index in np.ndindex(self.open_areas_m2[axis].shape):
                xyz = np.array(index[::-1])
                lo = np.array([e[i] for e, i in zip(xyz_edges, xyz)])
                hi = np.array(
                    [
                        e[i + (d != normal)]
                        for d, (e, i) in enumerate(zip(xyz_edges, xyz))
                    ]
                )
                if np.any(hi < bounds[0]) or np.any(lo > bounds[1]):
                    continue
                points = []
                for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    point = lo.copy()
                    point[other[0]] = (lo, hi)[a][other[0]]
                    point[other[1]] = (lo, hi)[b][other[1]]
                    points.append(point / scale)
                points = np.array(points)
                signed = points @ solid.normals.T - solid.offsets
                if np.any(signed.min(axis=0) > 1e-12):
                    continue
                full = self.grid.face_areas[axis][index]
                blocked, moment = solid.section_moments(points)
                area = full - blocked * scale**2
                if area < -1e-10 * full or area > (1 + 1e-10) * full:
                    raise ArithmeticError("Invalid clipped face aperture")
                if abs(area) <= 1e-12 * full:
                    area = 0.0
                elif abs(area - full) <= 1e-12 * full:
                    area = full
                self.open_areas_m2[axis][index] = area
                self.face_centroids_m[axis][index] = (
                    (full * self.face_centroids_m[axis][index] - moment * scale**3)
                    / area
                    if area
                    else 0.0
                )

    def closure(self):
        """Return oriented surface-area and first-moment defects per liquid cell."""
        area = self.wall_area_vectors_m2.copy()
        moment = self.wall_normal_moments_m3.copy()
        for axis in range(3):
            xyz = 2 - axis
            area[..., xyz] += np.diff(self.open_areas_m2[axis], axis=axis)
            face_moment = (
                self.open_areas_m2[axis][..., None] * self.face_centroids_m[axis]
            )
            moment[..., :, xyz] += np.diff(face_moment, axis=axis)
        moment -= self.volumes_m3[..., None, None] * np.eye(3)
        area[~self.fluid] = 0
        moment[~self.fluid] = 0
        return area, moment

    def measurements(self):
        area, moment = self.closure()
        fractions = self.volumes_m3[self.fluid] / self.grid.volumes[self.fluid]
        area_scale = self.grid.volumes ** (2 / 3)
        return dict(
            grid_counts_xyz=list(self.grid.shape[::-1]),
            axis_refinement_count=len(self.refinement_history),
            liquid_volume_m3=float(self.volumes_m3.sum()),
            cut_cell_count=int(
                np.count_nonzero((self.solid_fractions > 0) & self.fluid)
            ),
            min_liquid_fraction=float(fractions.min(initial=1)),
            normalized_area_closure=float(
                np.max(np.linalg.norm(area, axis=-1) / area_scale, initial=0)
            ),
            normalized_moment_closure=float(
                np.max(
                    np.linalg.norm(moment, axis=(-2, -1)) / self.grid.volumes, initial=0
                )
            ),
            embedded_wall_area_m2=float(self.wall_areas_m2.sum()),
        )
