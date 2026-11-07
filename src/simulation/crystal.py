# HOW THIS FILE WORKS
# Represent a convex 3D crystal by outward-facing planes: normals @ point <= offsets.
# Normalize each plane so its offset and outward displacement have length units.
# Check that the planes enclose a finite solid, then find a point strictly inside.
# SciPy intersects the planes to recover vertices and calculates the convex hull.
# Group the vertices on each original plane into an ordered face polygon; hull
# triangles are for rendering and are not independent crystal growth faces.
# Geometry changes return a new solid so cached vertices/volume remain consistent.
# Solid/cell intersections and shared-face sections define liquid capacities and
# transport apertures. Wetted areas and planar sections also drive growth/display.
# The engine supplies face motion and appends fixed tank planes; this file handles
# geometry and an optional synthetic defect counter, not calibrated KDP kinetics.

"""Convex polyhedral geometry in illustrative length units."""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


def _vector3(value, name):
    """Require one finite three-component coordinate or displacement."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


def _edge_plane_points(starts, ends, normals, offsets):
    """Intersections of line segments with planes, without testing other bounds."""
    directions = ends - starts
    denominator = directions @ normals.T
    parameters = np.full_like(denominator, np.nan)
    np.divide(
        offsets - starts @ normals.T,
        denominator,
        out=parameters,
        where=abs(denominator) > 1e-14,
    )
    valid = (parameters >= 0.0) & (parameters <= 1.0)
    rows, columns = np.nonzero(valid)
    return starts[rows] + parameters[rows, columns, None] * directions[rows]


def _polygon_area(points):
    """Area of a planar convex polygon whose corners follow its perimeter."""
    if len(points) < 3:
        return 0.0
    relative = points - points[0]
    return float(np.linalg.norm(np.cross(relative[:-1], relative[1:]).sum(axis=0)) / 2)


class Crystal:
    """A bounded convex solid defined by N planes, including optional redundant ones.

    normals[i] points outward; points inside satisfy dot(normals[i], point)
    <= offsets[i]. Input normals need not be unit vectors: both normal and
    offset are divided by the same length. Arrays are exposed read-only.
    Face indices keep the input plane order; redundant planes have empty faces.
    """

    def __init__(self, normals, offsets):
        normals = np.array(normals, dtype=float, copy=True)
        offsets = np.array(offsets, dtype=float, copy=True)
        if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) < 4:
            raise ValueError("normals must have shape (N, 3), with N >= 4")
        if offsets.shape != (len(normals),):
            raise ValueError("offsets must contain one value per plane")
        if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
            raise ValueError("Plane coefficients must be finite")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0) or not np.isfinite(lengths).all():
            raise ValueError("Each plane must have a finite nonzero normal")
        self.normals = normals / lengths[:, None]
        self.offsets = offsets / lengths
        if not np.isfinite(self.offsets).all():
            raise ValueError("Normalized offsets must be finite")

        # Optimizing both signs of each coordinate detects open, unbounded sets.
        # Coordinates may be negative: linprog's default nonnegative bounds would
        # silently change the solid, so all coordinate bounds are explicitly free.
        bounds = []
        for direction in np.vstack((np.eye(3), -np.eye(3))):
            result = linprog(
                direction,
                A_ub=self.normals,
                b_ub=self.offsets,
                bounds=[(None, None)] * 3,
                method="highs",
            )
            if not result.success:
                raise ValueError("Planes must enclose a feasible, bounded 3D solid")
            bounds.append(result.fun)
        extents = -np.asarray(bounds[3:]) - np.asarray(bounds[:3])
        tolerance = 1e-9 * max(1.0, float(extents.max()))

        # Find the center of the largest inscribed sphere. With unit normals,
        # normals @ center + radius <= offsets expresses distance from every face.
        # A positive radius rules out empty, flat and line-like intersections.
        center_result = linprog(
            [0.0, 0.0, 0.0, -1.0],
            A_ub=np.column_stack((self.normals, np.ones(len(normals)))),
            b_ub=self.offsets,
            bounds=[(None, None)] * 3 + [(0.0, None)],
            method="highs",
        )
        if not center_result.success or center_result.x[3] <= tolerance:
            raise ValueError("Planes must enclose a solid with a nonzero interior")
        self.interior_point = center_result.x[:3]

        self.defects = 0
        self._reconstruct(tolerance)

    def _reconstruct(self, tolerance):
        """Rebuild derived geometry from validated planes and a strict interior point."""
        # Work relative to the interior point to reduce cancellation for translated
        # seeds. SciPy expects inequalities A @ point + b <= 0, hence -local_offsets.
        local_offsets = self.offsets - self.normals @ self.interior_point
        try:
            intersection = HalfspaceIntersection(
                np.column_stack((self.normals, -local_offsets)), np.zeros(3)
            )
            local_vertices = intersection.intersections
            hull = ConvexHull(local_vertices)
        except QhullError as error:
            raise ValueError("Plane intersection is numerically degenerate") from error
        self.vertices = local_vertices + self.interior_point
        self.volume = float(hull.volume)
        self.surface_area = float(hull.area)
        self.triangles = hull.simplices.copy()

        # Recover each kinetic face instead of treating coplanar hull triangles
        # as different faces. A redundant plane touches fewer than three corners.
        self.faces = []
        self.face_areas = np.zeros(len(self.normals))
        seen_faces = set()
        for i, (normal, offset) in enumerate(zip(self.normals, local_offsets)):
            indices = np.flatnonzero(abs(local_vertices @ normal - offset) <= tolerance)
            # A repeated plane is redundant too; do not count its surface twice.
            key = tuple(indices)
            if len(indices) < 3 or key in seen_faces:
                self.faces.append(np.empty(0, dtype=int))
                continue
            seen_faces.add(key)
            # Build an orthogonal in-plane basis and sort corners around the face.
            # u cross v = normal, so the order is counterclockwise from outside.
            axis = np.eye(3)[np.argmin(abs(normal))]
            u = np.cross(normal, axis)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            points = local_vertices[indices]
            relative = points - points.mean(axis=0)
            angles = np.arctan2(relative @ v, relative @ u)
            indices = indices[np.argsort(angles)]
            self.faces.append(indices)
            # Triangulate an ordered face about its first vertex to measure area.
            points = local_vertices[indices]
            cross = np.cross(points[1:-1] - points[0], points[2:] - points[0])
            self.face_areas[i] = float(np.sum(cross @ normal) / 2)
        self.faces = tuple(self.faces)
        edges = {
            tuple(sorted((int(a), int(b))))
            for face in self.faces
            for a, b in zip(face, np.roll(face, -1))
        }
        self.edges = np.asarray(sorted(edges), dtype=int)

        # Prevent ordinary item assignment from changing a plane without updating
        # derived geometry. Use advanced()/translated()/rotated() for changes.
        for array in (
            self.normals,
            self.offsets,
            self.interior_point,
            self.vertices,
            self.triangles,
            self.face_areas,
            self.edges,
            *self.faces,
        ):
            array.setflags(write=False)

    @classmethod
    def box(cls, center, size):
        """Build an axis-aligned box; size is its full x/y/z side lengths."""
        center, size = _vector3(center, "center"), _vector3(size, "size")
        if np.any(size <= 0):
            raise ValueError("Box side lengths must be positive")
        # Plane order is +x, -x, +y, -y, +z, -z, useful for per-face displacements.
        normals = np.repeat(np.eye(3), 2, axis=0)
        normals[1::2] *= -1
        return cls(normals, normals @ center + np.repeat(size / 2, 2))

    def contains(self, points, tolerance=1e-9):
        """Return an inclusion mask for a point or an array ending in x/y/z."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError(
                "points must have a final axis of three finite coordinates"
            )
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        # Subtract the center first so this also works for translated geometry.
        local = points - self.interior_point
        offsets = self.offsets - self.normals @ self.interior_point
        return np.all(local @ self.normals.T <= offsets + tolerance, axis=-1)

    def advanced(self, distances):
        """Move each face outward by a nonnegative length; no kinetics are applied."""
        distances = np.asarray(distances, dtype=float)
        if (
            distances.shape != self.offsets.shape
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError("distances needs one finite nonnegative length per plane")
        # Outward-only motion preserves boundedness and the strict interior point.
        # Reuse those proofs instead of solving seven linear programs per substep.
        result = type(self).__new__(type(self))
        result.normals = self.normals
        result.offsets = self.offsets + distances
        result.interior_point = self.interior_point
        result.defects = self.defects
        scale = max(
            1.0,
            float(np.max(abs(result.offsets - result.normals @ result.interior_point))),
        )
        result._reconstruct(1e-9 * scale)
        return result

    def translated(self, displacement):
        """Return the same shape shifted by an x/y/z displacement."""
        displacement = _vector3(displacement, "displacement")
        result = type(self)(self.normals, self.offsets + self.normals @ displacement)
        result.defects = self.defects
        return result

    def rotated(self, rotation, origin=(0.0, 0.0, 0.0)):
        """Apply a proper 3x3 rotation about origin, preserving volume and areas."""
        rotation = np.asarray(rotation, dtype=float)
        origin = _vector3(origin, "origin")
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError(
                "rotation must be an orthogonal 3x3 matrix with determinant +1"
            )
        normals = self.normals @ rotation.T
        offsets = self.offsets + normals @ origin - self.normals @ origin
        result = type(self)(normals, offsets)
        result.defects = self.defects
        return result

    def bounded_by_box(self, size):
        """Append six fixed tank-wall planes for the box [0, size] in x/y/z.

        The growth engine appends six zero kinetic coefficients to these planes,
        so growth clips against walls even when crystal faces are sloped.
        """
        size = _vector3(size, "tank size")
        if np.any(size <= 0):
            raise ValueError("Tank side lengths must be positive")
        walls = type(self).box(size / 2, size)
        result = type(self)(
            np.vstack((self.normals, walls.normals)), np.r_[self.offsets, walls.offsets]
        )
        result.defects = self.defects
        return result

    def clipped_volume(self, lower, upper):
        """Actual solid volume inside one axis-aligned box, including cut cells.

        Vertices of a convex intersection are original corners, box corners,
        solid edges crossing box planes, or box edges crossing solid planes.
        Recover these points and measure their hull, rather than counting occupied
        voxel centers. This resolves sloped surfaces and seeds smaller than a cell.
        """
        lower, upper = _vector3(lower, "lower"), _vector3(upper, "upper")
        if np.any(upper <= lower):
            return 0.0
        bounds = self.bounds
        if np.any(upper <= bounds[0]) or np.any(lower >= bounds[1]):
            return 0.0
        if np.all(lower <= bounds[0]) and np.all(upper >= bounds[1]):
            return self.volume
        corners = np.array(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        inside = self.contains(corners, tolerance=1e-12)
        if inside.all():
            return float(np.prod(upper - lower))
        box_normals = np.vstack((np.eye(3), -np.eye(3)))
        box_offsets = np.r_[upper, -lower]
        on_solid = _edge_plane_points(
            self.vertices[self.edges[:, 0]],
            self.vertices[self.edges[:, 1]],
            box_normals,
            box_offsets,
        )
        on_solid = on_solid[
            np.all(on_solid @ box_normals.T <= box_offsets + 1e-11, axis=1)
        ]
        box_edges = np.array(
            [
                (i, j)
                for i in range(8)
                for j in range(i + 1, 8)
                if np.count_nonzero(corners[i] != corners[j]) == 1
            ]
        )
        on_box = _edge_plane_points(
            corners[box_edges[:, 0]],
            corners[box_edges[:, 1]],
            self.normals,
            self.offsets,
        )
        on_box = on_box[self.contains(on_box, tolerance=1e-11)]
        original = self.vertices[
            np.all(self.vertices @ box_normals.T <= box_offsets + 1e-11, axis=1)
        ]
        points = np.unique(
            np.vstack((corners[inside], original, on_box, on_solid)), axis=0
        )
        if len(points) < 4:
            return 0.0
        try:
            return float(ConvexHull(points).volume)
        except QhullError:
            # A box may only touch a face or edge: its solid intersection has no volume.
            return 0.0

    def section_area(self, rectangle):
        """Solid area cutting a grid-face rectangle, in true world coordinates."""
        points = np.asarray(rectangle, dtype=float)
        if points.shape != (4, 3) or not np.isfinite(points).all():
            raise ValueError("rectangle must contain four ordered finite 3D corners")
        # Clip the rectangle by each solid half-space. The result remains planar
        # and ordered, so it measures blocked area without staircase boundaries.
        for normal, offset in zip(self.normals, self.offsets):
            if len(points) < 3:
                return 0.0
            distances = points @ normal - offset
            output = []
            for a, b, da, db in zip(
                points, np.roll(points, -1, axis=0), distances, np.roll(distances, -1)
            ):
                if da <= 1e-12:
                    output.append(a)
                if (da < 0 < db) or (db < 0 < da):
                    output.append(a + (b - a) * da / (da - db))
            points = np.asarray(output)
        return _polygon_area(points)

    def wetted_face_area(self, index, level):
        """Area of a crystal face below a horizontal liquid level."""
        return _polygon_area(self.wetted_face(index, level))

    def wetted_face(self, index, level):
        """Ordered corners of the wetted portion of a facet, in its own plane."""
        points = self.vertices[self.faces[index]]
        if len(points) < 3:
            return np.empty((0, 3))
        if points[:, 2].max() <= level:
            return points
        if points[:, 2].min() >= level:
            return np.empty((0, 3))
        output = []
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            da, db = a[2] - level, b[2] - level
            if da <= 0.0:
                output.append(a)
            if (da < 0 < db) or (db < 0 < da):
                output.append(a + (b - a) * da / (da - db))
        return np.asarray(output)

    def sampling_prism(self, index, thickness, level):
        """Extrude a wetted facet outward by a fixed physical sampling distance.

        This convex prism defines the continuous neighborhood independently of
        the grid. Prisms of distinct facet interiors do not overlap: each point
        projects normally onto a unique closest point of the convex crystal.
        Tank cells/free-surface clipping are handled by the field sampler.
        """
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("Sampling thickness must be positive and finite")
        points = self.wetted_face(index, level)
        if len(points) < 3:
            raise ValueError("Cannot extrude a dry or inactive crystal face")
        normal = self.normals[index]
        # Nearly symmetric facets can produce coincident Qhull intersections
        # separated only by roundoff. Project onto the defining plane and merge
        # consecutive corners at the same geometric tolerance used for the face.
        tolerance = 1e-9 * max(1.0, float(np.ptp(points, axis=0).max()))
        points = points - (points @ normal - self.offsets[index])[:, None] * normal
        keep = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > tolerance
        points = points[keep]
        if len(points) < 3:
            raise ValueError("Cannot extrude a numerically degenerate facet")
        # A nearly coincident vertex from an adjacent face can project into this
        # face's interior. Angular sorting alone would make a concave perimeter;
        # recover the convex planar boundary before constructing its side planes.
        axis = np.eye(3)[np.argmin(abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        relative = points - points.mean(axis=0)
        planar = np.column_stack((relative @ u, relative @ v))
        points = points[ConvexHull(planar).vertices]
        edges = np.roll(points, -1, axis=0) - points
        # Tangential edge normals bound the facet footprint; opposite base/top
        # planes bound the outward distance. All normals have unit length.
        side_normals = np.cross(edges, normal)
        side_normals /= np.linalg.norm(side_normals, axis=1)[:, None]
        result = type(self).__new__(type(self))
        result.normals = np.vstack((-normal, normal, side_normals))
        result.offsets = np.r_[
            -self.offsets[index],
            self.offsets[index] + thickness,
            np.sum(side_normals * points, axis=1),
        ]
        result.interior_point = points.mean(axis=0) + normal * thickness / 2
        result.defects = 0
        # Convexity, boundedness and an interior point follow from extrusion of
        # the validated convex face, avoiding a new linear program for every face.
        result._reconstruct(
            1e-9 * max(1.0, thickness, float(np.ptp(points, axis=0).max()))
        )
        return result

    def cross_section(self, axis, coordinate):
        """Ordered 3D corners of a planar slice, for displaying the true boundary."""
        if axis not in (0, 1, 2) or not np.isfinite(coordinate):
            raise ValueError("Slice needs an x/y/z axis and finite coordinate")
        normal = np.eye(3)[axis]
        points = _edge_plane_points(
            self.vertices[self.edges[:, 0]],
            self.vertices[self.edges[:, 1]],
            normal[None, :],
            np.array([coordinate]),
        )
        on_plane = self.vertices[abs(self.vertices[:, axis] - coordinate) < 1e-11]
        points = np.unique(np.vstack((points, on_plane)), axis=0)
        if len(points) < 3:
            return np.empty((0, 3))
        other = [i for i in range(3) if i != axis]
        relative = points[:, other] - points[:, other].mean(axis=0)
        order = np.argsort(np.arctan2(relative[:, 1], relative[:, 0]))
        return points[order]

    @property
    def bounds(self):
        """Minimum and maximum world coordinates, each an x/y/z vector."""
        return np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    @property
    def dimensions(self):
        """Bounding width/depth/height; these are measurements, not stored faces."""
        return np.ptp(self.vertices, axis=0)

    @property
    def width(self):
        return float(self.dimensions[0])

    @property
    def depth(self):
        return float(self.dimensions[1])

    @property
    def height(self):
        return float(self.dimensions[2])
