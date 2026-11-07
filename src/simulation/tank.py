# HOW THIS FILE WORKS
# SimulationConfig validates the complete 3D experiment, including seed face
# matrices, optional orientation, field parameters and the final simulation time.
# x is width, y is depth, and z is vertical height. Scalar fields use [z, y, x].
# Tank creates cell-center coordinates and a flattened [n_cells, 3] position
# matrix so the engine can sample solution beside arbitrary crystal faces.
# Uniform or focused rectilinear cells use their actual edge-defined volumes;
# solute mass uses only each cell's liquid portion, at its actual liquid centroid.
# Exact partial-cell solid volumes exclude crystal from the solution. Open-face
# transport avoids diffusion through solid. Liquid first moments locate cell
# averages and support local field reconstruction. A volume-conserving displacement
# potential raises the free surface and carries solute/heat as the solid grows.
# Resolved mechanical stirring, inertia, buoyancy and Navier-Stokes remain future
# physics; current stirring is an effective diffusivity through liquid apertures.

"""Three-dimensional experiment configuration and volumetric solution fields."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
import math

import numpy as np

from .crystal import Crystal
from .physics import (
    diffuse_liquid,
    displace_liquid,
    liquid_gradient,
    nonnegative_gradient,
)


def _axis_edges(length, count, center, focus_width):
    """Fixed rectilinear mesh with a uniform central band and stretched tails."""
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


@dataclass(frozen=True)
class SimulationConfig:
    # World axes: width=x, depth=y, height=z. Arrays index these in reverse order.
    width: float = 10.0
    depth: float = 8.0
    height: float = 8.0
    nx: int = 20
    ny: int = 16
    nz: int = 16
    # Zero gives uniform cells. Positive values give the full physical width
    # of a fine band centered on the seed, with stretched cells toward the walls.
    grid_focus_width: float = 0.0
    # Open tank with headspace. None uses 75% of tank height, with the seed present.
    initial_liquid_height: float | None = None
    temperature: float = 25.0
    concentration: float = 1.35
    temperature_gradient: float = 0.0
    concentration_noise: float = 0.005
    solubility_reference: float = 1.0
    solubility_slope: float = 0.02
    reference_temperature: float = 25.0
    # Box dimensions apply when custom seed planes are not supplied. Custom
    # planes use local coordinates; rotation then translation place them in the tank.
    seed_width: float = 0.6
    seed_depth: float = 0.6
    seed_height: float = 0.8
    seed_normals: list | None = None
    seed_offsets: list | None = None
    seed_rotation: list | None = None
    seed_center: list | None = None
    crystal_density: float = 8.0
    growth_rate: float = 0.025
    face_growth: list | None = None
    # Diffusion has length**2/time units; other coefficients remain illustrative.
    solute_diffusivity: float = 0.02
    thermal_diffusivity: float = 0.04
    # Isotropic effective stirring, applied through liquid apertures only.
    # This is a configurable model coefficient, not resolved circulation flow.
    mixing_diffusivity: float = 0.02
    thermal_relaxation: float = 0.01
    rotation: float = 0.5
    circulation: float = 0.5
    impurity: float = 0.0
    defect_rate: float = 0.0
    # Sampling depth = this count * max(width/nx, depth/ny, height/nz).
    # These are nominal mean widths, independent of focused minimum spacing.
    boundary_layer_cells: float = 1.5
    dt: float = 1.0
    duration: float = 120.0
    seed: int = 42

    def __post_init__(self):
        structured = {
            "seed_normals",
            "seed_offsets",
            "seed_rotation",
            "seed_center",
            "face_growth",
            "initial_liquid_height",
        }
        for name, value in asdict(self).items():
            if name not in structured and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        for name in (
            "width",
            "depth",
            "height",
            "seed_width",
            "seed_depth",
            "seed_height",
            "crystal_density",
            "solubility_reference",
            "boundary_layer_cells",
            "dt",
            "duration",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("nx", "ny", "nz", "seed"):
            if not isinstance(getattr(self, name), Integral):
                raise ValueError(f"{name} must be an integer")
        if min(self.nx, self.ny, self.nz) < 3 or self.seed < 0:
            raise ValueError("Grid counts must be >= 3 and seed must be nonnegative")
        for name in (
            "concentration",
            "grid_focus_width",
            "concentration_noise",
            "solute_diffusivity",
            "thermal_diffusivity",
            "thermal_relaxation",
            "rotation",
            "circulation",
            "impurity",
            "defect_rate",
            "growth_rate",
            "mixing_diffusivity",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (self.seed_normals is None) != (self.seed_offsets is None):
            raise ValueError("seed_normals and seed_offsets must be supplied together")
        if self.initial_liquid_height is not None:
            value = self.initial_liquid_height
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 < value <= self.height
            ):
                raise ValueError(
                    "initial_liquid_height must be positive and no higher than the rim"
                )
        # Validate the actual rotated seed rather than its unrotated bounding box.
        crystal = self.make_crystal()
        if np.any(crystal.bounds[0] <= 0) or np.any(
            crystal.bounds[1] >= [self.width, self.depth, self.height]
        ):
            raise ValueError("The seed must fit strictly inside all six tank walls")
        if crystal.bounds[1, 2] > self.fill_height:
            raise ValueError(
                "The initial seed must be submerged below the liquid level"
            )
        if self.face_growth is not None:
            rates = np.asarray(self.face_growth, dtype=float)
            if (
                rates.shape != (len(crystal.normals),)
                or not np.isfinite(rates).all()
                or np.any(rates < 0)
            ):
                raise ValueError(
                    "face_growth needs one finite nonnegative coefficient per seed plane"
                )

    def make_crystal(self):
        """Build the local seed, rotate it, then place it in world coordinates."""
        if self.seed_normals is None:
            crystal = Crystal.box(
                [0.0, 0.0, 0.0], [self.seed_width, self.seed_depth, self.seed_height]
            )
        else:
            crystal = Crystal(self.seed_normals, self.seed_offsets)
        if self.seed_rotation is not None:
            crystal = crystal.rotated(self.seed_rotation)
        center = (
            self.seed_center
            if self.seed_center is not None
            else [self.width / 2, self.depth / 2, self.height / 2]
        )
        return crystal.translated(center)

    @property
    def fill_height(self):
        return (
            0.75 * self.height
            if self.initial_liquid_height is None
            else float(self.initial_liquid_height)
        )


@dataclass
class LiquidGeometry:
    # Volumes and unobstructed areas are geometric quantities, not center masks.
    volumes: np.ndarray
    solid_fraction: np.ndarray
    openings: tuple
    level: float
    centroids: np.ndarray
    opening_centroids: tuple
    bounds: tuple


class Tank:
    """Open 3D tank with moving solid exclusion and an incompressible carrier.

    Dilute approximation: dissolved solute has negligible partial volume. Growth
    removes solute mass, while carrier volume stays constant and its level rises.
    This is a stated approximation, not a full solution-density/thermodynamic model.
    """

    def __init__(self, config=None, rng=None, crystal=None):
        self.config = config if config is not None else SimulationConfig()
        c = self.config
        center = (
            c.seed_center
            if c.seed_center is not None
            else [c.width / 2, c.depth / 2, c.height / 2]
        )
        self.x_edges, self.y_edges, self.z_edges = [
            _axis_edges(length, count, coordinate, c.grid_focus_width)
            for length, count, coordinate in zip(
                (c.width, c.depth, c.height), (c.nx, c.ny, c.nz), center
            )
        ]
        self.widths = tuple(
            np.diff(edges) for edges in (self.x_edges, self.y_edges, self.z_edges)
        )
        # Scalar spacings summarize the smallest cells; all capacities and
        # transport distances below use the full actual-coordinate arrays.
        self.dx, self.dy, self.dz = [float(width.min()) for width in self.widths]
        self.x, self.y, self.z = [
            (edge[:-1] + edge[1:]) / 2
            for edge in (self.x_edges, self.y_edges, self.z_edges)
        ]
        zz, yy, xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        self.positions = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        self.shape = (c.nz, c.ny, c.nx)
        self.cell_lower, self.cell_upper = self._boxes()
        self.cell_volumes = np.prod(self.cell_upper - self.cell_lower, axis=-1)
        self.cell_volume = float(self.cell_volumes.min())
        self.distances = (
            np.diff(self.z)[:, None, None],
            np.diff(self.y)[None, :, None],
            np.diff(self.x)[None, None, :],
        )
        self.crystal = crystal
        self.geometry = self._geometry(crystal, c.fill_height)
        self.carrier_volume = float(self.geometry.volumes.sum())
        self.displacement_velocity = np.zeros((*self.shape, 3))
        self.continuity_error = 0.0
        self.diffusion_diagnostics = {}
        self.remap_diagnostics = {}
        # An affine initial temperature profile averages exactly at the actual
        # liquid centroid, including partial solid/free-surface cells.
        self.temperature = c.temperature + c.temperature_gradient * (
            self.liquid_centroids[..., 2] / c.height - 0.5
        )
        rng = rng if rng is not None else np.random.default_rng(c.seed)
        self.concentration = np.maximum(
            0.0, c.concentration + rng.normal(0.0, c.concentration_noise, self.shape)
        )
        self.temperature[~self.liquid_mask] = 0.0
        self.concentration[~self.liquid_mask] = 0.0

    @property
    def volume(self):
        c = self.config
        return c.width * c.depth * c.height

    @property
    def liquid_level(self):
        return self.geometry.level

    @property
    def liquid_volumes(self):
        return self.geometry.volumes

    @property
    def liquid_centroids(self):
        """Actual wet-volume centroids in [z,y,x,component] order; zero if dry."""
        return self.geometry.centroids

    @property
    def liquid_mask(self):
        return self.liquid_volumes > 0.0

    @property
    def liquid_fraction(self):
        return self.liquid_volumes / self.cell_volumes

    @property
    def liquid_volume(self):
        return float(self.liquid_volumes.sum())

    @property
    def solute_mass(self):
        return float(np.sum(self.concentration * self.liquid_volumes))

    def liquid_mean(self, field):
        return float(
            np.sum(np.asarray(field) * self.liquid_volumes) / self.liquid_volume
        )

    def field_gradient(self, field, targets=None, nonnegative=False):
        """Reconstruct local field variation without changing stored cell means."""
        gradient = liquid_gradient(
            field,
            self.liquid_volumes,
            self.liquid_centroids,
            self.geometry.openings,
            targets,
        )
        if nonnegative:
            gradient = nonnegative_gradient(
                field,
                gradient,
                self.liquid_centroids,
                np.array([self.dx, self.dy, self.dz]),
                self.liquid_level,
                bounds=self.geometry.bounds,
            )
        return gradient

    def sampling_volumes(self, prism):
        """Exact liquid volume inside a facet prism in each Cartesian cell."""
        return self.sampling_moments(prism)[0]

    def _boxes(self, normal=None):
        """Cartesian cell or face bounds on the configured rectilinear grid."""
        edges = (self.x_edges, self.y_edges, self.z_edges)
        lows = [edge[:-1] for edge in edges]
        highs = [edge[1:] for edge in edges]
        if normal is not None:
            lows[normal] = highs[normal] = edges[normal][1:-1]
        lo = np.meshgrid(lows[2], lows[1], lows[0], indexing="ij")
        hi = np.meshgrid(highs[2], highs[1], highs[0], indexing="ij")
        return np.stack(lo[::-1], axis=-1), np.stack(hi[::-1], axis=-1)

    def _bounds_indices(self, bounds):
        edges = (self.x_edges, self.y_edges, self.z_edges)
        lower = [
            max(0, int(np.searchsorted(e, bounds[0, i], side="right")) - 1)
            for i, e in enumerate(edges)
        ]
        upper = [
            min(len(e) - 2, int(np.searchsorted(e, bounds[1, i], side="left")))
            for i, e in enumerate(edges)
        ]
        return lower, upper

    def sampling_moments(self, prism):
        """Exact prism/cell moments on uniform or locally focused 3D cells."""
        lower, upper = self._bounds_indices(prism.bounds)
        volumes = np.zeros(self.shape)
        moments = np.zeros((*self.shape, 3))
        for iz in range(lower[2], upper[2] + 1):
            for iy in range(lower[1], upper[1] + 1):
                for ix in range(lower[0], upper[0] + 1):
                    cell = (iz, iy, ix)
                    if self.liquid_volumes[cell] <= 0:
                        continue
                    lo = self.cell_lower[cell]
                    hi = self.cell_upper[cell].copy()
                    hi[2] = min(hi[2], self.liquid_level)
                    volumes[cell], moments[cell] = prism.clipped_moments(lo, hi)
        return volumes.ravel(), moments.reshape(-1, 3)

    def _geometry(self, crystal, level):
        """Actual liquid volumes, moments and open-face geometry on 3D boxes."""
        lower = self.cell_lower
        upper = self.cell_upper.copy()
        upper[..., 2] = np.maximum(lower[..., 2], np.minimum(upper[..., 2], level))
        volumes = np.prod(upper - lower, axis=-1)
        moments = volumes[..., None] * (lower + upper) / 2
        solid = np.zeros(self.shape)
        if crystal is not None:
            first, last = self._bounds_indices(crystal.bounds)
            for iz in range(first[2], last[2] + 1):
                for iy in range(first[1], last[1] + 1):
                    for ix in range(first[0], last[0] + 1):
                        cell = (iz, iy, ix)
                        lo = lower[cell]
                        hi = self.cell_upper[cell]
                        full, full_moment = crystal.clipped_moments(lo, hi)
                        solid[cell] = full
                        if hi[2] <= level:
                            submerged, submoment = full, full_moment
                        elif lo[2] >= level:
                            submerged, submoment = 0.0, np.zeros(3)
                        else:
                            submerged, submoment = crystal.clipped_moments(
                                lo, upper[cell]
                            )
                        volumes[cell] -= submerged
                        moments[cell] -= submoment
        volumes[abs(volumes) < 1e-12 * self.cell_volumes] = 0.0
        if np.any(volumes < -1e-9 * self.cell_volumes):
            raise ValueError("Invalid liquid volume")
        openings = []
        opening_centroids = []
        for axis in range(3):
            normal = 2 - axis
            lo, hi = self._boxes(normal)
            full_area = np.prod(np.delete(hi - lo, normal, axis=-1), axis=-1)
            hi[..., 2] = np.maximum(lo[..., 2], np.minimum(hi[..., 2], level))
            area = np.prod(np.delete(hi - lo, normal, axis=-1), axis=-1)
            area = np.where(lo[..., 2] <= level, area, 0.0)
            face_moment = area[..., None] * (lo + hi) / 2
            if crystal is not None:
                bounds = crystal.bounds
                near = np.all(
                    (hi >= bounds[0] - 1e-12) & (lo <= bounds[1] + 1e-12), axis=-1
                ) & (area > 0)
                other = [a for a in range(3) if a != normal]
                for index in np.argwhere(near):
                    cell = tuple(index)
                    rectangle = []
                    for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = lo[cell].copy()
                        point[other[0]] = (lo[cell], hi[cell])[a][other[0]]
                        point[other[1]] = (lo[cell], hi[cell])[b][other[1]]
                        rectangle.append(point)
                    blocked, blocked_moment = crystal.section_moments(rectangle)
                    area[cell] -= blocked
                    face_moment[cell] -= blocked_moment
            area[abs(area) < 1e-12 * full_area] = 0.0
            if np.any(area < -1e-9 * full_area):
                raise ValueError("Invalid liquid opening")
            openings.append(area)
            opening_centroids.append(
                np.divide(
                    face_moment,
                    area[..., None],
                    out=np.zeros_like(face_moment),
                    where=area[..., None] > 0,
                )
            )
        centers = (lower + upper) / 2
        np.divide(
            moments,
            volumes[..., None],
            out=centers,
            where=((solid > 0) & (volumes > 0))[..., None],
        )
        centers[volumes <= 0] = 0
        return LiquidGeometry(
            volumes,
            solid / self.cell_volumes,
            tuple(openings),
            float(level),
            centers,
            tuple(opening_centroids),
            (lower, upper),
        )

    def level_for(self, crystal):
        """Free level that preserves carrier volume; refuse untracked overflow."""
        c = self.config
        area = c.width * c.depth
        height = (self.carrier_volume + crystal.volume) / area
        if height >= crystal.bounds[1, 2] and height <= c.height + 1e-12:
            return min(height, c.height)
        capacity = area * c.height - crystal.clipped_volume(
            [0, 0, 0], [c.width, c.depth, c.height]
        )
        if capacity < self.carrier_volume - 1e-10:
            raise ValueError("Open tank has no headspace left for displaced liquid")
        # An emerging crystal displaces only its submerged portion.
        lo, hi = 0.0, c.height
        for _ in range(44):
            mid = (lo + hi) / 2
            available = area * mid - crystal.clipped_volume(
                [0, 0, 0], [c.width, c.depth, mid]
            )
            if available < self.carrier_volume:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def move_boundary(self, crystal, dt, concentration):
        """Commit new geometry only after conservative displacement succeeds."""
        if not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Boundary motion needs a positive finite dt")
        geometry = self._geometry(crystal, self.level_for(crystal))
        updated_c, updated_t, fluxes, error = displace_liquid(
            self.liquid_volumes,
            geometry.volumes,
            self.geometry.openings,
            geometry.openings,
            concentration,
            self.temperature,
            self.dx,
            self.dy,
            self.dz,
            geometry=(self.geometry, geometry),
            distances=self.distances,
            diagnostics=self.remap_diagnostics,
        )
        # First-moment continuity gives volume-averaged material displacement,
        # including the moving solid/free surface beside partially wet cells.
        movement = (
            geometry.volumes[..., None] * geometry.centroids
            - self.liquid_volumes[..., None] * self.liquid_centroids
        )
        for axis, (flux, old_area, new_area, old_center, new_center) in enumerate(
            zip(
                fluxes,
                self.geometry.openings,
                geometry.openings,
                self.geometry.opening_centroids,
                geometry.opening_centroids,
            )
        ):
            area = old_area + new_area
            center = np.divide(
                old_area[..., None] * old_center + new_area[..., None] * new_center,
                area[..., None],
                out=np.zeros_like(old_center),
                where=area[..., None] > 0,
            )
            moment_flux = flux[..., None] * center
            lower, upper = [slice(None)] * 3, [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            movement[tuple(lower)] += moment_flux
            movement[tuple(upper)] -= moment_flux
        capacity = 0.5 * (self.liquid_volumes + geometry.volumes) * dt
        velocity = np.divide(
            movement,
            capacity[..., None],
            out=np.zeros_like(movement),
            where=capacity[..., None] > 0,
        )
        velocity[geometry.volumes <= 0] = 0
        self.geometry, self.crystal = geometry, crystal
        self.concentration, self.temperature = updated_c, updated_t
        self.displacement_velocity, self.continuity_error = velocity, error

    def diffuse(self, dt):
        """Conservative diffusion through actual liquid portions and open areas."""
        c = self.config
        agitation = c.rotation + c.circulation
        stirring = c.mixing_diffusivity * agitation / (1.0 + agitation)
        temperature = diffuse_liquid(
            self.temperature,
            self.liquid_volumes,
            self.geometry.openings,
            c.thermal_diffusivity + stirring,
            dt,
            self.dx,
            self.dy,
            self.dz,
            centroids=self.liquid_centroids,
            distances=self.distances,
            nominal_positions=self.positions,
            diagnostics=self.diffusion_diagnostics,
        )
        concentration = diffuse_liquid(
            self.concentration,
            self.liquid_volumes,
            self.geometry.openings,
            c.solute_diffusivity + stirring,
            dt,
            self.dx,
            self.dy,
            self.dz,
            centroids=self.liquid_centroids,
            distances=self.distances,
            nominal_positions=self.positions,
            diagnostics=self.diffusion_diagnostics,
        )
        self.temperature, self.concentration = temperature, concentration
