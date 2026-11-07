# HOW THIS FILE WORKS
# SimulationConfig validates the complete 3D experiment, including seed face
# matrices, optional orientation, field parameters and the final simulation time.
# x is width, y is depth, and z is vertical height. Scalar fields use [z, y, x].
# Tank creates cell-center coordinates and a flattened [n_cells, 3] position
# matrix so the engine can sample solution beside arbitrary crystal faces.
# Each voxel has volume dx * dy * dz; solute mass uses only its liquid portion.
# Exact partial-cell solid volumes exclude crystal from the solution. Open-face
# transport avoids diffusion through solid. A volume-conserving displacement
# potential raises the free surface and carries solute/heat as the solid grows.
# Resolved mechanical stirring, inertia, buoyancy and Navier-Stokes remain future
# physics; current stirring is an effective diffusivity through liquid apertures.

"""Three-dimensional experiment configuration and volumetric solution fields."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
import math

import numpy as np

from .crystal import Crystal
from .physics import diffuse_liquid, displace_liquid


@dataclass(frozen=True)
class SimulationConfig:
    # World axes: width=x, depth=y, height=z. Arrays index these in reverse order.
    width: float = 10.0
    depth: float = 8.0
    height: float = 8.0
    nx: int = 20
    ny: int = 16
    nz: int = 16
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


class Tank:
    """Open 3D tank with moving solid exclusion and an incompressible carrier.

    Dilute approximation: dissolved solute has negligible partial volume. Growth
    removes solute mass, while carrier volume stays constant and its level rises.
    This is a stated approximation, not a full solution-density/thermodynamic model.
    """

    def __init__(self, config=None, rng=None, crystal=None):
        self.config = config if config is not None else SimulationConfig()
        c = self.config
        self.dx, self.dy, self.dz = c.width / c.nx, c.depth / c.ny, c.height / c.nz
        self.cell_volume = self.dx * self.dy * self.dz
        self.x, self.y, self.z = [
            (np.arange(n) + 0.5) * d
            for n, d in ((c.nx, self.dx), (c.ny, self.dy), (c.nz, self.dz))
        ]
        zz, yy, xx = np.meshgrid(self.z, self.y, self.x, indexing="ij")
        self.positions = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
        self.shape = (c.nz, c.ny, c.nx)
        self.crystal = crystal
        self.geometry = self._geometry(crystal, c.fill_height)
        self.carrier_volume = float(self.geometry.volumes.sum())
        self.displacement_velocity = np.zeros((*self.shape, 3))
        self.continuity_error = 0.0
        vertical = c.temperature + c.temperature_gradient * (self.z / c.height - 0.5)
        self.temperature = np.broadcast_to(vertical[:, None, None], self.shape).copy()
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
    def liquid_mask(self):
        return self.liquid_volumes > 0.0

    @property
    def liquid_fraction(self):
        return self.liquid_volumes / self.cell_volume

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

    def sampling_volumes(self, prism):
        """Exact liquid volume inside a facet prism in each Cartesian cell.

        The prism is outside the solid by construction. Intersect its footprint
        with each cell and the free surface instead of accepting/rejecting whole
        liquid cells according to where their centers happen to lie.
        """
        c = self.config
        spacing = np.array([self.dx, self.dy, self.dz])
        bounds = prism.bounds
        lower = np.maximum(0, np.floor(bounds[0] / spacing).astype(int))
        upper = np.minimum(
            [c.nx - 1, c.ny - 1, c.nz - 1], np.floor(bounds[1] / spacing).astype(int)
        )
        upper[2] = min(upper[2], int(np.floor(self.liquid_level / self.dz)))
        volumes = np.zeros(self.shape)
        for iz in range(lower[2], upper[2] + 1):
            for iy in range(lower[1], upper[1] + 1):
                for ix in range(lower[0], upper[0] + 1):
                    if self.liquid_volumes[iz, iy, ix] <= 0.0:
                        continue
                    lo = np.array([ix, iy, iz]) * spacing
                    hi = np.minimum(lo + spacing, [c.width, c.depth, self.liquid_level])
                    volumes[iz, iy, ix] = prism.clipped_volume(lo, hi)
        return volumes.ravel()

    def _geometry(self, crystal, level):
        """Intersect the solid with cells and grid faces, including partial cells."""
        c = self.config
        heights = np.clip(level - np.arange(c.nz) * self.dz, 0.0, self.dz)
        volumes = np.broadcast_to(
            (heights * self.dx * self.dy)[:, None, None], self.shape
        ).copy()
        solid = np.zeros(self.shape)
        # Open areas start as the portion of each grid face below the free surface.
        openings = [
            np.broadcast_to(
                ((np.arange(1, c.nz) * self.dz <= level) * self.dx * self.dy)[
                    :, None, None
                ],
                (c.nz - 1, c.ny, c.nx),
            ).copy(),
            np.broadcast_to(
                (heights * self.dx)[:, None, None], (c.nz, c.ny - 1, c.nx)
            ).copy(),
            np.broadcast_to(
                (heights * self.dy)[:, None, None], (c.nz, c.ny, c.nx - 1)
            ).copy(),
        ]
        if crystal is not None:
            bounds = crystal.bounds
            spacing = np.array([self.dx, self.dy, self.dz])
            lower_indices = np.maximum(0, np.floor(bounds[0] / spacing).astype(int))
            upper_indices = np.minimum(
                [c.nx - 1, c.ny - 1, c.nz - 1],
                np.floor(bounds[1] / spacing).astype(int),
            )
            for iz in range(lower_indices[2], upper_indices[2] + 1):
                for iy in range(lower_indices[1], upper_indices[1] + 1):
                    for ix in range(lower_indices[0], upper_indices[0] + 1):
                        lo = np.array([ix, iy, iz]) * spacing
                        hi = lo + spacing
                        solid[iz, iy, ix] = crystal.clipped_volume(lo, hi)
                        if hi[2] <= level:
                            submerged = solid[iz, iy, ix]
                        elif lo[2] >= level:
                            submerged = 0.0
                        else:
                            submerged = crystal.clipped_volume(
                                lo, np.minimum(hi, [c.width, c.depth, level])
                            )
                        volumes[iz, iy, ix] -= submerged
            # Only faces near the solid need polygon intersection work.
            positions = self.positions.reshape((*self.shape, 3))
            for axis, area in enumerate(openings):
                lower, upper = [slice(None)] * 3, [slice(None)] * 3
                lower[axis], upper[axis] = slice(None, -1), slice(1, None)
                centers = (positions[tuple(lower)] + positions[tuple(upper)]) / 2
                physical_axis = 2 - axis
                radius = spacing / 2
                radius[physical_axis] = 0.0
                near = np.all(
                    (centers + radius >= bounds[0] - 1e-12)
                    & (centers - radius <= bounds[1] + 1e-12),
                    axis=-1,
                )
                for iz, iy, ix in np.argwhere(near & (area > 0.0)):
                    center = centers[iz, iy, ix]
                    lo, hi = center - radius, center + radius
                    if physical_axis != 2:
                        hi[2] = min(hi[2], level)
                    other = [a for a in range(3) if a != physical_axis]
                    rectangle = []
                    for first, second in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        point = lo.copy()
                        point[other[0]] = (lo, hi)[first][other[0]]
                        point[other[1]] = (lo, hi)[second][other[1]]
                        rectangle.append(point)
                    area[iz, iy, ix] -= crystal.section_area(rectangle)
        # Geometric operations may leave roundoff at exactly dry/blocked cells.
        # This only snaps geometric zero, not unstable concentrations or lost mass.
        volumes[abs(volumes) < 1e-12 * self.cell_volume] = 0.0
        if volumes.min() < -1e-9 * self.cell_volume:
            raise ValueError("Invalid negative liquid cut-cell volume")
        for area in openings:
            area[
                abs(area)
                < 1e-12 * max(self.dx * self.dy, self.dy * self.dz, self.dz * self.dx)
            ] = 0.0
            if area.min(initial=0.0) < -1e-9:
                raise ValueError("Invalid negative liquid aperture")
        return LiquidGeometry(
            volumes, solid / self.cell_volume, tuple(openings), float(level)
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
        )
        velocity = np.zeros((*self.shape, 3))
        for axis, (flux, old_area, new_area) in enumerate(
            zip(fluxes, self.geometry.openings, geometry.openings)
        ):
            area = np.maximum(old_area, new_area)
            speed = np.divide(
                flux, area * dt, out=np.zeros_like(flux), where=area > 0.0
            )
            lower, upper = [slice(None)] * 3, [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            velocity[tuple(lower) + (2 - axis,)] += speed / 2
            velocity[tuple(upper) + (2 - axis,)] += speed / 2
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
        )
        self.temperature, self.concentration = temperature, concentration
