# HOW THE SIMULATION WORKS
# Create a 3D face-plane seed and volumetric temperature/concentration fields.
# Append six stationary wall planes so even sloped crystal faces stay in the tank.
# For each stable internal step, diffuse both fields, relax temperature and mix.
# Integrate exact cell overlaps with outward face prisms of fixed physical depth.
# Advance face offsets using local supersaturation and individual coefficients.
# Scale growth to the solute available in those surface neighborhoods; remove
# exactly density * volume_gain, conserving dissolved plus crystal mass.
# Stop at duration, including when dt does not divide it. Defects are optional
# synthetic counts, not measured quality. Partial liquid volumes exclude solid,
# a rising free surface conserves carrier volume, and displacement advects solute/heat.
# The potential-flow remap is quasistatic; mixing/kinetics remain uncalibrated.

"""Faceted 3D growth with liquid exclusion and open-tank displacement."""

import math
from numbers import Real

import numpy as np

from .tank import SimulationConfig, Tank
from .physics import supersaturation


class Simulation:
    def __init__(self, config=None):
        self.config = config if config is not None else SimulationConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        seed = c.make_crystal()
        rates = (
            np.full(len(seed.normals), c.growth_rate)
            if c.face_growth is None
            else np.asarray(c.face_growth, dtype=float)
        )
        # Growth coefficients follow crystal faces, not global left/right directions.
        # Wall planes are fixed, with zero growth even when they become exposed.
        self.crystal = seed.bounded_by_box([c.width, c.depth, c.height])
        self.coefficients = np.r_[rates, np.zeros(6)]
        self.face_rates = np.zeros(len(self.coefficients))
        self.tank = Tank(c, self.rng, crystal=self.crystal)
        self.time = 0.0
        self.history = [self.measurements()]

    @property
    def total_mass(self):
        return self.tank.solute_mass + self.crystal.volume * self.config.crystal_density

    def supersaturation(self):
        c = self.config
        return supersaturation(
            self.tank.concentration,
            self.tank.temperature,
            c.solubility_reference,
            c.solubility_slope,
            c.reference_temperature,
        )

    def measurements(self):
        """Record scalar volume-based outcomes; full fields remain 3D matrices."""
        k, c = self.crystal, self.config
        return dict(
            time=self.time,
            width=k.width,
            depth=k.depth,
            height=k.height,
            volume=k.volume,
            surface_area=k.surface_area,
            crystal_mass=k.volume * c.crystal_density,
            dissolved_mass=self.tank.solute_mass,
            liquid_level=self.tank.liquid_level,
            liquid_volume=self.tank.liquid_volume,
            solid_occupied_volume=float(
                self.tank.geometry.solid_fraction.sum() * self.tank.cell_volume
            ),
            displacement_speed=float(
                np.linalg.norm(self.tank.displacement_velocity, axis=-1).max()
            ),
            continuity_error=self.tank.continuity_error,
            concentration=self.tank.liquid_mean(self.tank.concentration),
            temperature=self.tank.liquid_mean(self.tank.temperature),
            supersaturation=self.tank.liquid_mean(self.supersaturation()),
            defects=k.defects,
            total_mass=self.total_mass,
        )

    def step(self, dt=None):
        c, tank = self.config, self.tank
        elapsed = c.dt if dt is None else dt
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, Real)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("dt must be positive and finite")
        remaining = c.duration - self.time
        if remaining <= 0:
            return self.history[-1].copy()
        elapsed = min(elapsed, remaining)
        # Liquid cut-cell diffusion is implicit; substep only the evolving
        # geometry/kinetics here, not the tiny-cell explicit stability limit.
        count = max(1, math.ceil(elapsed))
        h = elapsed / count
        for _ in range(count):
            tank.diffuse(h)
            wet = tank.liquid_mask
            tank.temperature[wet] += (c.temperature - tank.temperature[wet]) * (
                -np.expm1(-c.thermal_relaxation * h)
            )
            # Stirring is an effective diffusivity through open liquid faces.
            # No global mean relaxation may shortcut through solid barriers.
            self._grow(h)
        # Snap the last step to the configured endpoint, with no replay or overshoot.
        self.time = c.duration if elapsed == remaining else self.time + elapsed
        self.history.append(self.measurements())
        return self.history[-1].copy()

    def _face_conditions(self):
        """Geometric facet-neighborhood averages and exact local source volumes."""
        k, tank, c = self.crystal, self.tank, self.config
        sigma = self.supersaturation().ravel()
        source_volumes = np.zeros(len(tank.positions))
        values = np.zeros(len(k.normals))
        layer = c.boundary_layer_cells * max(tank.dx, tank.dy, tank.dz)
        for i, face in enumerate(k.faces):
            if len(face) < 3 or self.coefficients[i] == 0.0:
                continue
            wetted_area = k.wetted_face_area(i, tank.liquid_level)
            if wetted_area <= 0.0:
                continue
            prism = k.sampling_prism(i, layer, tank.liquid_level)
            volumes = tank.sampling_volumes(prism)
            total = float(volumes.sum())
            if total <= 1e-14:
                continue
            values[i] = (
                float(np.dot(sigma, volumes) / total) * wetted_area / k.face_areas[i]
            )
            source_volumes += volumes
        capacities = tank.liquid_volumes.ravel()
        if np.any(source_volumes > capacities + 1e-9 * tank.cell_volume):
            raise ValueError("Facet sampling prisms exceed available liquid volume")
        # Only geometric roundoff is snapped. Sampling prisms are disjoint outside
        # a convex solid, so a liquid subvolume must never be counted twice.
        return values, np.minimum(source_volumes, capacities)

    def _grow(self, dt):
        c, k, tank = self.config, self.crystal, self.tank
        tank.displacement_velocity.fill(0.0)
        tank.continuity_error = 0.0
        self.face_rates.fill(0.0)
        if tank.liquid_level >= c.height - 1e-12:
            # This model tracks no overflow or compression. A full tank cannot
            # accept additional displaced volume, so growth must pause at the rim.
            return
        values, source_volumes = self._face_conditions()
        agitation = c.rotation + c.circulation
        transport = (1 + 0.2 * agitation / (1 + agitation)) / (1 + c.impurity)
        distances = self.coefficients * np.maximum(values, 0.0) * transport * dt
        if not np.any(distances > 0):
            return
        # Travel beyond twice the tank diagonal cannot change a clipped solid;
        # cap huge stress-test coefficients to avoid ill-conditioned offsets.
        distances = np.minimum(
            distances, 2 * math.sqrt(c.width**2 + c.depth**2 + c.height**2)
        )
        available = float(np.dot(tank.concentration.ravel(), source_volumes))
        if available <= 0:
            return

        def proposal(scale):
            candidate = k.advanced(distances * scale)
            used = max(0.0, candidate.volume - k.volume) * c.crystal_density
            try:
                tank.level_for(candidate)
                fits = True
            except ValueError:
                fits = False
            return candidate, used, fits

        candidate, used, fits = proposal(1.0)
        scale = 1.0
        if used > available or not fits:
            # Bisection scales all advancing faces together. Crucially, recompute
            # the actual solid volume each time; bounding-box volume is incorrect.
            lo, hi = 0.0, 1.0
            for _ in range(45):
                mid = (lo + hi) / 2
                _candidate, needed, fits = proposal(mid)
                if needed <= available and fits:
                    lo = mid
                else:
                    hi = mid
            scale = lo
            candidate, used, _fits = proposal(scale)
        if used <= 1e-14:
            return
        # Withdraw deposited solute, then conservatively move the remaining
        # solution to the new liquid geometry. No state changes if remapping fails.
        concentration = tank.concentration.copy().ravel()
        fraction = np.divide(
            source_volumes,
            tank.liquid_volumes.ravel(),
            out=np.zeros_like(source_volumes),
            where=tank.liquid_volumes.ravel() > 0.0,
        )
        # Uniform fractional withdrawal inside the physical prism, not inside all
        # neighboring voxels. Sum(C_i * V_prism_i) defines the same mass budget.
        concentration *= 1.0 - min(1.0, used / available) * fraction
        tank.move_boundary(
            candidate, dt, concentration.reshape(tank.concentration.shape)
        )
        candidate.defects = k.defects + int(
            self.rng.poisson(c.defect_rate * used * (1 + c.impurity))
        )
        self.crystal = candidate
        self.face_rates = distances * scale / dt

    def run(self):
        while self.time < self.config.duration:
            self.step()
        return self.history
