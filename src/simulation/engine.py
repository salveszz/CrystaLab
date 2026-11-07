# HOW THE SIMULATION WORKS
# 1. Create a seeded random generator, temperature/concentration grid, and a
#    rectangular crystal seed at the center of the tank.
# 2. Split each requested timestep into stable internal steps. Diffuse heat and
#    solute, relax temperature toward its target, and mix concentration toward
#    its mean according to the rotation and circulation settings.
# 3. Sample supersaturation near each face. Advance each face by its growth
#    coefficient times positive supersaturation, a transport factor, and time.
#    Impurities reduce this factor; rotation and circulation increase it.
# 4. Limit growth to the tank walls and available dissolved mass. Remove the
#    deposited mass proportionally from ALL solution cells, conserving combined
#    dissolved and crystal mass. This does not create local surface depletion.
# 5. Optionally sample synthetic defect events and record measurements over time.
# run() repeats steps until the configured duration is reached.
# This is an uncalibrated dilute-reservoir model: no actual fluid velocity,
# moving liquid boundary, dissolution, or prediction of real material defects.

"""Seeded growth engine independent of plotting and AI.

Approximations: fixed-volume dilute reservoir; rectangular crystal; no dissolution;
linear face kinetics; rotation/flow modeled as mixing and transport enhancement.
The crystal overlay does not displace liquid. Dissolved + crystal mass is conserved.
"""
from dataclasses import asdict
import math
import numpy as np
from .tank import SimulationConfig, Tank
from .crystal import Crystal
from .physics import diffuse, supersaturation


class Simulation:
    def __init__(self, config=None):
        self.config = config or SimulationConfig()
        c = self.config
        # One seeded generator drives initial concentration noise and synthetic events.
        self.rng = np.random.default_rng(c.seed)
        self.tank = Tank(c, self.rng)
        # Place the rectangular seed symmetrically about the tank center.
        self.crystal = Crystal((c.width-c.seed_width)/2, (c.width+c.seed_width)/2,
                               (c.height-c.seed_height)/2, (c.height+c.seed_height)/2)
        self.time = 0.0
        # Keep the initial measurement so exports include the starting state.
        self.history = [self.measurements()]

    @property
    def total_mass(self):
        # Mass balance includes dissolved solute plus crystal area times density.
        return self.tank.solute_mass + self.crystal.area * self.config.crystal_density

    # Keep field physics in physics.py; the engine supplies its configured coefficients.
    def supersaturation(self):
        c = self.config
        return supersaturation(self.tank.concentration, self.tank.temperature,
                               c.solubility_reference, c.solubility_slope, c.reference_temperature)

    # Build a scalar snapshot for plotting/CSV; full fields remain in the tank.
    def measurements(self):
        return dict(time=self.time, width=self.crystal.width, height=self.crystal.height,
                    area=self.crystal.area, concentration=float(self.tank.concentration.mean()),
                    temperature=float(self.tank.temperature.mean()),
                    supersaturation=float(self.supersaturation().mean()),
                    defects=self.crystal.defects, total_mass=self.total_mass)

    def step(self, dt=None):
        c, tank = self.config, self.tank
        # A caller may request a different external step, but it must be positive/finite.
        elapsed = c.dt if dt is None else dt
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("dt must be positive and finite")
        # Substep both diffusion and kinetics, including for large requested steps.
        # Finer cells and larger diffusivity require smaller internal diffusion steps.
        stability = max(c.solute_diffusivity, c.thermal_diffusivity) * (1/tank.dx**2 + 1/tank.dy**2)
        # Keep the explicit diffusion factor below 0.45 and kinetics steps at most one.
        count = max(1, math.ceil(elapsed * max(stability / 0.45, 1.0)))
        h = elapsed / count
        for _ in range(count):
            # Spread temperature spatially, then relax it toward the configured target.
            tank.temperature = diffuse(tank.temperature, c.thermal_diffusivity, h, tank.dx, tank.dy)
            tank.temperature += (c.temperature - tank.temperature) * (-np.expm1(-c.thermal_relaxation*h))
            # Spread solute before applying the separate bulk mixing approximation.
            tank.concentration = diffuse(tank.concentration, c.solute_diffusivity, h, tank.dx, tank.dy)
            # Exponential relaxation stays between zero and one for nonnegative settings.
            mixing = -np.expm1(-0.1 * (c.rotation + c.circulation) * h)
            tank.concentration += mixing * (tank.concentration.mean() - tank.concentration)
            # Deposit crystal material using the updated fields for this internal step.
            self._grow(h)
        # Record only the completed external step, not every numerical substep.
        self.time += elapsed
        row = self.measurements()
        self.history.append(row)
        return row

    def _grow(self, dt):
        c, k, tank = self.config, self.crystal, self.tank
        sigma = self.supersaturation()
        # Sample the solution next to each face, averaging along that face.
        def local(x, y, vertical):
            # Map requested physical coordinates into valid grid-cell indices.
            ix = int(np.clip(x / tank.dx, 0, c.nx-1))
            iy = int(np.clip(y / tank.dy, 0, c.ny-1))
            # Average cells along the face; fall back to one cell for a sub-grid seed.
            if vertical:
                indices = (tank.y >= k.bottom) & (tank.y <= k.top)
                return float(sigma[indices, ix].mean()) if indices.any() else float(sigma[iy, ix])
            indices = (tank.x >= k.left) & (tank.x <= k.right)
            return float(sigma[iy, indices].mean()) if indices.any() else float(sigma[iy, ix])
        # Face midpoints locate the rows/columns sampled just outside each boundary.
        cx, cy = (k.left+k.right)/2, (k.bottom+k.top)/2
        # Keep the same ordering throughout: left, right, bottom, top.
        values = [local(k.left-tank.dx, cy, True), local(k.right+tank.dx, cy, True),
                  local(cx, k.bottom-tank.dy, False), local(cx, k.top+tank.dy, False)]
        # Illustrative enhancement: mixing increases growth; impurity suppresses it.
        transport = (1 + 0.2 * (c.rotation+c.circulation)) / (1+c.impurity)
        # Convert each kinetic coefficient into an outward travel distance.
        delta = np.array([c.growth_left, c.growth_right, c.growth_bottom, c.growth_top])
        # Only positive supersaturation grows material; dissolution is not modeled.
        delta *= np.maximum(values, 0) * transport * dt
        # A face cannot travel farther than the distance to its corresponding wall.
        delta = np.minimum(delta, [k.left, c.width-k.right, k.bottom, c.height-k.top])
        # Compute the combined area increase, including growth at the corners.
        # Density converts that extra unit-depth area into required dissolved mass.
        def needed(scale):
            return ((k.width + scale*(delta[0]+delta[1])) *
                    (k.height + scale*(delta[2]+delta[3])) - k.area) * c.crystal_density
        available = tank.solute_mass
        scale = 1.0
        # If solute is insufficient, bisect a shared scale so every face slows together.
        if needed(scale) > available:
            low, high = 0.0, 1.0
            for _ in range(60):
                mid = (low+high)/2
                if needed(mid) <= available:
                    low = mid
                else:
                    high = mid
            scale = low
        # Withdraw exactly the deposited mass to conserve the combined mass balance.
        used = needed(scale)
        if available > 0:
            # Global proportional withdrawal is simple, but does not model local depletion.
            tank.concentration *= max(0.0, 1-used/available)
        delta *= scale
        # Outward means decreasing left/bottom and increasing right/top coordinates.
        k.left -= delta[0]
        k.right += delta[1]
        k.bottom -= delta[2]
        k.top += delta[3]
        # Synthetic defect count proportional to deposited mass and impurity.
        k.defects += int(self.rng.poisson(c.defect_rate * used * (1+c.impurity)))

    def run(self):
        # Trim the last step to finish at duration; the tolerance avoids tiny extra steps.
        while self.time < self.config.duration - 1e-12:
            self.step(min(self.config.dt, self.config.duration-self.time))
        return self.history
