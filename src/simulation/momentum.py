# HOW THIS FILE WORKS
# Advect liquid-volume mean momentum through the existing native open faces.
# One implicit upwind flux is shared with opposite signs by the adjacent cells;
# actual positive liquid volumes are retained even at large outgoing Courant
# numbers. Row scaling conditions the solve without clipping or capacity floors.
# Combine this bounded transport block with the existing coupled Stokes step,
# using its old divergence-free trace velocity as the nonlinear advecting flow.
# The resulting split update is first order in time/space; diagnostics separate
# donor-cell damping, both Euler damping terms, viscosity and pressure work.
# This fixed-domain operator does not change the canonical runtime or growth.

"""Conservative implicit momentum transport and a fixed cut-cell split step."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from .pressure import PressureProjection
from .stokes import CutCellStokes, StokesState
from .units import finite_number


@dataclass(frozen=True)
class MomentumStep:
    """Final constrained state, intermediate volume means and measured budgets."""

    state: StokesState
    advected_velocity_m_per_s: np.ndarray
    diagnostics: dict


class ImplicitMomentumTransport:
    """Closed/periodic first-order upwind transport of Cartesian volume means.

    Incompressible native face velocities prescribe the advecting flow. Their
    volume flux is independent of the transported vector in this building block.
    Implicit Euler removes the explicit small-cell stability restriction; it
    does not remove timestep/spatial accuracy requirements or donor diffusion.
    """

    def __init__(self, projection):
        if not isinstance(projection, PressureProjection):
            raise ValueError("Momentum transport requires shared PressureProjection")
        self.projection = projection
        self.fluid = projection.fluid
        self.volumes = projection._volumes
        self._left = np.concatenate([f.left for f in projection.faces])
        self._right = np.concatenate([f.right for f in projection.faces])
        self._cache = None

    def _values(self, value):
        array = self.projection._array(
            value, (*self.fluid.shape, 3), "Momentum velocity"
        )
        if np.any(array[~self.fluid]):
            raise ValueError("Momentum velocity must be zero in solid cells")
        return array[self.fluid].copy()

    def _flux(self, native_velocity):
        fields = self.projection.volume_fluxes(native_velocity)
        flux = np.concatenate(
            [q[tuple(f.cells.T)] for q, f in zip(fields, self.projection.faces)]
        )
        net = np.bincount(
            self._left, weights=flux, minlength=len(self.volumes)
        ) - np.bincount(self._right, weights=flux, minlength=len(self.volumes))
        scale = max(float(np.max(abs(flux), initial=0)), np.finfo(float).tiny)
        if np.max(abs(net), initial=0) > 1e-10 * scale:
            raise ValueError(
                "Momentum transport requires divergence-free volume fluxes"
            )
        return flux, net

    def _system(self, flux, dt):
        if (
            self._cache is not None
            and self._cache[0] == dt
            and np.array_equal(self._cache[1], flux)
        ):
            return self._cache[2:]
        left, right = self._left, self._right
        donor = np.where(flux >= 0, left, right)
        c = coo_matrix(
            (np.r_[flux, -flux], (np.r_[left, right], np.r_[donor, donor])),
            shape=(len(self.volumes),) * 2,
        ).tocsr()
        system = diags(self.volumes) + dt * c
        scale = 1 / system.diagonal()
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ArithmeticError("Invalid implicit momentum diagonal")
        factor = splu((diags(scale) @ system).tocsc())
        self._cache = (dt, flux.copy(), c, system, scale, factor)
        return self._cache[2:]

    def step(self, velocity_m_per_s, advecting_native_velocity_m_per_s, dt_s):
        """Return transported means and integrated conservation/energy budgets."""
        dt = finite_number(dt_s, "time step", positive=True)
        old = self._values(velocity_m_per_s)
        flux, net = self._flux(advecting_native_velocity_m_per_s)
        if np.max(abs(dt * net / self.volumes), initial=0) > 1e-9:
            raise ValueError(
                "Volume-flux divergence is too large over this momentum step"
            )
        c, system, scale, factor = self._system(flux, dt)
        rhs = self.volumes[:, None] * old
        values = factor.solve(scale[:, None] * rhs)
        residual = system @ values - rhs
        reference = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        if (
            not np.isfinite(values).all()
            or np.linalg.norm(residual) > 3e-12 * reference
        ):
            raise ArithmeticError("Implicit momentum solve failed its residual check")
        outgoing = np.bincount(
            self._left, weights=np.maximum(flux, 0), minlength=len(self.volumes)
        ) + np.bincount(
            self._right, weights=np.maximum(-flux, 0), minlength=len(self.volumes)
        )
        jump = values[self._left] - values[self._right]
        donor_dissipation = 0.5 * float(np.sum(abs(flux)[:, None] * jump**2))
        divergence_work = 0.5 * float(np.sum(net[:, None] * values**2))
        old_energy = 0.5 * float(np.sum(self.volumes[:, None] * old**2))
        new_energy = 0.5 * float(np.sum(self.volumes[:, None] * values**2))
        euler_damping = 0.5 * float(np.sum(self.volumes[:, None] * (values - old) ** 2))
        momentum_change = np.sum(self.volumes[:, None] * (values - old), axis=0)
        result = np.zeros((*self.fluid.shape, 3))
        result[self.fluid] = values
        result.setflags(write=False)
        lower, upper = old.min(axis=0), old.max(axis=0)
        metrics = dict(
            max_outgoing_courant=float(np.max(dt * outgoing / self.volumes, initial=0)),
            old_energy_m5_per_s2=old_energy,
            energy_m5_per_s2=new_energy,
            donor_dissipation_m5_per_s3=donor_dissipation,
            divergence_work_m5_per_s3=divergence_work,
            euler_damping_m5_per_s2=euler_damping,
            work_residual_m5_per_s3=(new_energy - old_energy + euler_damping) / dt
            + donor_dissipation
            + divergence_work,
            momentum_change_m4_per_s=momentum_change.tolist(),
            bound_violation_m_per_s=float(
                max(
                    np.max(lower - values, initial=0), np.max(values - upper, initial=0)
                )
            ),
            max_integrated_momentum_residual_m4_per_s=float(
                np.max(abs(residual), initial=0)
            ),
        )
        return result, metrics


class CutCellMomentum:
    """Lagged nonlinear advection followed by coupled viscosity/pressure.

    The state supplies the old divergence-free advecting face means. Transport
    is implicit in the advected cell means, then the existing monolithic Stokes
    solve restores the new constrained traces. This is a first-order split
    method, not a fully implicit nonlinear solve or a second-order MAC scheme.
    Outer/embedded walls must be stationary; no inlet/outlet or moving wall API.
    """

    def __init__(self, stokes):
        if not isinstance(stokes, CutCellStokes):
            raise ValueError("Cut-cell momentum requires shared CutCellStokes")
        self.stokes = stokes
        self.transport = ImplicitMomentumTransport(stokes.projection)

    def advance(self, state, dt_s, *, acceleration_m_per_s2=None):
        if not isinstance(state, StokesState):
            raise ValueError("Supply a constrained StokesState")
        dt = finite_number(dt_s, "time step", positive=True)
        v, g = self.stokes.viscosity, self.stokes.geometry
        traces = v._array(
            state.trace_velocity_m_per_s, (len(v.boundary), 3), "Trace velocity"
        )
        pressure = v._array(state.pressure_pa, g.grid.shape, "Pressure")
        if np.any(pressure[~g.fluid]):
            raise ValueError("Pressure must be zero in solid cells")
        if np.any(traces[v.boundary]):
            raise ValueError(
                "Nonlinear step requires stationary no-slip boundary traces"
            )
        advected, advection = self.transport.step(
            state.cell_velocity_m_per_s, self.stokes.normal_face_velocity(state), dt
        )
        new = self.stokes.step(
            advected, dt, acceleration_m_per_s2=acceleration_m_per_s2
        )
        metrics = self.stokes.balances(
            new, previous=advected, dt_s=dt, acceleration=acceleration_m_per_s2
        )
        metrics.update({"advection_" + key: value for key, value in advection.items()})
        # The transport and Stokes substeps use different intermediate states.
        # Keep the Stokes work/energy values separately before reporting the
        # combined budget, including both backward-Euler damping terms.
        final_values = new.cell_velocity_m_per_s[g.fluid]
        intermediate = advected[g.fluid]
        final_energy = 0.5 * float(np.sum(v.volumes[:, None] * final_values**2))
        stokes_damping = 0.5 * float(
            np.sum(v.volumes[:, None] * (final_values - intermediate) ** 2)
        )
        combined = (
            (
                final_energy
                - advection["old_energy_m5_per_s2"]
                + advection["euler_damping_m5_per_s2"]
                + stokes_damping
            )
            / dt
            + advection["donor_dissipation_m5_per_s3"]
            + advection["divergence_work_m5_per_s3"]
            + metrics["dissipation_m5_per_s3"]
            + metrics["pressure_work_m5_per_s3"]
            - metrics["body_power_m5_per_s3"]
            - metrics["boundary_power_m5_per_s3"]
        )
        boundary_flux = np.zeros_like(new.trace_velocity_m_per_s)
        for i, (ids, k, _b, _s) in enumerate(v._local):
            np.add.at(
                boundary_flux,
                ids,
                self.stokes.nu
                * (k @ (final_values[i] - new.trace_velocity_m_per_s[ids])),
            )
        pi = new.pressure_pa[g.fluid] / self.stokes.density
        boundary_flux += (self.stokes.divergence_integral.T @ pi).reshape(-1, 3)
        wall_impulse = -dt * boundary_flux[v.boundary].sum(axis=0)
        force = (
            np.zeros_like(final_values)
            if acceleration_m_per_s2 is None
            else v._cells_array(acceleration_m_per_s2, "Acceleration")
        )
        body_impulse = dt * np.sum(v.volumes[:, None] * force, axis=0)
        momentum_change = np.sum(
            v.volumes[:, None] * (final_values - state.cell_velocity_m_per_s[g.fluid]),
            axis=0,
        )
        metrics.update(
            energy_m5_per_s2=final_energy,
            advection_energy_m5_per_s2=advection["energy_m5_per_s2"],
            stokes_euler_damping_m5_per_s2=stokes_damping,
            combined_work_residual_m5_per_s3=combined,
            max_outgoing_courant=advection["max_outgoing_courant"],
            wall_momentum_impulse_m4_per_s=wall_impulse.tolist(),
            body_momentum_impulse_m4_per_s=body_impulse.tolist(),
            combined_momentum_residual_m4_per_s=(
                momentum_change - body_impulse - wall_impulse
            ).tolist(),
        )
        return MomentumStep(new, advected, metrics)
