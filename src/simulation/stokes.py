# HOW THIS FILE WORKS
# Couple the existing mimetic cut-cell viscosity to one cell pressure using the
# integrated divergence of its shared face velocities. The pressure force is
# the exact negative transpose of this divergence, so closed-domain pressure
# does no work. Condense volume means locally and solve the remaining symmetric
# indefinite face/pressure system with one pressure gauge per liquid region.
# Implicit Euler retains cell-mean kinetic energy and stationary no-slip traces.
# Export native normal fluxes and audit momentum, continuity and work together.
# This is fixed-domain linear Stokes, not nonlinear or growing-crystal flow.

"""Conservative pressure/viscosity coupling on stationary cut polyhedra."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, coo_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

from .pressure import PressureProjection
from .units import finite_number
from .viscosity import CutCellViscosity, ViscousState


@dataclass(frozen=True)
class StokesState(ViscousState):
    """Liquid/face velocity means and volume-mean gauge pressure in Pa."""

    pressure_pa: np.ndarray


class CutCellStokes:
    """Implicit or steady constant-property Stokes with fixed no-slip walls.

    Pressure couples directly to the viscosity's face means, not by projecting
    and interpolating unrelated MAC values. Cell means carry mass; face means
    enforce integrated continuity. The existing MAC pressure solver supplies
    topology, gauges and an independent native-flux check, not its A*d metric.
    This lowest-order hybrid scheme is not pressure-robust; gradient-force
    velocity error must be measured under refinement, especially at small nu.
    """

    def __init__(self, geometry, *, kinematic_viscosity_m2_per_s, density_kg_per_m3):
        self.nu = finite_number(
            kinematic_viscosity_m2_per_s, "kinematic viscosity", positive=True
        )
        self.density = finite_number(density_kg_per_m3, "density", positive=True)
        self.viscosity = CutCellViscosity(geometry)
        self.geometry = geometry
        self.projection = PressureProjection(
            geometry.grid, geometry.volumes_m3, geometry.open_areas_m2
        )
        v = self.viscosity
        self._free_dofs = (3 * v._free[:, None] + np.arange(3)).ravel()
        rows, columns, data = [], [], []
        lookup = np.full(geometry.grid.shape, -1, dtype=int)
        lookup[geometry.fluid] = np.arange(len(v.cells))
        for index, raw in enumerate(v.cells):
            for axis in range(3):
                for side in (0, 1):
                    face = list(raw)
                    face[axis] += side
                    face = tuple(face)
                    face_id = v.face_ids[axis][face]
                    if face_id < 0:
                        continue
                    rows.append(index)
                    columns.append(3 * face_id + 2 - axis)
                    data.append((2 * side - 1) * geometry.open_areas_m2[axis][face])
        for wall, raw in enumerate(geometry.wall_cells):
            for component in range(3):
                rows.append(lookup[tuple(raw)])
                columns.append(3 * v.wall_ids[wall] + component)
                data.append(
                    geometry.wall_areas_m2[wall]
                    * geometry.wall_normals[wall, component]
                )
        self.divergence_integral = coo_matrix(
            (data, (rows, columns)), shape=(len(v.cells), 3 * len(v.boundary))
        ).tocsr()
        self._groups = [nodes for nodes, _matrix, _factor in self.projection.groups]
        self._pressure_rows = np.concatenate([nodes[:-1] for nodes in self._groups])
        self._coupling = self.divergence_integral[:, self._free_dofs] / geometry.scale_m
        self._cache = None
        self._free_dofs.setflags(write=False)
        self._pressure_rows.setflags(write=False)

    def _system(self, ratio):
        if self._cache is not None and self._cache[0] == ratio:
            return self._cache[1:]
        v = self.viscosity
        matrix, reduced, _factor, factors = v._system(ratio)
        velocity = kron(reduced, eye(3, format="csr"), format="csr")
        coupling = self._coupling[self._pressure_rows]
        saddle = bmat([[velocity, -coupling.T], [-coupling, None]], format="csc")
        # Congruent diagonal scaling preserves symmetry. Pressure scales use
        # the diagonal Schur estimate; no penalty or pressure stabilization is
        # added, and no cell continuity equation is regularized.
        diagonal = velocity.diagonal()
        pressure_diagonal = np.asarray(coupling.power(2) @ (1 / diagonal)).ravel()
        scale = 1 / np.sqrt(np.r_[diagonal, pressure_diagonal])
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ArithmeticError("Degenerate coupled Stokes scaling")
        scaled = diags(scale) @ saddle @ diags(scale)
        factor = splu(scaled.tocsc()) if saddle.shape[0] else None
        self._cache = (ratio, matrix, factors, saddle, scale, factor)
        return self._cache[1:]

    def _solve(self, acceleration, *, previous=None, dt=None, boundary_velocity=None):
        v = self.viscosity
        force = v._cells_array(acceleration, "Acceleration")
        boundary = v._boundary_values(boundary_velocity)
        if dt is None:
            if not v._steady_anchored:
                raise ValueError(
                    "Steady Stokes needs a Dirichlet wall in every liquid region"
                )
            ratio = None
            load = v.volumes[:, None] * force / self.nu
        else:
            dt = finite_number(dt, "time step", positive=True)
            ratio = finite_number(dt * self.nu, "dt*nu", positive=True)
            old = v._cells_array(previous, "Previous velocity")
            load = v.volumes[:, None] * (old / ratio + force / self.nu)
        boundary_flux = self.divergence_integral @ boundary.ravel()
        flux_scale = max(
            float(
                np.max(abs(self.divergence_integral) @ abs(boundary.ravel()), initial=0)
            ),
            np.finfo(float).tiny,
        )
        for nodes in self._groups:
            if abs(boundary_flux[nodes].sum()) > 1e-10 * flux_scale * len(nodes):
                raise ValueError(
                    "Incompatible prescribed flux in a closed liquid region"
                )
        matrix, factors, saddle, scale, factor = self._system(ratio)
        rhs = np.zeros_like(boundary)
        for i, (ids, _k, b, _s) in enumerate(v._local):
            np.add.at(rhs, ids, b[:, None] * (load[i] * factors[i]))
        rhs -= matrix @ boundary
        right = np.r_[
            rhs[v._free].ravel(),
            boundary_flux[self._pressure_rows] / self.geometry.scale_m,
        ]
        answer = (
            np.zeros_like(right)
            if not np.any(right)
            else scale * factor.solve(scale * right)
        )
        reference = max(float(np.linalg.norm(right)), np.finfo(float).tiny)
        residual = saddle @ answer - right
        if (
            not np.isfinite(answer).all()
            or np.linalg.norm(residual) > 3e-12 * reference
        ):
            raise ArithmeticError("Coupled Stokes solve failed its residual check")
        trace = boundary.copy()
        count = len(self._free_dofs)
        trace[v._free] = answer[:count].reshape(-1, 3)
        means = np.zeros((*self.geometry.grid.shape, 3))
        means[self.geometry.fluid] = np.array(
            [
                (load[i] + b @ trace[ids]) * factors[i]
                for i, (ids, _k, b, _s) in enumerate(v._local)
            ]
        )
        pi = np.zeros(len(v.cells))
        pi[self._pressure_rows] = answer[count:] * self.nu / self.geometry.scale_m
        pressure = np.zeros(self.geometry.grid.shape)
        for nodes in self._groups:
            pi[nodes] -= np.average(pi[nodes], weights=v.volumes[nodes])
        pressure[self.geometry.fluid] = self.density * pi
        if not np.isfinite(means).all() or not np.isfinite(pressure).all():
            raise ArithmeticError("Nonfinite coupled Stokes state")
        for field in (means, trace, pressure):
            field.setflags(write=False)
        return StokesState(means, trace, pressure)

    def step(
        self,
        velocity_m_per_s,
        dt_s,
        *,
        acceleration_m_per_s2=None,
        boundary_velocity_m_per_s=None,
    ):
        """Implicit Euler; embedded walls fixed, optional prescribed outer means."""
        force = (
            np.zeros((*self.geometry.grid.shape, 3))
            if acceleration_m_per_s2 is None
            else acceleration_m_per_s2
        )
        return self._solve(
            force,
            previous=velocity_m_per_s,
            dt=dt_s,
            boundary_velocity=boundary_velocity_m_per_s,
        )

    def steady(self, acceleration_m_per_s2, *, boundary_velocity_m_per_s=None):
        """Steady Stokes, with optional manufactured outer velocity means."""
        return self._solve(
            acceleration_m_per_s2, boundary_velocity=boundary_velocity_m_per_s
        )

    def normal_face_velocity(self, state):
        return self.viscosity.normal_face_velocity(state)

    def net_flux(self, state):
        trace = self.viscosity._array(
            state.trace_velocity_m_per_s,
            (len(self.viscosity.boundary), 3),
            "Trace velocity",
        )
        result = np.zeros(self.geometry.grid.shape)
        result[self.geometry.fluid] = self.divergence_integral @ trace.ravel()
        return result

    def balances(self, state, *, previous=None, dt_s=None, acceleration=None):
        """Audit the combined equations, pressure work and cell-mean energy.

        Density is factored out: multiply energy/powers by rho for joules/watts.
        Pressure is returned in Pa and converted to p/rho here. Report both
        integrated flux and divergence so tiny volumes cannot conceal errors.
        """
        v = self.viscosity
        means = v._cells_array(state.cell_velocity_m_per_s, "Velocity")
        trace = v._array(
            state.trace_velocity_m_per_s, (len(v.boundary), 3), "Trace velocity"
        )
        pressure = v._array(state.pressure_pa, self.geometry.grid.shape, "Pressure")
        pi = pressure[self.geometry.fluid] / self.density
        force = (
            np.zeros_like(means)
            if acceleration is None
            else v._cells_array(acceleration, "Acceleration")
        )
        residual = -v.volumes[:, None] * force
        viscous_flux = np.zeros_like(trace)
        dissipation = 0.0
        for i, (ids, k, _b, _s) in enumerate(v._local):
            difference = means[i] - trace[ids]
            flux = self.nu * (k @ difference)
            residual[i] += flux.sum(axis=0)
            np.add.at(viscous_flux, ids, flux)
            dissipation += float(np.sum(difference * flux))
        total_flux = viscous_flux + (self.divergence_integral.T @ pi).reshape(-1, 3)
        net = self.divergence_integral @ trace.ravel()
        pressure_work = -float(pi @ net)
        boundary_power = -float(np.sum(trace[v.boundary] * total_flux[v.boundary]))
        body_power = float(np.sum(v.volumes[:, None] * means * force))
        energy = 0.5 * float(np.sum(v.volumes[:, None] * means**2))
        work_residual = dissipation + pressure_work - body_power - boundary_power
        if previous is not None:
            dt_s = finite_number(dt_s, "time step", positive=True)
            old = v._cells_array(previous, "Previous velocity")
            residual += v.volumes[:, None] * (means - old) / dt_s
            old_energy = 0.5 * float(np.sum(v.volumes[:, None] * old**2))
            change_energy = 0.5 * float(np.sum(v.volumes[:, None] * (means - old) ** 2))
            work_residual += (energy - old_energy + change_energy) / dt_s
        return dict(
            energy_m5_per_s2=energy,
            dissipation_m5_per_s3=dissipation,
            body_power_m5_per_s3=body_power,
            boundary_power_m5_per_s3=boundary_power,
            pressure_work_m5_per_s3=pressure_work,
            work_residual_m5_per_s3=work_residual,
            max_cell_momentum_residual_m4_per_s2=float(np.max(abs(residual))),
            max_face_momentum_residual_m4_per_s2=float(
                np.max(abs(total_flux[v._free]), initial=0)
            ),
            max_integral_flux_m3_per_s=float(np.max(abs(net))),
            max_divergence_per_s=float(np.max(abs(net / v.volumes))),
        )
