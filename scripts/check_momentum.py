# HOW THIS FILE WORKS
# Verify conservative implicit momentum transport and its split coupling to the
# existing cut-cell Stokes operator. Analytic periodic transport/ABC references
# and a polynomial curl field give independent spatial/time accuracy evidence.
# Retain exact liquid geometry, boundedness, momentum/energy budgets and a tiny
# cell circulation at large Courant number. Archive inputs/sources/results in a
# fresh directory; failed gates remain visible. This is fixed-domain verification,
# not physical growth calibration, AI training or canonical runner integration.

"""Independent momentum transport, small-cell and nonlinear coupling checks."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import leggauss
from scipy.signal import convolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation.cutcell import CutCellGeometry
from simulation.crystal import Crystal
from simulation.mesh import RectilinearGrid, focused_axis_edges
from simulation.momentum import ImplicitMomentumTransport, CutCellMomentum
from simulation.stokes import CutCellStokes, StokesState
from simulation.units import finite_number
from check_cutcell import wall_geometry, cube_geometry
from check_viscosity import rms_error


def operator_for(geometry, protocol):
    return CutCellStokes(
        geometry,
        kinematic_viscosity_m2_per_s=protocol["kinematic_viscosity_m2_per_s"],
        density_kg_per_m3=protocol["density_kg_per_m3"],
        linear_solver=protocol.get("linear_solver", "direct"),
    )


def momentum_wall_geometry(protocol, n, *, focused=False):
    """Preserve the focus band and explicitly resolve the coarse background.

    Subdivide intervals wider than the declared L/n cap. Record actual grid
    counts/edges in the study; extra resolution is not an equal-cell-budget
    comparison. Domain, wall plane and liquid geometry are unchanged.
    """
    settings = protocol["sloped_wall"]
    cap = settings.get("max_background_width_factor")
    length = protocol["length_m"]
    if not focused or cap is None:
        return wall_geometry(length, settings, n, focused=focused)
    limit = finite_number(cap, "background width factor", positive=True) * length / n
    original = focused_axis_edges(length, n, length / 2, settings["focus_width_m"])
    edge = np.r_[
        np.concatenate(
            [
                np.linspace(
                    lo, hi, max(1, int(np.ceil((hi - lo) / limit - 1e-12))) + 1
                )[:-1]
                for lo, hi in zip(original, original[1:])
            ]
        ),
        length,
    ]
    grid = RectilinearGrid([length] * 3, [len(edge) - 1] * 3, [False] * 3, [edge] * 3)
    box = Crystal.box([length / 2] * 3, [length] * 3)
    solid = Crystal(
        np.vstack((box.normals, [0, -settings["slope_dz_dy"], 1])),
        np.r_[box.offsets, settings["intercept_m"]],
    )
    return CutCellGeometry(grid, solid)


def polynomial_fields(protocol):
    """Coefficients of u=curl(psi*[1,1,1]) and analytic NS forcing.

    Dimensionless psi=x^2(1-x)^2*y^2(1-y)^2*z^2(1-z)^2*(z-a-b*y)^2.
    Its gradient vanishes on every outer/embedded wall. Derivatives and products
    of its coefficients give the exact Laplacian and (u dot grad)u independently
    of every discrete operator. Coefficient axes are x/y/z, not grid indexing.
    """
    settings = protocol["sloped_wall"]
    a, b = settings["intercept_m"] / protocol["length_m"], settings["slope_dz_dy"]
    f = np.array([0, 0, 1, -2, 1.0])
    d2 = np.array([[a * a, -2 * a, 1], [2 * a * b, -2 * b, 0], [b * b, 0, 0]])
    yz = convolve(np.outer(f, f), d2, method="direct")
    psi = f[:, None, None] * yz[None, :, :]
    derivatives = [poly.polyder(psi, axis=axis) for axis in range(3)]

    def add(*terms):
        shape = np.max([term.shape for term in terms], axis=0)
        out = np.zeros(shape)
        for term in terms:
            out[tuple(slice(0, n) for n in term.shape)] += term
        return out

    amplitude = protocol["speed_m_per_s"] * settings["streamfunction_scale"]
    u = [
        amplitude * add(derivatives[1], -derivatives[2]),
        amplitude * add(derivatives[2], -derivatives[0]),
        amplitude * add(derivatives[0], -derivatives[1]),
    ]
    length, nu = protocol["length_m"], protocol["kinematic_viscosity_m2_per_s"]
    viscous, advective = [], []
    for component in range(3):
        viscous.append(
            -nu
            / length**2
            * add(*[poly.polyder(u[component], 2, axis=axis) for axis in range(3)])
        )
        advective.append(
            add(
                *[
                    convolve(
                        u[axis], poly.polyder(u[component], axis=axis), method="direct"
                    )
                    / length
                    for axis in range(3)
                ]
            )
        )
    pressure_scale = nu * protocol["speed_m_per_s"] / length
    gradient = pressure_scale / length * np.array([1, 2, -3])
    linear = [add(viscous[i], np.full((1, 1, 1), gradient[i])) for i in range(3)]
    return u, linear, advective


def _yz_moments(bounds_y, bounds_z, a, b, shape, order):
    """Analytic z monomials with Gaussian y integration over true liquid.

    Split where the plane crosses a box edge. Constant y or z gives a face mean;
    both ranges give a cross-section mean. No clipped geometric moments or
    numerical stiffness enter the reference integrals.
    """
    yl, yh = bounds_y
    zl, zh = bounds_z
    ny, nz = shape
    nodes, weights = leggauss(order)
    if yl == yh:
        bottom = max(zl, a + b * yl)
        if bottom >= zh:
            return np.zeros(shape)
        zmean = np.array(
            [
                (zh ** (k + 1) - bottom ** (k + 1)) / ((k + 1) * (zh - bottom))
                for k in range(nz)
            ]
        )
        return yl ** np.arange(ny)[:, None] * zmean
    if zl == zh:
        upper = min(yh, (zl - a) / b)
        if upper <= yl:
            return np.zeros(shape)
        ymean = np.array(
            [
                (upper ** (k + 1) - yl ** (k + 1)) / ((k + 1) * (upper - yl))
                for k in range(ny)
            ]
        )
        return ymean[:, None] * zl ** np.arange(nz)
    breaks = sorted({yl, yh, *[v for v in ((zl - a) / b, (zh - a) / b) if yl < v < yh]})
    moments = np.zeros(shape)
    for lo, hi in zip(breaks, breaks[1:]):
        if a + b * (lo + hi) / 2 >= zh:
            continue
        y = (lo + hi) / 2 + (hi - lo) / 2 * nodes
        bottom = np.maximum(zl, a + b * y)
        yp = y[None, :] ** np.arange(ny)[:, None]
        zi = np.array(
            [(zh ** (k + 1) - bottom ** (k + 1)) / (k + 1) for k in range(nz)]
        )
        moments += (hi - lo) / 2 * ((yp * weights) @ zi.T)
    return moments / moments[0, 0] if moments[0, 0] else moments


def polynomial_means(operator, protocol, coefficients, *, faces=False):
    g, v = operator.geometry, operator.viscosity
    settings = protocol["sloped_wall"]
    a, b = settings["intercept_m"] / protocol["length_m"], settings["slope_dz_dy"]
    shape = tuple(np.max([c.shape for c in coefficients], axis=0))
    edges = [e / protocol["length_m"] for e in g.grid.edges]

    def average(bounds):
        xl, xh = bounds[2]
        xp = (
            xl ** np.arange(shape[0])
            if xl == xh
            else np.array(
                [
                    (xh ** (k + 1) - xl ** (k + 1)) / ((k + 1) * (xh - xl))
                    for k in range(shape[0])
                ]
            )
        )
        yz = _yz_moments(
            bounds[1], bounds[0], a, b, shape[1:], settings["quadrature_order"]
        )
        return np.array(
            [
                np.einsum(
                    "i,jk,ijk->", xp[: c.shape[0]], yz[: c.shape[1], : c.shape[2]], c
                )
                for c in coefficients
            ]
        )

    if not faces:
        result = np.zeros((*g.grid.shape, len(coefficients)))
        for raw in v.cells:
            result[tuple(raw)] = average([e[k : k + 2] for e, k in zip(edges, raw)])
        return result
    result = np.zeros((len(v.boundary), len(coefficients)))
    for axis, ids in enumerate(v.face_ids):
        for raw in np.argwhere(ids >= 0):
            bounds = [e[k : k + 2] for e, k in zip(edges, raw)]
            coordinate = edges[axis][raw[axis]]
            bounds[axis] = [coordinate, coordinate]
            result[ids[tuple(raw)]] = average(bounds)
    # The analytic curl is exactly zero on these walls; avoid polynomial
    # cancellation noise when prescribing an exact stationary boundary trace.
    result[v.boundary] = 0
    return result


def wall_reference(operator, protocol):
    u, linear, advective = polynomial_fields(protocol)
    values = polynomial_means(operator, protocol, u)
    traces = polynomial_means(operator, protocol, u, faces=True)
    normal_flux = curl_face_fluxes(operator, protocol)
    for axis, ids in enumerate(operator.viscosity.face_ids):
        active = ids >= 0
        traces[ids[active], 2 - axis] = (
            normal_flux[axis][active] / operator.geometry.open_areas_m2[axis][active]
        )
    force_linear = polynomial_means(operator, protocol, linear)
    force_advective = polynomial_means(operator, protocol, advective)
    scale = operator.nu * protocol["speed_m_per_s"] / protocol["length_m"]
    pressure = operator.density * operator.projection.mean_zero(
        (operator.geometry.centroids_m / protocol["length_m"])
        @ np.array([1, 2, -3])
        * scale
    )
    return StokesState(values, traces, pressure), force_linear, force_advective


def curl_face_fluxes(operator, protocol):
    """Analytic curl flux by shared edge integrals (Stokes' theorem).

    The potential is zero on the sloped wall; its cut-edge contribution is zero.
    Shared Cartesian edge integrals telescope, avoiding cancellation in separate
    high-degree face quadratures. This constructs the independent reference,
    never corrects a numerical velocity or projects a manufactured forcing.
    """
    g = operator.geometry
    length = protocol["length_m"]
    settings = protocol["sloped_wall"]
    a, b = settings["intercept_m"] / length, settings["slope_dz_dy"]
    edges = [e / length for e in g.grid.edges[::-1]]
    f = np.array([0, 0, 1, -2, 1.0])
    cache = {}

    def edge(axis, vertex):
        key = (axis, *vertex)
        if key in cache:
            return cache[key]
        xyz = np.array([e[i] for e, i in zip(edges, vertex)])
        lo, hi = edges[axis][vertex[axis] : vertex[axis] + 2]
        if axis == 0:
            coefficients = f
            multiplier = (
                poly.polyval(xyz[1], f)
                * poly.polyval(xyz[2], f)
                * max(0, xyz[2] - a - b * xyz[1]) ** 2
            )
        elif axis == 1:
            hi = min(hi, (xyz[2] - a) / b)
            c = xyz[2] - a
            coefficients = poly.polymul(f, [c * c, -2 * b * c, b * b])
            multiplier = poly.polyval(xyz[0], f) * poly.polyval(xyz[2], f)
        else:
            lo = max(lo, a + b * xyz[1])
            c = a + b * xyz[1]
            coefficients = poly.polymul(f, [c * c, -2 * c, 1])
            multiplier = poly.polyval(xyz[0], f) * poly.polyval(xyz[1], f)
        primitive = poly.polyint(coefficients)
        value = (
            multiplier * (poly.polyval(hi, primitive) - poly.polyval(lo, primitive))
            if hi > lo
            else 0.0
        )
        cache[key] = value
        return value

    result = []
    factor = protocol["speed_m_per_s"] * settings["streamfunction_scale"] * length**2
    for axis, area in enumerate(g.open_areas_m2):
        c = 2 - axis
        t1, t2 = (c + 1) % 3, (c + 2) % 3
        field = np.zeros_like(area)
        for raw in np.argwhere(area > 0):
            vertex = list(raw[::-1])
            hi1, hi2 = vertex.copy(), vertex.copy()
            hi1[t1] += 1
            hi2[t2] += 1
            field[tuple(raw)] = factor * (
                edge(t2, hi1) - edge(t2, vertex) - edge(t1, hi2) + edge(t1, vertex)
            )
        result.append(field)
    return tuple(result)


def abc_reference(operator, protocol, t):
    """Exact rectangular cell/face averages, including cross-product pressure."""
    g, v = operator.geometry, operator.viscosity
    length, speed = protocol["length_m"], protocol["speed_m_per_s"]
    k = 2 * np.pi / length
    amplitude = speed * np.exp(-operator.nu * k * k * t)
    position = g.grid.positions()
    widths = np.stack(
        np.broadcast_arrays(
            *[
                w.reshape(tuple(len(w) if j == i else 1 for j in range(3)))
                for i, w in enumerate(g.grid.widths)
            ]
        ),
        axis=-1,
    )[..., ::-1]

    def velocity(p, factor):
        sine = np.sin(k * p) * factor
        cosine = np.cos(k * p) * factor
        return amplitude * np.stack(
            (
                sine[..., 2] + cosine[..., 1],
                sine[..., 0] + cosine[..., 2],
                sine[..., 1] + cosine[..., 0],
            ),
            axis=-1,
        )

    factor = np.sinc(widths / length)
    values = velocity(position, factor)
    sine, cosine = np.sin(k * position) * factor, np.cos(k * position) * factor
    pi = -(amplitude**2) * (
        sine[..., 2] * cosine[..., 1]
        + sine[..., 0] * cosine[..., 2]
        + sine[..., 1] * cosine[..., 0]
    )
    pressure = operator.density * operator.projection.mean_zero(pi)
    traces = np.zeros((len(v.boundary), 3))
    for axis, ids in enumerate(v.face_ids):
        p = g.grid.positions(axis)
        factors = []
        for xyz in range(3):
            metric = 2 - xyz
            shape = [1, 1, 1]
            shape[metric] = (
                len(g.grid.widths[metric])
                if metric != axis
                else g.grid.shape[metric] + 1
            )
            f = (
                np.ones(shape)
                if metric == axis
                else np.sinc(g.grid.widths[metric] / length).reshape(shape)
            )
            factors.append(np.broadcast_to(f, ids.shape))
        result = velocity(p, np.stack(factors, axis=-1))
        active = ids >= 0
        traces[ids[active]] = result[active]
    return StokesState(values, traces, pressure)


def tiny_cycle(protocol):
    """An exact four-face circulation through a retained triangular sliver."""
    settings = protocol["tiny_cycle"]
    n, delta = settings["n"], settings["relative_gap"]
    length = protocol["length_m"]
    h = length / n
    wall = dict(intercept_m=(3 - delta) * h, slope_dz_dy=0.5, focus_width_m=length)
    g = wall_geometry(length, wall, n)
    op = operator_for(g, protocol)
    # (z,y,x): through a tiny triangle, its lower-y neighbor, and the two
    # cells one z-layer above. One flux value closes the entire loop.
    cells = np.array([[3, 2, 2], [3, 1, 2], [4, 1, 2], [4, 2, 2]])
    links = [
        (1, (3, 2, 2), -1),
        (0, (4, 1, 2), 1),
        (1, (4, 2, 2), 1),
        (0, (4, 2, 2), -1),
    ]
    minimum = min(g.open_areas_m2[axis][index] for axis, index, _sign in links)
    flux = protocol["speed_m_per_s"] * minimum
    native = [np.zeros_like(area) for area in g.open_areas_m2[::-1]]
    for axis, index, sign in links:
        native[2 - axis][index] = sign * flux / g.open_areas_m2[axis][index]
    smallest = min(g.volumes_m3[tuple(cell)] for cell in cells)
    dt = settings["target_outgoing_courant"] * smallest / flux
    old = np.zeros((*g.grid.shape, 3))
    old[tuple(cells[0])] = [1, -0.5, 0.25]
    return op, tuple(native), old, dt


def combined_metrics(step, operator, protocol, dt):
    metrics = dict(step.diagnostics)
    speed, length = protocol["speed_m_per_s"], protocol["length_m"]
    energy_scale = max(metrics["advection_old_energy_m5_per_s2"], np.finfo(float).tiny)
    power_scale = max(
        energy_scale / dt, abs(metrics["body_power_m5_per_s3"]), np.finfo(float).tiny
    )
    metrics.update(
        relative_work_error=abs(metrics["combined_work_residual_m5_per_s3"])
        / power_scale,
        relative_momentum_error=max(
            abs(np.array(metrics["combined_momentum_residual_m4_per_s"]))
        )
        / (speed * length**3),
        relative_advection_momentum_error=max(
            abs(np.array(metrics["advection_momentum_change_m4_per_s"]))
        )
        / (speed * length**3),
        relative_volume_flux=metrics["max_integral_flux_m3_per_s"]
        / (speed * length**2),
        relative_pressure_work=abs(metrics["pressure_work_m5_per_s3"]) / power_scale,
        relative_bound_violation=metrics["advection_bound_violation_m_per_s"] / speed,
        relative_energy_increase=metrics["energy_m5_per_s2"] / energy_scale - 1,
        wall_trace_exactly_zero=not np.any(
            step.state.trace_velocity_m_per_s[operator.viscosity.boundary]
        ),
    )
    return metrics


def transport_wave_case(protocol, n):
    length, speed = protocol["length_m"], protocol["speed_m_per_s"]
    g = CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
    op = operator_for(g, protocol)
    transport = ImplicitMomentumTransport(op.projection)
    native = tuple(
        np.full(f.mask.shape, speed if i == 0 else 0)
        for i, f in enumerate(op.projection.faces)
    )
    phase = 2 * np.pi * g.grid.positions()[..., 0] / length
    old = np.stack(
        (
            np.zeros_like(phase),
            speed * np.sinc(1 / n) * np.sin(phase),
            0.5 * speed * np.sinc(1 / n) * np.cos(phase),
        ),
        axis=-1,
    )
    start = old.copy()
    settings = protocol["transport_wave"]
    steps = int(
        np.ceil(
            settings["duration_s"] / (settings["outgoing_courant"] * length / n / speed)
        )
    )
    dt = settings["duration_s"] / steps
    worst = {}
    for _ in range(steps):
        old, metrics = transport.step(old, native, dt)
        energy_scale = metrics["old_energy_m5_per_s2"]
        row = dict(
            relative_work_error=abs(metrics["work_residual_m5_per_s3"])
            / (energy_scale / dt),
            relative_momentum_error=max(
                abs(np.array(metrics["momentum_change_m4_per_s"]))
            )
            / (speed * length**3),
            relative_bound_violation=metrics["bound_violation_m_per_s"] / speed,
            relative_energy_increase=metrics["energy_m5_per_s2"] / energy_scale - 1,
        )
        for key, value in row.items():
            worst[key] = max(worst.get(key, -float("inf")), value)
    exact_phase = phase - 2 * np.pi * speed * settings["duration_s"] / length
    exact = np.stack(
        (
            np.zeros_like(phase),
            speed * np.sinc(1 / n) * np.sin(exact_phase),
            0.5 * speed * np.sinc(1 / n) * np.cos(exact_phase),
        ),
        axis=-1,
    )
    metrics.update(
        worst,
        n=n,
        steps=steps,
        dt_s=dt,
        relative_rms_error=rms_error(old, exact, g.volumes_m3),
    )
    return (
        op,
        None,
        metrics,
        dict(
            velocity_m_per_s=old,
            reference_velocity_m_per_s=exact,
            initial_velocity_m_per_s=start,
        ),
    )


def nonlinear_case(protocol, n, *, focused=False, abc=False):
    settings = protocol["abc"] if abc else protocol["sloped_wall"]
    length = protocol["length_m"]
    g = (
        CutCellGeometry(RectilinearGrid([length] * 3, [n] * 3, [True] * 3))
        if abc
        else momentum_wall_geometry(protocol, n, focused=focused)
    )
    op = operator_for(g, protocol)
    momentum = CutCellMomentum(op)
    if abc:
        state = abc_reference(op, protocol, 0)
        reference = abc_reference(op, protocol, settings["duration_s"])
        force = None
    else:
        state, linear, advection = wall_reference(op, protocol)
        reference = state
        force = linear + advection
    initial = state.cell_velocity_m_per_s.copy()
    steps = int(np.ceil(settings["duration_s"] / settings["dt_s"]))
    dt = settings["duration_s"] / steps
    worst = {}
    for _ in range(steps):
        step = momentum.advance(state, dt, acceleration_m_per_s2=force)
        state = step.state
        metrics = combined_metrics(step, op, protocol, dt)
        for key in (
            "relative_work_error",
            "relative_momentum_error",
            "relative_advection_momentum_error",
            "relative_volume_flux",
            "relative_pressure_work",
            "relative_bound_violation",
            "relative_energy_increase",
            "max_outgoing_courant",
        ):
            worst[key] = max(worst.get(key, -float("inf")), metrics[key])
    metrics.update(
        worst,
        n=n,
        steps=steps,
        dt_s=dt,
        relative_velocity_error=rms_error(
            state.cell_velocity_m_per_s, reference.cell_velocity_m_per_s, g.volumes_m3
        ),
        forced=not abc,
        max_cell_width_m=max(float(w.max()) for w in g.grid.widths),
        max_cell_width_over_background=max(float(w.max()) for w in g.grid.widths)
        / (length / n),
    )
    pscale = op.density * op.nu * protocol["speed_m_per_s"] / length
    if abc:
        metrics["relative_pressure_error"] = float(
            np.sqrt(
                np.sum(g.volumes_m3 * (state.pressure_pa - reference.pressure_pa) ** 2)
                / np.sum(g.volumes_m3 * reference.pressure_pa**2)
            )
        )
    else:
        metrics["pressure_rms_over_scale"] = (
            float(
                np.sqrt(
                    np.sum(
                        g.volumes_m3 * (state.pressure_pa - reference.pressure_pa) ** 2
                    )
                    / g.volumes_m3.sum()
                )
            )
            / pscale
        )
    return (
        op,
        state,
        metrics,
        dict(
            reference_velocity_m_per_s=reference.cell_velocity_m_per_s,
            reference_pressure_pa=reference.pressure_pa,
            initial_velocity_m_per_s=initial,
        ),
    )


def unforced_case(protocol, geometry):
    op = operator_for(geometry, protocol)
    rng = np.random.default_rng(protocol["seed"])
    old = rng.normal(size=(*geometry.grid.shape, 3)) * protocol["speed_m_per_s"]
    old[~geometry.fluid] = 0
    dt = protocol["unforced"]["dt_s"]
    state = op.step(old, dt)
    initial = state.cell_velocity_m_per_s.copy()
    momentum = CutCellMomentum(op)
    worst = {}
    for _ in range(protocol["unforced"]["steps"]):
        step = momentum.advance(state, dt)
        state = step.state
        metrics = combined_metrics(step, op, protocol, dt)
        for key, value in metrics.items():
            if key.startswith("relative_") or key == "max_outgoing_courant":
                worst[key] = max(worst.get(key, -float("inf")), value)
    metrics.update(worst, forced=False)
    return op, state, metrics, dict(initial_velocity_m_per_s=initial)


def tiny_case(protocol):
    op, native, old, dt = tiny_cycle(protocol)
    result, metrics = ImplicitMomentumTransport(op.projection).step(old, native, dt)
    speed = float(abs(old).max())
    old_energy = metrics["old_energy_m5_per_s2"]
    mass_scale = float(np.sum(op.geometry.volumes_m3[..., None] * abs(old)))
    metrics.update(
        dt_s=dt,
        relative_work_error=abs(metrics["work_residual_m5_per_s3"]) / (old_energy / dt),
        relative_momentum_error=max(abs(np.array(metrics["momentum_change_m4_per_s"])))
        / mass_scale,
        relative_bound_violation=metrics["bound_violation_m_per_s"] / speed,
        relative_energy_increase=metrics["energy_m5_per_s2"] / old_energy - 1,
    )
    return (
        op,
        None,
        metrics,
        dict(
            velocity_m_per_s=result,
            initial_velocity_m_per_s=old,
            advecting_x_m_per_s=native[0],
            advecting_y_m_per_s=native[1],
            advecting_z_m_per_s=native[2],
        ),
    )


def tiny_coupled_case(protocol):
    op, native, old, dt = tiny_cycle(protocol)
    v = op.viscosity
    trace = np.zeros((len(v.boundary), 3))
    for values, axis in zip(native, (2, 1, 0)):
        ids = v.face_ids[axis]
        active = ids >= 0
        trace[ids[active], 2 - axis] = values[active]
    initial = StokesState(old, trace, np.zeros(op.geometry.grid.shape))
    step = CutCellMomentum(op).advance(initial, dt)
    return (
        op,
        step.state,
        combined_metrics(step, op, protocol, dt),
        dict(
            initial_velocity_m_per_s=old,
            advected_velocity_m_per_s=step.advected_velocity_m_per_s,
        ),
    )


def temporal_case(protocol, steps):
    settings = protocol["temporal"]
    g = wall_geometry(protocol["length_m"], protocol["sloped_wall"], settings["n"])
    op = operator_for(g, protocol)
    reference, linear, advection = wall_reference(op, protocol)
    momentum = CutCellMomentum(op)
    state = reference
    dt = settings["duration_s"] / steps
    worst = {}
    for index in range(1, steps + 1):
        factor = np.exp(-index * dt / settings["decay_time_s"])
        force = (
            factor
            * (linear - reference.cell_velocity_m_per_s / settings["decay_time_s"])
            + factor**2 * advection
        )
        step = momentum.advance(state, dt, acceleration_m_per_s2=force)
        state = step.state
        metrics = combined_metrics(step, op, protocol, dt)
        for key in (
            "relative_work_error",
            "relative_momentum_error",
            "relative_advection_momentum_error",
            "relative_volume_flux",
            "relative_pressure_work",
            "relative_bound_violation",
        ):
            worst[key] = max(worst.get(key, -float("inf")), metrics[key])
    metrics.update(worst, steps=steps, dt_s=dt, forced=True)
    return op, state, metrics, {}


def _validate(protocol):
    for name in (
        "length_m",
        "kinematic_viscosity_m2_per_s",
        "density_kg_per_m3",
        "speed_m_per_s",
    ):
        finite_number(protocol[name], name, positive=True)
    for name in ("transport_wave", "abc", "sloped_wall"):
        values = protocol[name]["counts"]
        if (
            len(values) < 3
            or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in values
            )
            or any(a >= b for a, b in zip(values, values[1:]))
        ):
            raise ValueError("Declare three increasing integer refinement levels")
    for name, value in protocol["gates"].items():
        finite_number(value, name, positive=True)
    if protocol["sloped_wall"]["quadrature_order"] < 12:
        raise ValueError("Polynomial NS source needs Gaussian order >=12")
    steps = protocol["temporal"]["steps"]
    if (
        len(steps) < 3
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in steps)
        or any(a >= b for a, b in zip(steps, steps[1:]))
        or protocol["temporal"]["reference_steps"] < 4 * steps[-1]
    ):
        raise ValueError("Declare increasing time levels and a separate fine reference")


def run_study(protocol, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh momentum verification directory")
    _validate(protocol)
    names = [
        "src/simulation/" + name + ".py"
        for name in (
            "momentum",
            "stokes",
            "viscosity",
            "pressure",
            "cutcell",
            "crystal",
            "mesh",
            "linear",
            "units",
        )
    ]
    names += [
        "scripts/check_momentum.py",
        "scripts/check_stokes.py",
        "scripts/check_cutcell.py",
        "scripts/check_viscosity.py",
        "configs/momentum_verification.json",
        "tests/test_momentum.py",
    ]
    sources = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    output.mkdir(parents=True)
    manifest = dict(
        protocol=protocol,
        sources=sources,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=importlib.metadata.version("scipy"),
            threads={
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
            },
        ),
    )
    (output / "study.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    for name in names:
        p = output / "sources" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    rows = []

    def record(name, action):
        started = time.perf_counter()
        try:
            op, state, metrics, fields = action()
            g, v = op.geometry, op.viscosity
            if state is not None:
                fields.update(
                    velocity_m_per_s=state.cell_velocity_m_per_s,
                    trace_velocity_m_per_s=state.trace_velocity_m_per_s,
                    pressure_pa=state.pressure_pa,
                )
            fields.update(
                liquid_volumes_m3=g.volumes_m3,
                liquid_centroids_m=g.centroids_m,
                wall_cells=g.wall_cells,
                wall_face_ids=g.wall_face_ids,
                wall_areas_m2=g.wall_areas_m2,
                wall_normals=g.wall_normals,
                wall_centroids_m=g.wall_centroids_m,
                axis_refinement_history=g.refinement_history,
                wall_ids=v.wall_ids,
                boundary_ids=v.boundary_ids,
                trace_centroids_m=v.face_centroids_m,
            )
            for axis, label in enumerate(("z", "y", "x")):
                fields[label + "_edges_m"] = g.grid.edges[axis]
                fields[label + "_open_areas_m2"] = g.open_areas_m2[axis]
                fields[label + "_face_ids"] = v.face_ids[axis]
            if not all(np.isfinite(value).all() for value in fields.values()):
                raise ArithmeticError("Nonfinite momentum evidence")
            np.savez_compressed(output / (name + ".npz"), **fields)
            row = dict(case=name, **metrics, **g.measurements())
        except (ValueError, ArithmeticError, RuntimeError) as error:
            row = dict(case=name, error=str(error))
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        (output / "cases.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row, allow_nan=False), flush=True)

    for n in protocol["transport_wave"]["counts"]:
        record(f"transport_wave_{n}", lambda n=n: transport_wave_case(protocol, n))
    for n in protocol["abc"]["counts"]:
        record(f"abc_{n}", lambda n=n: nonlinear_case(protocol, n, abc=True))
    for focused in (False, True):
        for n in protocol["sloped_wall"]["counts"]:
            record(
                f"{'focused' if focused else 'uniform'}_wall_{n}",
                lambda n=n, focused=focused: nonlinear_case(
                    protocol, n, focused=focused
                ),
            )
    record("tiny_cycle", lambda: tiny_case(protocol))
    record("tiny_coupled", lambda: tiny_coupled_case(protocol))
    for name in ("rotated_cube", "subcell_cube"):
        settings = protocol[name]
        record(
            name,
            lambda settings=settings: unforced_case(
                protocol,
                cube_geometry(protocol["length_m"], settings, settings["counts_xyz"]),
            ),
        )
    record(
        "periodic_wall",
        lambda: unforced_case(
            protocol,
            wall_geometry(
                protocol["length_m"],
                protocol["sloped_wall"],
                6,
                periodic=(True, False, False),
            ),
        ),
    )
    try:
        _op, fine, _metrics, _fields = temporal_case(
            protocol, protocol["temporal"]["reference_steps"]
        )

        def measure_time(steps):
            op, state, metrics, fields = temporal_case(protocol, steps)
            metrics["relative_temporal_velocity_error"] = rms_error(
                state.cell_velocity_m_per_s,
                fine.cell_velocity_m_per_s,
                op.geometry.volumes_m3,
            )
            pscale = (
                op.density * op.nu * protocol["speed_m_per_s"] / protocol["length_m"]
            )
            metrics["temporal_pressure_rms_over_scale"] = (
                float(
                    np.sqrt(
                        np.sum(
                            op.geometry.volumes_m3
                            * (state.pressure_pa - fine.pressure_pa) ** 2
                        )
                        / op.geometry.volumes_m3.sum()
                    )
                )
                / pscale
            )
            fields.update(
                fine_time_velocity_m_per_s=fine.cell_velocity_m_per_s,
                fine_time_pressure_pa=fine.pressure_pa,
            )
            return op, state, metrics, fields

        for steps in protocol["temporal"]["steps"]:
            record(f"temporal_{steps}", lambda steps=steps: measure_time(steps))
    except (ValueError, ArithmeticError, RuntimeError) as error:
        rows.append(dict(case="temporal_reference", error=str(error)))
    limits = protocol["gates"]
    gates = dict(all_cases_completed=all("error" not in row for row in rows))
    for key, limit in (
        ("relative_work_error", "maximum_relative_work_error"),
        ("relative_momentum_error", "maximum_relative_momentum_error"),
        ("relative_bound_violation", "maximum_relative_bound_violation"),
    ):
        gates[key] = all(row.get(key, float("inf")) <= limits[limit] for row in rows)
    coupled = [row for row in rows if "relative_volume_flux" in row]
    gates["volume_and_walls"] = len(coupled) == len(rows) - len(
        protocol["transport_wave"]["counts"]
    ) - 1 and all(
        row["relative_volume_flux"] < limits["maximum_relative_volume_flux"]
        and row["relative_pressure_work"] < limits["maximum_relative_work_error"]
        and row["wall_trace_exactly_zero"]
        for row in coupled
    )
    gates["unforced_energy"] = all(
        row.get("relative_energy_increase", float("inf"))
        <= limits["maximum_relative_energy_increase"]
        for row in rows
        if not row.get("forced", False) and "error" not in row
    )
    for family, metric, tolerance in (
        ("transport_wave", "relative_rms_error", "finest_transport_relative_rms_error"),
        ("abc", "relative_velocity_error", "finest_abc_relative_velocity_error"),
        ("abc", "relative_pressure_error", "finest_abc_relative_pressure_error"),
        (
            "uniform_wall",
            "relative_velocity_error",
            "finest_wall_relative_velocity_error",
        ),
        (
            "uniform_wall",
            "pressure_rms_over_scale",
            "finest_wall_pressure_rms_over_scale",
        ),
        (
            "focused_wall",
            "relative_velocity_error",
            "finest_wall_relative_velocity_error",
        ),
        (
            "focused_wall",
            "pressure_rms_over_scale",
            "finest_wall_pressure_rms_over_scale",
        ),
    ):
        selected = [r for r in rows if r["case"].startswith(family + "_")]
        values = [r.get(metric, float("inf")) for r in selected]
        rates = [
            np.log(a / b) / np.log(r2["n"] / r1["n"])
            if "n" in r1 and "n" in r2 and np.isfinite(a) and np.isfinite(b) and b > 0
            else -float("inf")
            for a, b, r1, r2 in zip(values, values[1:], selected, selected[1:])
        ]
        gates[family + "_" + metric] = bool(
            values
            and values[-1] < limits[tolerance]
            and all(rate >= limits["minimum_spatial_rate"] for rate in rates)
        )
    selected = [r for r in rows if r["case"].startswith("temporal_") and "steps" in r]
    for metric in (
        "relative_temporal_velocity_error",
        "temporal_pressure_rms_over_scale",
    ):
        rates = [
            np.log(a[metric] / b[metric]) / np.log(b["steps"] / a["steps"])
            for a, b in zip(selected, selected[1:])
        ]
        gates[metric] = bool(
            len(selected) == len(protocol["temporal"]["steps"])
            and all(rate >= limits["minimum_temporal_rate"] for rate in rates)
        )
    tiny = next(r for r in rows if r["case"] == "tiny_cycle")
    gates["tiny_capacity_large_courant"] = (
        tiny.get("max_outgoing_courant", 0)
        >= 0.999 * protocol["tiny_cycle"]["target_outgoing_courant"]
        and 0 < tiny.get("min_liquid_fraction", 1) < 1e-8
    )
    for name, digest in sources.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during momentum verification")
    report = dict(
        protocol=protocol, results=rows, gates=gates, overall_pass=all(gates.values())
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_report(report, output)
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for family in ("abc", "uniform_wall", "focused_wall"):
        rows = [
            r
            for r in report["results"]
            if r["case"].startswith(family + "_") and "n" in r
        ]
        for axis, metric in zip(
            axes[:2],
            (
                "relative_velocity_error",
                "relative_pressure_error"
                if family == "abc"
                else "pressure_rms_over_scale",
            ),
        ):
            if rows:
                axis.loglog(
                    [r["n"] for r in rows],
                    [r[metric] for r in rows],
                    "o-",
                    label=family,
                )
    rows = [
        r
        for r in report["results"]
        if r["case"].startswith("temporal_") and "steps" in r
    ]
    for metric in (
        "relative_temporal_velocity_error",
        "temporal_pressure_rms_over_scale",
    ):
        if rows:
            axes[2].loglog(
                [r["dt_s"] for r in rows], [r[metric] for r in rows], "o-", label=metric
            )
    for axis in axes[:2]:
        counts = sorted(
            set(
                report["protocol"]["abc"]["counts"]
                + report["protocol"]["sloped_wall"]["counts"]
            )
        )
        axis.set_xticks(counts, [str(n) for n in counts])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Nominal background count")
    if rows:
        dt = [r["dt_s"] for r in rows]
        axes[2].set_xticks(dt, [f"{v:g}" for v in dt])
        axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].set_xlabel("Time step (s)")
    for axis, label in zip(
        axes,
        (
            "Velocity relative RMS error",
            "Pressure RMS / reference or prescribed scale",
            "Time error / reference or prescribed scale",
        ),
    ):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(True, which="both", alpha=0.25)
    fig.savefig(output / "refinement.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/momentum_verification.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study(json.loads(args.config.read_text()), args.output)
    print("PASS" if report["overall_pass"] else "FAIL")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
