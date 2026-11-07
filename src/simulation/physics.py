# HOW THIS FILE WORKS
# Scalar fields use [z, y, x]. The full-box diffuse() helper uses stable explicit
# steps; the growth engine uses diffuse_liquid(), an implicit finite-volume solve
# weighted by actual liquid capacities and unobstructed shared-face areas.
# Liquid-only diffusion conserves sum(field * liquid_volume), not the raw field
# sum, and cannot cross solid, air or the outside walls. No field clipping hides
# instability. displace_liquid() conservatively carries solute/heat when liquid
# capacities change, using a quasistatic volume potential rather than Navier-Stokes.
# Solubility/supersaturation supply illustrative growth conditions on the same grid.

"""3D diffusion, liquid-boundary transport and illustrative solution physics."""

import math
from numbers import Real

import numpy as np


def _finite_scalar(value, name, minimum=0.0, strict=False):
    """Validate scalar coefficients, time intervals and cell spacings."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < minimum
        or (strict and value == minimum)
    ):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return float(value)


def diffuse(field, coefficient, dt, dx, dy, dz):
    """Diffuse for dt, returning a floating array with the same [nz, ny, nx] shape.

    On a uniform grid, the explicit stability factor is
    D * substep * (1/dx**2 + 1/dy**2 + 1/dz**2). Keep it <= 0.45,
    below the 0.5 bound, so each update preserves the input value range.
    Zero dt or zero diffusivity returns an independent unchanged copy.
    Units must be consistent: D has length**2/time and spacings have length.
    """
    coefficient = _finite_scalar(coefficient, "coefficient")
    dt = _finite_scalar(dt, "dt")
    dx = _finite_scalar(dx, "dx", strict=True)
    dy = _finite_scalar(dy, "dy", strict=True)
    dz = _finite_scalar(dz, "dz", strict=True)
    if np.iscomplexobj(field):
        raise ValueError("field must contain real values")
    values = np.array(field, dtype=float, copy=True)
    if values.ndim != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("field must be a nonempty finite 3D array in [z, y, x] order")
    if coefficient == 0 or dt == 0:
        return values

    # Smaller cells impose a stricter timestep, even if only one axis is refined.
    # Include every axis in the stability limit, including vertical z transport.
    factor = coefficient * dt * (1 / dx**2 + 1 / dy**2 + 1 / dz**2)
    if not math.isfinite(factor):
        raise ValueError("Diffusion factor is too large for finite arithmetic")
    count = max(1, math.ceil(factor / 0.45))
    h = dt / count
    weights = (
        coefficient * h / dz**2,
        coefficient * h / dy**2,
        coefficient * h / dx**2,
    )
    for _ in range(count):
        updated = values.copy()
        # Axis 0 is z, axis 1 is y, axis 2 is x. Read all fluxes from the OLD
        # field so this is one simultaneous explicit step, not an axis-order sweep.
        for axis, weight in enumerate(weights):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            lower, upper = tuple(lower), tuple(upper)
            flux = weight * (values[upper] - values[lower])
            # Shared internal faces cancel in the total; outside faces are omitted.
            updated[lower] += flux
            updated[upper] -= flux
        values = updated
    return values


def solubility(temperature, reference=1.0, slope=0.02, reference_temperature=25.0):
    """Illustrative equilibrium concentration; parameters require calibration."""
    # A positive floor prevents division by zero in supersaturation.
    return np.maximum(1e-9, reference + slope * (temperature - reference_temperature))


def supersaturation(
    concentration, temperature, reference=1.0, slope=0.02, reference_temperature=25.0
):
    """Relative excess concentration at each [z, y, x] cell."""
    equilibrium = solubility(temperature, reference, slope, reference_temperature)
    return concentration / equilibrium - 1.0


# CUT-CELL TRANSPORT
# Capacities below are actual liquid volumes; openings are unobstructed areas
# between neighboring cells, ordered z/y/x. Dry cells are excluded from every
# solve. Sparse graph fluxes conserve scalar amounts, including partial cells.
def _liquid_graph(volumes, openings, spacing):
    from scipy.sparse import coo_matrix

    volumes = np.asarray(volumes, dtype=float)
    if (
        volumes.ndim != 3
        or not volumes.size
        or not np.isfinite(volumes).all()
        or np.any(volumes < 0.0)
    ):
        raise ValueError("Liquid capacities must be a finite nonnegative 3D matrix")
    if len(openings) != 3 or len(spacing) != 3:
        raise ValueError("Liquid openings and spacings must follow z/y/x order")
    active = np.flatnonzero(volumes.ravel() > 0.0)
    lookup = np.full(volumes.size, -1, dtype=int)
    lookup[active] = np.arange(len(active))
    indices = np.arange(volumes.size).reshape(volumes.shape)
    first, second, conductances = [], [], []
    records = []
    for axis, (area, distance) in enumerate(zip(openings, spacing)):
        distance = _finite_scalar(distance, "spacing", strict=True)
        expected = list(volumes.shape)
        expected[axis] -= 1
        if (
            np.shape(area) != tuple(expected)
            or not np.isfinite(area).all()
            or np.any(np.asarray(area) < 0.0)
        ):
            raise ValueError(
                "Open areas must be finite, nonnegative and match adjacent grid faces"
            )
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        a, b = (
            lookup[indices[tuple(lower)]].ravel(),
            lookup[indices[tuple(upper)]].ravel(),
        )
        g = np.asarray(area).ravel() / distance
        valid = (a >= 0) & (b >= 0) & (g > 0.0)
        first.append(a[valid])
        second.append(b[valid])
        conductances.append(g[valid])
        records.append(valid)
    i, j, g = (
        np.concatenate(first),
        np.concatenate(second),
        np.concatenate(conductances),
    )
    laplacian = coo_matrix(
        (np.r_[g, g, -g, -g], (np.r_[i, j, i, j], np.r_[i, j, j, i])),
        shape=(len(active), len(active)),
    ).tocsr()
    return active, i, j, g, laplacian, records


def diffuse_liquid(field, volumes, openings, coefficient, dt, dx, dy, dz):
    """Implicit finite-volume diffusion through liquid apertures only.

    Solve (diag(V) + D*dt*L) c_new = V*c_old. Backward Euler handles tiny
    cut cells without unstable explicit steps. No flux crosses solid, air or walls.
    This two-point flux uses Cartesian center spacing; it is not a high-order
    reconstruction at an oblique interface.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    coefficient = _finite_scalar(coefficient, "coefficient")
    dt = _finite_scalar(dt, "dt")
    values, volumes = np.asarray(field, float), np.asarray(volumes, float)
    if values.shape != volumes.shape or not np.isfinite(values).all():
        raise ValueError("field must be finite and match liquid cell volumes")
    active, _i, _j, _g, laplacian, _records = _liquid_graph(
        volumes, openings, (dz, dy, dx)
    )
    result = np.zeros(values.size)
    if len(active):
        capacity = volumes.ravel()[active]
        if coefficient == 0 or dt == 0:
            result[active] = values.ravel()[active]
        else:
            system = diags(capacity) + coefficient * dt * laplacian
            result[active] = spsolve(system.tocsc(), capacity * values.ravel()[active])
    return result.reshape(values.shape)


def displace_liquid(
    old_volumes,
    new_volumes,
    old_openings,
    new_openings,
    concentration,
    temperature,
    dx,
    dy,
    dz,
):
    """Conserve carrier, solute and sensible-heat proxy as the solid boundary moves.

    A graph potential routes volume from closing cells to opening cells. Its
    divergence satisfies new_volume-old_volume + net_outflow = 0. This is a
    quasistatic displacement field, not Navier-Stokes pressure or stirring flow.
    Upwind transport follows decreasing potential, so all inflows reach a cell
    before its outflows. Newly opened cells receive actual donor solute and heat.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    old, new = np.asarray(old_volumes, float), np.asarray(new_volumes, float)
    if (
        old.ndim != 3
        or old.shape != new.shape
        or not np.isfinite(old).all()
        or not np.isfinite(new).all()
        or np.any(old < 0.0)
        or np.any(new < 0.0)
    ):
        raise ValueError(
            "Old/new liquid volumes must be finite nonnegative 3D matrices of equal shape"
        )
    concentration, temperature = (
        np.asarray(concentration, float),
        np.asarray(temperature, float),
    )
    if (
        concentration.shape != old.shape
        or temperature.shape != old.shape
        or not np.isfinite(concentration).all()
        or not np.isfinite(temperature).all()
    ):
        raise ValueError("Displaced fields must be finite and match the liquid grid")
    total = float(old.sum())
    tolerance = 2e-10 * max(1.0, total)
    if old.shape != new.shape or abs(float(new.sum()) - total) > tolerance:
        raise ValueError("Displacement must preserve the incompressible carrier volume")
    openings = tuple(np.maximum(a, b) for a, b in zip(old_openings, new_openings))
    active, i, j, g, laplacian, records = _liquid_graph(
        np.maximum(old, new), openings, (dz, dy, dx)
    )
    source = (old - new).ravel()[active]
    potential = np.zeros(len(active))
    count, labels = connected_components(laplacian, directed=False)
    for component in range(count):
        nodes = np.flatnonzero(labels == component)
        rhs = source[nodes].copy()
        if abs(float(rhs.sum())) > tolerance:
            raise ValueError("Moving solid would compress trapped, disconnected liquid")
        # Remove roundoff in the geometric volume sum; one potential is a gauge.
        rhs[-1] -= rhs.sum()
        if len(nodes) > 1:
            matrix = laplacian[nodes][:, nodes].tocsc()
            potential[nodes[:-1]] = spsolve(matrix[:-1, :-1], rhs[:-1])
    signed_flux = g * (potential[i] - potential[j])
    donor = np.where(signed_flux >= 0.0, i, j)
    receiver = np.where(signed_flux >= 0.0, j, i)
    amount = abs(signed_flux)
    # Group outflow edges once. Sorting nodes by potential gives a directed
    # acyclic transport order and avoids a CFL restriction on boundary remapping.
    order = np.argsort(donor, kind="stable")
    starts = np.searchsorted(donor[order], np.arange(len(active) + 1))
    carrier = old.ravel()[active].copy()
    masses = concentration.ravel()[active] * carrier
    heat = temperature.ravel()[active] * carrier
    targets = new.ravel()[active]
    for node in np.argsort(-potential, kind="stable"):
        edges = order[starts[node] : starts[node + 1]]
        outflow = float(amount[edges].sum())
        if outflow <= 0.0:
            continue
        if outflow > carrier[node] + tolerance:
            raise ValueError("Displacement outflow exceeds donor liquid capacity")
        # A closing cell transfers its entire scalar content, eliminating only
        # solver roundoff rather than leaving solute inside newly solid material.
        if targets[node] == 0.0:
            weights = amount[edges] / outflow
            mass_out, heat_out = weights * masses[node], weights * heat[node]
        else:
            inverse = 1.0 / max(carrier[node], np.finfo(float).tiny)
            mass_out, heat_out = (
                amount[edges] * masses[node] * inverse,
                amount[edges] * heat[node] * inverse,
            )
        np.add.at(carrier, receiver[edges], amount[edges])
        np.add.at(masses, receiver[edges], mass_out)
        np.add.at(heat, receiver[edges], heat_out)
        carrier[node] -= outflow
        masses[node] -= mass_out.sum()
        heat[node] -= heat_out.sum()
    continuity = laplacian @ potential - source
    result_c, result_t = np.zeros(old.size), np.zeros(old.size)
    wet = targets > 0.0
    result_c[active[wet]], result_t[active[wet]] = (
        masses[wet] / targets[wet],
        heat[wet] / targets[wet],
    )
    # Reconstruct the signed integrated volume flux at every Cartesian grid face.
    face_fluxes, offset = [], 0
    for area, valid in zip(openings, records):
        flat = np.zeros(np.asarray(area).size)
        n = int(valid.sum())
        flat[valid] = signed_flux[offset : offset + n]
        face_fluxes.append(flat.reshape(np.asarray(area).shape))
        offset += n
    return (
        result_c.reshape(old.shape),
        result_t.reshape(old.shape),
        tuple(face_fluxes),
        float(np.max(abs(continuity), initial=0.0)),
    )
