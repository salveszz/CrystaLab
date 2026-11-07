# HOW THIS FILE WORKS
# Scalar fields use [z, y, x]. The full-box diffuse() helper uses stable explicit
# steps; the growth engine uses diffuse_liquid(), an implicit finite-volume solve
# weighted by actual liquid capacities and unobstructed shared-face areas.
# Liquid-only diffusion conserves sum(field * liquid_volume), not the raw field
# sum, and cannot cross solid, air or the outside walls. No field clipping hides
# instability. displace_liquid() conservatively carries solute/heat when liquid
# capacities change, using a quasistatic volume potential rather than Navier-Stokes.
# Solubility/supersaturation supply illustrative growth conditions on the same grid.
# Local least-squares gradients reconstruct sampling values from liquid averages;
# diffusion uses those gradients to correct liquid-centroid offsets. Opposite
# edge corrections conserve mass; bounded RHS transfers protect the input range.
# Displacement uses aperture/cell moments and bounded MUSCL midpoint scalar
# transfers; the donor remap supplies its conservative low-order base.

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


def _gradient_stencil(volumes, centroids, openings, targets=None):
    """Geometry-only least-squares coefficients shared by sampling and diffusion."""
    volumes = np.asarray(volumes, dtype=float)
    values = volumes
    centers = np.asarray(centroids, dtype=float)
    if (
        values.ndim != 3
        or centers.shape != (*values.shape, 3)
        or not np.isfinite(values).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(volumes).all()
        or np.any(volumes < 0)
        or len(openings) != 3
    ):
        raise ValueError("Gradient fields, liquid volumes and centroids must match")
    wet = volumes > 0
    selected = (
        wet
        if targets is None
        else wet & np.asarray(targets, dtype=bool).reshape(values.shape)
    )
    cells = np.argwhere(selected)
    matrix = np.zeros((len(cells), 3, 3))
    records = []
    for axis, area in enumerate(openings):
        expected = list(values.shape)
        expected[axis] -= 1
        if (
            np.shape(area) != tuple(expected)
            or not np.isfinite(area).all()
            or np.any(area < 0)
        ):
            raise ValueError("Gradient openings must match adjacent cell faces")
        for direction in (-1, 1):
            other = cells.copy()
            other[:, axis] += direction
            valid = (other[:, axis] >= 0) & (other[:, axis] < values.shape[axis])
            rows = np.flatnonzero(valid)
            neighbor = tuple(other[rows].T)
            rows = rows[wet[neighbor]]
            if not len(rows):
                continue
            current, neighbor = tuple(cells[rows].T), tuple(other[rows].T)
            faces = cells[rows].copy()
            if direction < 0:
                faces[:, axis] -= 1
            aperture = area[tuple(faces.T)]
            delta = centers[neighbor] - centers[current]
            squared = np.einsum("ij,ij->i", delta, delta)
            weight = np.divide(
                aperture, squared, out=np.zeros_like(aperture), where=squared > 0
            )
            matrix[rows] += (
                weight[:, None, None] * delta[:, :, None] * delta[:, None, :]
            )
            records.append(
                (
                    rows,
                    np.ravel_multi_index(neighbor, values.shape),
                    weight[:, None] * delta,
                )
            )
    if not records:
        return np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 3))
    inverse = np.linalg.pinv(matrix, rcond=1e-12, hermitian=True)
    rows = np.concatenate([record[0] for record in records])
    neighbors = np.concatenate([record[1] for record in records])
    weighted = np.concatenate([record[2] for record in records])
    coefficients = np.einsum("nij,nj->ni", inverse[rows], weighted)
    current = np.ravel_multi_index(tuple(cells[rows].T), values.shape)
    return current, neighbors, coefficients


def liquid_gradient(field, volumes, centroids, openings, targets=None):
    """Fit gradients from liquid-cell averages through open-face neighbors.

    A pseudoinverse recovers observable components in a deficient stencil, with
    zero gradient for isolated cells. No analytic gradient enters the runtime.
    """
    values = np.asarray(field, dtype=float)
    if values.shape != np.shape(volumes) or not np.isfinite(values).all():
        raise ValueError("Gradient field must match finite liquid volumes")
    current, neighbors, coefficients = _gradient_stencil(
        volumes, centroids, openings, targets
    )
    result = np.zeros((values.size, 3))
    np.add.at(
        result,
        current,
        coefficients * (values.ravel()[neighbors] - values.ravel()[current])[:, None],
    )
    return result.reshape((*values.shape, 3))


def _reconstruction_extrema(field, gradient, centroids, spacing, level, bounds=None):
    """Bounds of a linear reconstruction on the wet Cartesian bounding box."""
    if bounds is None:
        z, y, x = np.indices(np.shape(field))
        lower = np.stack((x, y, z), axis=-1) * np.asarray(spacing)
        upper = lower + spacing
        upper[..., 2] = np.minimum(upper[..., 2], level)
    else:
        lower, upper = bounds
    low_delta = np.where(gradient >= 0, lower - centroids, upper - centroids)
    high_delta = np.where(gradient >= 0, upper - centroids, lower - centroids)
    return field + np.sum(gradient * low_delta, axis=-1), field + np.sum(
        gradient * high_delta, axis=-1
    )


def nonnegative_gradient(field, gradient, centroids, spacing, level, bounds=None):
    """Scale slopes, preserving stored means, so every reconstructed value is >=0."""
    minimum, _maximum = _reconstruction_extrema(
        field, gradient, centroids, spacing, level, bounds=bounds
    )
    change = minimum - field
    factor = np.ones(np.shape(field))
    np.divide(
        np.maximum(field, 0.0), -change, out=factor, where=(minimum < 0) & (change < 0)
    )
    return gradient * np.minimum(factor, 1.0)[..., None]


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
        distance = np.asarray(distance, dtype=float)
        if not np.isfinite(distance).all() or np.any(distance <= 0):
            raise ValueError("Liquid face distances must be finite and positive")
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
        g = (np.asarray(area) / np.broadcast_to(distance, np.shape(area))).ravel()
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


def _solve_spd(system, rhs, guess=None):
    """Solve the same symmetric positive-definite equations without dense LU fill.

    Jacobi-preconditioned conjugate gradients is much cheaper on refined 3D
    grids. Require a tight true residual and scalar-sum residual; otherwise use
    the original sparse direct solve. Small systems use the direct path too.
    This changes linear algebra only, not fluxes, boundary conditions or kinetics.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg, spsolve

    matrix = system.tocsr()
    if matrix.shape[0] < 256:
        return spsolve(matrix.tocsc(), rhs)
    diagonal = matrix.diagonal()
    if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
        raise ValueError("Transport system must have positive finite diagonal entries")
    solution, status = cg(
        matrix,
        rhs,
        x0=guess,
        M=diags(1.0 / diagonal),
        rtol=1e-13,
        atol=0.0,
        maxiter=3000,
    )
    residual = matrix @ solution - rhs
    scale = max(float(np.linalg.norm(rhs)), 1e-30)
    sum_scale = max(abs(float(np.sum(rhs))), scale)
    if (
        status != 0
        or not np.isfinite(solution).all()
        or np.linalg.norm(residual) > 1e-12 * scale
        or abs(float(residual.sum())) > 1e-12 * sum_scale
    ):
        return spsolve(matrix.tocsc(), rhs)
    return solution


def _centroid_flux_correction(
    volumes, openings, spacing, centroids, active, i, j, g, nominal_positions=None
):
    """Sparse edge-flux correction for the offset of actual liquid centroids.

    F_ij = g_ij * (c_i-c_j) - g_ij * grad_face dot (shift_i-shift_j).
    Least-squares gradients are linear in the unknown cell means. Construct that
    operator once per solve, retaining opposite fluxes on each shared open face.
    """
    from scipy.sparse import coo_matrix, csr_matrix

    centers = np.asarray(centroids, dtype=float)
    if centers.shape != (*volumes.shape, 3) or not np.isfinite(centers).all():
        raise ValueError("Diffusion centroids must match liquid volumes")
    coordinates = np.array(np.unravel_index(active, volumes.shape)).T[:, ::-1]
    nominal = (
        (coordinates + 0.5) * np.asarray(spacing)[::-1]
        if nominal_positions is None
        else np.asarray(nominal_positions).reshape(-1, 3)[active]
    )
    shifts = centers.reshape(-1, 3)[active] - nominal
    difference = shifts[i] - shifts[j]
    edges = np.flatnonzero(np.linalg.norm(difference, axis=1) > 1e-12 * min(spacing))
    n = len(active)
    if not len(edges):
        return csr_matrix((n, n)), csr_matrix((0, n)), i[edges], j[edges]
    targets = np.zeros(volumes.size, dtype=bool)
    targets[active[np.r_[i[edges], j[edges]]]] = True
    current, neighbors, coefficients = _gradient_stencil(
        volumes, centers, openings, targets
    )
    lookup = np.full(volumes.size, -1, dtype=int)
    lookup[active] = np.arange(n)
    rows = (lookup[current, None] * 3 + np.arange(3)).ravel()
    gradient = coo_matrix(
        (
            np.r_[coefficients.ravel(), -coefficients.ravel()],
            (
                np.r_[rows, rows],
                np.r_[np.repeat(lookup[neighbors], 3), np.repeat(lookup[current], 3)],
            ),
        ),
        shape=(3 * n, n),
    ).tocsr()
    first, second = i[edges], j[edges]
    columns = (np.stack((first, second), axis=1)[..., None] * 3 + np.arange(3)).ravel()
    weights = np.broadcast_to(
        -0.5 * g[edges, None, None] * difference[edges, None, :], (len(edges), 2, 3)
    ).ravel()
    average = coo_matrix(
        (weights, (np.repeat(np.arange(len(edges)), 6), columns)),
        shape=(len(edges), 3 * n),
    ).tocsr()
    correction = average @ gradient
    incidence = coo_matrix(
        (
            np.r_[np.ones(len(edges)), -np.ones(len(edges))],
            (np.r_[first, second], np.tile(np.arange(len(edges)), 2)),
        ),
        shape=(n, len(edges)),
    ).tocsr()
    return incidence @ correction, correction, first, second


def _solve_corrected(system, rhs, guess, diagnostics):
    """Residual-checked nonsymmetric solve for geometry-corrected diffusion."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import gmres, spsolve

    diagonal = np.maximum(abs(system.diagonal()), 1e-30)
    answer, status = gmres(
        system,
        rhs,
        x0=guess,
        M=diags(1.0 / diagonal),
        rtol=1e-13,
        atol=0,
        restart=40,
        maxiter=100,
    )
    residual = system @ answer - rhs
    scale = max(float(np.linalg.norm(rhs)), 1e-30)
    if (
        status != 0
        or not np.isfinite(answer).all()
        or np.linalg.norm(residual) > 1e-12 * scale
        or abs(float(residual.sum())) > 1e-13 * max(abs(float(rhs.sum())), scale)
    ):
        diagnostics["direct_fallbacks"] = diagnostics.get("direct_fallbacks", 0) + 1
        answer = spsolve(system.tocsc(), rhs)
        residual = system @ answer - rhs
        if (
            not np.isfinite(answer).all()
            or np.linalg.norm(residual) > 5e-12 * scale
            or abs(float(residual.sum())) > 1e-12 * max(abs(float(rhs.sum())), scale)
        ):
            raise ArithmeticError(
                "Corrected diffusion failed its linear residual check"
            )
    return answer


def _bounded_diffusion_correction(
    system, rhs, old, capacity, correction, first, second, scale, diagnostics
):
    """Limit a predicted correction by cell mass budgets, then solve the M-matrix.

    Limiting opposite edge transfers conserves mass. The corrected RHS stays
    between capacity*min(old) and capacity*max(old); the base implicit diffusion
    matrix then preserves those bounds without clipping the final field.
    """
    amount = scale * correction
    donor = np.where(amount >= 0, first, second)
    receiver = np.where(amount >= 0, second, first)
    magnitude = abs(amount)
    outflow = np.bincount(donor, weights=magnitude, minlength=len(old))
    inflow = np.bincount(receiver, weights=magnitude, minlength=len(old))
    outgoing = np.divide(
        capacity * (old - old.min()), outflow, out=np.ones_like(old), where=outflow > 0
    )
    incoming = np.divide(
        capacity * (old.max() - old), inflow, out=np.ones_like(old), where=inflow > 0
    )
    fraction = np.minimum(1.0, np.minimum(outgoing[donor], incoming[receiver]))
    limited = amount * fraction
    corrected_rhs = rhs.copy()
    np.add.at(corrected_rhs, first, -limited)
    np.add.at(corrected_rhs, second, limited)
    diagnostics["bounded_solves"] = diagnostics.get("bounded_solves", 0) + 1
    diagnostics["limited_edges"] = diagnostics.get("limited_edges", 0) + int(
        np.count_nonzero(fraction < 1 - 1e-12)
    )
    return _solve_spd(system, corrected_rhs, guess=old)


def diffuse_liquid(
    field,
    volumes,
    openings,
    coefficient,
    dt,
    dx,
    dy,
    dz,
    centroids=None,
    diagnostics=None,
    distances=None,
    nominal_positions=None,
):
    """Implicit finite-volume diffusion through liquid apertures only.

    Solve (diag(V) + D*dt*L) c_new = V*c_old. Backward Euler handles tiny
    cut cells without unstable explicit steps. No flux crosses solid, air or walls.
    With actual liquid centroids, add a least-squares correction to the two-point
    flux and solve it implicitly. A conservative bounded correction handles any
    candidate outside the input range; fields are never clipped after solving.
    """
    from scipy.sparse import diags

    coefficient = _finite_scalar(coefficient, "coefficient")
    dt = _finite_scalar(dt, "dt")
    values, volumes = np.asarray(field, float), np.asarray(volumes, float)
    if values.shape != volumes.shape or not np.isfinite(values).all():
        raise ValueError("field must be finite and match liquid cell volumes")
    active, _i, _j, _g, laplacian, _records = _liquid_graph(
        volumes, openings, (dz, dy, dx) if distances is None else distances
    )
    result = np.zeros(values.size)
    diagnostics = {} if diagnostics is None else diagnostics
    if len(active):
        capacity = volumes.ravel()[active]
        old = values.ravel()[active]
        if coefficient == 0 or dt == 0 or np.ptp(old) == 0:
            result[active] = values.ravel()[active]
        else:
            system = diags(capacity) + coefficient * dt * laplacian
            rhs = capacity * old
            diagnostics["solves"] = diagnostics.get("solves", 0) + 1
            if centroids is None:
                result[active] = _solve_spd(system, rhs, guess=old)
            else:
                extra, edge_correction, first, second = _centroid_flux_correction(
                    np.asarray(volumes),
                    openings,
                    (dz, dy, dx),
                    centroids,
                    active,
                    _i,
                    _j,
                    _g,
                    nominal_positions=nominal_positions,
                )
                if not extra.nnz:
                    result[active] = _solve_spd(system, rhs, guess=old)
                else:
                    answer = _solve_corrected(
                        system + coefficient * dt * extra, rhs, old, diagnostics
                    )
                    tolerance = 1e-12 * max(1.0, float(abs(old).max()))
                    if (
                        answer.min() < old.min() - tolerance
                        or answer.max() > old.max() + tolerance
                    ):
                        answer = _bounded_diffusion_correction(
                            system,
                            rhs,
                            old,
                            capacity,
                            edge_correction @ answer,
                            first,
                            second,
                            coefficient * dt,
                            diagnostics,
                        )
                    result[active] = answer
    return result.reshape(values.shape)


def _remap_prediction(
    old, new, fields, geometry, active, i, j, signed_flux, records, spacing
):
    """MUSCL midpoint face values with a first-moment geometric transport velocity.

    Integrated material displacement follows the first-moment balance:
    V_avg * displacement = moment_new - moment_old + div(q * aperture_center).
    This includes moving solid/free-surface boundaries, rather than assigning
    zero normal velocity where a Cartesian neighbor happens to be dry.
    """
    before, after = geometry
    face_centers = []
    for old_area, new_area, old_center, new_center, valid in zip(
        before.openings,
        after.openings,
        before.opening_centroids,
        after.opening_centroids,
        records,
    ):
        total = old_area + new_area
        center = np.divide(
            old_area[..., None] * old_center + new_area[..., None] * new_center,
            total[..., None],
            out=np.zeros_like(old_center),
            where=total[..., None] > 0,
        )
        face_centers.append(center.reshape(-1, 3)[valid])
    face_centers = np.concatenate(face_centers)
    old_centers = before.centroids.reshape(-1, 3)[active]
    new_centers = after.centroids.reshape(-1, 3)[active]
    movement = (
        new.ravel()[active, None] * new_centers
        - old.ravel()[active, None] * old_centers
    )
    moment_flux = signed_flux[:, None] * face_centers
    np.add.at(movement, i, moment_flux)
    np.add.at(movement, j, -moment_flux)
    movement /= 0.5 * (old + new).ravel()[active, None]
    donor = np.where(signed_flux >= 0, i, j)
    targets = np.zeros(old.size, dtype=bool)
    targets[active[donor]] = True
    current, neighbors, coefficients = _gradient_stencil(
        old, before.centroids, before.openings, targets
    )
    data = np.stack(fields, axis=-1).reshape(-1, 2)
    gradients = np.zeros((old.size, 2, 3))
    np.add.at(
        gradients,
        current,
        (data[neighbors] - data[current])[:, :, None] * coefficients[:, None, :],
    )
    gradients[:, 0] = nonnegative_gradient(
        fields[0],
        gradients[:, 0].reshape((*old.shape, 3)),
        before.centroids,
        spacing,
        before.level,
        bounds=before.bounds,
    ).reshape(-1, 3)
    delta = face_centers - old_centers[donor] - 0.5 * movement[donor]
    predicted = data[active[donor]] + np.einsum(
        "nfc,nc->nf", gradients[active[donor]], delta
    )
    wet = old > 0
    bounds = []
    for index, field in enumerate(fields):
        low, high = _reconstruction_extrema(
            field,
            gradients[:, index].reshape((*old.shape, 3)),
            before.centroids,
            spacing,
            before.level,
            bounds=before.bounds,
        )
        bounds.append((float(low[wet].min()), float(high[wet].max())))
    return predicted, old.ravel()[active[donor]] > 0, np.asarray(bounds)


def _limit_remap_outflow(
    low, predicted, valid, amount, contents, target, bounds, diagnostics
):
    """Bound shared outgoing fluxes and retained content, never clip the fields."""
    correction = np.where(valid[:, None], amount[:, None] * predicted - low, 0.0)
    if target == 0:
        # A closing cell must donate all content. Redistribute only a zero-sum
        # correction among its outlets; no scalar may remain inside the solid.
        correction -= amount[:, None] / amount.sum() * correction.sum(axis=0)
    result = low.copy()
    limited = False
    for scalar in range(2):
        change = correction[:, scalar]
        theta = 1.0
        positive = change > 0
        negative = change < 0
        lower, upper = bounds[scalar]
        if positive.any():
            theta = min(
                theta,
                float(
                    np.min(
                        (amount[positive] * upper - low[positive, scalar])
                        / change[positive]
                    )
                ),
            )
        if negative.any():
            theta = min(
                theta,
                float(
                    np.min(
                        (low[negative, scalar] - amount[negative] * lower)
                        / -change[negative]
                    )
                ),
            )
        total_change = float(change.sum())
        if target > 0:
            remaining = contents[scalar] - float(low[:, scalar].sum())
            if total_change > 0:
                theta = min(theta, (remaining - target * lower) / total_change)
            elif total_change < 0:
                theta = min(theta, (target * upper - remaining) / -total_change)
        theta = max(0.0, min(1.0, theta))
        limited |= theta < 1 - 1e-12
        result[:, scalar] += theta * change
    diagnostics["corrected_nodes"] = diagnostics.get("corrected_nodes", 0) + 1
    if limited:
        diagnostics["limited_nodes"] = diagnostics.get("limited_nodes", 0) + 1
    return result


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
    geometry=None,
    diagnostics=None,
    distances=None,
):
    """Conserve carrier, solute and sensible-heat proxy as the solid boundary moves.

    A graph potential routes volume from closing cells to opening cells. Its
    divergence satisfies new_volume-old_volume + net_outflow = 0. This is a
    quasistatic displacement field, not Navier-Stokes pressure or stirring flow.
    Transport follows decreasing potential, so inflows arrive before outflows.
    Geometry enables bounded MUSCL face values and a first-moment midpoint
    predictor. Newly opened cells receive actual donor solute and heat.
    """
    from scipy.sparse.csgraph import connected_components

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
        np.maximum(old, new), openings, (dz, dy, dx) if distances is None else distances
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
            potential[nodes[:-1]] = _solve_spd(matrix[:-1, :-1], rhs[:-1])
    signed_flux = g * (potential[i] - potential[j])
    donor = np.where(signed_flux >= 0.0, i, j)
    receiver = np.where(signed_flux >= 0.0, j, i)
    amount = abs(signed_flux)
    diagnostics = {} if diagnostics is None else diagnostics
    prediction = None
    if geometry is not None:
        prediction = _remap_prediction(
            old,
            new,
            (concentration, temperature),
            geometry,
            active,
            i,
            j,
            signed_flux,
            records,
            np.array([dx, dy, dz]),
        )
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
        if prediction is not None:
            predicted, valid, bounds = prediction
            corrected = _limit_remap_outflow(
                np.stack((mass_out, heat_out), axis=1),
                predicted[edges],
                valid[edges],
                amount[edges],
                np.array([masses[node], heat[node]]),
                targets[node],
                bounds,
                diagnostics,
            )
            mass_out, heat_out = corrected[:, 0], corrected[:, 1]
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
