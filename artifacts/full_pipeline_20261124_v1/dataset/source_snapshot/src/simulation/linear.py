# HOW THIS FILE WORKS
# Solve symmetric positive-definite finite-volume systems with a cached direct
# factorization or diagonally preconditioned conjugate gradients. Check the
# actual residual and finiteness after every solve. Momentum and pressure share
# this solver so cut-cell projection inherits the same failure criteria.

"""Residual-checked SPD solves shared by momentum and pressure operators."""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg


_PRODUCT_BACKEND = ContextVar("flow_spd_product_backend", default=None)


@contextmanager
def using_product_backend(backend):
    """Scope an experimental iterative solver to this execution context.

    Cached small-system direct solves retain their normal CPU path. A backend
    failure propagates; it must never silently substitute CPU work for GPU work.
    """
    if not callable(getattr(backend, "solve", None)):
        raise ValueError("Product backend must provide solve(system, rhs, guess)")
    token = _PRODUCT_BACKEND.set(backend)
    try:
        yield backend
    finally:
        _PRODUCT_BACKEND.reset(token)


def solve_spd(system, rhs, guess=None, direct=None):
    if not np.any(rhs):
        return np.zeros_like(rhs)
    if direct is not None:
        answer = direct(rhs)
        status = 0
    elif (backend := _PRODUCT_BACKEND.get()) is not None:
        answer, status = backend.solve(system, rhs, guess)
    else:
        answer, status = cg(
            system,
            rhs,
            x0=guess,
            M=diags(1 / system.diagonal()),
            rtol=2e-13,
            atol=0.0,
            maxiter=4000,
        )
    scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    if (
        status != 0
        or not np.isfinite(answer).all()
        or np.linalg.norm(system @ answer - rhs) > 3e-12 * scale
    ):
        raise ArithmeticError("Flow linear solve failed its residual check")
    return answer
