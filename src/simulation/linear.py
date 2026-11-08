# HOW THIS FILE WORKS
# Solve symmetric positive-definite finite-volume systems with a cached direct
# factorization or diagonally preconditioned conjugate gradients. Check the
# actual residual and finiteness after every solve. Momentum and pressure share
# this solver so cut-cell projection inherits the same failure criteria.

"""Residual-checked SPD solves shared by momentum and pressure operators."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg


def solve_spd(system, rhs, guess=None, direct=None):
    if not np.any(rhs):
        return np.zeros_like(rhs)
    if direct is not None:
        answer = direct(rhs)
        status = 0
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
