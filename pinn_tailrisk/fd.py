"""Finite-difference solver for the 1-D forward jump-diffusion PIDE.

    p_t = -a p_x + 0.5 sigma^2 p_xx - lam p + lam * (p * f_Y)(x)

Scheme: Crank-Nicolson for the local (drift, diffusion, -lam p) part,
second-order Adams-Bashforth for the nonlocal jump convolution (IMEX CN-AB2),
so the global error is O(dx^2 + dt^2). The convolution uses FFT with a
cell-averaged jump kernel (exact mass on the grid). Boundaries: p = 0
(the domain is chosen wide enough that the mass outside is negligible).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse, stats
from scipy.signal import fftconvolve
from scipy.sparse.linalg import splu

from .jumpdiff import MertonParams, merton_pdf


def jump_kernel(dx: float, n: int, p: MertonParams) -> np.ndarray:
    """Cell-averaged jump density at offsets m*dx, m = -(n-1)..(n-1)."""
    m = np.arange(-(n - 1), n) * dx
    return (stats.norm.cdf(m + dx / 2, p.mu_j, p.sigma_j) - stats.norm.cdf(m - dx / 2, p.mu_j, p.sigma_j))


def solve_pide_1d(p: MertonParams, x_min: float, x_max: float, nx: int, t0: float,
                  times, dt: float, p0=None):
    """Return (x, {t: pdf}) and wall-clock time. Initial condition is p(., t0).

    p0 defaults to the exact Merton density at t0, the same initial condition
    the PINN is given, so both solve exactly the same problem.
    """
    tic = time.perf_counter()
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    u = merton_pdf(x, t0, p) if p0 is None else np.asarray(p0, float).copy()

    # local operator L on interior points (Dirichlet zero at both ends)
    c_xx = 0.5 * p.sigma**2 / dx**2
    c_x = p.a / (2 * dx)
    main = np.full(nx, -2 * c_xx - p.lam)
    lower = np.full(nx - 1, c_xx + c_x)     # coefficient of p_{i-1}
    upper = np.full(nx - 1, c_xx - c_x)     # coefficient of p_{i+1}
    L = sparse.diags([lower, main, upper], [-1, 0, 1], format="csc")
    I = sparse.identity(nx, format="csc")

    kern = jump_kernel(dx, nx, p)

    def jump(v):
        return p.lam * fftconvolve(v, kern, mode="full")[nx - 1: 2 * nx - 1]

    times = sorted(times)
    out, t = {}, t0
    j_prev = None
    lu_cache = {}
    for t_target in times:
        while t < t_target - 1e-15:
            h = min(dt, t_target - t)
            if h not in lu_cache:
                lu_cache[h] = (splu((I - 0.5 * h * L).tocsc()), (I + 0.5 * h * L).tocsr())
            lu, rhs_op = lu_cache[h]
            j_now = jump(u)
            j_ext = j_now if j_prev is None else 1.5 * j_now - 0.5 * j_prev
            u = lu.solve(rhs_op @ u + h * j_ext)
            u[0] = u[-1] = 0.0
            j_prev, t = j_now, t + h
        out[t_target] = u.copy()
    return x, out, time.perf_counter() - tic
