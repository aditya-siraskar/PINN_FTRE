"""Finite-difference solver for the 2-asset common-jump PIDE.

    p_t = -a . grad p + 0.5 sum_ij C_ij d_ij p - lam p + lam * int p(x - beta y) f_Y(y) dy

Scheme: Douglas ADI (theta = 1/2). The axis-aligned drift/diffusion parts (each
with half of -lam p) are implicit, one tridiagonal sweep per axis. The mixed
derivative C_01 d_xy and the jump integral are explicit. The jump integral is a 2-D
convolution with a kernel supported on the line {beta * y}: the jump-size density
is sampled finely and splatted bilinearly onto grid offsets (mass-preserving), then
applied by FFT, so each step costs O(n^2 log n). Dirichlet p = 0 on the boundary.

Its purpose in the study is twofold: a second independent reference in 2-D, and a
measured demonstration that grid cost grows like n^d.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats
from scipy.linalg import solve_banded
from scipy.signal import fftconvolve

from .jumpdiff import MultiMertonParams, multi_merton_logpdf


def jump_kernel_2d(hx: float, hy: float, n: int, mp: MultiMertonParams, n_y: int = 20001) -> np.ndarray:
    """Mass of the jump displacement beta*Y in each grid offset, shape (2n-1, 2n-1)."""
    y = np.linspace(mp.mu_j - 9 * mp.sigma_j, mp.mu_j + 9 * mp.sigma_j, n_y)
    w = stats.norm.pdf(y, mp.mu_j, mp.sigma_j)
    w /= w.sum()
    u, v = mp.beta[0] * y / hx, mp.beta[1] * y / hy
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    K = np.zeros((2 * n - 1, 2 * n - 1))
    for di, dj, wt in [(0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)]:
        ii, jj = i0 + di + n - 1, j0 + dj + n - 1
        ok = (ii >= 0) & (ii < 2 * n - 1) & (jj >= 0) & (jj < 2 * n - 1)   # jumps leaving the grid are lost
        np.add.at(K, (ii[ok], jj[ok]), w[ok] * wt[ok])
    return K


def _banded(n: int, h: float, a: float, c: float, lam_half: float, theta_dt: float):
    """Banded form of (I - theta dt A) for A = -a d + 0.5 c d2 - lam/2, Dirichlet rows = identity."""
    lo = c / (2 * h**2) + a / (2 * h)
    mid = -c / h**2 - lam_half
    up = c / (2 * h**2) - a / (2 * h)
    ab = np.zeros((3, n))
    ab[0, 1:] = -theta_dt * up      # super-diagonal (coefficient of p_{i+1})
    ab[1, :] = 1 - theta_dt * mid
    ab[2, :-1] = -theta_dt * lo     # sub-diagonal (coefficient of p_{i-1})
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = ab[2, -2] = 0.0
    return ab, (lo, mid, up)


def _apply_axis(U, coef, axis):
    lo, mid, up = coef
    out = mid * U
    if axis == 0:
        out[1:-1] += lo * U[:-2] + up * U[2:]
    else:
        out[:, 1:-1] += lo * U[:, :-2] + up * U[:, 2:]
    return out


def solve_pide_2d(mp: MultiMertonParams, lo, hi, n: int, t0: float, times, dt: float):
    """Return (x, y, {t: P[i, j] = p(x_i, y_j)}, wall seconds). IC: exact density at t0."""
    assert mp.d == 2
    tic = time.perf_counter()
    x, y = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    hx, hy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.exp(multi_merton_logpdf(np.stack([X, Y], -1), t0, mp))
    C = mp.cov
    K = jump_kernel_2d(hx, hy, n, mp)

    def explicit(V):
        mixed = np.zeros_like(V)
        mixed[1:-1, 1:-1] = C[0, 1] * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4 * hx * hy)
        jump = mp.lam * fftconvolve(V, K, mode="full")[n - 1: 2 * n - 1, n - 1: 2 * n - 1]
        return mixed + jump

    out, t, cache = {}, t0, {}
    for t_target in sorted(times):
        while t < t_target - 1e-15:
            h = min(dt, t_target - t)
            if h not in cache:
                cache[h] = (_banded(n, hx, mp.a[0], C[0, 0], mp.lam / 2, 0.5 * h),
                            _banded(n, hy, mp.a[1], C[1, 1], mp.lam / 2, 0.5 * h))
            (ab1, c1), (ab2, c2) = cache[h]
            A1U, A2U = _apply_axis(U, c1, 0), _apply_axis(U, c2, 1)
            Y0 = U + h * (A1U + A2U + explicit(U))
            R1 = Y0 - 0.5 * h * A1U
            R1[0], R1[-1] = 0.0, 0.0
            Y1 = solve_banded((1, 1), ab1, R1)
            R2 = Y1 - 0.5 * h * A2U
            R2[:, 0], R2[:, -1] = 0.0, 0.0
            U = solve_banded((1, 1), ab2, R2.T).T
            U[0], U[-1], U[:, 0], U[:, -1] = 0.0, 0.0, 0.0, 0.0
            t += h
        out[t_target] = U.copy()
    return x, y, out, time.perf_counter() - tic
