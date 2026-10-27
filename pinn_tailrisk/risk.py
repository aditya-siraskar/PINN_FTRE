"""VaR / ES from a density on a grid, from samples, and for a Gaussian model.

Convention: X is a log return, loss L = -X, alpha is the confidence level
(0.99 means "99% VaR"). VaR and ES are reported as positive loss numbers.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid

ALPHAS = (0.95, 0.975, 0.99, 0.995)


def var_es_from_pdf(x: np.ndarray, pdf: np.ndarray, alpha: float, renormalise: bool = True):
    """VaR/ES from a density sampled on an increasing grid x.

    The CDF is the cumulative trapezoid of pdf; the quantile is found by linear
    interpolation; ES integrates x * pdf up to the quantile (with the partial
    last cell handled exactly under linear interpolation of pdf).
    """
    pdf = np.clip(pdf, 0.0, None)
    cdf = cumulative_trapezoid(pdf, x, initial=0.0)
    if renormalise:
        pdf, cdf = pdf / cdf[-1], cdf / cdf[-1]
    tail = 1.0 - alpha
    i = int(np.searchsorted(cdf, tail))
    if i == 0:
        raise ValueError("quantile lies left of the grid; widen the domain")
    # linear interpolation inside cell [i-1, i]
    frac = (tail - cdf[i - 1]) / max(cdf[i] - cdf[i - 1], 1e-300)
    q = x[i - 1] + frac * (x[i] - x[i - 1])
    xm = np.append(x[:i], q)
    pm = np.append(pdf[:i], np.interp(q, x, pdf))
    partial = np.trapezoid(xm * pm, xm)
    # rescale so the integrated tail mass is exactly `tail`
    mass = np.trapezoid(pm, xm)
    es = -partial / mass if mass > 0 else np.nan
    return -q, es


def var_es_from_samples(x: np.ndarray, alpha: float):
    q = np.quantile(x, 1.0 - alpha)
    return -q, -x[x <= q].mean()


def var_es_from_weighted(z: np.ndarray, w: np.ndarray, alpha: float):
    """VaR/ES of a discrete distribution: values z with probability weights w (renormalised).

    Used for a portfolio return z = w . x read off a d-dimensional density grid
    (weights = p * cell volume). The cell holding the quantile is split exactly.
    """
    z, w = np.ravel(z), np.clip(np.ravel(w), 0.0, None)
    order = np.argsort(z)
    z, w = z[order], w[order] / w.sum()
    tail = 1.0 - alpha
    c = np.cumsum(w)
    k = int(np.searchsorted(c, tail))
    taken = tail - (c[k - 1] if k > 0 else 0.0)
    es = -(np.dot(z[:k], w[:k]) + taken * z[k]) / tail
    return -z[k], es


def gaussian_var_es(mean: float, std: float, alpha: float):
    z = stats.norm.ppf(1.0 - alpha)
    return -(mean + std * z), -(mean - std * stats.norm.pdf(z) / (1.0 - alpha))


def bulk_tail_masks(x: np.ndarray, cdf: np.ndarray, tail: float = 0.01):
    """Boolean masks for the bulk [q_tail, q_{1-tail}] and the two tails."""
    lo, hi = np.interp([tail, 1 - tail], cdf, x)
    bulk = (x >= lo) & (x <= hi)
    return bulk, x < lo, x > hi
