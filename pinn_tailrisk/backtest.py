"""VaR/ES backtests: Kupiec, Christoffersen, the Fissler-Ziegel FZ0 score and Diebold-Mariano.

Conventions: y is the realised P&L (negative = loss); VaR and ES are positive losses;
a breach is y < -VaR.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _xlogy(x, y):
    return x * math.log(y) if x > 0 else 0.0


def kupiec(hits: np.ndarray, alpha: float) -> dict:
    """Unconditional coverage (POF) likelihood-ratio test."""
    n, x, p = len(hits), int(hits.sum()), 1 - alpha
    ph = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - ph) - _xlogy(x, ph))
    return dict(n=n, breaches=x, rate=ph, lr_uc=lr, p_uc=float(stats.chi2.sf(lr, 1)))


def christoffersen(hits: np.ndarray, alpha: float) -> dict:
    """Independence (first-order Markov) and conditional coverage tests."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _xlogy(n00 + n10, 1 - p1) + _xlogy(n01 + n11, p1)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    lr_cc = kupiec(hits, alpha)["lr_uc"] + lr_ind
    return dict(lr_ind=lr_ind, p_ind=float(stats.chi2.sf(lr_ind, 1)), lr_cc=lr_cc,
                p_cc=float(stats.chi2.sf(lr_cc, 2)), n11=n11)


def fz0_loss(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """FZ0 loss of Patton, Ziegel & Chen (2019): strictly consistent for (VaR, ES); lower is better."""
    p = 1 - alpha
    v, e = -var, -np.maximum(np.maximum(es, var), 1e-8)   # quantile and ES of y; FZ0 needs ES of y < 0
    return -((y <= v) * (v - y)) / (p * e) + v / e + np.log(-e) - 1


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, lag: int | None = None) -> dict:
    """DM test of equal mean loss with a Newey-West variance; stat < 0 means model 1 is better."""
    dl = l1 - l2
    n = len(dl)
    lag = int(math.floor(4 * (n / 100) ** (2 / 9))) if lag is None else lag
    u = dl - dl.mean()
    s = u @ u / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    if s <= 0:                                        # identical losses (a model against itself)
        return dict(stat=float("nan"), p=float("nan"), mean_diff=0.0)
    stat = dl.mean() / math.sqrt(s / n)
    return dict(stat=float(stat), p=float(2 * stats.norm.sf(abs(stat))), mean_diff=float(dl.mean()))


def summarise(y: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict:
    """All the backtests for one model and level, plus the breach-day ES ratio (realised / forecast)."""
    hits = y < -var
    out = dict(**kupiec(hits, alpha), **christoffersen(hits, alpha))
    out["fz0"] = float(fz0_loss(y, var, es, alpha).mean())
    out["es_ratio"] = float((-y[hits]).mean() / es[hits].mean()) if hits.any() else float("nan")
    return out
