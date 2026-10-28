"""Stage 6 benchmark risk models on real returns (one-day-ahead, no look-ahead).

Univariate models act on the portfolio P&L series y_t; multivariate ones produce joint
scenarios of the asset log returns, so they can also revalue a non-linear portfolio:
  * Gaussian            rolling mean / std of y
  * historical sim.     the last `window` joint returns, equal weight
  * filtered HS         historical joint returns rescaled by per-asset EWMA vols (today / then)
  * GARCH(1,1)-t        on y, Student-t innovations (arch package)
  * GARCH-EVT           McNeil-Frey: GPD tail on the GARCH-t standardised residuals
  * DCC-GARCH-t         GARCH(1,1)-t per asset + DCC(1,1) correlation; scenarios from a
                        multivariate t with the median marginal dof, unit variance
GARCH and DCC parameters are refitted on a rolling window every `refit` days and the
filters run daily with the fitted parameters.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import optimize, stats


# --------------------------------------------------------------------------- closed-form tails
def t_var_es(alpha: float, nu: float) -> tuple[float, float]:
    """VaR/ES (as positive losses) of a unit-variance Student t."""
    sc = math.sqrt((nu - 2) / nu)
    q = stats.t.ppf(1 - alpha, nu)
    es = (nu + q**2) / (nu - 1) * stats.t.pdf(q, nu) / (1 - alpha)
    return -q * sc, es * sc


def norm_var_es(alpha: float) -> tuple[float, float]:
    q = stats.norm.ppf(alpha)
    return q, stats.norm.pdf(q) / (1 - alpha)


def sample_var_es(z: np.ndarray, alpha: float) -> tuple[float, float]:
    """VaR/ES (positive losses) of equally weighted P&L draws z (exact tail average)."""
    z = np.sort(z)
    n = len(z)
    m = (1 - alpha) * n
    k = int(math.floor(m))
    var = -z[k]
    es = -(z[:k].sum() + (m - k) * z[k]) / m
    return var, es


# --------------------------------------------------------------------------- filters
def ewma_vol(x: np.ndarray, decay: float = 0.94, warmup: int = 250) -> np.ndarray:
    """Per-column EWMA vol at day t from data up to t-1 (x (T,) or (T,d))."""
    x2 = np.atleast_2d(x.T).T ** 2
    v = np.empty_like(x2)
    v[0] = x2[:warmup].mean(0)
    for t in range(1, len(x2)):
        v[t] = decay * v[t - 1] + (1 - decay) * x2[t - 1]
    v = np.sqrt(v)
    return v if x.ndim == 2 else v[:, 0]


def garch_fit(y: np.ndarray) -> dict:
    """GARCH(1,1)-t with constant mean on y (fitted in percent for numerical stability)."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = arch_model(100 * y, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = r.params
    return dict(mu=p["mu"] / 100, omega=p["omega"] / 1e4, alpha=p["alpha[1]"], beta=p["beta[1]"], nu=p["nu"])


def garch_filter(y: np.ndarray, g: dict, v0: float | None = None) -> np.ndarray:
    """Conditional vol sigma_t from y up to t-1 (length len(y) + 1: the last entry is the next day)."""
    e2 = (y - g["mu"]) ** 2
    v = np.empty(len(y) + 1)
    v[0] = e2.mean() if v0 is None else v0
    for t in range(len(y)):
        v[t + 1] = g["omega"] + g["alpha"] * e2[t] + g["beta"] * v[t]
    return np.sqrt(v)


def gpd_tail(loss: np.ndarray, alpha: float, u_frac: float = 0.10) -> tuple[float, float]:
    """POT: GPD on exceedances of the top u_frac losses; VaR/ES at alpha (McNeil-Frey formulas)."""
    u = np.quantile(loss, 1 - u_frac)
    exc = loss[loss > u] - u
    xi, _, b = stats.genpareto.fit(exc, floc=0)
    xi = min(xi, 0.9)
    var = u + b / xi * ((len(loss) / len(exc) * (1 - alpha)) ** (-xi) - 1) if abs(xi) > 1e-6 else \
        u - b * math.log(len(loss) / len(exc) * (1 - alpha))
    es = (var + b - xi * u) / (1 - xi)
    return var, es


# --------------------------------------------------------------------------- DCC
def dcc_fit(e: np.ndarray) -> tuple[float, float]:
    """DCC(1,1) (a, b) by Gaussian quasi-likelihood on standardised residuals e (T,d)."""
    Qbar = np.cov(e.T)

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        Q, f = Qbar.copy(), 0.0
        for t in range(len(e)):
            dq = np.sqrt(np.diag(Q))
            R = Q / np.outer(dq, dq)
            _, logdet = np.linalg.slogdet(R)
            f += logdet + e[t] @ np.linalg.solve(R, e[t])
            Q = (1 - a - b) * Qbar + a * np.outer(e[t], e[t]) + b * Q
        return 0.5 * f

    r = optimize.minimize(nll, [0.03, 0.95], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-3, maxiter=200))
    return tuple(r.x)


def dcc_filter(e: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    """Correlation R_t from e up to t-1 (len(e) + 1 matrices)."""
    Q = Qbar.copy()
    out = np.empty((len(e) + 1, *Qbar.shape))
    for t in range(len(e) + 1):
        dq = np.sqrt(np.diag(Q))
        out[t] = Q / np.outer(dq, dq)
        if t < len(e):
            Q = (1 - a - b) * Qbar + a * np.outer(e[t], e[t]) + b * Q
    return out


def mvt_draws(n: int, R: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance multivariate t draws with correlation R."""
    z = rng.standard_normal((n, len(R))) @ np.linalg.cholesky(R).T
    return z * np.sqrt((nu - 2) / rng.chisquare(nu, n))[:, None]
