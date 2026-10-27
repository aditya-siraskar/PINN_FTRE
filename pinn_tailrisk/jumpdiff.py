"""Merton jump-diffusion for log returns: exact densities and exact simulation.

Model (log return X_t, X_0 = 0):

    dX_t = a dt + sigma dW_t + dJ_t,     J_t = sum_{i<=N_t} Y_i,
    N_t ~ Poisson(lam t),                Y_i ~ N(mu_j, sigma_j^2).

Forward (Fokker-Planck-Kolmogorov) PIDE for the density p(x, t):

    p_t = -a p_x + 0.5 sigma^2 p_xx + lam * int p(x - y, t) f_Y(y) dy - lam p.

Exact solution (Merton 1976): a Poisson-weighted mixture of Gaussians,

    p(x, t) = sum_k Pois(k; lam t) * N(x; a t + k mu_j, sigma^2 t + k sigma_j^2).

The multivariate version uses one *common* jump factor: on a jump all assets
move by beta * Y with scalar Y ~ N(mu_j, sigma_j^2). The jump integral is
therefore 1-D in any dimension d, and the density is again a Gaussian mixture.

Units: time in years (252 trading days), returns as log returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize, stats

DAY = 1.0 / 252.0


@dataclass(frozen=True)
class MertonParams:
    a: float = 0.05          # drift of the log return (per year)
    sigma: float = 0.20      # diffusion volatility (per sqrt year)
    lam: float = 5.0         # jump intensity (jumps per year)
    mu_j: float = -0.05      # mean jump size (log return)
    sigma_j: float = 0.07    # jump-size std

    def as_dict(self) -> dict:
        return asdict(self)

    def variance(self, t):
        return (self.sigma**2 + self.lam * (self.mu_j**2 + self.sigma_j**2)) * t

    def mean(self, t):
        return (self.a + self.lam * self.mu_j) * t


def _n_terms(lam_t: float, tol: float = 1e-16) -> int:
    """Number of Poisson terms so that the neglected mass is < tol."""
    return int(stats.poisson.ppf(1.0 - tol, lam_t)) + 3 if lam_t > 0 else 1


def mixture(p: MertonParams, t: float):
    """Weights, means and stds of the Gaussian mixture at time t."""
    k = np.arange(_n_terms(p.lam * t))
    w = stats.poisson.pmf(k, p.lam * t)
    m = p.a * t + k * p.mu_j
    s = np.sqrt(p.sigma**2 * t + k * p.sigma_j**2)
    return w, m, s


def merton_pdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    return (w * stats.norm.pdf(x[..., None], m, s)).sum(-1)


def merton_logpdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    lw = np.log(w, where=w > 0, out=np.full_like(w, -np.inf))
    return np.logaddexp.reduce(lw + stats.norm.logpdf(x[..., None], m, s), axis=-1)


def merton_cdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    return (w * stats.norm.cdf(x[..., None], m, s)).sum(-1)


def merton_var_es(alpha: float, t: float, p: MertonParams) -> tuple[float, float]:
    """Exact VaR and ES of the loss L = -X at level alpha (e.g. 0.99).

    q = F^{-1}(1 - alpha);  VaR = -q;
    ES = -E[X | X <= q] = -(1/(1-alpha)) * sum_k w_k (m_k Phi(z_k) - s_k phi(z_k)),
    with z_k = (q - m_k) / s_k.
    """
    w, m, s = mixture(p, t)
    sd = np.sqrt(p.variance(t))
    lo, hi = p.mean(t) - 50 * sd, p.mean(t) + 50 * sd
    q = optimize.brentq(lambda x: (w * stats.norm.cdf(x, m, s)).sum() - (1 - alpha), lo, hi, xtol=1e-14)
    z = (q - m) / s
    partial = (w * (m * stats.norm.cdf(z) - s * stats.norm.pdf(z))).sum()
    return -q, -partial / (1 - alpha)


def simulate_merton(n: int, t: float, p: MertonParams, rng: np.random.Generator) -> np.ndarray:
    """Exact samples of X_t (no time discretisation error)."""
    k = rng.poisson(p.lam * t, size=n)
    return (p.a * t + p.sigma * np.sqrt(t) * rng.standard_normal(n)
            + k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal(n))


def simulate_merton_paths(n_paths: int, n_steps: int, dt: float, p: MertonParams,
                          rng: np.random.Generator) -> np.ndarray:
    """Daily increments of a path; shape (n_paths, n_steps). Used for synthetic data."""
    k = rng.poisson(p.lam * dt, size=(n_paths, n_steps))
    return (p.a * dt + p.sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
            + k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal((n_paths, n_steps)))


# --------------------------------------------------------------------------- multi-asset
@dataclass(frozen=True)
class MultiMertonParams:
    """d assets: correlated diffusion + one common jump factor beta * Y."""
    a: np.ndarray                 # (d,) drifts
    cov: np.ndarray               # (d, d) diffusion covariance (per year)
    lam: float = 5.0
    mu_j: float = -0.05
    sigma_j: float = 0.07
    beta: np.ndarray = field(default=None)  # (d,) jump loadings; default ones

    def __post_init__(self):
        if self.beta is None:
            object.__setattr__(self, "beta", np.ones(len(self.a)))

    @property
    def d(self) -> int:
        return len(self.a)

    @classmethod
    def equicorrelated(cls, d: int, sigma=0.20, rho=0.5, a=0.05, beta=None, **jump):
        vols = np.full(d, sigma) if np.isscalar(sigma) else np.asarray(sigma)
        corr = np.full((d, d), rho) + (1 - rho) * np.eye(d)
        return cls(a=np.full(d, a), cov=np.outer(vols, vols) * corr,
                   beta=None if beta is None else np.asarray(beta), **jump)

    def marginal(self, i: int) -> MertonParams:
        return MertonParams(a=self.a[i], sigma=float(np.sqrt(self.cov[i, i])), lam=self.lam,
                            mu_j=self.beta[i] * self.mu_j, sigma_j=abs(self.beta[i]) * self.sigma_j)

    def portfolio(self, w: np.ndarray) -> MertonParams:
        """Exact 1-D law of w . X (linear portfolio of log returns)."""
        w = np.asarray(w, float)
        bw = float(w @ self.beta)
        return MertonParams(a=float(w @ self.a), sigma=float(np.sqrt(w @ self.cov @ w)), lam=self.lam,
                            mu_j=bw * self.mu_j, sigma_j=abs(bw) * self.sigma_j)


def multi_mixture(p: MultiMertonParams, t: float):
    k = np.arange(_n_terms(p.lam * t))
    w = stats.poisson.pmf(k, p.lam * t)
    means = p.a[None] * t + k[:, None] * p.mu_j * p.beta[None]
    bb = np.outer(p.beta, p.beta) * p.sigma_j**2
    covs = p.cov[None] * t + k[:, None, None] * bb[None]
    return w, means, covs


def multi_merton_logpdf(x, t: float, p: MultiMertonParams):
    """x: (..., d). Exact log density of the d-dimensional common-jump model."""
    x = np.asarray(x, float)
    w, means, covs = multi_mixture(p, t)
    terms = [np.log(wk) + stats.multivariate_normal.logpdf(x, mk, ck)
             for wk, mk, ck in zip(w, means, covs) if wk > 0]
    return np.logaddexp.reduce(np.stack(terms, -1), axis=-1)


def simulate_multi_merton(n: int, t: float, p: MultiMertonParams, rng: np.random.Generator) -> np.ndarray:
    k = rng.poisson(p.lam * t, size=n)
    diff = rng.multivariate_normal(np.zeros(p.d), p.cov * t, size=n)
    y = k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal(n)
    return p.a[None] * t + diff + y[:, None] * p.beta[None]
