"""Maximum-likelihood calibration of the d-asset common-jump Merton model to daily returns.

Each day is one draw of X over dt = 1 trading day; its density is the Poisson mixture
over k = 0..k_max jumps (the neglected mass is < 1e-7 while lam dt < 0.1). An optional
per-day vol scale s_t multiplies every diffusion vol (covariance s_t^2 Sigma), as in the
Stage 5 amortised model, so a filtered model is calibrated with the very density the PINN
learns. Identification: the jump loadings beta are non-negative (softplus) and normalised to mean 1, so the
jump is a common shock to all assets; without the sign constraint the NIFTY fit loads it on one stock.

Parameters are optimised jointly with L-BFGS in float64; any subset can be held fixed
(the rolling recalibration of (lam, mu_J) keeps everything else at the development fit).
Standard errors come from the inverse Hessian of the negative log-likelihood.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .jumpdiff import DAY, MultiMertonParams

FREE = ("a", "chol", "beta", "log_lam", "mu_j", "log_sigma_j")


def _unpack(v: dict, d: int):
    Lr = torch.zeros(d, d, dtype=torch.float64)
    Lr[tuple(torch.tril_indices(d, d))] = v["chol"]
    L = torch.tril(Lr, -1) + torch.diag(torch.exp(torch.diagonal(Lr)))
    b = torch.nn.functional.softplus(v["beta"])                    # loadings >= 0: a common crash factor
    beta = b / b.mean()
    return v["a"], L @ L.T, beta, torch.exp(v["log_lam"]), v["mu_j"], torch.exp(v["log_sigma_j"])


def merton_loglik(x: torch.Tensor, s: torch.Tensor, a, cov, beta, lam, mu_j, sigma_j, dt: float = DAY,
                  k_max: int = 4) -> torch.Tensor:
    """Per-day log density of x (N,d) with per-day vol scale s (N,); differentiable in the parameters."""
    # Component k has covariance h Sigma + c beta beta' (h = s^2 dt, c = k sigma_J^2): a rank-1 update of
    # h Sigma, so one Cholesky of Sigma and the matrix determinant lemma / Sherman-Morrison suffice.
    d = x.shape[1]
    k = torch.arange(k_max + 1, dtype=x.dtype)
    L = torch.linalg.cholesky(cov)
    w0 = torch.linalg.solve_triangular(L, (x - a * dt).T, upper=False).T       # (N,d)
    g = torch.linalg.solve_triangular(L, beta[:, None], upper=False).squeeze(-1)
    w = w0[:, None, :] - k[None, :, None] * mu_j * g                            # L^-1 (x - mean_k), (N,K,d)
    h = (s**2 * dt)[:, None]
    c = k[None] * sigma_j**2
    r = 1 + c * (g @ g) / h
    quad = (w**2).sum(-1) / h - c * ((w @ g) / h) ** 2 / r
    logdet = d * torch.log(h) + 2 * torch.log(torch.diagonal(L)).sum() + torch.log(r)
    logpi = k * torch.log(lam * dt) - lam * dt - torch.lgamma(k + 1)
    return torch.logsumexp(logpi - 0.5 * (d * math.log(2 * math.pi) + logdet + quad), -1)


def init_vector(x: np.ndarray, s: np.ndarray | None = None, lam: float = 10.0, mu_j: float = -0.02,
                sigma_j: float = 0.03) -> dict:
    """Moment-based start: sample mean and covariance of the vol-standardised returns."""
    s = np.ones(len(x)) if s is None else s
    xs = x / s[:, None]
    L = np.linalg.cholesky(np.cov(xs.T) / DAY)
    Lr = np.tril(L, -1) + np.diag(np.log(np.diag(L)))
    d = x.shape[1]
    return dict(a=x.mean(0) / DAY, chol=Lr[np.tril_indices(d)], beta=np.full(d, math.log(math.e - 1)), log_lam=math.log(lam),
                mu_j=mu_j, log_sigma_j=math.log(sigma_j))


def fit_merton(x: np.ndarray, s: np.ndarray | None = None, init: dict | None = None, fixed=(), k_max: int = 4,
               max_iter: int = 300, se: bool = False, prior: dict | None = None):
    """MLE (MAP with `prior`) of the common-jump model. Returns (MultiMertonParams, info with the vector, nll, SEs).

    prior: {name: (mean, sd)} Gaussian priors on entries of the parameter vector (e.g. log_lam, mu_j);
    the rolling recalibration shrinks towards the development fit this way, since (lam, mu_J) are weakly
    identified on a 1000-day window.
    """
    d = x.shape[1]
    X = torch.tensor(x, dtype=torch.float64)
    Sv = torch.ones(len(x), dtype=torch.float64) if s is None else torch.tensor(s, dtype=torch.float64)
    v0 = init or init_vector(x, s)
    v = {k: torch.tensor(np.asarray(v0[k], float), dtype=torch.float64, requires_grad=k not in fixed) for k in FREE}
    free = [v[k] for k in FREE if k not in fixed]

    def nll():
        f = -merton_loglik(X, Sv, *_unpack(v, d), k_max=k_max).mean()
        for k, (m, sd) in (prior or {}).items():
            f = f + 0.5 * (((v[k] - m) / sd) ** 2).sum() / len(x)
        return f

    opt = torch.optim.LBFGS(free, lr=1.0, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        f = nll()
        f.backward()
        return f

    opt.step(closure)
    a, cov, beta, lam, mu, sj = (t.detach().numpy() for t in _unpack(v, d))
    p = MultiMertonParams(a=a, cov=cov, lam=float(lam), mu_j=float(mu), sigma_j=float(sj), beta=beta)
    info = dict(vector={k: v[k].detach().numpy().copy() for k in FREE}, nll=float(nll().detach()) * len(x), n=len(x))
    if se:
        info["se"] = jump_se(X, Sv, v, d, k_max)
    return p, info


def jump_se(X, Sv, v: dict, d: int, k_max: int = 4) -> dict:
    """Standard errors of (lam, mu_J, sigma_J) from the inverse Hessian of the full negative log-likelihood."""
    sizes = [v[k].numel() for k in FREE]
    theta0 = torch.cat([v[k].detach().reshape(-1) for k in FREE])

    def f(th):
        parts = dict(zip(FREE, torch.split(th, sizes)))
        parts = {k: p.reshape(v[k].shape) for k, p in parts.items()}
        return -merton_loglik(X, Sv, *_unpack(parts, d), k_max=k_max).sum()

    H = torch.autograd.functional.hessian(f, theta0)
    C = torch.linalg.pinv(H)
    i = np.cumsum([0] + sizes)
    j_lam, j_mu, j_sj = i[3], i[4], i[5]
    lam, sj = math.exp(float(theta0[j_lam])), math.exp(float(theta0[j_sj]))
    return dict(lam=lam * math.sqrt(max(float(C[j_lam, j_lam]), 0)), mu_j=math.sqrt(max(float(C[j_mu, j_mu]), 0)),
                sigma_j=sj * math.sqrt(max(float(C[j_sj, j_sj]), 0)))


def ewma_scale(x: np.ndarray, cov_ref: np.ndarray, decay: float = 0.94, warmup: int = 250) -> np.ndarray:
    """Vol scale s_t from the returns up to day t-1 (no look-ahead).

    v_t = decay v_{t-1} + (1 - decay) |L^-1 x_{t-1}|^2 / d, with L L' = cov_ref (daily), started at
    the mean of the first `warmup` days; s_t = sqrt(v_t).
    """
    u2 = (np.linalg.solve(np.linalg.cholesky(cov_ref), x.T) ** 2).mean(0)
    v = np.empty(len(x))
    v[0] = u2[:warmup].mean()
    for t in range(1, len(x)):
        v[t] = decay * v[t - 1] + (1 - decay) * u2[t - 1]
    return np.sqrt(v)
