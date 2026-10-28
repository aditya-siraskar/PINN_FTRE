"""Amortised d-asset PINN (Stage 5): one network for a whole box of market parameters.

Stage 4 trained one network per parameter set. Here the parameters
theta = (s, lam, mu_J) are extra network inputs and are drawn afresh for every
collocation, initial-condition and normalisation point, so after one training run a
new market state costs a forward pass instead of a retrain:
  * s     scales every diffusion vol (covariance s^2 Sigma; Sigma, the correlations and
          the loadings beta stay fixed),
  * lam   jump intensity (sampled log-uniformly),
  * mu_J  mean common jump; sigma_J stays fixed.

Everything else is Stage 4: log-density output, whitened similarity input (now also
divided by s), exact or Hutchinson trace, 1-D Gauss-Hermite jump integral, the
jump-mixture Student-t proposal (now evaluated at each row's theta) and IS VaR/ES.
The initial condition is the exact mixture at t0, evaluated in torch for each row's theta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .jumpdiff import MultiMertonParams
from .pinn_nd import NDConfig, NDTrainer, Proposal, _grad

THETA_NAMES = ("s", "lam", "mu_j")


@dataclass
class ThetaBox:
    """Training box of theta; lam is sampled and encoded on a log scale."""
    s: tuple = (0.6, 1.6)
    lam: tuple = (1.0, 15.0)
    mu_j: tuple = (-0.10, -0.02)

    def _lo_hi(self):
        lo = torch.tensor([self.s[0], math.log(self.lam[0]), self.mu_j[0]])
        hi = torch.tensor([self.s[1], math.log(self.lam[1]), self.mu_j[1]])
        return lo, hi

    def sample(self, n: int, device) -> torch.Tensor:
        lo, hi = (v.to(device) for v in self._lo_hi())
        u = lo + (hi - lo) * torch.rand(n, 3, device=device)
        return torch.stack([u[:, 0], torch.exp(u[:, 1]), u[:, 2]], -1)

    def encode(self, th: torch.Tensor) -> torch.Tensor:
        """theta -> [-1, 1]^3 network features."""
        lo, hi = (v.to(th.device) for v in self._lo_hi())
        u = torch.stack([th[:, 0], torch.log(th[:, 1]), th[:, 2]], -1)
        return 2 * (u - lo) / (hi - lo) - 1


def params_at(base: MultiMertonParams, th) -> MultiMertonParams:
    """The MultiMertonParams of one theta = (s, lam, mu_j)."""
    s, lam, mu = (float(v) for v in th)
    return MultiMertonParams(a=base.a, cov=s**2 * base.cov, lam=lam, mu_j=mu, sigma_j=base.sigma_j, beta=base.beta)


def exact_logpdf_torch(x, t, th, p: MultiMertonParams, k_max: int = 8):
    """Exact log density of the common-jump model at per-row (t, theta); torch, any dtype.

    x (N,d); t and th have N rows or one row. Poisson terms k = 0..k_max (enough while
    lam t stays below about 1: the neglected mass is then < 1e-6).
    """
    f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=x.dtype, device=x.device)  # noqa: E731
    a, cov, beta = f(p.a), f(p.cov), f(p.beta)
    k = torch.arange(k_max + 1, device=x.device, dtype=x.dtype)
    s2, lam, mu = th[:, 0, None, None, None] ** 2, th[:, 1, None], th[:, 2, None, None]
    loc = a * t[:, None, None] + k[None, :, None] * mu * beta
    S = s2 * t[:, None, None, None] * cov + (k[:, None, None] * p.sigma_j**2 * torch.outer(beta, beta))[None]
    chol = torch.linalg.cholesky(S)
    lt = lam * t[:, None]
    logpi = k * torch.log(lt) - lt - torch.lgamma(k + 1)
    diff = x[:, None, :] - loc
    cb = chol.expand(diff.shape[0], -1, -1, -1) if chol.shape[0] == 1 else chol
    z = torch.linalg.solve_triangular(cb, diff[..., None], upper=False).squeeze(-1)
    logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    d = x.shape[1]
    return torch.logsumexp(logpi - 0.5 * (d * math.log(2 * math.pi) + logdet + (z**2).sum(-1)), -1)


class AmortisedPINN(nn.Module):
    def __init__(self, p: MultiMertonParams, t0: float, T: float, cfg: NDConfig, box: ThetaBox, z_scale: float = 10.0):
        super().__init__()
        self.t0, self.T, self.cfg, self.box, self.z_scale, self.d = t0, T, cfg, box, z_scale, p.d
        L = np.linalg.cholesky(p.cov)
        self.register_buffer("a", torch.tensor(p.a, dtype=torch.float32))
        self.register_buffer("Linv", torch.tensor(np.linalg.inv(L), dtype=torch.float32))
        self.logdet = float(np.linalg.slogdet(p.cov)[1])
        layers, n_in = [], p.d + 1 + len(THETA_NAMES)
        for _ in range(cfg.depth):
            layers += [nn.Linear(n_in, cfg.width), nn.Tanh()]
            n_in = cfg.width
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """x (N,d), t (N,), th (N,3) -> log p (N,)."""
        s = th[:, 0]
        z = (x - self.a * t[:, None]) @ self.Linv.T / (s * torch.sqrt(t))[:, None] / self.z_scale
        tau = 2 * (torch.log(t) - math.log(self.t0)) / (math.log(self.T) - math.log(self.t0)) - 1
        g = self.net(torch.cat([z, tau[:, None], self.box.encode(th)], -1)).squeeze(-1)
        gauss = -0.5 * (self.d * torch.log(2 * math.pi * t) + self.logdet) - self.d * torch.log(s)
        return gauss + self.cfg.out_scale * g


class AmortisedTrainer(NDTrainer):
    """NDTrainer with theta as an input; `fit` and the training loop are inherited."""

    def __init__(self, p: MultiMertonParams, t0: float, T: float, cfg: NDConfig, box: ThetaBox | None = None,
                 device=None):
        self.p, self.t0, self.T, self.cfg, self.d = p, t0, T, cfg, p.d
        self.box = box or ThetaBox()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.model = AmortisedPINN(p, t0, T, cfg, self.box).to(self.device)
        self.trace = cfg.trace if cfg.trace != "auto" else ("exact" if p.d <= cfg.exact_max_d else "hutchinson")
        f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=self.device)  # noqa: E731
        self.a, self.cov, self.beta = f(p.a), f(p.cov), f(p.beta)
        self.L = f(np.linalg.cholesky(p.cov))
        gh_z, gh_w = np.polynomial.hermite.hermgauss(cfg.n_gh)
        self.gh_z = f(math.sqrt(2) * p.sigma_j * gh_z)       # jump nodes are mu_J + gh_z, mu_J per row
        self.jump_w = f(gh_w / math.sqrt(math.pi))
        self.q_col = Proposal(p, cfg.colloc_scale, cfg.colloc_df, cfg.k_max, self.device)
        self.q_is = Proposal(p, cfg.is_scale, cfg.df, cfg.k_max, self.device)
        self.history: list[dict] = []

    def theta(self, th, n: int = 1) -> torch.Tensor:
        """One theta (sequence of 3) -> (n,3) tensor on the device."""
        return torch.as_tensor(np.asarray(th, float), dtype=torch.float32, device=self.device).reshape(1, 3).expand(n, 3)

    # ---- losses -------------------------------------------------------------
    def residual(self, x, t, th):
        m, K = self.model, len(self.gh_z)
        u = m(x, t, th)
        u_t, g = _grad(u, t), _grad(u, x)
        s2, lam, mu = th[:, 0] ** 2, th[:, 1], th[:, 2]
        drift = g @ self.a
        quad = ((g @ self.cov) * g).sum(-1)
        if self.trace == "exact":
            dirs, scale = [self.L[:, i].expand_as(x) for i in range(self.d)], 1.0
        else:
            dirs = [(torch.randint(0, 2, x.shape, device=self.device) * 2.0 - 1.0) @ self.L.T
                    for _ in range(self.cfg.n_probes)]
            scale = 1.0 / self.cfg.n_probes
        tr = sum((_grad((g * v).sum(-1), x) * v).sum(-1) for v in dirs) * scale
        y = mu[:, None] + self.gh_z[None]                                   # (N,K) jump nodes
        xs = (x[:, None, :] - y[..., None] * self.beta).reshape(-1, self.d)
        u_shift = m(xs, t[:, None].expand(-1, K).reshape(-1), th[:, None].expand(-1, K, -1).reshape(-1, 3))
        S = (self.jump_w * torch.exp(torch.clamp(u_shift.reshape(len(x), K) - u[:, None], max=30.0))).sum(-1)
        r = u_t + drift - 0.5 * s2 * (tr + quad) - lam * (S - 1.0)
        return t * r

    def losses(self):
        cfg, m, dev = self.cfg, self.model, self.device
        t = torch.exp(self._logt(cfg.n_colloc))
        th = self.box.sample(cfg.n_colloc, dev)
        x = self.q_col.sample(t, th=th).requires_grad_(True)
        t = t.requires_grad_(True)
        l_pde = self.residual(x, t, th).pow(2).mean()

        t_ic = torch.full((cfg.n_ic,), self.t0, device=dev)
        th_ic = self.box.sample(cfg.n_ic, dev)
        with torch.no_grad():
            x_ic = self.q_col.sample(t_ic, th=th_ic)
            u_ic = exact_logpdf_torch(x_ic, t_ic, th_ic, self.p)
        l_ic = (m(x_ic, t_ic, th_ic) - u_ic).pow(2).mean()

        per = cfg.n_norm // cfg.n_norm_t
        tn = torch.exp(self._logt(cfg.n_norm_t)).repeat_interleave(per)
        thn = self.box.sample(cfg.n_norm_t, dev).repeat_interleave(per, 0)
        with torch.no_grad():
            xn = self.q_is.sample(tn, th=thn)
            lq = self.q_is.log_prob(xn, tn, thn)
        lw = (m(xn, tn, thn) - lq).reshape(cfg.n_norm_t, -1)
        l_norm = (torch.logsumexp(lw, -1) - math.log(per)).pow(2).mean()

        total = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_norm * l_norm
        return total, dict(pde=l_pde.item(), ic=l_ic.item(), norm=l_norm.item())

    # ---- inference ------------------------------------------------------------
    @torch.no_grad()
    def logpdf(self, x: np.ndarray, t: float, th, chunk: int = 1 << 16) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x, float), dtype=torch.float32, device=self.device)
        out = [self.model(xc, torch.full((len(xc),), t, device=self.device), self.theta(th, len(xc)))
               for xc in x.split(chunk)]
        return torch.cat(out).cpu().numpy().astype(float)

    @torch.no_grad()
    def weighted_samples_t(self, t: float, th, n: int, chunk: int = 1 << 16):
        """Proposal draws X (n,d), log q and PINN log p at (t, theta), as torch tensors on the device."""
        tv, tht = torch.full((1,), t, device=self.device), self.theta(th)
        X = self.q_is.sample(tv, n, th=tht)
        lq = torch.cat([self.q_is.log_prob(xc, tv, tht) for xc in X.split(chunk)])
        lp = torch.cat([self.model(xc, torch.full((len(xc),), t, device=self.device), self.theta(th, len(xc)))
                        for xc in X.split(chunk)])
        return X, lq, lp

    def weighted_samples(self, t: float, th, n: int):
        return tuple(v.cpu().numpy().astype(float) for v in self.weighted_samples_t(t, th, n))


# --------------------------------------------------------------------------- torch risk queries (for latency)
def es_weighted_t(z: torch.Tensor, w: torch.Tensor | None, alpha: float):
    """VaR/ES of returns z (N,) or (N,P) with probability weights w (N,) (None = equal); torch, on device."""
    z2 = z if z.dim() == 2 else z[:, None]
    zs, idx = torch.sort(z2, 0)
    ws = torch.full_like(zs, 1.0 / len(zs)) if w is None else (w / w.sum())[idx]
    c = torch.cumsum(ws, 0)
    tail = 1.0 - alpha
    k = torch.searchsorted(c.T.contiguous(), torch.full((z2.shape[1], 1), tail, device=z.device, dtype=c.dtype)).squeeze(-1)
    k = k.clamp(max=len(zs) - 1)
    mask = torch.arange(len(zs), device=z.device)[:, None] < k[None]
    below = (zs * ws * mask).sum(0)
    c_prev = (ws * mask).sum(0)
    zk = zs.gather(0, k[None]).squeeze(0)
    es = -(below + (tail - c_prev) * zk) / tail
    var = -zk
    return (var, es) if z.dim() == 2 else (var[0], es[0])


def simulate_torch(n: int, t: float, p: MultiMertonParams, th, device, L: torch.Tensor | None = None):
    """Exact Monte Carlo draws of X_t at theta, on the device (the latency competitor)."""
    s, lam, mu = (float(v) for v in th)
    f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=device)  # noqa: E731
    L = f(np.linalg.cholesky(p.cov)) if L is None else L
    k = torch.poisson(torch.full((n,), lam * t, device=device))
    diff = (torch.randn(n, p.d, device=device) @ L.T) * (s * math.sqrt(t))
    y = k * mu + torch.sqrt(k) * p.sigma_j * torch.randn(n, device=device)
    return f(p.a) * t + diff + y[:, None] * f(p.beta)
