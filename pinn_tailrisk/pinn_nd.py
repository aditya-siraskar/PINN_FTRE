"""Physics-informed network for the d-asset common-jump PIDE (Stage 4).

Forward equation for the joint log-return density p(x, t), x in R^d:
    p_t = -a . grad p + 0.5 tr(Sigma Hess p) + lam int p(x - beta y) f_Y(y) dy - lam p.
Divided by p (u = log p) this is
    r = u_t + a . grad u - 0.5 [tr(Sigma Hess u) + grad u' Sigma grad u] - lam (S - 1),
    S = int exp(u(x - beta y) - u(x)) f_Y(y) dy,
i.e. the Stage 2 log residual with the diffusion term made d-dimensional. The jump
integral stays 1-D (one common jump along beta), so Gauss-Hermite is reused as is.

Carried over from the 1-D study: log-density output, log residual times t,
start at t0 from the exact density, normalisation penalty, 48 GH nodes.
Changed for d dimensions:

* **Whitened similarity input** z = L^{-1}(x - a t) / sqrt(t), L = chol(Sigma);
  the Gaussian normaliser -0.5 log det(2 pi Sigma t) is added to the output.
* **Trace term** tr(Sigma Hess u): exact (d Hessian-vector products along the
  columns of L) or the **Hutchinson** estimator E_v[(Lv)' Hess u (Lv)] with
  Rademacher v, whose cost does not grow with d.
* **No box domain.** Collocation, the normalisation integral and VaR/ES all use one
  importance proposal q(x | t): a mixture over k = 0..K jumps of multivariate
  Student-t components centred at a t + k mu_J beta with scale c^2 (Sigma t + k sigma_J^2 beta beta').
  It uses only the PIDE coefficients, never the solution.
* **No tail-adaptive sampling**: the Stage 2c ablation found it gave no gain.
* **Portfolio VaR/ES** by self-normalised importance sampling: draw from q once,
  weight by exp(u)/q, and any portfolio w is then a dot product and a sort.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .jumpdiff import MultiMertonParams, multi_merton_logpdf


@dataclass
class NDConfig:
    width: int = 128
    depth: int = 4
    out_scale: float = 10.0
    n_gh: int = 48
    n_colloc: int = 2048
    trace: str = "auto"              # "exact" | "hutchinson" | "auto" (exact for d <= exact_max_d)
    exact_max_d: int = 4
    n_probes: int = 4                # Rademacher probes per point for Hutchinson
    n_ic: int = 4096
    n_norm: int = 4096               # proposal samples per normalisation step
    n_norm_t: int = 8                # horizons per normalisation step
    k_max: int = 3                   # jump counts covered by the proposal
    colloc_scale: float = 1.5        # proposal widening for collocation
    is_scale: float = 1.2            # proposal widening for normalisation and VaR/ES
    df: float = 5.0                  # Student-t degrees of freedom of the IS proposal (heavy: safe weights)
    colloc_df: float = 30.0          # near-Gaussian for collocation: df = 5 put points so far out that their
                                     # residuals dominated the loss
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_norm: float = 10.0
    steps: int = 20000
    lr: float = 2e-3
    lr_min: float = 2e-5
    log_every: int = 250
    seed: int = 0
    extra: dict = field(default_factory=dict)


class Proposal:
    """Mixture over k = 0..K jumps of multivariate Student-t; samples and log-density q(x | t)."""

    def __init__(self, p: MultiMertonParams, scale: float, df: float, k_max: int, device, floor: float = 0.05):
        f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=device)  # noqa: E731
        self.a, self.cov, self.beta = f(p.a), f(p.cov), f(p.beta)
        self.lam, self.mu_j, self.sigma_j = p.lam, p.mu_j, p.sigma_j
        self.scale, self.df, self.floor, self.d = scale, df, floor, p.d
        self.k = torch.arange(k_max + 1, device=device, dtype=torch.float32)
        self.bb = torch.outer(self.beta, self.beta)
        self.device = device

    def components(self, t: torch.Tensor):
        """t: (M,) -> loc (M,K,d), chol (M,K,d,d), log weights (M,K)."""
        k = self.k
        loc = self.a * t[:, None, None] + (k[:, None] * self.mu_j * self.beta)[None]
        S = self.scale**2 * (t[:, None, None, None] * self.cov + (k[:, None, None] * self.sigma_j**2 * self.bb)[None])
        chol = torch.linalg.cholesky(S)
        lt = self.lam * t[:, None]
        logpois = k * torch.log(lt) - lt - torch.lgamma(k + 1)
        logpi = torch.log(torch.clamp(torch.exp(logpois), min=self.floor))
        return loc, chol, logpi - torch.logsumexp(logpi, -1, keepdim=True)

    def sample(self, t: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """One draw per entry of t (shape (N,)), or n draws at the single time t (shape (1,))."""
        loc, chol, logpi = self.components(t)
        n = n or len(t)
        idx = torch.arange(n, device=self.device) if len(t) > 1 else torch.zeros(n, dtype=torch.long, device=self.device)
        comp = torch.multinomial(torch.exp(logpi[idx]), 1).squeeze(-1)
        eps = torch.randn(n, self.d, device=self.device, dtype=loc.dtype)
        chi2 = torch.distributions.Chi2(torch.tensor(self.df, device=self.device, dtype=loc.dtype)).sample((n,))
        y = torch.einsum("nij,nj->ni", chol[idx, comp], eps) * torch.sqrt(self.df / chi2)[:, None]
        return loc[idx, comp] + y

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (N,d); t: (N,) or a single time (1,)."""
        loc, chol, logpi = self.components(t)
        d, df = self.d, self.df
        diff = x[:, None, :] - loc                                     # (N,K,d), loc broadcasts if M = 1
        cb = chol.expand(diff.shape[0], -1, -1, -1) if chol.shape[0] == 1 else chol
        z = torch.linalg.solve_triangular(cb, diff[..., None], upper=False).squeeze(-1)
        maha = (z**2).sum(-1)
        logdet = 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        lp = (math.lgamma((df + d) / 2) - math.lgamma(df / 2) - 0.5 * d * math.log(df * math.pi)
              - 0.5 * logdet - 0.5 * (df + d) * torch.log1p(maha / df))
        return torch.logsumexp(logpi + lp, -1)


class NDPINN(nn.Module):
    def __init__(self, p: MultiMertonParams, t0: float, T: float, cfg: NDConfig, z_scale: float = 10.0):
        super().__init__()
        self.t0, self.T, self.cfg, self.z_scale, self.d = t0, T, cfg, z_scale, p.d
        L = np.linalg.cholesky(p.cov)
        self.register_buffer("a", torch.tensor(p.a, dtype=torch.float32))
        self.register_buffer("Linv", torch.tensor(np.linalg.inv(L), dtype=torch.float32))
        self.logdet = float(np.linalg.slogdet(p.cov)[1])
        layers, n_in = [], p.d + 1
        for _ in range(cfg.depth):
            layers += [nn.Linear(n_in, cfg.width), nn.Tanh()]
            n_in = cfg.width
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (N,d), t: (N,) -> log p (N,)."""
        z = (x - self.a * t[:, None]) @ self.Linv.T / torch.sqrt(t)[:, None] / self.z_scale
        tau = 2 * (torch.log(t) - math.log(self.t0)) / (math.log(self.T) - math.log(self.t0)) - 1
        g = self.net(torch.cat([z, tau[:, None]], -1)).squeeze(-1)
        return -0.5 * (self.d * torch.log(2 * math.pi * t) + self.logdet) + self.cfg.out_scale * g


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class NDTrainer:
    def __init__(self, p: MultiMertonParams, t0: float, T: float, cfg: NDConfig, device=None):
        self.p, self.t0, self.T, self.cfg, self.d = p, t0, T, cfg, p.d
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.model = NDPINN(p, t0, T, cfg).to(self.device)
        self.trace = cfg.trace if cfg.trace != "auto" else ("exact" if p.d <= cfg.exact_max_d else "hutchinson")

        f = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float32, device=self.device)  # noqa: E731
        self.a, self.cov, self.beta = f(p.a), f(p.cov), f(p.beta)
        self.L = f(np.linalg.cholesky(p.cov))
        gh_z, gh_w = np.polynomial.hermite.hermgauss(cfg.n_gh)
        self.jump_y = f(p.mu_j + math.sqrt(2) * p.sigma_j * gh_z)
        self.jump_w = f(gh_w / math.sqrt(math.pi))
        self.q_col = Proposal(p, cfg.colloc_scale, cfg.colloc_df, cfg.k_max, self.device)
        self.q_is = Proposal(p, cfg.is_scale, cfg.df, cfg.k_max, self.device)

        # fixed initial-condition data from the exact density at t0
        t0v = torch.full((1,), t0, device=self.device)
        self.x_ic = self.q_col.sample(t0v, cfg.n_ic)
        self.u_ic = f(multi_merton_logpdf(self.x_ic.cpu().numpy(), t0, p))
        self.history: list[dict] = []

    def _logt(self, n):
        return torch.empty(n, device=self.device).uniform_(math.log(self.t0), math.log(self.T))

    # ---- losses -------------------------------------------------------------
    def residual(self, x, t):
        p, m = self.p, self.model
        u = m(x, t)
        u_t, g = _grad(u, t), _grad(u, x)                                  # (N,), (N,d)
        drift = g @ self.a
        quad = ((g @ self.cov) * g).sum(-1)
        if self.trace == "exact":
            dirs = [self.L[:, i].expand_as(x) for i in range(self.d)]
            scale = 1.0
        else:
            dirs = [(torch.randint(0, 2, x.shape, device=self.device) * 2.0 - 1.0) @ self.L.T
                    for _ in range(self.cfg.n_probes)]
            scale = 1.0 / self.cfg.n_probes
        tr = sum((_grad((g * v).sum(-1), x) * v).sum(-1) for v in dirs) * scale
        K = len(self.jump_y)
        xs = (x[:, None, :] - self.jump_y[None, :, None] * self.beta).reshape(-1, self.d)
        u_shift = m(xs, t[:, None].expand(-1, K).reshape(-1)).reshape(len(x), K)
        S = (self.jump_w * torch.exp(torch.clamp(u_shift - u[:, None], max=30.0))).sum(-1)
        r = u_t + drift - 0.5 * (tr + quad) - p.lam * (S - 1.0)
        return t * r

    def losses(self):
        cfg, m = self.cfg, self.model
        t = torch.exp(self._logt(cfg.n_colloc))
        x = self.q_col.sample(t).requires_grad_(True)
        t = t.requires_grad_(True)
        l_pde = self.residual(x, t).pow(2).mean()

        l_ic = (m(self.x_ic, torch.full((len(self.x_ic),), self.t0, device=self.device)) - self.u_ic).pow(2).mean()

        tn = torch.exp(self._logt(cfg.n_norm_t)).repeat_interleave(cfg.n_norm // cfg.n_norm_t)
        with torch.no_grad():
            xn = self.q_is.sample(tn)
            lq = self.q_is.log_prob(xn, tn)
        lw = (m(xn, tn) - lq).reshape(cfg.n_norm_t, -1)
        log_mass = torch.logsumexp(lw, -1) - math.log(lw.shape[1])
        l_norm = log_mass.pow(2).mean()   # = (mass - 1)^2 near 1; (mass - 1)^2 overflows for an untrained net

        total = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_norm * l_norm
        return total, dict(pde=l_pde.item(), ic=l_ic.item(), norm=l_norm.item())

    # ---- training -----------------------------------------------------------
    def fit(self, callback=None):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr_min)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        for step in range(1, cfg.steps + 1):
            opt.zero_grad(set_to_none=True)
            total, parts = self.losses()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % cfg.log_every == 0 or step == 1:
                rec = dict(step=step, total=total.item(), time=time.perf_counter() - tic, **parts)
                if callback:
                    rec.update(callback(self))
                self.history.append(rec)
        self.train_time = time.perf_counter() - tic
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else float("nan")
        return self.history

    # ---- inference ------------------------------------------------------------
    @torch.no_grad()
    def logpdf(self, x: np.ndarray, t: float, chunk: int = 1 << 16) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x, float), dtype=torch.float32, device=self.device)
        out = [self.model(xc, torch.full((len(xc),), t, device=self.device)) for xc in x.split(chunk)]
        return torch.cat(out).cpu().numpy().astype(float)

    @torch.no_grad()
    def weighted_samples(self, t: float, n: int, chunk: int = 1 << 16):
        """Proposal draws X (n,d), log q and PINN log p at horizon t (numpy)."""
        tv = torch.full((1,), t, device=self.device)
        X = self.q_is.sample(tv, n)
        lq = torch.cat([self.q_is.log_prob(xc, tv) for xc in X.split(chunk)])
        lp = torch.cat([self.model(xc, torch.full((len(xc),), t, device=self.device)) for xc in X.split(chunk)])
        return X.cpu().numpy().astype(float), lq.cpu().numpy().astype(float), lp.cpu().numpy().astype(float)


def is_weights(logp: np.ndarray, logq: np.ndarray):
    """Self-normalised importance weights, the mass estimate int p and the effective sample size."""
    lw = logp - logq
    m = lw.max()
    w = np.exp(lw - m)
    mass = w.mean() * math.exp(m)
    w = w / w.sum()
    return w, mass, 1.0 / np.sum(w**2)
