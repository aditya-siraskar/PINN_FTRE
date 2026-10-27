"""Physics-informed network for the forward jump-diffusion PIDE (1-D, Phase 1).

Design choices (each is a switch in the config so it can be ablated):

* **Log-density output.** The network returns u = log p, so p > 0 by
  construction and 1e-4-level tail densities are O(1) numbers for the network.
* **Similarity-variable inputs.** x enters as z = (x - a t) / (sigma sqrt t),
  t as normalised log t. The diffusive bulk then has width O(1) at every t,
  instead of collapsing as t -> t0. The known Gaussian normaliser
  -0.5 log(2 pi sigma^2 t) is added to the output.
* **Log-form residual.** Dividing the PIDE by p gives
      r = u_t + a u_x - 0.5 sigma^2 (u_xx + u_x^2) - lam (S - 1),
      S = int exp(u(x - y) - u(x)) f_Y(y) dy,
  which weighs a tail point as much as a bulk point (relative error).
  r is multiplied by t (a dimensionless residual). The "linear" option
  (r * p) is the standard PINN residual for the ablation.
* **Gauss-Hermite quadrature** for the jump integral S (Y Gaussian).
* **Start at t0 > 0** from the known short-time density (no Dirac delta).
* **Normalisation penalty** int p dx = 1, and a decay condition at both ends.
* **Tail-adaptive collocation**: a share of points is drawn around the model's
  own current 1% quantile, refreshed during training.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .jumpdiff import MertonParams, merton_logpdf


@dataclass
class PINNConfig:
    width: int = 64
    depth: int = 4
    out_scale: float = 10.0
    residual: str = "log"            # "log" | "linear"
    n_gh: int = 48                   # Gauss-Hermite nodes for the jump integral (24 is too few at 1 day)
    n_colloc: int = 4096
    tail_share: float = 0.4          # share of collocation points near the tail quantile
    tail_level: float = 0.01
    adapt_every: int = 250
    n_ic: int = 1024
    n_norm_t: int = 16
    n_norm_x: int = 1024
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_norm: float = 10.0
    w_bc: float = 1.0
    steps: int = 20000
    lr: float = 2e-3
    lr_min: float = 2e-5
    log_every: int = 250
    seed: int = 0
    extra: dict = field(default_factory=dict)


class TailPINN(nn.Module):
    def __init__(self, p: MertonParams, t0: float, T: float, cfg: PINNConfig, z_scale: float = 10.0):
        super().__init__()
        self.p, self.t0, self.T, self.cfg, self.z_scale = p, t0, T, cfg, z_scale
        layers, n_in = [], 2
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
        p = self.p
        s = p.sigma * torch.sqrt(t)
        z = (x - p.a * t) / s / self.z_scale
        tau = 2 * (torch.log(t) - math.log(self.t0)) / (math.log(self.T) - math.log(self.t0)) - 1
        g = self.net(torch.stack([z, tau], -1)).squeeze(-1)
        return -0.5 * torch.log(2 * math.pi * s**2) + self.cfg.out_scale * g


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class Trainer:
    def __init__(self, p: MertonParams, x_min: float, x_max: float, t0: float, T: float,
                 cfg: PINNConfig, device=None):
        self.p, self.x_min, self.x_max, self.t0, self.T, self.cfg = p, x_min, x_max, t0, T, cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.model = TailPINN(p, t0, T, cfg).to(self.device)

        gh_z, gh_w = np.polynomial.hermite.hermgauss(cfg.n_gh)
        self.jump_y = torch.tensor(p.mu_j + math.sqrt(2) * p.sigma_j * gh_z, dtype=torch.float32, device=self.device)
        self.jump_w = torch.tensor(gh_w / math.sqrt(math.pi), dtype=torch.float32, device=self.device)

        # fixed initial-condition data and normalisation grid
        x_ic = np.linspace(x_min, x_max, cfg.n_ic)
        self.x_ic = self._t(x_ic)
        self.u_ic = self._t(merton_logpdf(x_ic, t0, p))
        self.x_norm = torch.linspace(x_min, x_max, cfg.n_norm_x, device=self.device)
        # tail-quantile table used by the adaptive sampler (log t -> q); start from the bulk scale
        self.q_logt = np.linspace(math.log(t0), math.log(T), 16)
        self.q_tail = np.full(16, p.a * T - 2.3 * p.sigma * math.sqrt(T))
        self.history: list[dict] = []

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)

    # ---- sampling ---------------------------------------------------------
    def sample_collocation(self):
        n, cfg = self.cfg.n_colloc, self.cfg
        logt = self.rng.uniform(math.log(self.t0), math.log(self.T), n)
        x = self.rng.uniform(self.x_min, self.x_max, n)
        n_tail = int(cfg.tail_share * n)
        if n_tail:
            q = np.interp(logt[:n_tail], self.q_logt, self.q_tail)
            width = 0.5 * np.abs(q - self.p.a * np.exp(logt[:n_tail])) + 1e-3
            x[:n_tail] = np.clip(q + width * self.rng.standard_normal(n_tail), self.x_min, self.x_max)
        x, t = self._t(x), self._t(np.exp(logt))
        return x.requires_grad_(True), t.requires_grad_(True)

    @torch.no_grad()
    def update_tail_quantiles(self):
        ts = torch.exp(self._t(self.q_logt))
        X, Tt = torch.meshgrid(self.x_norm, ts, indexing="ij")
        pdf = torch.exp(self.model(X.reshape(-1), Tt.reshape(-1))).reshape(X.shape)
        dx = self.x_norm[1] - self.x_norm[0]
        cdf = torch.cumsum(pdf, 0) * dx
        cdf = cdf / cdf[-1:]
        idx = torch.searchsorted(cdf.T.contiguous(), torch.full((len(ts), 1), self.cfg.tail_level, device=self.device))
        self.q_tail = self.x_norm[idx.squeeze(-1).clamp(max=len(self.x_norm) - 1)].cpu().numpy()

    # ---- losses -------------------------------------------------------------
    def residual(self, x, t):
        p, m = self.p, self.model
        u = m(x, t)
        u_t, u_x = _grad(u, t), _grad(u, x)
        u_xx = _grad(u_x, x)
        xs = (x[:, None] - self.jump_y[None]).reshape(-1)
        ts = t[:, None].expand(-1, len(self.jump_y)).reshape(-1)
        u_shift = m(xs, ts).reshape(len(x), -1)
        # p = 0 outside the domain (same convention as the FD solver's Dirichlet edges)
        inside = ((xs >= self.x_min) & (xs <= self.x_max)).reshape(len(x), -1).float()
        S = (self.jump_w[None] * inside * torch.exp(torch.clamp(u_shift - u[:, None], max=30.0))).sum(-1)
        r = u_t + p.a * u_x - 0.5 * p.sigma**2 * (u_xx + u_x**2) - p.lam * (S - 1.0)
        r = t * r
        if self.cfg.residual == "linear":
            r = r * torch.exp(u - u.detach().max())
        return r

    def losses(self):
        cfg, m = self.cfg, self.model
        x, t = self.sample_collocation()
        l_pde = self.residual(x, t).pow(2).mean()

        l_ic = (m(self.x_ic, torch.full_like(self.x_ic, self.t0)) - self.u_ic).pow(2).mean()

        logt = torch.empty(cfg.n_norm_t, device=self.device).uniform_(math.log(self.t0), math.log(self.T))
        X, Tt = torch.meshgrid(self.x_norm, torch.exp(logt), indexing="ij")
        pdf = torch.exp(m(X.reshape(-1), Tt.reshape(-1))).reshape(X.shape)
        mass = torch.trapezoid(pdf, self.x_norm, dim=0)
        l_norm = (mass - 1.0).pow(2).mean()

        tb = torch.exp(logt).requires_grad_(True)
        xl = torch.full_like(tb, self.x_min).requires_grad_(True)
        xr = torch.full_like(tb, self.x_max).requires_grad_(True)
        ul_x, ur_x = _grad(m(xl, tb), xl), _grad(m(xr, tb), xr)
        l_bc = (torch.relu(-ul_x).pow(2) + torch.relu(ur_x).pow(2)).mean() / 100.0

        total = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_norm * l_norm + cfg.w_bc * l_bc
        return total, dict(pde=l_pde.item(), ic=l_ic.item(), norm=l_norm.item(), bc=l_bc.item())

    # ---- training -----------------------------------------------------------
    def fit(self, callback=None):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr_min)
        tic = time.perf_counter()
        for step in range(1, cfg.steps + 1):
            if cfg.tail_share and step % cfg.adapt_every == 0:
                self.update_tail_quantiles()
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
        return self.history

    @torch.no_grad()
    def logpdf(self, x: np.ndarray, t: float) -> np.ndarray:
        xt = self._t(x)
        return self.model(xt, torch.full_like(xt, t)).cpu().numpy().astype(float)
