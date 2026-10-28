"""Data-only baseline for Stage 7 (RQ3): the Stage 4 network fitted to samples by maximum likelihood.

Same architecture (NDPINN: whitened input, Gaussian normaliser, out_scale), same optimiser,
learning-rate schedule, step count and normaliser estimator as the PINN. The only change is
the loss: the PIDE residual and the exact initial condition are replaced by the negative
log-likelihood of N exact samples spread over a grid of times,

    NLL = -mean_i [ u(x_i, t_i) - log Z(t_i) ],   log Z(t) = log E_q[exp(u(X, t)) / q(X | t)],

with log Z estimated on fresh proposal draws every step (the PINN's normalisation term, used
here as the likelihood's normaliser). A held-out split gives early stopping: the state with the
best validation NLL is kept.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .jumpdiff import MultiMertonParams
from .pinn_nd import NDConfig, NDTrainer


class DataOnlyTrainer(NDTrainer):
    def __init__(self, p: MultiMertonParams, t0: float, T: float, cfg: NDConfig, X: np.ndarray, ti: np.ndarray,
                 times: np.ndarray, val_frac: float = 0.2, batch: int = 2048, n_val_norm: int = 4096, device=None):
        """X (N,d) samples; ti (N,) index of each sample's time in `times`."""
        super().__init__(p, t0, T, cfg, device)
        f = lambda v: torch.as_tensor(np.asarray(v), device=self.device)  # noqa: E731
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(X))
        n_val = max(int(val_frac * len(X)), len(times))
        va, trn = perm[:n_val], perm[n_val:]
        self.times = f(np.asarray(times, float)).float()
        self.Xtr, self.jtr = f(X[trn]).float(), f(ti[trn]).long()
        self.Xva, self.jva = f(X[va]).float(), f(ti[va]).long()
        self.batch = batch
        with torch.no_grad():   # fixed draws so the validation NLL is comparable across steps
            tv = self.times.repeat_interleave(n_val_norm)
            self.val_xn = self.q_is.sample(tv)
            self.val_lq = self.q_is.log_prob(self.val_xn, tv)
        self.n_val_norm = n_val_norm
        self.best = (math.inf, None, 0)
        self.step = 0

    def _log_z(self, xn, lq, per):
        tn = self.times.repeat_interleave(per)
        lw = (self.model(xn, tn) - lq).reshape(len(self.times), per)
        return torch.logsumexp(lw, -1) - math.log(per)

    def losses(self):
        cfg = self.cfg
        self.step += 1
        idx = torch.randint(len(self.Xtr), (min(self.batch, len(self.Xtr)),), device=self.device)
        x, j = self.Xtr[idx], self.jtr[idx]
        per = max(cfg.n_norm // len(self.times), 1)
        with torch.no_grad():
            tn = self.times.repeat_interleave(per)
            xn = self.q_is.sample(tn)
            lq = self.q_is.log_prob(xn, tn)
        log_z = self._log_z(xn, lq, per)
        nll = -(self.model(x, self.times[j]) - log_z[j]).mean()
        return nll, dict(nll=nll.item(), norm=log_z.pow(2).mean().item())

    @torch.no_grad()
    def val_nll(self) -> float:
        log_z = self._log_z(self.val_xn, self.val_lq, self.n_val_norm)
        return float(-(self.model(self.Xva, self.times[self.jva]) - log_z[self.jva]).mean())

    def _track(self, tr) -> dict:
        v = self.val_nll()
        if v < self.best[0]:
            self.best = (v, copy.deepcopy(self.model.state_dict()), self.step)
        return dict(val_nll=v)

    def fit(self, callback=None):
        super().fit(callback=self._track)
        self._track(self)                               # the final state is a candidate too
        self.model.load_state_dict(self.best[1])
        self.best_step = self.best[2]
        return self.history
