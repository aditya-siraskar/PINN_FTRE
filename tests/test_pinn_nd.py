"""Checks of the Stage 4 machinery that do not need training."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, multi_merton_logpdf  # noqa: E402
from pinn_tailrisk.pinn_nd import NDConfig, NDTrainer, is_weights  # noqa: E402


class ExactLogPdf(torch.nn.Module):
    """Exact common-jump mixture density in torch, standing in for the network."""

    def __init__(self, p: MultiMertonParams, k_max: int = 12):
        super().__init__()
        f = lambda v: torch.tensor(np.asarray(v, float), dtype=torch.float64)  # noqa: E731
        self.a, self.cov, self.beta, self.p, self.k = f(p.a), f(p.cov), f(p.beta), p, torch.arange(k_max + 1.0, dtype=torch.float64)

    def forward(self, x, t):
        p, k = self.p, self.k
        loc = self.a * t[:, None, None] + (k[:, None] * p.mu_j * self.beta)[None]
        S = t[:, None, None, None] * self.cov + (k[:, None, None] * p.sigma_j**2 * torch.outer(self.beta, self.beta))[None]
        mvn = torch.distributions.MultivariateNormal(loc, covariance_matrix=S)
        lt = p.lam * t[:, None]
        logw = k * torch.log(lt) - lt - torch.lgamma(k + 1)
        return torch.logsumexp(logw + mvn.log_prob(x[:, None, :]), -1)


@pytest.fixture(scope="module")
def setup():
    torch.set_default_dtype(torch.float64)
    p = MultiMertonParams.equicorrelated(3, sigma=[0.18, 0.24, 0.30], rho=0.4, beta=[0.8, 1.0, 1.2])
    cfg = NDConfig(n_ic=64, n_colloc=256, n_norm=512, n_norm_t=4, trace="exact")
    tr = NDTrainer(p, 0.5 * DAY, 10 * DAY, cfg, device="cpu")
    yield p, tr
    torch.set_default_dtype(torch.float32)


def _resid(tr, n=256):
    t = torch.exp(torch.empty(n).uniform_(math.log(1 * DAY), math.log(10 * DAY))).double()
    x = tr.q_col.sample(t.float()).double().requires_grad_(True)
    for name in ("a", "cov", "beta", "L", "jump_y", "jump_w"):
        setattr(tr, name, getattr(tr, name).double())
    return tr.residual(x, t.requires_grad_(True)).detach()


@pytest.mark.parametrize("trace", ["exact", "hutchinson"])
def test_exact_density_has_zero_residual(setup, trace):
    p, tr = setup
    tr.model, tr.trace = ExactLogPdf(p), trace
    tr.cfg.n_probes = 64
    r = _resid(tr)
    # quadrature is the only error for "exact" (median 5e-4 / 1.6e-5 / 1.5e-7 at 24 / 48 / 96 nodes);
    # Hutchinson adds probe noise
    assert r.abs().median() < (1e-4 if trace == "exact" else 2e-2)


def test_proposal_is_a_density_and_covers_p(setup):
    p, tr = setup
    t = 5 * DAY
    tv = torch.full((1,), t)
    X = tr.q_is.sample(tv, 200_000)
    lq = tr.q_is.log_prob(X, tv).double().numpy()
    lp = multi_merton_logpdf(X.double().numpy(), t, p)
    _, mass, ess = is_weights(lp, lq)
    assert abs(mass - 1) < 0.02
    assert ess > 0.2 * len(X)
