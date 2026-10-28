"""Checks of the Stage 5 (amortised) machinery that do not need training."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, merton_var_es, multi_merton_logpdf  # noqa: E402
from pinn_tailrisk.pinn_amortised import (AmortisedTrainer, es_weighted_t, exact_logpdf_torch,  # noqa: E402
                                          params_at)
from pinn_tailrisk.pinn_nd import NDConfig, is_weights  # noqa: E402
from pinn_tailrisk.risk import var_es_from_weighted  # noqa: E402

P3 = MultiMertonParams.equicorrelated(3, sigma=[0.18, 0.24, 0.30], rho=0.4, beta=[0.8, 1.0, 1.2])
TH = [(0.7, 2.0, -0.08), (1.0, 5.0, -0.05), (1.5, 14.0, -0.03)]


class Exact(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, t, th):
        return exact_logpdf_torch(x, t, th, self.p, k_max=12)


@pytest.fixture(scope="module")
def tr():
    cfg = NDConfig(n_ic=64, n_colloc=256, n_norm=512, n_norm_t=4, trace="exact")
    return AmortisedTrainer(P3, 0.5 * DAY, 10 * DAY, cfg, device="cpu")


@pytest.mark.parametrize("th", TH)
def test_torch_exact_logpdf_matches_numpy(th):
    x = np.random.default_rng(0).normal(scale=0.05, size=(500, 3))
    t = 3 * DAY
    lt = exact_logpdf_torch(torch.tensor(x), torch.full((1,), t, dtype=torch.float64),
                            torch.tensor([th], dtype=torch.float64), P3, k_max=12).numpy()
    assert np.allclose(lt, multi_merton_logpdf(x, t, params_at(P3, th)), atol=1e-8)


def test_exact_density_has_zero_residual(tr):
    torch.set_default_dtype(torch.float64)
    try:
        tr.model = Exact(P3)
        for name in ("a", "cov", "beta", "L", "gh_z", "jump_w"):
            setattr(tr, name, getattr(tr, name).double())
        n = 256
        t = torch.exp(torch.empty(n).uniform_(math.log(DAY), math.log(10 * DAY)))
        th = tr.box.sample(n, "cpu")
        x = tr.q_col.sample(t.float(), th=th.float()).double().requires_grad_(True)
        r = tr.residual(x, t.requires_grad_(True), th.double()).detach()
    finally:
        torch.set_default_dtype(torch.float32)
    assert r.abs().median() < 1e-4       # 48-node quadrature is the only error (as in Stage 4)


@pytest.mark.parametrize("th", TH)
def test_proposal_covers_p_across_theta(tr, th):
    t = 5 * DAY
    tv, tht = torch.full((1,), t), torch.tensor([th], dtype=torch.float32)
    X = tr.q_is.sample(tv, 200_000, th=tht)
    lq = tr.q_is.log_prob(X, tv, tht).double().numpy()
    _, mass, ess = is_weights(multi_merton_logpdf(X.double().numpy(), t, params_at(P3, th)), lq)
    assert abs(mass - 1) < 0.02
    assert ess > 0.2 * len(X)


def test_torch_es_matches_numpy_and_exact():
    th, t, w = TH[1], 10 * DAY, np.full(3, 1 / 3)
    rng = np.random.default_rng(1)
    z = rng.normal(size=10_000)
    wt = rng.uniform(size=10_000)
    v, e = es_weighted_t(torch.tensor(z), torch.tensor(wt), 0.99)
    vn, en = var_es_from_weighted(z, wt, 0.99)
    assert abs(float(v) - vn) < 1e-12 and abs(float(e) - en) < 1e-12
    # exact-density IS through the torch path recovers the closed-form ES
    tv, tht = torch.full((1,), t), torch.tensor([th], dtype=torch.float32)
    tr_ = AmortisedTrainer(P3, 0.5 * DAY, 10 * DAY, NDConfig(trace="exact"), device="cpu")
    X = tr_.q_is.sample(tv, 400_000, th=tht)
    lw = exact_logpdf_torch(X.double(), tv.double(), tht.double(), P3) - tr_.q_is.log_prob(X, tv, tht).double()
    _, es = es_weighted_t(X.double() @ torch.tensor(w), torch.softmax(lw, 0), 0.99)
    es_true = merton_var_es(0.99, t, params_at(P3, th).portfolio(w))[1]
    assert abs(float(es) / es_true - 1) < 0.01
