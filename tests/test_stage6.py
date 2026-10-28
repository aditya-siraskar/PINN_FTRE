"""Stage 6 building blocks: calibration recovers known parameters; backtests and benchmarks are correct."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_tailrisk import backtest as B  # noqa: E402
from pinn_tailrisk import benchmarks as BM  # noqa: E402
from pinn_tailrisk.calibrate import ewma_scale, fit_merton, merton_loglik  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, multi_merton_logpdf, simulate_multi_merton  # noqa: E402

P5 = MultiMertonParams.equicorrelated(5, sigma=[0.18, 0.2, 0.22, 0.25, 0.3], rho=0.4, beta=[0.8, 0.9, 1, 1.1, 1.2],
                                      lam=8.0, mu_j=-0.04, sigma_j=0.04)


def test_loglik_matches_numpy_with_vol_scale():
    rng = np.random.default_rng(0)
    x = simulate_multi_merton(500, DAY, P5, rng)
    s = rng.uniform(0.5, 2.0, 500)
    f = lambda v: torch.tensor(np.asarray(v, float))  # noqa: E731
    ll = merton_loglik(f(x), f(s), f(P5.a), f(P5.cov), f(P5.beta), f(P5.lam), f(P5.mu_j), f(P5.sigma_j), k_max=8).numpy()
    ref = [multi_merton_logpdf(x[i], DAY, MultiMertonParams(a=P5.a, cov=s[i]**2 * P5.cov, lam=P5.lam, mu_j=P5.mu_j,
                                                             sigma_j=P5.sigma_j, beta=P5.beta)) for i in range(20)]
    assert np.allclose(ll[:20], ref, atol=1e-9)


def test_mle_recovers_parameters():
    x = simulate_multi_merton(4000, DAY, P5, np.random.default_rng(1))
    q, info = fit_merton(x, se=True)
    assert abs(q.lam - P5.lam) < 3 * info["se"]["lam"]
    assert abs(q.mu_j - P5.mu_j) < 3 * info["se"]["mu_j"]
    assert np.allclose(np.sqrt(np.diag(q.cov)), np.sqrt(np.diag(P5.cov)), rtol=0.08)
    assert np.allclose(q.beta, P5.beta / P5.beta.mean(), atol=0.1)


def test_ewma_scale_has_no_lookahead():
    x = np.random.default_rng(2).normal(size=(400, 3)) * 0.01
    s1 = ewma_scale(x, np.eye(3) * 1e-4, warmup=100)
    x2 = x.copy()
    x2[300:] *= 10
    s2 = ewma_scale(x2, np.eye(3) * 1e-4, warmup=100)
    assert np.allclose(s1[:301], s2[:301]) and s2[301] > s1[301]


def test_kupiec_and_christoffersen():
    hits = np.zeros(1000, bool)
    hits[::100] = True                                  # exactly 1%, evenly spaced
    assert B.kupiec(hits, 0.99)["p_uc"] > 0.99
    assert B.christoffersen(hits, 0.99)["p_ind"] > 0.5
    clustered = np.zeros(1000, bool)
    clustered[500:510] = True
    assert B.christoffersen(clustered, 0.99)["p_ind"] < 1e-3
    assert B.kupiec(np.r_[np.ones(30), np.zeros(970)].astype(bool), 0.99)["p_uc"] < 1e-5


def test_fz0_prefers_the_true_forecast():
    y = np.random.default_rng(3).standard_t(5, 200_000) * math.sqrt(3 / 5)
    v, e = BM.t_var_es(0.975, 5.0)
    true = B.fz0_loss(y, np.full_like(y, v), np.full_like(y, e), 0.975).mean()
    for f in (0.8, 1.25):
        assert B.fz0_loss(y, np.full_like(y, f * v), np.full_like(y, f * e), 0.975).mean() > true


def test_t_var_es_and_gpd_against_samples():
    rng = np.random.default_rng(4)
    z = rng.standard_t(6, 2_000_000) * math.sqrt(4 / 6)
    v, e = BM.t_var_es(0.99, 6.0)
    vs, es = BM.sample_var_es(z, 0.99)
    assert abs(v / vs - 1) < 0.01 and abs(e / es - 1) < 0.01
    vg, eg = BM.gpd_tail(-z[:200_000], 0.99)
    assert abs(vg / vs - 1) < 0.03 and abs(eg / es - 1) < 0.05


def test_dcc_filter_unit_diagonal_and_draw_moments():
    e = np.random.default_rng(5).normal(size=(300, 3))
    R = BM.dcc_filter(e, 0.05, 0.9, np.cov(e.T))
    assert R.shape == (301, 3, 3) and np.allclose(np.diagonal(R, axis1=1, axis2=2), 1)
    Rc = np.array([[1, 0.5], [0.5, 1]])
    z = BM.mvt_draws(400_000, Rc, 6.0, np.random.default_rng(6))
    assert np.allclose(np.cov(z.T), Rc, atol=0.02)
