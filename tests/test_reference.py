import numpy as np
import pytest

from pinn_tailrisk.fd import solve_pide_1d
from pinn_tailrisk.fd2d import solve_pide_2d
from pinn_tailrisk.jumpdiff import (DAY, MertonParams, MultiMertonParams, merton_cdf, merton_pdf,
                                    merton_var_es, multi_merton_logpdf, simulate_merton,
                                    simulate_multi_merton)
from pinn_tailrisk.risk import gaussian_var_es, var_es_from_pdf, var_es_from_samples, var_es_from_weighted

P = MertonParams()
T = 10 * DAY


def test_density_integrates_to_one():
    x = np.linspace(-1, 1, 20001)
    assert np.trapezoid(merton_pdf(x, T, P), x) == pytest.approx(1.0, abs=1e-8)
    assert merton_cdf(np.array([1.0]), T, P)[0] == pytest.approx(1.0, abs=1e-10)


def test_var_es_exact_matches_grid_and_mc():
    x = np.linspace(-1, 0.5, 40001)
    for a in (0.95, 0.99):
        v, e = merton_var_es(a, T, P)
        vg, eg = var_es_from_pdf(x, merton_pdf(x, T, P), a)
        assert vg == pytest.approx(v, rel=1e-5) and eg == pytest.approx(e, rel=1e-5)
        vm, em = var_es_from_samples(simulate_merton(2_000_000, T, P, np.random.default_rng(1)), a)
        assert vm == pytest.approx(v, rel=0.02) and em == pytest.approx(e, rel=0.02)


def test_no_jumps_reduces_to_gaussian():
    q = MertonParams(lam=0.0)
    v, e = merton_var_es(0.99, T, q)
    vg, eg = gaussian_var_es(q.a * T, q.sigma * np.sqrt(T), 0.99)
    assert v == pytest.approx(vg, rel=1e-8) and e == pytest.approx(eg, rel=1e-8)


def test_fd_matches_exact():
    x, sol, _ = solve_pide_1d(P, -1.0, 0.5, 2001, 0.5 * DAY, [T], 0.05 * DAY)
    l1 = np.trapezoid(np.abs(sol[T] - merton_pdf(x, T, P)), x)
    assert l1 < 1e-3
    assert var_es_from_pdf(x, sol[T], 0.99)[1] == pytest.approx(merton_var_es(0.99, T, P)[1], rel=1e-4)


def test_weighted_var_es_matches_samples():
    x = simulate_merton(400_000, T, P, np.random.default_rng(3))
    v, e = var_es_from_weighted(x, np.ones_like(x), 0.99)
    vs, es = var_es_from_samples(x, 0.99)
    assert v == pytest.approx(vs, rel=1e-3) and e == pytest.approx(es, rel=1e-3)


def test_fd2d_matches_exact():
    mp = MultiMertonParams.equicorrelated(2, sigma=[0.20, 0.25], rho=0.5, beta=[1.0, 1.2])
    x, y, sol, _ = solve_pide_2d(mp, (-1.0, -1.0), (0.45, 0.45), 151, 0.5 * DAY, [T], 0.05 * DAY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    h = (x[1] - x[0]) * (y[1] - y[0])
    exact = np.exp(multi_merton_logpdf(np.stack([X, Y], -1), T, mp))
    assert sol[T].sum() * h == pytest.approx(1.0, abs=1e-4)
    assert np.abs(sol[T] - exact).sum() * h < 2e-2
    w = np.array([0.5, 0.5])
    es = var_es_from_weighted(w[0] * X + w[1] * Y, sol[T] * h, 0.99)[1]
    assert es == pytest.approx(merton_var_es(0.99, T, mp.portfolio(w))[1], rel=3e-3)


def test_multi_asset_portfolio_projection():
    mp = MultiMertonParams.equicorrelated(3, rho=0.4, beta=[1.0, 0.8, 1.2])
    w = np.array([0.5, 0.3, 0.2])
    xs = simulate_multi_merton(1_000_000, T, mp, np.random.default_rng(2)) @ w
    v, e = merton_var_es(0.99, T, mp.portfolio(w))
    assert var_es_from_samples(xs, 0.99)[1] == pytest.approx(e, rel=0.02)


def test_multi_logpdf_marginal_consistency():
    mp = MultiMertonParams.equicorrelated(1)
    x = np.linspace(-0.3, 0.2, 7)
    np.testing.assert_allclose(multi_merton_logpdf(x[:, None], T, mp),
                               np.log(merton_pdf(x, T, mp.marginal(0))), rtol=1e-8)
