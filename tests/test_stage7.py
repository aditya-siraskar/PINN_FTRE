"""Stage 7 data-only baseline: the likelihood loss and early stopping."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_tailrisk.data_only import DataOnlyTrainer  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, multi_merton_logpdf, simulate_multi_merton  # noqa: E402
from pinn_tailrisk.pinn_nd import NDConfig  # noqa: E402


def _trainer(steps):
    p = MultiMertonParams.equicorrelated(2, sigma=[0.18, 0.30], rho=0.4, beta=[0.8, 1.2])
    times = np.array([1.0, 5.0]) * DAY
    rng = np.random.default_rng(0)
    X = np.vstack([simulate_multi_merton(2000, t, p, rng) for t in times])
    ti = np.repeat([0, 1], 2000)
    cfg = NDConfig(n_ic=64, n_norm=1024, steps=steps, log_every=20, trace="exact")
    return p, times, X, ti, DataOnlyTrainer(p, 0.5 * DAY, 10 * DAY, cfg, X, ti, times, batch=512, n_val_norm=1024, device="cpu")


def test_exact_density_minimises_the_loss():
    """At the exact density the estimated log Z is ~0 and the NLL is the samples' exact NLL."""
    p, times, X, ti, tr = _trainer(1)
    nll_exact = -np.mean([multi_merton_logpdf(X[ti == j], t, p).mean() for j, t in enumerate(times)])

    class Exact(torch.nn.Module):
        def forward(self, x, t):
            out = np.empty(len(x))
            tn = t.numpy()
            for tv in np.unique(tn):
                m = tn == tv
                out[m] = multi_merton_logpdf(x[m].numpy().astype(float), float(tv), p)
            return torch.as_tensor(out, dtype=torch.float32)

    tr.model = Exact()
    assert abs(tr.val_nll() - nll_exact) < 0.05 * abs(nll_exact) + 0.1


def test_fit_improves_and_keeps_best_state():
    *_, tr = _trainer(120)
    v0 = tr.val_nll()
    tr.fit()
    assert tr.val_nll() < v0
    assert tr.val_nll() == pytest.approx(tr.best[0], rel=1e-5)
    assert 1 <= tr.best_step <= 120
