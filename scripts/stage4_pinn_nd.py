"""Stage 4: d-asset PINN on the common-jump PIDE, checked against the exact mixture.

One run per dimension d. Portfolio VaR/ES come from self-normalised importance
sampling with the PINN density; two baselines use the same number of draws:
  * Monte Carlo from the true model (what a risk desk would run), and
  * importance sampling with the *exact* density (the proposal's noise floor, so
    PINN error = density error + this floor).
Ground truth: the exact 1-D law of w . X (`MultiMertonParams.portfolio`).

Produces figures/stage4_pinn_nd_d<d>/fig01..fig04 and results/stage4_pinn_nd_d<d>/{metrics.json,
history.csv, risk_table.csv, model.pt}.
Run:  python scripts/stage4_pinn_nd.py --d 5 [--trace exact|hutchinson] [--steps N] [--seed N] [--tag name]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import figs_pinn1d as F  # noqa: E402
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config, save_results  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, merton_var_es, multi_merton_logpdf, simulate_multi_merton  # noqa: E402
from pinn_tailrisk.pinn_nd import NDConfig, NDTrainer, is_weights  # noqa: E402
from pinn_tailrisk.risk import var_es_from_samples, var_es_from_weighted  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage4_pinn_nd.yaml")
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--trace", choices=["exact", "hutchinson"])
ap.add_argument("--steps", type=int)
ap.add_argument("--seed", type=int)
ap.add_argument("--tag", default="")
args = ap.parse_args()

d = args.d
STAGE = f"stage4_pinn_nd_d{d}" + (f"_{args.tag}" if args.tag else "")
base, s3, c4 = load_config("configs/base.yaml"), load_config("configs/stage3_multiasset.yaml"), load_config(args.config)
pc, ev = c4["pinn_nd"], c4["eval"]
for k in ("trace", "steps", "seed"):
    if getattr(args, k) is not None:
        pc[k] = getattr(args, k)
cfg = NDConfig(**pc)
sc = s3["scaling"]
mp = MultiMertonParams.equicorrelated(
    d, sigma=np.linspace(*sc["sigma_range"], d) if d > 1 else [sc["sigma_range"][0]], rho=sc["rho"],
    a=base["merton"]["a"], beta=np.linspace(*sc["beta_range"], d) if d > 1 else [1.0],
    **{k: base["merton"][k] for k in ("lam", "mu_j", "sigma_j")})
dom = base["domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H, ALPHAS = base["eval"]["horizons_days"], base["eval"]["alphas"]
P.use_style()

# portfolios: equal weight, random long-only, random long-short (all sum to 1)
prng = np.random.default_rng(ev["portfolio_seed"])
if d == 1:
    W, kinds = np.ones((1, 1)), ["single asset"]
else:
    ls = prng.normal(size=(ev["n_long_short"], d))
    ls[:, 0] = 1 - ls[:, 1:].sum(1)
    W = np.vstack([np.full(d, 1 / d), prng.dirichlet(np.ones(d), ev["n_long_only"]), ls])
    kinds = ["equal weight"] + ["long-only"] * ev["n_long_only"] + ["long-short"] * ev["n_long_short"]
exact = {(h, a, i): merton_var_es(a, h * DAY, mp.portfolio(w)) for h in H for a in ALPHAS for i, w in enumerate(W)}


def track(tr: NDTrainer) -> dict:
    out = {}
    for h in (1, 10):
        X, lq, lp = tr.weighted_samples(h * DAY, ev["n_track"])
        w, _, _ = is_weights(lp, lq)
        _, es = var_es_from_weighted(X @ W[0], w, 0.99)
        out[f"es99_rel_{h}d"] = abs(es - exact[(h, 0.99, 0)][1]) / exact[(h, 0.99, 0)][1]
    return out


tr = NDTrainer(mp, t0, T, cfg)
print(f"d={d}  device={tr.device}  trace={tr.trace}  params={sum(q.numel() for q in tr.model.parameters())}  steps={cfg.steps}")
tr.fit(callback=track)
print(f"trained in {tr.train_time:.0f}s  peak GPU memory {tr.peak_mem_mb:.0f} MB")
hist = pd.DataFrame(tr.history)
(RESULTS / STAGE).mkdir(parents=True, exist_ok=True)
hist.to_csv(RESULTS / STAGE / "history.csv", index=False)
torch.save(dict(state=tr.model.state_dict(), cfg=pc, d=d, cov=mp.cov, beta=mp.beta), RESULTS / STAGE / "model.pt")

# ---------------------------------------------------------------- evaluation
rng = np.random.default_rng(1000 + cfg.seed)
rows, dens, timing, is_diag = [], {}, {}, {}
for h in H:
    t = h * DAY
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync(); tic = time.perf_counter()
    X, lq, lp = tr.weighted_samples(t, ev["n_is"])
    sync(); t_pinn_draw = time.perf_counter() - tic
    w_pinn, mass, ess = is_weights(lp, lq)
    w_ex, _, ess_ex = is_weights(multi_merton_logpdf(X, t, mp), lq)
    tic = time.perf_counter()
    Xmc = simulate_multi_merton(ev["n_is"], t, mp, rng)
    t_mc_draw = time.perf_counter() - tic
    tic = time.perf_counter()
    for i, wv in enumerate(W):
        z, zmc = X @ wv, Xmc @ wv
        for a in ALPHAS:
            ve, ee = exact[(h, a, i)]
            vp, ep = var_es_from_weighted(z, w_pinn, a)
            vi, ei = var_es_from_weighted(z, w_ex, a)
            vm, em = var_es_from_samples(zmc, a)
            rows.append(dict(h=h, alpha=a, port=i, kind=kinds[i], var_exact=ve, es_exact=ee,
                             var_pinn=vp, es_pinn=ep, var_is_exact=vi, es_is_exact=ei, var_mc=vm, es_mc=em))
    t_query = (time.perf_counter() - tic) / len(W)
    timing[f"{h}d"] = dict(pinn_draw_s=t_pinn_draw, mc_draw_s=t_mc_draw, per_portfolio_s=t_query)
    is_diag[f"{h}d"] = dict(pinn_mass=mass, ess_frac_pinn=ess / ev["n_is"], ess_frac_exact=ess_ex / ev["n_is"])

    # density error at exact samples; "tail" = equal-weight portfolio in its worst 1%
    Xs = simulate_multi_merton(ev["n_density"], t, mp, rng)
    err = np.abs(tr.logpdf(Xs, t) - multi_merton_logpdf(Xs, t, mp))
    tail = Xs @ W[0] <= -exact[(h, 0.99, 0)][0]
    dens[f"{h}d"] = dict(mae_log=err.mean(), tail_mae_log=err[tail].mean(), p95_log=np.quantile(err, 0.95))
risk = pd.DataFrame(rows)
for m in ("pinn", "is_exact", "mc"):
    risk[f"var_rel_{m}"] = (risk[f"var_{m}"] - risk.var_exact) / risk.var_exact
    risk[f"es_rel_{m}"] = (risk[f"es_{m}"] - risk.es_exact) / risk.es_exact
risk.to_csv(RESULTS / STAGE / "risk_table.csv", index=False)
med = risk.groupby(["h", "alpha"])[[c for c in risk if "_rel_" in c]].agg(lambda s: s.abs().median()).reset_index()

# ---------------------------------------------------------------- fig01: training
F.plot_training(hist, STAGE)

# ---------------------------------------------------------------- fig02: density slices
beta_hat = mp.beta / np.linalg.norm(mp.beta)
lines = [("along the jump direction $\\beta$", beta_hat, np.linspace(-0.5, 0.2, 400))]
if d > 1:
    v = np.eye(d)[0] - beta_hat[0] * beta_hat
    lines.append(("orthogonal to $\\beta$", v / np.linalg.norm(v), np.linspace(-0.25, 0.25, 400)))
fig, axes = plt.subplots(len(lines), 3, figsize=(11.5, 3.2 * len(lines)), squeeze=False)
for r, (lab, u, s) in enumerate(lines):
    for j, h in enumerate(H):
        t, ax = h * DAY, axes[r, j]
        Xl = mp.a * t + s[:, None] * u
        le = multi_merton_logpdf(Xl, t, mp) / np.log(10)
        keep = le > le.max() - 10
        ax.plot(100 * s[keep], tr.logpdf(Xl, t)[keep] / np.log(10), **P.style("pinn"))
        ax.plot(100 * s[keep], le[keep], **P.style("exact", lw=1.4))
        ax.set(title=f"{h}-day, {lab}", xlabel="distance from the drift point (%)")
    axes[r, 0].set_ylabel(r"$\log_{10}$ joint density")
axes[0, 0].legend(loc="lower center")
P.panel_labels(axes.ravel())
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_density_slices")

# ---------------------------------------------------------------- fig03: ES99 for every portfolio
IS_FLOOR = dict(color=P.MUTED, marker="d", ls="", ms=5, label="IS, exact density (sampling floor)")
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
for ax, h in zip(axes, H):
    r = risk[(risk.h == h) & (risk.alpha == 0.99)]
    ax.plot(100 * r.es_exact, 100 * r.es_rel_mc.abs(), marker="^", ls="", ms=6, color=P.METHOD["mc"]["color"],
            label=f"Monte Carlo ({ev['n_is'] // 1024}k paths)")
    ax.plot(100 * r.es_exact, 100 * r.es_rel_is_exact.abs(), **IS_FLOOR)
    ax.plot(100 * r.es_exact, 100 * r.es_rel_pinn.abs(), marker="o", ls="", ms=6, color=P.METHOD["pinn"]["color"],
            label="PINN (IS)")
    ax.axhline(1.0, color=P.AXIS, lw=0.8)
    ax.set(yscale="log", title=f"{h}-day", xlabel="exact ES$_{99}$ of the portfolio (%)")
axes[0].set_ylabel("|ES$_{99}$ error| vs exact (%)")
axes[-1].text(1.0, 1.0, "1%", transform=axes[-1].get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=P.INK_2)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
P.panel_labels(axes)
fig.tight_layout(rect=(0, 0, 1, 0.91))
P.savefig(fig, STAGE, "fig03_portfolio_es")

# ---------------------------------------------------------------- fig04: median VaR/ES error vs alpha
F.plot_var_es_error(med, H, ALPHAS, STAGE, methods=("pinn", "mc"),
                    ylabel="{} error| vs exact (%), median over portfolios", name="fig04_var_es_error")

save_results(STAGE, "metrics", dict(
    d=d, trace=tr.trace, pinn_config=pc, cov=mp.cov.tolist(), beta=mp.beta.tolist(), n_portfolios=len(W),
    train_time_s=tr.train_time, peak_mem_mb=tr.peak_mem_mb, n_params=sum(q.numel() for q in tr.model.parameters()),
    timing=timing, importance_sampling=is_diag, density_error=dens,
    median_abs_rel=med.to_dict(orient="records")))
print(med[["h", "alpha", "es_rel_pinn", "es_rel_is_exact", "es_rel_mc"]].to_string(index=False))
print({h: {k: f"{v:.2e}" for k, v in x.items()} for h, x in dens.items()})
print(is_diag)
