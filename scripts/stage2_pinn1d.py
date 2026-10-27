"""Stage 2b: train the 1-D PINN on the Merton PIDE and compare it with exact and FD.

Produces figures/stage2_pinn1d/fig01..fig05, results/stage2_pinn1d/{metrics.json, history.csv, model.pt}.
Run:  python scripts/stage2_pinn1d.py [--config configs/stage2_pinn1d.yaml] [--steps N] [--tag name]
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
from pinn_tailrisk.fd import solve_pide_1d  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MertonParams, merton_cdf, merton_logpdf, merton_var_es  # noqa: E402
from pinn_tailrisk.pinn import PINNConfig, Trainer  # noqa: E402
from pinn_tailrisk.risk import bulk_tail_masks, var_es_from_pdf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage2_pinn1d.yaml")
ap.add_argument("--steps", type=int)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int)
args = ap.parse_args()

STAGE = "stage2_pinn1d" + (f"_{args.tag}" if args.tag else "")
base = load_config("configs/base.yaml")
pc = load_config(args.config)["pinn"]
if args.steps:
    pc["steps"] = args.steps
if args.seed is not None:
    pc["seed"] = args.seed
cfg = PINNConfig(**pc)
p = MertonParams(**base["merton"])
dom = base["domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H, ALPHAS = base["eval"]["horizons_days"], base["eval"]["alphas"]
x = np.linspace(dom["x_min"], dom["x_max"], base["eval"]["nx"])
P.use_style()
pct = lambda v: v * 100  # noqa: E731
exact_es = {h: merton_var_es(0.99, h * DAY, p)[1] for h in (1, 10)}


def track(tr: Trainer) -> dict:
    out = {}
    for h in (1, 10):
        _, es = var_es_from_pdf(x, np.exp(tr.logpdf(x, h * DAY)), 0.99)
        out[f"es99_rel_{h}d"] = abs(es - exact_es[h]) / exact_es[h]
    return out


tr = Trainer(p, dom["x_min"], dom["x_max"], t0, T, cfg)
print(f"device={tr.device}  params={sum(q.numel() for q in tr.model.parameters())}  steps={cfg.steps}")
tr.fit(callback=track)
print(f"trained in {tr.train_time:.0f}s")
hist = pd.DataFrame(tr.history)
(RESULTS / STAGE).mkdir(parents=True, exist_ok=True)
hist.to_csv(RESULTS / STAGE / "history.csv", index=False)
torch.save(dict(state=tr.model.state_dict(), cfg=pc, params=p.as_dict(), domain=dom), RESULTS / STAGE / "model.pt")

# ---------------------------------------------------------------- evaluation
# FD reference on its wider domain, so its p = 0 edges do not distort the far tail
fdd = base["fd_domain"]
xf, fd_sol, fd_time = solve_pide_1d(p, fdd["x_min"], fdd["x_max"], fdd["nx"], t0, [h * DAY for h in H], 0.02 * DAY)
torch.cuda.synchronize() if torch.cuda.is_available() else None
tic = time.perf_counter()
for h in H:
    _ = var_es_from_pdf(x, np.exp(tr.logpdf(x, h * DAY)), 0.99)
pinn_query_ms = (time.perf_counter() - tic) / len(H) * 1e3

rows, dens_err = [], {}
for h in H:
    t = h * DAY
    lp_ex, lp_pi = merton_logpdf(x, t, p), tr.logpdf(x, t)
    bulk, left, right = bulk_tail_masks(x, merton_cdf(x, t, p), 0.01)
    pex, ppi = np.exp(lp_ex), np.exp(lp_pi)
    lp_fd = np.log(np.clip(np.interp(x, xf, fd_sol[t]), 1e-300, None))
    sig = lp_ex > np.log(1e-8)  # ignore regions with negligible mass
    dens_err[h] = dict(
        bulk_rel_l1=np.trapezoid(np.abs(ppi - pex)[bulk], x[bulk]) / np.trapezoid(pex[bulk], x[bulk]),
        left_tail_mae_log=np.mean(np.abs(lp_pi - lp_ex)[left & sig]),
        right_tail_mae_log=np.mean(np.abs(lp_pi - lp_ex)[right & sig]),
        fd_bulk_rel_l1=np.trapezoid(np.abs(np.exp(lp_fd) - pex)[bulk], x[bulk]) / np.trapezoid(pex[bulk], x[bulk]),
        fd_left_tail_mae_log=np.mean(np.abs(lp_fd - lp_ex)[left & sig]),
    )
    for a in ALPHAS:
        ve, ee = merton_var_es(a, t, p)
        vp, ep = var_es_from_pdf(x, ppi, a)
        vf, ef = var_es_from_pdf(xf, fd_sol[t], a)
        rows.append(dict(h=h, alpha=a, var_exact=ve, es_exact=ee, var_pinn=vp, es_pinn=ep, var_fd=vf, es_fd=ef))
risk = pd.DataFrame(rows)
for m in ("pinn", "fd"):
    risk[f"var_rel_{m}"] = (risk[f"var_{m}"] - risk.var_exact) / risk.var_exact
    risk[f"es_rel_{m}"] = (risk[f"es_{m}"] - risk.es_exact) / risk.es_exact
risk.to_csv(RESULTS / STAGE / "risk_table.csv", index=False)

# ---------------------------------------------------------------- fig01: training dynamics
F.plot_training(hist, STAGE)

# ---------------------------------------------------------------- fig02: densities
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
for ax, h in zip(axes, H):
    t = h * DAY
    v99, _ = merton_var_es(0.99, t, p)
    P.shade_tail(ax, -60, -pct(v99), "worst 1%")
    ax.plot(pct(xf), np.log10(np.clip(fd_sol[t], 1e-300, None)), **P.style("fd", lw=1.2))
    ax.plot(pct(x), tr.logpdf(x, t) / np.log(10), **P.style("pinn"))
    ax.plot(pct(x), merton_logpdf(x, t, p) / np.log(10), **P.style("exact", lw=1.4))
    ax.set(xlim=(-50, 20), ylim=(-8, 2.2), xlabel="log return (%)", title=f"{h}-day horizon")
axes[0].set_ylabel(r"$\log_{10}$ density")
axes[0].legend(loc="lower center")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_density")

# ---------------------------------------------------------------- fig03: error map over (x, t)
tt = np.exp(np.linspace(np.log(t0 * 1.01), np.log(T), 120))
xx = np.linspace(dom["x_min"], dom["x_max"], 600)
E = np.full((len(tt), len(xx)), np.nan)
qlo, qhi = [], []
for i, t in enumerate(tt):
    le = merton_logpdf(xx, t, p)
    e = (tr.logpdf(xx, t) - le) / np.log(10)
    e[le < np.log(1e-8)] = np.nan
    E[i] = e
    c = merton_cdf(xx, t, p)
    qlo.append(np.interp(0.01, c, xx))
    qhi.append(np.interp(0.99, c, xx))
lim = np.nanquantile(np.abs(E), 0.99)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
im = ax.pcolormesh(pct(xx), tt / DAY, E, cmap="div_br", vmin=-lim, vmax=lim, shading="auto", rasterized=True)
ax.plot(pct(np.array(qlo)), tt / DAY, color=P.INK, lw=1, ls=(0, (4, 2)))
ax.plot(pct(np.array(qhi)), tt / DAY, color=P.INK, lw=1, ls=(0, (4, 2)))
ax.text(pct(qlo[-1]) - 1, tt[-1] / DAY, "1% quantile", ha="right", va="top", fontsize=8, color=P.INK)
ax.set(yscale="log", xlim=(-55, 25), xlabel="log return (%)", ylabel="horizon (days)",
       title=r"PINN error $\log_{10}\hat p - \log_{10} p$ (blank where p < 1e-8)")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("decades (signed)", color=P.INK_2)
cb.outline.set_visible(False)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_error_map")

# ---------------------------------------------------------------- fig04: bulk vs tail (RQ1)
F.plot_bulk_vs_tail(dens_err, H, STAGE)

# ---------------------------------------------------------------- fig05: VaR/ES errors (RQ2, controlled)
F.plot_var_es_error(risk, H, ALPHAS, STAGE)

save_results(STAGE, "metrics", dict(
    pinn_config=pc, params=p.as_dict(), domain=dom,
    train_time_s=tr.train_time, pinn_query_ms=pinn_query_ms, fd_solve_s=fd_time,
    density_error={f"{h}d": v for h, v in dens_err.items()},
    risk=risk.to_dict(orient="records")))
print(risk[["h", "alpha", "es_exact", "es_pinn", "es_rel_pinn", "es_rel_fd"]].to_string(index=False))
print({h: {k: f"{v:.2e}" for k, v in d.items()} for h, d in dens_err.items()})
