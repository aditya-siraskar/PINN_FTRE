"""Stage 3: multi-asset ground truth (exact, 2-D finite differences, Monte Carlo).

Produces figures/stage3_multiasset_reference/fig01..fig04 and results/stage3_multiasset_reference/metrics.json.
Run:  python scripts/stage3_multiasset_reference.py      (about 5 min on a laptop CPU; no GPU needed)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import load_config, save_results  # noqa: E402
from pinn_tailrisk.fd import solve_pide_1d  # noqa: E402
from pinn_tailrisk.fd2d import solve_pide_2d  # noqa: E402
from pinn_tailrisk.jumpdiff import (DAY, MertonParams, MultiMertonParams, merton_var_es,  # noqa: E402
                                    multi_merton_logpdf, simulate_multi_merton)
from pinn_tailrisk.risk import var_es_from_samples, var_es_from_weighted  # noqa: E402

STAGE = "stage3_multiasset_reference"
base = load_config("configs/base.yaml")
cfg = load_config("configs/stage3_multiasset.yaml")
jump = {k: base["merton"][k] for k in ("lam", "mu_j", "sigma_j")}
t0, T = base["domain"]["t0_days"] * DAY, base["domain"]["T_days"] * DAY
ta = cfg["two_asset"]
mp = MultiMertonParams.equicorrelated(2, sigma=ta["sigma"], rho=ta["rho"], a=base["merton"]["a"],
                                      beta=ta["beta"], **jump)
w_ref = np.array(ta["weights"])
f2 = cfg["fd2d"]
lo, hi, dt2 = (f2["x_min"],) * 2, (f2["x_max"],) * 2, f2["dt_days"] * DAY
rng = np.random.default_rng(0)
P.use_style()
pct = lambda v: v * 100  # noqa: E731
metrics: dict = {"two_asset": dict(sigma=ta["sigma"], rho=ta["rho"], beta=ta["beta"], **jump), "fd2d": f2}


def grid_eval(x, y, pdf, w):
    X, Y = np.meshgrid(x, y, indexing="ij")
    return var_es_from_weighted(w[0] * X + w[1] * Y, pdf * (x[1] - x[0]) * (y[1] - y[0]), 0.99)


# ---------------------------------------------------------------- reference 2-D solve
print("2-D FD reference solve ...")
x, y, sol, wall_ref = solve_pide_2d(mp, lo, hi, f2["n"], t0, [T], dt2)
X, Y = np.meshgrid(x, y, indexing="ij")
h2 = (x[1] - x[0]) * (y[1] - y[0])
lp_ex = multi_merton_logpdf(np.stack([X, Y], -1), T, mp)
p_fd = sol[T]
metrics["fd2d_reference"] = dict(n=f2["n"], wall_s=wall_ref, mass=float(p_fd.sum() * h2),
                                 l1=float(np.abs(p_fd - np.exp(lp_ex)).sum() * h2))

# ---------------------------------------------------------------- fig01: 2-D density, exact vs FD
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
ax = axes[0]
levels = np.arange(-6, 2.5, 1.0)
L_ex = lp_ex / np.log(10)
cf = ax.contourf(pct(X), pct(Y), L_ex, levels=np.arange(-6, 2.6, 0.5), cmap="seq_blue", extend="min")
ax.contour(pct(X), pct(Y), np.log10(np.clip(p_fd, 1e-300, None)), levels=levels,
           colors=P.METHOD["fd"]["color"], linewidths=1.1)
ax.contour(pct(X), pct(Y), L_ex, levels=levels, colors=P.INK, linewidths=0.9, linestyles=[(0, (4, 3))])
ax.set(xlim=(-55, 25), ylim=(-55, 25), xlabel="asset 1 log return (%)", ylabel="asset 2 log return (%)",
       title="10-day joint density: exact vs 2-D FD", aspect="equal")
ax.grid(False)
ax.plot([], [], color=P.METHOD["fd"]["color"], lw=1.1, label="FD contours")
ax.plot([], [], color=P.INK, lw=0.9, ls=(0, (4, 3)), label="exact contours")
ax.legend(loc="upper left")
cb = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.9)
cb.set_label(r"$\log_{10}$ density (exact)", color=P.INK_2)
cb.outline.set_visible(False)
ax.annotate("common-jump ridge\n(both assets fall together)", xy=(-28, -33), xytext=(-8, -50), fontsize=8,
            color=P.INK_2, arrowprops=dict(arrowstyle="-", color=P.INK_2, lw=0.8))
ax = axes[1]
E = np.log10(np.clip(p_fd, 1e-300, None)) - L_ex
E[L_ex < -6] = np.nan
lim = np.nanquantile(np.abs(E), 0.99)
im = ax.pcolormesh(pct(x), pct(y), E.T, cmap="div_br", vmin=-lim, vmax=lim, shading="auto", rasterized=True)
ax.set(xlim=(-55, 25), ylim=(-55, 25), xlabel="asset 1 log return (%)", ylabel="asset 2 log return (%)",
       title=r"FD error $\log_{10}p_{FD}-\log_{10}p$ (p > 1e-6)", aspect="equal")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
cb.set_label("decades (signed)", color=P.INK_2)
cb.outline.set_visible(False)
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig01_density_2d")

# ---------------------------------------------------------------- fig02: grid convergence and the cost of grids
print("2-D grid convergence ...")
e_true = merton_var_es(0.99, T, mp.portfolio(w_ref))[1]
conv = []
for n in f2["n_list"]:
    xs, ys, s, wall = solve_pide_2d(mp, lo, hi, n, t0, [T], dt2)
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    pe = np.exp(multi_merton_logpdf(np.stack([Xs, Ys], -1), T, mp))
    hh = (xs[1] - xs[0]) * (ys[1] - ys[0])
    conv.append(dict(n=n, h=float(xs[1] - xs[0]), wall_s=wall, l1=float(np.abs(s[T] - pe).sum() * hh),
                     es99_rel=float(abs(grid_eval(xs, ys, s[T], w_ref)[1] - e_true) / e_true)))
    print(f"  n={n}: {wall:.1f}s  L1={conv[-1]['l1']:.2e}  ES rel={conv[-1]['es99_rel']:.2e}")
p1 = mp.portfolio(w_ref)
conv1 = []
for n in [251, 501, 1001, 2001, 4001, 8001]:
    _, _, wall = solve_pide_1d(p1, -1.0, 0.5, n, t0, [T], dt2)
    conv1.append(dict(n=n, wall_s=wall))
metrics["fd2d_convergence"], metrics["fd1d_timing"] = conv, conv1

sc = cfg["scaling"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ax = axes[0]
hs = np.array([c["h"] for c in conv]) * 100
l1 = np.array([c["l1"] for c in conv])
ax.loglog(hs, l1, marker="s", **P.style("fd", label=r"$L^1$ density error"))
ax.loglog(hs, [c["es99_rel"] for c in conv], marker="o", color=P.INK_2, lw=1.4, label="portfolio ES$_{99}$ rel. error")
ax.loglog(hs, l1[-1] * (hs / hs[-1]) ** 2, color=P.MUTED, ls=(0, (2, 2)), lw=1, label="slope 2")
ax.set(xlabel=r"grid step $\Delta x$ (%)", ylabel="error at 10 days", title="2-D FD converges")
ax.text(0.98, 0.04, f"ES floor ~1e-4 from the\nfirst-order $\\Delta t$ = {f2['dt_days']} d",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=P.INK_2)
ax.legend(loc="upper left")
ax = axes[1]
ax.loglog([c["n"] for c in conv1], [c["wall_s"] for c in conv1], marker="s", **P.style("fd", label="1-D FD"))
ax.loglog([c["n"] ** 2 for c in conv], [c["wall_s"] for c in conv], marker="s",
          **P.style("fd", ls="--", label="2-D FD"))
ax.set(xlabel="grid unknowns", ylabel="wall time (s, CPU)", title="Cost follows the number of unknowns")
ax.legend(loc="upper left")
ax = axes[2]
d = np.arange(1, 11)
mem = float(sc["grid_points_per_axis"]) ** d * 8
ax.semilogy(d, mem, marker="s", **P.style("fd", label=f"one density grid, {sc['grid_points_per_axis']} pts/axis"))
for cap, lab, va, f in [(4e9, "GTX 1650 (4 GB)", "top", 0.6), (16e9, "T4 (16 GB)", "bottom", 1.5),
                        (1e12, "1 TB", "bottom", 1.5)]:
    ax.axhline(cap, color=P.AXIS, lw=0.8)
    ax.text(10.3, cap * f, lab, ha="right", va=va, fontsize=8, color=P.INK_2)
ax.set(xlabel="number of assets d", ylabel="memory (bytes)", title="…and grids stop fitting beyond d = 3",
       xticks=d)
ax.legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_fd_cost")

# ---------------------------------------------------------------- fig03: Monte Carlo vs dimension
print("Monte Carlo vs dimension ...")
mc_rows = []
for dd in sc["d_list"]:
    mpd = MultiMertonParams.equicorrelated(
        dd, sigma=np.linspace(*sc["sigma_range"], dd) if dd > 1 else [sc["sigma_range"][0]], rho=sc["rho"],
        a=base["merton"]["a"], beta=np.linspace(*sc["beta_range"], dd) if dd > 1 else [1.0], **jump)
    wd = np.full(dd, 1.0 / dd)
    et = merton_var_es(0.99, T, mpd.portfolio(wd))[1]
    errs, tic = [], time.perf_counter()
    for _ in range(sc["mc_reps"]):
        errs.append(abs(var_es_from_samples(simulate_multi_merton(sc["mc_paths"], T, mpd, rng) @ wd, 0.99)[1] - et) / et)
    mc_rows.append(dict(d=dd, es_true=et, wall_ms=(time.perf_counter() - tic) / sc["mc_reps"] * 1e3,
                        err_med=float(np.median(errs)), err_q10=float(np.quantile(errs, 0.1)),
                        err_q90=float(np.quantile(errs, 0.9))))
    print(f"  d={dd}: {mc_rows[-1]['wall_ms']:.1f} ms  median err {mc_rows[-1]['err_med']:.2e}")
metrics["mc_vs_d"] = mc_rows
dd = np.array([r["d"] for r in mc_rows])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
ax.fill_between(dd, [r["err_q10"] for r in mc_rows], [r["err_q90"] for r in mc_rows],
                color=P.METHOD["mc"]["color"], alpha=0.18, lw=0, label="10–90% of runs")
ax.plot(dd, [r["err_med"] for r in mc_rows], marker="^", **P.style("mc", label="median"))
ax.set(xscale="log", yscale="log", xticks=dd, xticklabels=dd, xlabel="number of assets d",
       ylabel="portfolio ES$_{99}$ relative error",
       title=f"MC accuracy barely depends on d (N = {sc['mc_paths']:,})")
ax.legend(loc="upper left")
ax = axes[1]
ax.plot(dd, [r["wall_ms"] for r in mc_rows], marker="^", **P.style("mc", label="Monte Carlo"))
ax.set(xscale="log", yscale="log", xticks=dd, xticklabels=dd, xlabel="number of assets d",
       ylabel="wall time per estimate (ms, CPU)", title="…but every query pays the full simulation")
ax.legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_mc_vs_d")

# ---------------------------------------------------------------- fig04: portfolio ES across weights, three routes
print("portfolio ES across weights ...")
ws = np.linspace(0, 1, 11)
rows = []
sims = simulate_multi_merton(1_000_000, T, mp, rng)
for w1 in ws:
    w = np.array([w1, 1 - w1])
    v_ex, e_ex = merton_var_es(0.99, T, mp.portfolio(w))
    v_fd, e_fd = grid_eval(x, y, p_fd, w)
    v_mc, e_mc = var_es_from_samples(sims @ w, 0.99)
    rows.append(dict(w1=w1, es_exact=e_ex, es_fd=e_fd, es_mc=e_mc, var_exact=v_ex, var_fd=v_fd, var_mc=v_mc))
metrics["portfolio_weights"] = rows
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
wf = np.linspace(0, 1, 101)
ax.plot(wf * 100, pct(np.array([merton_var_es(0.99, T, mp.portfolio(np.array([a, 1 - a])))[1] for a in wf])),
        **P.style("exact"))
ax.plot(ws * 100, pct(np.array([r["es_fd"] for r in rows])), lw=0, marker="s", ms=6,
        color=P.METHOD["fd"]["color"], label="2-D FD (grid)")
ax.plot(ws * 100, pct(np.array([r["es_mc"] for r in rows])), lw=0, marker="^", ms=6,
        color=P.METHOD["mc"]["color"], label="Monte Carlo (1M)")
ax.set(xlabel="weight in asset 1 (%)", ylabel="10-day ES$_{99}$ (% of portfolio)",
       title="Portfolio ES across weights")
ax.legend(loc="upper right")
ax = axes[1]
for key, m in [("es_fd", "fd"), ("es_mc", "mc")]:
    ax.plot(ws * 100, [abs(r[key] - r["es_exact"]) / r["es_exact"] * 100 for r in rows], marker=P.METHOD[m]["marker"],
            ms=6, **P.style(m, lw=1.2))
ax.set(yscale="log", xlabel="weight in asset 1 (%)", ylabel="|ES$_{99}$ error| (%)",
       title="Error vs the exact 1-D projection")
ax.legend(loc="center left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig04_portfolio_es")

print(save_results(STAGE, "metrics", metrics))
