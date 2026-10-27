"""Stage 2a: ground truth. Exact Merton density vs finite differences vs Monte Carlo.

Produces figures/stage2_reference/fig01..fig05 and results/stage2_reference/metrics.json.
Run:  python scripts/stage2_reference.py
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
from pinn_tailrisk.jumpdiff import (DAY, MertonParams, merton_cdf, merton_logpdf,  # noqa: E402
                                    merton_pdf, merton_var_es, simulate_merton)
from pinn_tailrisk.risk import gaussian_var_es, var_es_from_pdf, var_es_from_samples  # noqa: E402

STAGE = "stage2_reference"
cfg = load_config("configs/base.yaml")
p = MertonParams(**cfg["merton"])
dom = cfg["domain"]
fdd = cfg["fd_domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H = cfg["eval"]["horizons_days"]
ALPHAS = cfg["eval"]["alphas"]
rng = np.random.default_rng(0)
P.use_style()
metrics: dict = {"params": p.as_dict(), "domain": dom}
x = np.linspace(dom["x_min"], dom["x_max"], cfg["eval"]["nx"])
horizon_colors = [P.SEQ_BLUE(v) for v in (0.35, 0.65, 0.95)]
pct = lambda v: v * 100  # noqa: E731  (log return -> %)

# ---------------------------------------------------------------- fig01: the target
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for h, c in zip(H, horizon_colors):
    axes[0].plot(pct(x), merton_pdf(x, h * DAY, p) / 100, color=c, label=f"{h}-day")
    axes[1].plot(pct(x), merton_logpdf(x, h * DAY, p) / np.log(10), color=c, label=f"{h}-day")
axes[0].set(xlim=(-12, 8), xlabel="log return (%)", ylabel="density (per %)", title="Density, linear scale")
axes[1].set(xlim=(-45, 20), ylim=(-8, 2), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
            title="Same densities, log scale")
axes[0].legend(title="horizon", title_fontsize=8.5)

h = 10
sd = np.sqrt(p.variance(h * DAY))
gauss = -0.5 * ((x - p.mean(h * DAY)) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))
axes[2].plot(pct(x), merton_logpdf(x, h * DAY, p) / np.log(10), **P.style("exact"))
axes[2].plot(pct(x), gauss / np.log(10), **P.style("gaussian", label="Gaussian, same mean & variance"))
v99, _ = merton_var_es(0.99, h * DAY, p)
P.shade_tail(axes[2], -45, -pct(v99), "worst 1%")
axes[2].set(xlim=(-45, 20), ylim=(-8, 2), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
            title="10-day: jumps fatten the left tail")
axes[2].legend(loc="lower center")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig01_target_density")

# ---------------------------------------------------------------- fig02: FD and MC agree with exact
nx_fd, dt_fd = fdd["nx"], 0.02 * DAY
xf, sol, fd_time = solve_pide_1d(p, fdd["x_min"], fdd["x_max"], nx_fd, t0, [h * DAY for h in H], dt_fd)
mc = simulate_merton(2_000_000, T, p, rng)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
edges = np.linspace(dom["x_min"], dom["x_max"], 241)
cnt, _ = np.histogram(mc, edges)
dens = cnt / (len(mc) * np.diff(edges))
ok = cnt > 0
mid = 0.5 * (edges[1:] + edges[:-1])
ax.plot(pct(mid[ok]), np.log10(dens[ok]), lw=0, marker=P.METHOD["mc"]["marker"], ms=3.5,
        color=P.METHOD["mc"]["color"], label="Monte Carlo (2M paths)")
ax.plot(pct(xf), np.log10(np.clip(sol[T], 1e-300, None)), **P.style("fd"))
ax.plot(pct(x), merton_logpdf(x, T, p) / np.log(10), **P.style("exact"))
ax.set(xlim=(-50, 20), ylim=(-6.5, 1.5), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
       title="10-day density: three independent routes")
ax.legend(loc="lower center")
ax = axes[1]
for hh, c in zip(H, horizon_colors):
    ex = merton_pdf(xf, hh * DAY, p)
    m = ex > 1e-6
    ax.plot(pct(xf[m]), np.abs(np.log10(np.clip(sol[hh * DAY][m], 1e-300, None)) - np.log10(ex[m])),
            color=c, lw=1.4, label=f"{hh}-day")
ax.set(yscale="log", xlim=(-50, 20), ylim=(1e-10, 1e-2), xlabel="log return (%)",
       ylabel=r"$|\log_{10}p_{FD} - \log_{10}p|$", title="Finite-difference error where p > 1e-6")
ax.legend(title="horizon", title_fontsize=8.5)
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_fd_mc_validation")

metrics["fd_reference"] = dict(nx=nx_fd, dt_days=0.02, x_min=fdd["x_min"], x_max=fdd["x_max"],
                               wall_s=fd_time, var_es={})
for hh in H:
    row = {}
    for a in ALPHAS:
        ve = merton_var_es(a, hh * DAY, p)
        vf = var_es_from_pdf(xf, sol[hh * DAY], a)
        row[str(a)] = dict(exact=ve, fd=vf)
    metrics["fd_reference"]["var_es"][f"{hh}d"] = row

# ---------------------------------------------------------------- fig03: FD convergence
def fd_errors(nx, dt):
    xg, s, wall = solve_pide_1d(p, fdd["x_min"], fdd["x_max"], nx, t0, [T], dt)
    l1 = np.trapezoid(np.abs(s[T] - merton_pdf(xg, T, p)), xg)
    v, e = var_es_from_pdf(xg, s[T], 0.99)
    vt, et = merton_var_es(0.99, T, p)
    return l1, abs(e - et) / et, wall

nxs = [251, 501, 1001, 2001, 4001]
dx_rows = [fd_errors(n, 0.005 * DAY) for n in nxs]
dts = [1.0, 0.5, 0.25, 0.125, 0.0625]
dt_rows = [fd_errors(8001, d * DAY) for d in dts]
dxv = [(fdd["x_max"] - fdd["x_min"]) / (n - 1) for n in nxs]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, xs, rows, xl, title in [(axes[0], np.array(dxv) * 100, dx_rows, r"grid step $\Delta x$ (%)", "Space refinement"),
                                (axes[1], np.array(dts), dt_rows, r"time step $\Delta t$ (days)", "Time refinement")]:
    l1 = np.array([r[0] for r in rows])
    es = np.array([r[1] for r in rows])
    ax.loglog(xs, l1, marker="s", **P.style("fd", label=r"$L^1$ density error"))
    ax.loglog(xs, es, marker="o", color=P.INK_2, ls="-", lw=1.4, label="ES$_{99}$ relative error")
    ref = l1[-1] * (xs / xs[-1]) ** 2
    ax.loglog(xs, ref, color=P.MUTED, ls=(0, (2, 2)), lw=1, label="slope 2")
    ax.set(xlabel=xl, ylabel="error at 10 days", title=title)
    ax.legend()
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_fd_convergence")
metrics["fd_convergence"] = dict(
    dx=[dict(nx=n, l1=r[0], es99_rel=r[1], wall_s=r[2]) for n, r in zip(nxs, dx_rows)],
    dt=[dict(dt_days=d, l1=r[0], es99_rel=r[1], wall_s=r[2]) for d, r in zip(dts, dt_rows)])

# ---------------------------------------------------------------- fig04: why the tail matters
al = np.linspace(0.90, 0.999, 120)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, hh in zip(axes, [1, 10]):
    t = hh * DAY
    ve = np.array([merton_var_es(a, t, p) for a in al])
    vg = np.array([gaussian_var_es(p.mean(t), np.sqrt(p.variance(t)), a) for a in al])
    ax.plot(al * 100, pct(ve[:, 0]), **P.style("exact", ls="-", label="VaR, jump-diffusion"))
    ax.plot(al * 100, pct(ve[:, 1]), **P.style("exact", label="ES, jump-diffusion"))
    ax.plot(al * 100, pct(vg[:, 0]), **P.style("gaussian", ls="-", label="VaR, Gaussian"))
    ax.plot(al * 100, pct(vg[:, 1]), **P.style("gaussian", label="ES, Gaussian"))
    for a in (0.99, 0.975):
        ax.axvline(a * 100, color=P.AXIS, lw=0.8, zorder=0)
    ax.set(xlabel=r"confidence level $\alpha$ (%)", title=f"{hh}-day horizon")
    i = np.argmin(np.abs(al - 0.99))
    e_true, e_gauss = pct(ve[i, 1]), pct(vg[i, 1])
    gap = (e_true - e_gauss) / e_true * 100
    ax.annotate("", xy=(99, e_gauss), xytext=(99, e_true),
                arrowprops=dict(arrowstyle="<->", color=P.INK_2, lw=1, shrinkA=0, shrinkB=0))
    ax.text(98.8, e_true + 0.6, f"Gaussian ES$_{{99}}$\n{gap:.0f}% too low",
            ha="right", va="bottom", fontsize=8.5, color=P.INK_2)
axes[0].set_ylabel("loss (% of portfolio)")
axes[0].legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig04_var_es_vs_alpha")

# ---------------------------------------------------------------- fig05: Monte Carlo cost of tail accuracy
ns = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]).astype(int)
_, es_true = merton_var_es(0.99, T, p)
reps = 200
err, wall = [], []
for n in ns:
    tic = time.perf_counter()
    e = [var_es_from_samples(simulate_merton(n, T, p, rng), 0.99)[1] for _ in range(reps)]
    wall.append((time.perf_counter() - tic) / reps)
    err.append(np.abs(np.array(e) - es_true) / es_true)
err = np.array(err)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
med, lo, hi = np.median(err, 1), np.quantile(err, 0.1, 1), np.quantile(err, 0.9, 1)
ax.fill_between(ns, lo, hi, color=P.METHOD["mc"]["color"], alpha=0.18, lw=0, label="10–90% of runs")
ax.loglog(ns, med, marker="^", **P.style("mc", label="median"))
ax.loglog(ns, med[0] * np.sqrt(ns[0] / ns), color=P.MUTED, ls=(0, (2, 2)), lw=1, label=r"$N^{-1/2}$")
ax.axhline(0.01, color=P.AXIS, lw=0.8)
ax.text(ns[0], 0.0105, "1% error", fontsize=8, color=P.INK_2, va="bottom")
ax.set(xlabel="Monte Carlo paths N", ylabel="ES$_{99}$ relative error", title="Tail accuracy improves slowly")
ax.legend(loc="lower left")
ax = axes[1]
ax.loglog(ns, np.array(wall) * 1e3, marker="^", **P.style("mc", label="Monte Carlo"))
ax.axhline(fd_time * 1e3, **P.style("fd", label=f"FD, full density ({fd_time*1e3:.0f} ms)"))
ax.set(xlabel="Monte Carlo paths N", ylabel="wall time per estimate (ms)", title="…and the cost grows linearly")
ax.legend(loc="lower right")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig05_mc_cost")
metrics["mc_convergence"] = [dict(n=int(n), median_rel_err=float(m), wall_ms=w * 1e3)
                             for n, m, w in zip(ns, med, wall)]

print(save_results(STAGE, "metrics", metrics))
