"""Stage 5: amortised PINN over market parameters theta = (s, lam, mu_J); real-time VaR/ES.

One training run covers a box of markets. Every evaluation below queries the same
trained network at a new theta, with no retraining:
  1. accuracy across theta: median |ES99 error| over the Stage 4 portfolios on a grid of
     (s, lam) and a sweep of mu_J, against the closed form; Monte Carlo and exact-density
     IS (the sampling floor) use the same number of draws;
  2. latency per query vs Monte Carlo on the same device, error vs draws for both;
  3. break-even: queries after which the training time is paid back, at matched error;
  4. Euler risk contributions w_i dES/dw_i (IS tail average) vs finite differences of
     the exact ES, and what-if curves ES99(lam), ES99(s).

Produces figures/stage5_amortised_d<d>/fig01..fig05 and results/stage5_amortised_d<d>/{metrics.json,
history.csv, risk_grid.csv, latency.csv, model.pt}.
Run:  python scripts/stage5_amortised.py --d 5 [--steps N] [--eval-only] [--quick] [--tag name]
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
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import figs_pinn1d as F  # noqa: E402
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config, save_results  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, merton_var_es  # noqa: E402
from pinn_tailrisk.pinn_amortised import (AmortisedTrainer, ThetaBox, es_weighted_t,  # noqa: E402
                                          exact_logpdf_torch, params_at, simulate_torch)
from pinn_tailrisk.pinn_nd import NDConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage5_amortised.yaml")
ap.add_argument("--d", type=int, required=True)
ap.add_argument("--steps", type=int)
ap.add_argument("--seed", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--tag", default="")
ap.add_argument("--eval-only", action="store_true", help="load results/<stage>/model.pt instead of training")
ap.add_argument("--quick", action="store_true", help="small grid and draw counts (CPU smoke test)")
args = ap.parse_args()

d = args.d
STAGE = f"stage5_amortised_d{d}" + (f"_{args.tag}" if args.tag else "")
base, s3, c5 = load_config("configs/base.yaml"), load_config("configs/stage3_multiasset.yaml"), load_config(args.config)
pc, ev = c5["pinn"], c5["eval"]
for k in ("steps", "seed", "width"):
    if getattr(args, k) is not None:
        pc[k] = getattr(args, k)
if args.quick:
    ev.update(grid_s=ev["grid_s"][::2], grid_lam=ev["grid_lam"][::2], grid_mu=ev["grid_mu"][::2], n_is=8192,
              n_track=8192, latency_n=[1024, 4096, 16384], latency_n_mc=[4096, 16384, 65536], latency_reps=4,
              whatif_n=8192)
cfg = NDConfig(**pc)
box = ThetaBox(**{k: tuple(v) for k, v in c5["theta_box"].items()})
sc = s3["scaling"]
mp = MultiMertonParams.equicorrelated(
    d, sigma=np.linspace(*sc["sigma_range"], d), rho=sc["rho"], a=base["merton"]["a"],
    beta=np.linspace(*sc["beta_range"], d), **{k: base["merton"][k] for k in ("lam", "mu_j", "sigma_j")})
dom = base["domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H = base["eval"]["horizons_days"]
C = tuple(ev["centre"])
P.use_style()

prng = np.random.default_rng(ev["portfolio_seed"])          # the Stage 4 portfolios
ls = prng.normal(size=(ev["n_long_short"], d))
ls[:, 0] = 1 - ls[:, 1:].sum(1)
W = np.vstack([np.full(d, 1 / d), prng.dirichlet(np.ones(d), ev["n_long_only"]), ls])
_exact: dict = {}


def exact_var_es(th, h, i, alpha=0.99):
    key = (tuple(th), h, i, alpha)
    if key not in _exact:
        _exact[key] = merton_var_es(alpha, h * DAY, params_at(mp, th).portfolio(W[i]))
    return _exact[key]


def track(tr: AmortisedTrainer) -> dict:
    out = {}
    for h in (1, 10):
        X, lq, lp = tr.weighted_samples_t(h * DAY, C, ev["n_track"])
        _, es = es_weighted_t(X @ tr.a.new_tensor(W[0]), torch.softmax(lp - lq, 0), 0.99)
        out[f"es99_rel_{h}d"] = abs(float(es) / exact_var_es(C, h, 0)[1] - 1)
    return out


tr = AmortisedTrainer(mp, t0, T, cfg, box)
dev = tr.device
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
print(f"d={d}  device={dev}  trace={tr.trace}  params={sum(q.numel() for q in tr.model.parameters())}  steps={cfg.steps}")
(RESULTS / STAGE).mkdir(parents=True, exist_ok=True)
if args.eval_only:
    ck = torch.load(RESULTS / STAGE / "model.pt", map_location=dev, weights_only=False)  # our own file
    tr.model.load_state_dict(ck["state"])
    tr.train_time, tr.peak_mem_mb = ck["train_time_s"], ck["peak_mem_mb"]
    hist = pd.read_csv(RESULTS / STAGE / "history.csv")
else:
    tr.fit(callback=track)
    print(f"trained in {tr.train_time:.0f}s  peak GPU memory {tr.peak_mem_mb:.0f} MB")
    hist = pd.DataFrame(tr.history)
    hist.to_csv(RESULTS / STAGE / "history.csv", index=False)
    torch.save(dict(state=tr.model.state_dict(), cfg=pc, box=c5["theta_box"], d=d, cov=mp.cov.tolist(), beta=mp.beta.tolist(),
                    train_time_s=tr.train_time, peak_mem_mb=tr.peak_mem_mb), RESULTS / STAGE / "model.pt")
Wt = torch.tensor(W, dtype=torch.float32, device=dev)
torch.manual_seed(1000 + cfg.seed)

# ---------------------------------------------------------------- 1. accuracy across theta
thetas = [(s, lam, C[2]) for s in ev["grid_s"] for lam in ev["grid_lam"]]
thetas += [(C[0], C[1], mu) for mu in ev["grid_mu"] if (C[0], C[1], mu) not in thetas]
rows, is_diag = [], []
for th in thetas:
    for h in H:
        t = h * DAY
        X, lq, lp = tr.weighted_samples_t(t, th, ev["n_is"])
        lw_p = lp - lq
        lw_e = exact_logpdf_torch(X.double(), torch.full((1,), t, device=dev, dtype=torch.float64),
                                  torch.tensor([th], device=dev, dtype=torch.float64), mp) - lq.double()
        Xmc = simulate_torch(ev["n_is"], t, mp, th, dev)
        _, es_p = es_weighted_t(X @ Wt.T, torch.softmax(lw_p, 0), 0.99)
        _, es_e = es_weighted_t((X.double() @ Wt.double().T), torch.softmax(lw_e, 0), 0.99)
        _, es_m = es_weighted_t(Xmc @ Wt.T, None, 0.99)
        wp = torch.softmax(lw_p.double(), 0)
        is_diag.append(dict(s=th[0], lam=th[1], mu_j=th[2], h=h, ess_frac=float(1 / (wp**2).sum()) / len(wp),
                            mass=float(torch.exp(torch.logsumexp(lw_p.double(), 0)) / len(wp))))
        for i in range(len(W)):
            ee = exact_var_es(th, h, i)[1]
            rows.append(dict(s=th[0], lam=th[1], mu_j=th[2], h=h, port=i, es_exact=ee,
                             es_rel_pinn=float(es_p[i]) / ee - 1, es_rel_is_exact=float(es_e[i]) / ee - 1,
                             es_rel_mc=float(es_m[i]) / ee - 1))
grid = pd.DataFrame(rows)
grid.to_csv(RESULTS / STAGE / "risk_grid.csv", index=False)
gmed = grid.groupby(["s", "lam", "mu_j", "h"])[["es_rel_pinn", "es_rel_is_exact", "es_rel_mc"]].agg(
    lambda v: v.abs().median()).reset_index()
print(gmed.groupby("h")[["es_rel_pinn", "es_rel_is_exact", "es_rel_mc"]].median())


# ---------------------------------------------------------------- 2. latency and error vs draws (centre theta)
def query(method: str, n: int, t: float):
    """One risk query at a new market state: draws -> weights -> ES99 of the equal-weight portfolio."""
    if method == "pinn":
        X, lq, lp = tr.weighted_samples_t(t, C, n, chunk=n)
        w = torch.softmax(lp - lq, 0)
    else:
        X, w = simulate_torch(n, t, mp, C, dev, L=tr.L), None
    es_weighted_t(X @ Wt[0], w, 0.99)
    return X, w


lat_rows = []
for h in (1, 10):
    t = h * DAY
    ex = np.array([exact_var_es(C, h, i)[1] for i in range(len(W))])
    for method, ns in (("pinn", ev["latency_n"]), ("mc", ev["latency_n_mc"])):
        for n in ns:
            for _ in range(3):
                query(method, n, t)                                          # warm-up
            times, errs = [], []
            for _ in range(ev["latency_reps"]):
                sync(); tic = time.perf_counter()
                X, w = query(method, n, t)
                sync(); times.append(time.perf_counter() - tic)
                _, es = es_weighted_t(X @ Wt.T, w, 0.99)
                errs.append(np.abs(es.cpu().numpy() / ex - 1))
            lat_rows.append(dict(h=h, method=method, n=n, latency_ms=1e3 * np.median(times),
                                 err_median=np.median(errs), err_p90=np.quantile(errs, 0.9)))
lat = pd.DataFrame(lat_rows)
lat.to_csv(RESULTS / STAGE / "latency.csv", index=False)
print(lat.to_string(index=False))

# ---------------------------------------------------------------- 3. matched error and break-even
breakeven = {}
for h in (1, 10):
    lp_, lm_ = lat[(lat.h == h) & (lat.method == "pinn")], lat[(lat.h == h) & (lat.method == "mc")]
    floor = lp_.err_median.min()
    op = lp_[lp_.err_median <= 1.2 * floor].iloc[0]                    # cheapest PINN query near its floor
    slope, icpt = np.polyfit(np.log(lm_.n), np.log(lm_.err_median), 1)  # MC error ~ N^slope (slope ~ -1/2)
    n_mc = float(np.exp((np.log(op.err_median) - icpt) / slope))
    l1, l0 = np.polyfit(lm_.n, lm_.latency_ms, 1)                       # MC latency ~ l0 + l1 N
    lat_mc = float(max(l0 + l1 * n_mc, lm_.latency_ms.min()))
    gain = lat_mc - op.latency_ms
    breakeven[f"{h}d"] = dict(target_err=float(op.err_median), pinn_n=int(op.n), pinn_ms=float(op.latency_ms),
                              mc_n=n_mc, mc_ms=lat_mc, mc_err_slope=float(slope), speedup=lat_mc / op.latency_ms,
                              breakeven_queries=tr.train_time * 1e3 / gain if gain > 0 else float("inf"))
print(breakeven)

# ---------------------------------------------------------------- 4. Euler contributions and what-if curves
def euler(X: np.ndarray, w: np.ndarray, wp: np.ndarray, alpha=0.99) -> np.ndarray:
    """w_i dES/dw_i = -w_i E[X_i | tail] from weighted draws; sums to the ES."""
    z = X @ wp
    o = np.argsort(z)
    c = np.cumsum(w[o] / w.sum())
    tail = 1 - alpha
    k = int(np.searchsorted(c, tail))
    om = np.where(np.arange(len(o)) < k, w[o] / w.sum(), 0.0)
    om[k] = tail - (c[k - 1] if k else 0.0)
    return -wp * (om @ X[o]) / tail


he, n_e = 10, 4 * ev["whatif_n"]
X, lq, lp = tr.weighted_samples(he * DAY, C, n_e)
eu_pinn = euler(X, np.exp(lp - lq - (lp - lq).max()), W[0])
Xmc = simulate_torch(n_e, he * DAY, mp, C, dev).cpu().numpy().astype(float)
eu_mc = euler(Xmc, np.ones(len(Xmc)), W[0])
bump = 1e-4
eu_ex = np.array([W[0][i] * (merton_var_es(0.99, he * DAY, params_at(mp, C).portfolio(W[0] + bump * e))[1]
                             - merton_var_es(0.99, he * DAY, params_at(mp, C).portfolio(W[0] - bump * e))[1]) / (2 * bump)
                  for i, e in enumerate(np.eye(d))])

whatif = {}
for name, vals in (("lam", np.geomspace(*box.lam, 25)), ("s", np.linspace(*box.s, 21))):
    out = []
    for v in vals:
        th = (v, C[1], C[2]) if name == "s" else (C[0], v, C[2])
        torch.manual_seed(7)                                     # common random numbers along the sweep
        Xp, lqp, lpp = tr.weighted_samples_t(he * DAY, th, ev["whatif_n"])
        _, es_p = es_weighted_t(Xp @ Wt[0], torch.softmax(lpp - lqp, 0), 0.99)
        torch.manual_seed(7)
        _, es_m = es_weighted_t(simulate_torch(ev["whatif_n"], he * DAY, mp, th, dev) @ Wt[0], None, 0.99)
        out.append(dict(v=float(v), exact=merton_var_es(0.99, he * DAY, params_at(mp, th).portfolio(W[0]))[1],
                        pinn=float(es_p), mc=float(es_m)))
    whatif[name] = pd.DataFrame(out)

# ---------------------------------------------------------------- fig01: training
F.plot_training(hist, STAGE)

# ---------------------------------------------------------------- fig02: accuracy across theta
fig = plt.figure(figsize=(13.5, 4.2))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.3, 1.15], wspace=0.3)    # column 3: spacer
axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4)]
cax = fig.add_subplot(gs[0, 2])
g0 = gmed[gmed.mu_j == C[2]]
norm = LogNorm(vmin=max(1e-2, 100 * g0.es_rel_pinn.min()), vmax=max(10.0, 100 * g0.es_rel_pinn.max()))
for ax, h in zip(axes[:2], (1, 10)):
    M = g0[g0.h == h].pivot(index="lam", columns="s", values="es_rel_pinn") * 100
    im = ax.imshow(M.values, origin="lower", cmap="seq_blue", norm=norm, aspect="auto")
    for (r, c), v in np.ndenumerate(M.values):
        ax.text(c, r, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > np.sqrt(norm.vmin * norm.vmax) else P.INK)
    mc_med = 100 * g0[g0.h == h].es_rel_mc.median()
    ax.set(xticks=range(M.shape[1]), xticklabels=[f"{v:g}" for v in M.columns], yticks=range(M.shape[0]),
           yticklabels=[f"{v:g}" for v in M.index], xlabel="vol multiplier s", ylabel="jump intensity λ (per year)",
           title=f"PINN |ES$_{{99}}$ error| (%), {h}-day\n(MC at same N: median {mc_med:.2f}%)")
    ax.grid(False)
fig.colorbar(im, cax=cax, label="median over portfolios (%)")
ax = axes[2]
gm = gmed[(gmed.s == C[0]) & (gmed.lam == C[1])].sort_values("mu_j")
for h, lsty in ((1, "-"), (10, "--")):
    g = gm[gm.h == h]
    ax.plot(100 * g.mu_j, 100 * g.es_rel_pinn, marker="o", **P.style("pinn", ls=lsty, label=f"PINN, {h}-day"))
    ax.plot(100 * g.mu_j, 100 * g.es_rel_mc, marker="^", **P.style("mc", ls=lsty, label=f"Monte Carlo, {h}-day"))
    ax.plot(100 * g.mu_j, 100 * g.es_rel_is_exact, marker="d", color=P.MUTED, ls=lsty, lw=1.2, label=f"IS floor, {h}-day")
ax.axhline(1.0, color=P.AXIS, lw=0.8)
ax.set(yscale="log", xlabel="mean jump μ$_J$ (%)", ylabel="median |ES$_{99}$ error| (%)",
       title=f"Across μ$_J$ (s = {C[0]:g}, λ = {C[1]:g})")
ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
P.panel_labels(axes)
P.savefig(fig, STAGE, "fig02_accuracy_theta")

# ---------------------------------------------------------------- fig03: latency vs error
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, h in zip(axes, (1, 10)):
    for m in ("pinn", "mc"):
        g = lat[(lat.h == h) & (lat.method == m)]
        ax.plot(g.latency_ms, 100 * g.err_median, marker=P.METHOD[m]["marker"], **P.style(m))
        for _, r in g.iterrows():
            ax.annotate(f"{r.n // 1024}k", (r.latency_ms, 100 * r.err_median), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color=P.INK_2)
    b = breakeven[f"{h}d"]
    ax.axhline(100 * b["target_err"], color=P.AXIS, lw=0.8)
    ax.plot([b["pinn_ms"], b["mc_ms"]], [100 * b["target_err"]] * 2, ls="", marker="|", ms=12, color=P.INK)
    ax.set(xscale="log", yscale="log", xlabel=f"latency per query (ms, {dev})",
           title=f"{h}-day: matched error → {b['speedup']:.1f}× {'faster' if b['speedup'] >= 1 else 'slower'}")
axes[0].set_ylabel("median |ES$_{99}$ error| (%)")
axes[0].legend()
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_latency")

# ---------------------------------------------------------------- fig04: break-even
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
q = np.geomspace(1, 1e8, 200)
for ax, h in zip(axes, (1, 10)):
    b = breakeven[f"{h}d"]
    ax.plot(q, q * b["mc_ms"] / 1e3, **P.style("mc", label=f"Monte Carlo ({b['mc_ms']:.2f} ms/query)"))
    ax.plot(q, tr.train_time + q * b["pinn_ms"] / 1e3,
            **P.style("pinn", label=f"PINN: {tr.train_time / 60:.0f} min training + {b['pinn_ms']:.2f} ms/query"))
    if np.isfinite(b["breakeven_queries"]):
        ax.axvline(b["breakeven_queries"], color=P.AXIS, lw=0.8)
        ax.text(b["breakeven_queries"], 0.97, f" break-even\n {b['breakeven_queries']:.2g} queries",
                transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=P.INK_2)
    else:
        ax.text(0.03, 0.97, "no break-even: MC is faster per query", transform=ax.transAxes, va="top", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="number of risk queries",
           title=f"{h}-day, matched |ES$_{{99}}$ error| = {100 * b['target_err']:.2f}%")
    ax.legend(loc="lower right", fontsize=7.5)
axes[0].set_ylabel("cumulative compute (s)")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig04_breakeven")

# ---------------------------------------------------------------- fig05: Euler contributions and what-if
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ax, xi, bw = axes[0], np.arange(d), 0.27
ax.bar(xi - bw, 100 * eu_ex, bw, color=P.INK, label="Exact (finite diff.)")
ax.bar(xi, 100 * eu_pinn, bw, color=P.METHOD["pinn"]["color"], label="PINN (IS tail average)")
ax.bar(xi + bw, 100 * eu_mc, bw, color=P.METHOD["mc"]["color"], label="Monte Carlo")
ax.set(xticks=xi, xticklabels=[f"asset {i + 1}" for i in xi], ylabel="contribution to ES$_{99}$ (%)",
       title=f"Euler contributions, equal weight, {he}-day")
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5)
for ax, (name, lab) in zip(axes[1:], (("lam", "jump intensity λ (per year)"), ("s", "vol multiplier s"))):
    g = whatif[name]
    ax.plot(g.v, 100 * g.exact, **P.style("exact"))
    ax.plot(g.v, 100 * g.pinn, **P.style("pinn"))
    ax.plot(g.v, 100 * g.mc, marker="^", ls="", ms=4, color=P.METHOD["mc"]["color"], label=f"Monte Carlo ({ev['whatif_n'] // 1024}k)")
    ax.set(xlabel=lab, ylabel="ES$_{99}$ (%)", title=f"What-if: ES$_{{99}}$ vs {name if name == 's' else 'λ'}, {he}-day")
    ax.legend(fontsize=7.5)
axes[1].set_xscale("log")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig05_euler_whatif")

save_results(STAGE, "metrics", dict(
    d=d, device=dev, gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", trace=tr.trace, pinn_config=pc,
    theta_box=c5["theta_box"], centre=list(C), n_portfolios=len(W), train_time_s=tr.train_time,
    peak_mem_mb=tr.peak_mem_mb, n_params=sum(q.numel() for q in tr.model.parameters()),
    es99_median_over_grid={f"{h}d": gmed[gmed.h == h][["es_rel_pinn", "es_rel_is_exact", "es_rel_mc"]].median().to_dict()
                           for h in H},
    es99_worst_theta={f"{h}d": gmed[gmed.h == h].sort_values("es_rel_pinn").iloc[-1].to_dict() for h in H},
    importance_sampling={f"{h}d": dict(ess_frac_min=g.ess_frac.min(), ess_frac_median=g.ess_frac.median(),
                                       mass_min=g.mass.min(), mass_max=g.mass.max())
                         for h, g in pd.DataFrame(is_diag).groupby("h")},
    latency=lat.to_dict(orient="records"), breakeven=breakeven,
    euler=dict(exact=eu_ex.tolist(), pinn=eu_pinn.tolist(), mc=eu_mc.tolist()),
    whatif={k: v.to_dict(orient="records") for k, v in whatif.items()}))
