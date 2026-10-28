"""Stage 6: real data. One-day-ahead VaR/ES backtests of a linear and a non-linear portfolio.

For one market (5-asset basket):
  1. calibration: the d-asset common-jump Merton model by maximum likelihood on the
     development period, with a daily EWMA vol scale s_t (the "filtered" model, the density
     the Stage 5 PINN learns) and without it (unfiltered); rolling monthly recalibration
     over the test period: (lam, mu_J) of the filtered model with shrinkage to the
     development fit, the full unfiltered model;
  2. the amortised PINN over theta = (s, lam, mu_J), trained once on the development fit
     (box from development quantities only); every test day is one forward query at theta_t;
  3. daily forecasts from the PINN, the same model with its exact density (isolates the PINN
     error), the unfiltered Merton, Gaussian, HS, filtered HS, GARCH-t, GARCH-EVT, DCC-GARCH-t;
  4. backtests: Kupiec, Christoffersen, FZ0 score, Diebold-Mariano vs the PINN, for the
     equal-weight basket (linear) and basket + short 1-month put (non-linear: full revaluation).

Produces figures/stage6_realdata_<market>/fig01..fig05 and results/stage6_realdata_<market>/
{metrics.json, history.csv, calibration.csv, forecasts.csv.gz, backtest.csv, model.pt}.
Run:  python scripts/stage6_realdata.py --market nifty [--steps N] [--eval-only] [--quick]
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import backtest as B  # noqa: E402
from pinn_tailrisk import benchmarks as BM  # noqa: E402
from pinn_tailrisk import figs_pinn1d as F  # noqa: E402
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.calibrate import ewma_scale, fit_merton  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config, save_results  # noqa: E402
from pinn_tailrisk.data import load_returns  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, merton_var_es  # noqa: E402
from pinn_tailrisk.pinn_amortised import (AmortisedTrainer, ThetaBox, es_weighted_t,  # noqa: E402
                                          exact_logpdf_torch, params_at, simulate_torch)
from pinn_tailrisk.pinn_nd import NDConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage6_realdata.yaml")
ap.add_argument("--market", required=True)
ap.add_argument("--steps", type=int)
ap.add_argument("--eval-only", action="store_true", help="load results/<stage>/model.pt instead of training")
ap.add_argument("--quick", action="store_true", help="first 120 test days, small draw counts (CPU smoke test)")
ap.add_argument("--refresh-data", action="store_true", help="re-download prices instead of the committed returns")
args = ap.parse_args()

c6 = load_config(args.config)
mk, dc, mc, pf, bc, ev = (c6[k] for k in ("markets", "data", "model", "portfolio", "benchmarks", "eval"))
mk = mk[args.market]
pc = dict(c6["pinn"])
t0, T = pc.pop("t0_days") * DAY, pc.pop("T_days") * DAY
if args.steps is not None:
    pc["steps"] = args.steps
if args.quick:
    ev.update(n_is=4096)
    bc.update(n_scen=4096)
STAGE = f"stage6_realdata_{args.market}" + ("_quick" if args.quick else "")
OUT = RESULTS / STAGE
OUT.mkdir(parents=True, exist_ok=True)
P.use_style()
ALPHAS = ev["alphas"]
MODELS = ["gaussian", "hs", "fhs", "garch_t", "evt", "dcc", "merton_unf", "exact", "pinn"]
JOINT = ["hs", "fhs", "dcc", "merton_unf", "exact", "pinn"]            # can revalue the non-linear book
LABEL = {m: P.METHOD[m]["label"] for m in MODELS}
LABEL.update(exact="Merton, filtered (exact)", pinn="PINN (filtered Merton)", evt="GARCH-EVT")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 1. data and calibration
R = load_returns(args.market, mk["tickers"], dc["start"], dc["end"], refresh=args.refresh_data)
R = R[mk["tickers"]]
X, dates = R.values, R.index
nT, d = X.shape
ndev = int((dates < dc["test_start"]).sum())
test = np.arange(ndev, nT if not args.quick else ndev + 120)
w = np.full(d, 1 / d)
y_lin = (np.exp(X) - 1) @ w                                          # basket simple return (daily rebalanced)
print(f"{mk['name']}: {nT} days, development {dates[0].date()}..{dates[ndev - 1].date()}, "
      f"test {dates[test[0]].date()}..{dates[test[-1]].date()} ({len(test)} days)")

W, wu = mc["window"], mc["warmup"]
cref = np.cov(X[:ndev].T)
s_t = ewma_scale(X, cref, mc["ewma_decay"], wu)
fixed_beta = ("beta",) if mc["beta_fixed"] else ()
tic = time.time()
p_dev, info_dev = fit_merton(X[wu:ndev], s=s_t[wu:ndev], fixed=fixed_beta, se=True)
p_unf_dev, info_unf = fit_merton(X[:ndev], fixed=fixed_beta, se=True)
v_dev = info_dev["vector"]
print(f"filtered dev fit: lam={p_dev.lam:.1f}±{info_dev['se']['lam']:.1f} mu_J={p_dev.mu_j:.4f}±{info_dev['se']['mu_j']:.4f} "
      f"sigma_J={p_dev.sigma_j:.4f}±{info_dev['se']['sigma_j']:.4f}; unfiltered lam={p_unf_dev.lam:.1f}")

prior = dict(log_lam=(float(v_dev["log_lam"]), mc["prior_sd_log_lam"]), mu_j=(float(v_dev["mu_j"]), mc["prior_sd_mu_j"]))
cal_rows, v_unf = [], info_unf["vector"]
refits = list(range(test[0], test[-1] + 1, mc["refit"]))
for e in refits:
    pf_, _ = fit_merton(X[e - W:e], s=s_t[e - W:e], init=v_dev, prior=prior,
                        fixed=("a", "chol", "beta", "log_sigma_j"))
    pu_, iu = fit_merton(X[e - W:e], init=v_unf, fixed=fixed_beta)
    v_unf = iu["vector"]
    cal_rows.append(dict(day=e, date=dates[e], lam=pf_.lam, mu_j=pf_.mu_j, unf=pu_))
print(f"rolling calibration: {len(refits)} refits in {time.time() - tic:.0f}s")

# PINN box from development quantities only; test-day thetas outside it are clipped (and counted)
sd_ = s_t[wu:ndev]
box = ThetaBox(s=(float(sd_.min() * mc["box_s"][0]), float(sd_.max() * mc["box_s"][1])),
               lam=(float(p_dev.lam * np.exp(-mc["box_log_lam"])), float(p_dev.lam * np.exp(mc["box_log_lam"]))),
               mu_j=(float(p_dev.mu_j - mc["box_mu_j"]), float(p_dev.mu_j + mc["box_mu_j"])))
which = np.searchsorted(refits, test, side="right") - 1               # last refit on or before each test day
theta_raw = np.column_stack([s_t[test], [cal_rows[k]["lam"] for k in which], [cal_rows[k]["mu_j"] for k in which]])
lo = np.array([box.s[0], box.lam[0], box.mu_j[0]])
hi = np.array([box.s[1], box.lam[1], box.mu_j[1]])
theta = np.clip(theta_raw, lo, hi)
n_clip = int((np.abs(theta - theta_raw) > 0).any(1).sum())
print(f"box s={box.s} lam={box.lam} mu_j={box.mu_j}; test days clipped: {n_clip}")

# ---------------------------------------------------------------- 2. amortised PINN on the development fit
cfg = NDConfig(**pc)
tr = AmortisedTrainer(p_dev, t0, T, cfg, box)
dev = tr.device
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
C = (float(np.median(sd_)), p_dev.lam, p_dev.mu_j)


def track(tr_: AmortisedTrainer) -> dict:
    Xs, lq, lp = tr_.weighted_samples_t(DAY, C, 16384)
    _, es = es_weighted_t(Xs @ tr_.a.new_tensor(w), torch.softmax(lp - lq, 0), 0.99)
    return dict(es99_rel_1d=abs(float(es) / merton_var_es(0.99, DAY, params_at(p_dev, C).portfolio(w))[1] - 1))


print(f"device={dev}  params={sum(q.numel() for q in tr.model.parameters())}  steps={cfg.steps}")
if args.eval_only:
    ck = torch.load(OUT / "model.pt", map_location=dev, weights_only=False)  # our own file
    tr.model.load_state_dict(ck["state"])
    tr.train_time, tr.peak_mem_mb = ck["train_time_s"], ck["peak_mem_mb"]
    hist = pd.read_csv(OUT / "history.csv")
else:
    tr.fit(callback=track)
    print(f"trained in {tr.train_time:.0f}s  peak GPU memory {tr.peak_mem_mb:.0f} MB")
    hist = pd.DataFrame(tr.history)
    hist.to_csv(OUT / "history.csv", index=False)
    torch.save(dict(state=tr.model.state_dict(), cfg=pc, box=vars(box), d=d, a=p_dev.a.tolist(), cov=p_dev.cov.tolist(),
                    beta=p_dev.beta.tolist(), sigma_j=p_dev.sigma_j, train_time_s=tr.train_time,
                    peak_mem_mb=tr.peak_mem_mb), OUT / "model.pt")

# ---------------------------------------------------------------- 3. non-linear book: basket + short put
wt = torch.tensor(w, dtype=torch.float32, device=dev)
ndtr = torch.special.ndtr


def put(S, K, tau, sig):
    """Black-Scholes put, zero rate, per unit of today's basket value."""
    if tau <= 1e-12:
        return torch.clamp(K - S, min=0)
    v = sig * np.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * v**2) / v
    return K * ndtr(-(d1 - v)) - S * ndtr(-d1)


def pnl_nl(Xs: torch.Tensor, st: dict) -> torch.Tensor:
    """Full revaluation over one day; implied vol held at today's level (no vega P&L)."""
    rb = (torch.exp(Xs) - 1) @ (wt if Xs.dtype == torch.float32 else wt.double())
    one = torch.ones((), dtype=Xs.dtype, device=Xs.device)
    p0 = put(one, one * st["K"], st["tau"], st["sig"])
    return rb - pf["n_puts"] * (put(1 + rb, one * st["K"], st["tau"] - DAY, st["sig"]) - p0)


sig_imp = BM.ewma_vol(y_lin, mc["ewma_decay"], wu) * np.sqrt(252)       # basket vol, data up to t-1
state, V, K = {}, 1.0, None
for j, i in enumerate(test):
    if j % pf["put_months_days"] == 0:
        K = pf["put_strike"] * V
    state[i] = dict(K=K / V, tau=(pf["put_months_days"] - j % pf["put_months_days"]) * DAY, sig=float(sig_imp[i]))
    V *= 1 + y_lin[i]
y_nl = np.array([float(pnl_nl(torch.tensor(X[i:i + 1], dtype=torch.float32, device=dev), state[i])[0]) for i in test])

# ---------------------------------------------------------------- 4. benchmark filters (refitted every month)
sig_assets = BM.ewma_vol(X, mc["ewma_decay"], wu)                     # FHS
g_rows = {}                                                            # day -> GARCH/DCC state for that day
gw, gr = bc["garch_window"], bc["garch_refit"]
tic = time.time()
dcc_ab = None
for b0 in range(test[0], test[-1] + 1, gr):
    days = [i for i in test if b0 <= i < b0 + gr]
    lo_ = b0 - gw
    g = BM.garch_fit(y_lin[lo_:b0])
    sg = BM.garch_filter(y_lin[lo_:days[-1]], g)                       # sg[k] = vol of day lo_ + k
    z = (y_lin[lo_:b0] - g["mu"]) / sg[:gw]
    evt = {a: BM.gpd_tail(-z, a, bc["evt_u_frac"]) for a in ALPHAS}
    ga = [BM.garch_fit(X[lo_:b0, k]) for k in range(d)]
    sa = np.column_stack([BM.garch_filter(X[lo_:days[-1], k], ga[k]) for k in range(d)])
    mu_a = np.array([q["mu"] for q in ga])
    e = (X[lo_:days[-1]] - mu_a) / sa[:-1]
    if dcc_ab is None or (b0 - test[0]) % bc["dcc_refit"] < gr:
        dcc_ab = BM.dcc_fit(e[:gw])
    Qbar = np.cov(e[:gw].T)
    Rt = BM.dcc_filter(e, *dcc_ab, Qbar)
    nu = float(np.median([q["nu"] for q in ga]))
    for i in days:
        k = i - lo_
        g_rows[i] = dict(g=g, sig=sg[k], evt=evt, mu_a=mu_a, sig_a=sa[k], R=Rt[k], nu=nu)
print(f"GARCH / EVT / DCC filters in {time.time() - tic:.0f}s")

# ---------------------------------------------------------------- 5. daily forecasts
Xall = torch.tensor(X, dtype=torch.float32, device=dev)
rows, pin_rows, timing = [], [], {m: 0.0 for m in MODELS}
qg = {a: stats.norm.ppf(a) for a in ALPHAS}
tvar = {}


def add(i, model, book, z, wts):
    for a in ALPHAS:
        v, e_ = es_weighted_t(z, wts, a)
        rows.append(dict(day=i, model=model, book=book, alpha=a, var=float(v), es=float(e_)))


torch.manual_seed(1)
for j, i in enumerate(test):
    th, st, gs = theta[j], state[i], g_rows[i]
    # PINN and exact density on the same proposal draws
    sync(); tic = time.perf_counter()
    Xs, lq, lp = tr.weighted_samples_t(DAY, th, ev["n_is"], r_max=ev["r_max"])
    wp = torch.softmax(lp - lq, 0)
    zl, zn = (torch.exp(Xs) - 1) @ wt, pnl_nl(Xs, st)
    add(i, "pinn", "linear", zl, wp)
    add(i, "pinn", "nonlinear", zn, wp)
    sync(); timing["pinn"] += time.perf_counter() - tic
    tic = time.perf_counter()
    le = exact_logpdf_torch(Xs.double(), torch.full((1,), DAY, device=dev, dtype=torch.float64),
                            torch.tensor(np.asarray(th)[None], device=dev, dtype=torch.float64), p_dev, k_max=16) - lq.double()
    we = torch.softmax(le, 0).float()
    add(i, "exact", "linear", zl, we)
    add(i, "exact", "nonlinear", zn, we)
    sync(); timing["exact"] += time.perf_counter() - tic
    wpd = wp.double()
    pin_rows.append(dict(day=i, ess=float(1 / (wpd**2).sum()) / len(wpd), mass=float(torch.exp(torch.logsumexp((lp - lq).double(), 0)) / len(lq)),
                         dropped=float(torch.isinf(lp).double().mean())))
    # unfiltered Merton: exact Monte Carlo at the last full refit
    tic = time.perf_counter()
    pu = cal_rows[which[j]]["unf"]
    Xu = simulate_torch(bc["n_scen"], DAY, pu, (1.0, pu.lam, pu.mu_j), dev)
    add(i, "merton_unf", "linear", (torch.exp(Xu) - 1) @ wt, None)
    add(i, "merton_unf", "nonlinear", pnl_nl(Xu, st), None)
    sync(); timing["merton_unf"] += time.perf_counter() - tic
    # historical and filtered historical simulation
    tic = time.perf_counter()
    Xh = Xall[i - bc["hs_window"]:i]
    add(i, "hs", "linear", (torch.exp(Xh) - 1) @ wt, None)
    add(i, "hs", "nonlinear", pnl_nl(Xh, st), None)
    timing["hs"] += time.perf_counter() - tic
    tic = time.perf_counter()
    sa_ = torch.tensor(sig_assets[i - bc["hs_window"]:i + 1], dtype=torch.float32, device=dev)
    Xf = Xh / sa_[:-1] * sa_[-1]
    add(i, "fhs", "linear", (torch.exp(Xf) - 1) @ wt, None)
    add(i, "fhs", "nonlinear", pnl_nl(Xf, st), None)
    timing["fhs"] += time.perf_counter() - tic
    # DCC-GARCH-t scenarios
    tic = time.perf_counter()
    Xd = torch.tensor(gs["mu_a"] + gs["sig_a"] * BM.mvt_draws(bc["n_scen"], gs["R"], gs["nu"], rng),
                      dtype=torch.float32, device=dev)
    add(i, "dcc", "linear", (torch.exp(Xd) - 1) @ wt, None)
    add(i, "dcc", "nonlinear", pnl_nl(Xd, st), None)
    timing["dcc"] += time.perf_counter() - tic
    # univariate models on the basket return
    tic = time.perf_counter()
    yw = y_lin[i - bc["gauss_window"]:i]
    m_, sd = yw.mean(), yw.std(ddof=1)
    for a in ALPHAS:
        rows.append(dict(day=i, model="gaussian", book="linear", alpha=a, var=sd * qg[a] - m_,
                         es=sd * stats.norm.pdf(qg[a]) / (1 - a) - m_))
    timing["gaussian"] += time.perf_counter() - tic
    tic = time.perf_counter()
    g = gs["g"]
    for a in ALPHAS:
        key = (round(g["nu"], 6), a)
        if key not in tvar:
            tvar[key] = BM.t_var_es(a, g["nu"])
        v_, e_ = tvar[key]
        rows.append(dict(day=i, model="garch_t", book="linear", alpha=a, var=gs["sig"] * v_ - g["mu"],
                         es=gs["sig"] * e_ - g["mu"]))
        v_, e_ = gs["evt"][a]
        rows.append(dict(day=i, model="evt", book="linear", alpha=a, var=gs["sig"] * v_ - g["mu"],
                         es=gs["sig"] * e_ - g["mu"]))
    timing["garch_t"] += time.perf_counter() - tic
    if j % 250 == 0:
        print(f"  day {j}/{len(test)}  {dates[i].date()}  theta={np.round(th, 4)}")

fc = pd.DataFrame(rows)
fc["date"] = dates[fc.day]
realised = pd.DataFrame(dict(day=test, linear=y_lin[test], nonlinear=y_nl))
fc = fc.merge(realised.melt(id_vars="day", var_name="book", value_name="y"), on=["day", "book"])
fc.to_csv(OUT / "forecasts.csv.gz", index=False, float_format="%.6g")
pin = pd.DataFrame(pin_rows)

# ---------------------------------------------------------------- 6. backtests
bt_rows = []
for (model, book, a), g in fc.groupby(["model", "book", "alpha"]):
    g = g.sort_values("day")
    bt_rows.append(dict(model=model, book=book, alpha=a, **B.summarise(g.y.values, g["var"].values, g.es.values, a)))
bt = pd.DataFrame(bt_rows)
for (book, a), g in fc.groupby(["book", "alpha"]):
    ref = g[g.model == "pinn"].sort_values("day")
    l_ref = B.fz0_loss(ref.y.values, ref["var"].values, ref.es.values, a)
    for model, gm in g.groupby("model"):
        gm = gm.sort_values("day")
        dm = B.diebold_mariano(B.fz0_loss(gm.y.values, gm["var"].values, gm.es.values, a), l_ref)
        sel = (bt.model == model) & (bt.book == book) & (bt.alpha == a)
        bt.loc[sel, "dm_stat_vs_pinn"], bt.loc[sel, "dm_p_vs_pinn"] = dm["stat"], dm["p"]
bt.to_csv(OUT / "backtest.csv", index=False)
print(bt[bt.alpha == 0.99][["model", "book", "breaches", "rate", "p_uc", "p_cc", "fz0", "dm_p_vs_pinn"]].to_string(index=False))

# PINN vs exact density, day by day
pv = fc[fc.model.isin(["pinn", "exact"])].pivot_table(index=["day", "book", "alpha"], columns="model", values="es").reset_index()
pv["rel"] = pv.pinn / pv.exact - 1

cal = pd.DataFrame([dict(day=r["day"], date=r["date"], lam=r["lam"], mu_j=r["mu_j"], unf_lam=r["unf"].lam,
                         unf_mu_j=r["unf"].mu_j, unf_sigma_j=r["unf"].sigma_j,
                         unf_vol=float(np.sqrt(w @ r["unf"].cov @ w))) for r in cal_rows])
cal.to_csv(OUT / "calibration.csv", index=False)

# ---------------------------------------------------------------- fig01: data and calibration
F.plot_training(hist, STAGE)
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
tdates = dates[test]
ax = axes[0, 0]
ax.plot(dates, np.cumsum(np.log1p(y_lin)), color=P.INK, lw=1)
ax.set(ylabel="cumulative log return", title=f"{mk['name']}: equal-weight basket")
ax = axes[0, 1]
ax.plot(dates[wu:], s_t[wu:], color=P.INK_2, lw=0.8, label="s$_t$ (EWMA)")
for v_ in box.s:
    ax.axhline(v_, color=P.METHOD["pinn"]["color"], ls="--", lw=0.8)
ax.text(dates[wu], box.s[1], " PINN training box", va="bottom", fontsize=8, color=P.METHOD["pinn"]["color"])
ax.set(ylabel="vol scale s$_t$", title="Daily vol scale (data up to t−1)")
ax = axes[1, 0]
se = info_dev["se"]
ax.plot(cal.date, cal.lam, drawstyle="steps-post", **P.style("pinn", label="filtered (rolling, shrunk)"))
ax.plot(cal.date, cal.unf_lam, drawstyle="steps-post", **P.style("merton_unf", label="unfiltered (rolling)"))
ax.fill_between(cal.date, max(p_dev.lam - 2 * se["lam"], 1e-3), p_dev.lam + 2 * se["lam"], step="post", color=P.TAIL_FILL,
                label="development fit ± 2 SE")
ax.set(yscale="log", ylabel="jumps per year λ", title="Jump intensity")
ax.legend(fontsize=7.5)
ax = axes[1, 1]
ax.plot(cal.date, 100 * cal.mu_j, drawstyle="steps-post", **P.style("pinn", label="filtered"))
ax.plot(cal.date, 100 * cal.unf_mu_j, drawstyle="steps-post", **P.style("merton_unf", label="unfiltered"))
ax.fill_between(cal.date, 100 * (p_dev.mu_j - 2 * se["mu_j"]), 100 * (p_dev.mu_j + 2 * se["mu_j"]), step="post",
                color=P.TAIL_FILL, label="development fit ± 2 SE")
ax.axhline(0, color=P.AXIS, lw=0.8)
ax.set(ylabel="mean jump μ$_J$ (%)", title="Mean jump")
ax.legend(fontsize=7.5)
for ax in axes[0]:
    ax.axvspan(tdates[0], tdates[-1], color=P.TAIL_FILL, alpha=0.6, lw=0, zorder=0)
axes[0, 0].text(tdates[0], 0.02, " test period", transform=axes[0, 0].get_xaxis_transform(), fontsize=8, color=P.INK_2)
for ax in axes[1]:
    ax.set_title(ax.get_title(loc="left") + " (test period, monthly refits)", loc="left")
P.panel_labels(axes.flat)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_data_calibration")

# ---------------------------------------------------------------- fig03: VaR paths, linear book
SHOW = ["hs", "garch_t", "merton_unf", "pinn"]
fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
g99 = fc[(fc.book == "linear") & (fc.alpha == 0.99)]
crisis = (tdates >= pd.Timestamp("2020-02-01")) & (tdates <= pd.Timestamp("2020-07-31"))
for ax, sel, ttl in ((axes[0], np.ones(len(test), bool), "Test period"), (axes[1], crisis, "COVID crash (Feb–Jul 2020)")):
    if not sel.any():
        ax.set_visible(False)
        continue
    dd = tdates[sel]
    ax.plot(dd, 100 * y_lin[test][sel], ls="", marker=".", ms=2.5, color=P.MUTED, label="realised return")
    for m in SHOW:
        gm = g99[g99.model == m].sort_values("day")
        ax.plot(dd, -100 * gm["var"].values[sel], lw=1.1, **{k: v for k, v in P.style(m, label=LABEL[m]).items() if k != "lw"})
    gp = g99[g99.model == "pinn"].sort_values("day")
    hit = (y_lin[test] < -gp["var"].values) & sel
    ax.plot(tdates[hit], 100 * y_lin[test][hit], ls="", marker="x", ms=5, color=P.RED, label="PINN breach")
    ax.set(ylabel="daily return (%)", title=f"{ttl}: VaR$_{{99}}$ forecasts")
axes[0].legend(ncol=3, fontsize=7.5, loc="lower left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_var_paths")


# ---------------------------------------------------------------- fig04: backtest summary, both books
def kupiec_band(n, alpha, level=0.95):
    """Breach rates not rejected by the Kupiec test at `level`."""
    ks = np.arange(0, int(4 * (1 - alpha) * n) + 10)
    ok = [k for k in ks if B.kupiec(np.r_[np.ones(k), np.zeros(n - k)], alpha)["p_uc"] > 1 - level]
    return min(ok) / n, max(ok) / n


fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
for r_, book in enumerate(("linear", "nonlinear")):
    ms = MODELS if book == "linear" else JOINT
    yl = np.arange(len(ms))
    for c_, a in enumerate(ALPHAS):
        ax = axes[r_, c_]
        g = bt[(bt.book == book) & (bt.alpha == a)].set_index("model").loc[ms]
        lo_b, hi_b = kupiec_band(len(test), a)
        ax.axvspan(100 * lo_b, 100 * hi_b, color=P.TAIL_FILL, label="Kupiec 95% region")
        ax.axvline(100 * (1 - a), color=P.AXIS, lw=0.8)
        ax.barh(yl, 100 * g.rate, color=[P.METHOD[m]["color"] for m in ms], height=0.65)
        for k, m in enumerate(ms):
            ax.text(100 * g.rate.iloc[k], k, f" {g.breaches.iloc[k]:.0f}" + (" ×" if g.p_cc.iloc[k] < 0.05 else ""),
                    va="center", fontsize=7.5, color=P.INK_2)
        ax.set(yticks=yl, yticklabels=[LABEL[m] for m in ms], xlabel="breach rate (%)",
               title=f"{book.capitalize()} book, VaR$_{{{100 * a:g}}}$")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if c_:
            ax.set_yticklabels([])
    ax = axes[r_, 2]
    a = 0.975
    g = bt[(bt.book == book) & (bt.alpha == a)].set_index("model").loc[ms]
    ref = g.loc["pinn", "fz0"]
    ax.barh(yl, g.fz0 - ref, color=[P.METHOD[m]["color"] for m in ms], height=0.65)
    for k, m in enumerate(ms):
        p = g.dm_p_vs_pinn.iloc[k]
        star = "" if m == "pinn" or not np.isfinite(p) else ("**" if p < 0.01 else "*" if p < 0.05 else "")
        ax.text(g.fz0.iloc[k] - ref, k, f" {star}", va="center", fontsize=9)
    ax.axvline(0, color=P.INK, lw=0.8)
    ax.set(yticks=yl, yticklabels=[], xlabel="mean FZ0 − PINN (lower = better)",
           title="ES$_{97.5}$/VaR$_{97.5}$ joint score\n(* DM p < 0.05, ** p < 0.01)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
axes[0, 0].legend(fontsize=7, loc="lower right")
fig.text(0.01, 0.005, "× = Christoffersen conditional coverage rejected at 5%; numbers = breach counts", fontsize=7.5,
         color=P.INK_2)
P.panel_labels(axes.flat)
fig.tight_layout(rect=(0, 0.02, 1, 1))
P.savefig(fig, STAGE, "fig04_backtests")

# ---------------------------------------------------------------- fig05: PINN vs exact density, non-linear book
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
ax = axes[0]
rb = np.linspace(-0.12, 0.06, 200)
st_typ = dict(K=pf["put_strike"], tau=pf["put_months_days"] * DAY, sig=float(np.median(sig_imp[test])))
Xr = torch.tensor(np.log1p(rb)[:, None].repeat(d, 1), dtype=torch.float32, device=dev)
ax.plot(100 * rb, 100 * rb, color=P.MUTED, ls="--", lw=1, label="linear book")
ax.plot(100 * rb, 100 * pnl_nl(Xr, st_typ).cpu().numpy(), color=P.INK, lw=1.8, label="basket + short put")
ax.axhline(0, color=P.AXIS, lw=0.8)
ax.set(xlabel="basket return (%)", ylabel="one-day P&L (%)", title=f"Non-linear book (put {100 * pf['put_strike']:g}% strike, 1 month)")
ax.legend(fontsize=7.5)
ax = axes[1]
for book, ls_ in (("linear", "-"), ("nonlinear", "--")):
    g = pv[(pv.book == book) & (pv.alpha == 0.99)].sort_values("day")
    ax.plot(dates[g.day], 100 * g.rel, lw=0.8, color=P.METHOD["pinn"]["color"], ls=ls_, label=f"{book} book")
ax.axhline(0, color=P.INK, lw=0.8)
ax.set(ylabel="ES$_{99}$: PINN / exact − 1 (%)", title="PINN vs exact density, every test day")
ax.legend(fontsize=7.5)
ax = axes[2]
for book, ls_ in (("linear", "-"), ("nonlinear", "--")):
    v = np.sort(np.abs(pv[(pv.book == book) & (pv.alpha == 0.99)].rel.values)) * 100
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=P.METHOD["pinn"]["color"], ls=ls_, lw=1.6,
            label=f"{book}: median {np.median(v):.2f}%")
ax.set(xscale="log", xlabel="|ES$_{99}$ PINN / exact − 1| (%)", ylabel="fraction of test days", title="Distribution over days")
ax.legend(fontsize=7.5)
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig05_pinn_vs_exact")

# ---------------------------------------------------------------- metrics
bt_dict = {f"{r.book}_{r.alpha}_{r.model}": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                             for k, v in r._asdict().items() if k not in ("book", "alpha", "model")}
           for r in bt.itertuples(index=False)}
save_results(STAGE, "metrics", dict(
    market=args.market, tickers=mk["tickers"], n_days=nT, dev_end=str(dates[ndev - 1].date()),
    test=[str(tdates[0].date()), str(tdates[-1].date())], n_test=len(test), device=dev,
    gpu=torch.cuda.get_device_name() if dev == "cuda" else "cpu", pinn_config=pc, t_range_days=[t0 / DAY, T / DAY],
    dev_fit=dict(lam=p_dev.lam, mu_j=p_dev.mu_j, sigma_j=p_dev.sigma_j, se=info_dev["se"],
                 vols=np.sqrt(np.diag(p_dev.cov)).tolist(), beta=p_dev.beta.tolist(), nll=info_dev["nll"]),
    dev_fit_unfiltered=dict(lam=p_unf_dev.lam, mu_j=p_unf_dev.mu_j, sigma_j=p_unf_dev.sigma_j, se=info_unf["se"]),
    box=vars(box), n_clipped=n_clip, train_time_s=tr.train_time, peak_mem_mb=tr.peak_mem_mb,
    pinn_vs_exact={f"{b}_{a}": dict(median_abs=float(g.rel.abs().median()), p90_abs=float(g.rel.abs().quantile(0.9)),
                                    max_abs=float(g.rel.abs().max()))
                   for (b, a), g in pv.groupby(["book", "alpha"])},
    importance_sampling=dict(ess_frac_min=float(pin.ess.min()), ess_frac_median=float(pin.ess.median()),
                             mass_min=float(pin.mass.min()), mass_max=float(pin.mass.max()),
                             r_max=ev["r_max"], dropped_frac_max=float(pin.dropped.max())),
    ms_per_day={m: 1e3 * v / len(test) for m, v in timing.items() if v > 0}, backtest=bt_dict))
print(f"done: {OUT}")
