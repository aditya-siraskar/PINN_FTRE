"""Stage 7: physics vs data-only (RQ3), failure cases, final scorecard.

Three parts, each resumable:

  --part data --d 2|5     The Stage 4 PINN (no samples, only the PIDE and the t0 density) against the
                          same network fitted by maximum likelihood to N exact samples (pinn_tailrisk.data_only),
                          N = 1e3..1e6, 3 seeds; and against Monte Carlo on the same samples. Every model is
                          weighted on the same importance draws, so differences are model differences.
                          -> figures/results stage7_physics_vs_data_d<d>/
  --part failure --case   One change to the d = 2 Stage 4 setup that the method does not handle by default:
                          high_lam | high_lam_kfix | tiny_t0 (configs/stage7_ablation.yaml).
                          -> figures/results stage7_failure_<case>/
  --part summary          Scorecard over Stages 2-7, failure-case figure, far-tail extrapolation probe.
                          -> figures/results stage7_final/

Run:  python scripts/stage7_ablation.py --part data --d 2 [--steps N] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import figs_pinn1d as F  # noqa: E402
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config, save_results  # noqa: E402
from pinn_tailrisk.data_only import DataOnlyTrainer  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MultiMertonParams, merton_var_es, multi_merton_logpdf, simulate_multi_merton  # noqa: E402
from pinn_tailrisk.pinn_nd import NDPINN, NDConfig, NDTrainer  # noqa: E402
from pinn_tailrisk.risk import var_es_from_samples, var_es_from_weighted  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage7_ablation.yaml")
ap.add_argument("--part", choices=["data", "failure", "summary"], required=True)
ap.add_argument("--d", type=int, default=2)
ap.add_argument("--case", choices=["high_lam", "high_lam_kfix", "tiny_t0"])
ap.add_argument("--steps", type=int)
ap.add_argument("--quick", action="store_true", help="few steps, small N and draw counts (CPU smoke test)")
args = ap.parse_args()

base, s3, c4 = (load_config(f"configs/{n}.yaml") for n in ("base", "stage3_multiasset", "stage4_pinn_nd"))
c7 = load_config(args.config)
pc0, ev4, ev = c4["pinn_nd"], c4["eval"], c7["eval"]
dc = c7["data_only"]
if args.steps is not None:
    pc0["steps"] = args.steps
if args.quick:
    pc0.update(steps=args.steps or 300)
    ev.update(n_is=16384, n_density=4096)
    dc.update(n_list=[1000, 10000], seeds=[0])
SUFFIX = "_quick" if args.quick else ""
H, ALPHAS = base["eval"]["horizons_days"], ev["alphas"]
dom = base["domain"]
P.use_style()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)


# ---------------------------------------------------------------- shared helpers
def family(d: int, lam: float | None = None) -> MultiMertonParams:
    """Stage 4's d-asset family (base.yaml jumps, Stage 3 vols/loadings/correlation)."""
    sc, m = s3["scaling"], dict(base["merton"])
    if lam is not None:
        m["lam"] = lam
    return MultiMertonParams.equicorrelated(
        d, sigma=np.linspace(*sc["sigma_range"], d) if d > 1 else [sc["sigma_range"][0]], rho=sc["rho"],
        a=m["a"], beta=np.linspace(*sc["beta_range"], d) if d > 1 else [1.0],
        **{k: m[k] for k in ("lam", "mu_j", "sigma_j")})


def portfolios(d: int) -> np.ndarray:
    """Stage 4's portfolios: equal weight, Dirichlet long-only, long-short (same seed, so the same weights)."""
    prng = np.random.default_rng(ev4["portfolio_seed"])
    ls = prng.normal(size=(ev4["n_long_short"], d))
    ls[:, 0] = 1 - ls[:, 1:].sum(1)
    return np.vstack([np.full(d, 1 / d), prng.dirichlet(np.ones(d), ev4["n_long_only"]), ls])


def exact_table(mp, W) -> dict:
    return {(h, a, i): merton_var_es(a, h * DAY, mp.portfolio(w)) for h in H for a in ALPHAS for i, w in enumerate(W)}


@torch.no_grad()
def logp(model, X: torch.Tensor, t: float) -> torch.Tensor:
    return torch.cat([model(c, torch.full((len(c),), t, device=c.device)) for c in X.split(1 << 16)])


def is_draws(q, t: float, n: int, seed: int):
    """Common proposal draws: every model at horizon t is weighted on these."""
    torch.manual_seed(seed)
    tv = torch.full((1,), t, device=dev)
    X = q.sample(tv, n)
    return X, torch.cat([q.log_prob(c, tv) for c in X.split(1 << 16)])


def risk_rows(name, z, w, h, exact, **extra):
    """z (n, P) portfolio returns; w (n,) probability weights or None (equal-weight samples)."""
    rows = []
    for i in range(z.shape[1]):
        for a in ALPHAS:
            v, e = var_es_from_samples(z[:, i], a) if w is None else var_es_from_weighted(z[:, i], w, a)
            ve, ee = exact[(h, a, i)]
            rows.append(dict(model=name, h=h, alpha=a, port=i, var=v, es=e, var_rel=(v - ve) / ve, es_rel=(e - ee) / ee, **extra))
    return rows


def weights(lp: torch.Tensor, lq: torch.Tensor):
    lw = (lp - lq).double()
    w = torch.softmax(lw, 0)
    mass = float(torch.exp(torch.logsumexp(lw, 0)) / len(lw))
    return w.cpu().numpy(), dict(ess_frac=float(1 / (w**2).sum()) / len(w), mass=mass)


def evaluate(models: dict, mp, q, W, exact, n_is: int, extra: dict | None = None):
    """VaR/ES of every portfolio from each model's density on common draws; + exact-density IS floor."""
    rows, diag = [], {}
    for h in H:
        t = h * DAY
        X, lq = is_draws(q, t, n_is, seed=100 + h)
        z = X.cpu().numpy().astype(float) @ W.T
        for name, m in models.items():
            w, dg = weights(logp(m, X, t), lq)
            rows += risk_rows(name, z, w, h, exact, **(extra or {}))
            diag[f"{name}_{h}d"] = dg
        if "physics" in models:
            w, dg = weights(torch.as_tensor(multi_merton_logpdf(X.cpu().numpy().astype(float), t, mp), device=dev), lq)
            rows += risk_rows("is_exact", z, w, h, exact)
            diag[f"is_exact_{h}d"] = dg
    return rows, diag


def density_error(model, mp, W, exact, n: int) -> dict:
    """Mean |log p - log p_exact| on exact samples; tail = equal-weight portfolio beyond its VaR99."""
    out = {}
    for h in H:
        t = h * DAY
        Xs = simulate_multi_merton(n, t, mp, np.random.default_rng(500 + h))
        err = np.abs(logp(model, torch.as_tensor(Xs, dtype=torch.float32, device=dev), t).cpu().numpy() - multi_merton_logpdf(Xs, t, mp))
        tail = Xs @ W[0] <= -exact[(h, 0.99, 0)][0]
        out[f"{h}d"] = dict(mae_log=float(err.mean()), tail_mae_log=float(err[tail].mean()))
    return out


def med_err(df: pd.DataFrame, by=("model", "h", "alpha")) -> pd.DataFrame:
    """Median over portfolios of |ES rel. error| (and VaR)."""
    return df.assign(es_abs=df.es_rel.abs(), var_abs=df.var_rel.abs()).groupby(list(by))[["es_abs", "var_abs"]].median().reset_index()


def physics_trainer(mp, t0, T, pc, out: Path, W, exact):
    """Train the Stage 4 PINN (or load out/model.pt) and return the trainer and its history."""
    tr = NDTrainer(mp, t0, T, NDConfig(**pc))

    def track(tr_):
        res = {}
        for h in (1, 10):
            X, lq = is_draws(tr_.q_is, h * DAY, 65536, seed=7)
            w, _ = weights(logp(tr_.model, X, h * DAY), lq)
            _, es = var_es_from_weighted(X.cpu().numpy().astype(float) @ W[0], w, 0.99)
            res[f"es99_rel_{h}d"] = abs(es / exact[(h, 0.99, 0)][1] - 1)
        return res

    if (out / "model.pt").exists():
        ck = torch.load(out / "model.pt", map_location=tr.device, weights_only=False)  # our own file
        tr.model.load_state_dict(ck["state"])
        tr.train_time = ck["train_time_s"]
        return tr, pd.read_csv(out / "history.csv")
    print(f"training the PINN: d={mp.d} trace={tr.trace} steps={pc['steps']} device={tr.device}")
    tr.fit(callback=track)
    hist = pd.DataFrame(tr.history)
    hist.to_csv(out / "history.csv", index=False)
    torch.save(dict(state=tr.model.state_dict(), cfg=pc, train_time_s=tr.train_time), out / "model.pt")
    print(f"PINN trained in {tr.train_time / 60:.1f} min")
    return tr, hist


H_COL = dict(zip(H, F.HORIZON_COLORS))


# ================================================================ part 1: physics vs data-only
def part_data(d: int):
    STAGE = f"stage7_physics_vs_data_d{d}{SUFFIX}"
    out = RESULTS / STAGE
    (out / "runs").mkdir(parents=True, exist_ok=True)
    mp, W = family(d), portfolios(d)
    exact = exact_table(mp, W)
    t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
    tr, hist = physics_trainer(mp, t0, T, dict(pc0), out, W, exact)
    F.plot_training(hist, STAGE)
    rows, diag = evaluate({"physics": tr.model}, mp, tr.q_is, W, exact, ev["n_is"])
    phys = pd.DataFrame(rows)
    phys_dens = density_error(tr.model, mp, W, exact, ev["n_density"])

    times = np.unique(np.concatenate([np.geomspace(t0, T, dc["n_times"]), np.array(H) * DAY]))
    j_of_h = {h: int(np.argmin(np.abs(times - h * DAY))) for h in H}
    for N in dc["n_list"]:
        for seed in dc["seeds"]:
            f = out / "runs" / f"N{N}_s{seed}.csv"
            if f.exists():
                continue
            rng = np.random.default_rng(10_000 * seed + 17)
            n_per = max(N // len(times), 1)
            X = np.vstack([simulate_multi_merton(n_per, t, mp, rng) for t in times])
            ti = np.repeat(np.arange(len(times)), n_per)
            cfg = dict(pc0, seed=seed)
            dt = DataOnlyTrainer(mp, t0, T, NDConfig(**cfg), X, ti, times, dc["val_frac"], dc["batch"], dc["n_val_norm"])
            dt.fit()
            r, dg = evaluate({"data": dt.model}, mp, tr.q_is, W, exact, ev["n_is"], dict(N=N, seed=seed))
            for h in H:                              # Monte Carlo on the same samples at that horizon
                Z = X[ti == j_of_h[h]] @ W.T
                r += risk_rows("mc_same", Z, None, h, exact, N=N, seed=seed)
            de = density_error(dt.model, mp, W, exact, ev["n_density"])
            df = pd.DataFrame(r).assign(train_s=dt.train_time, best_step=dt.best_step, n_per_time=n_per,
                                        **{f"tail_mae_{h}d": de[f"{h}d"]["tail_mae_log"] for h in H},
                                        **{f"mae_{h}d": de[f"{h}d"]["mae_log"] for h in H},
                                        **{f"ess_{h}d": dg[f"data_{h}d"]["ess_frac"] for h in H})
            df.to_csv(f, index=False)
            if seed == dc["seeds"][0]:
                torch.save(dict(state=dt.model.state_dict()), out / "runs" / f"N{N}_s{seed}.pt")
            print(f"data-only N={N:>8} seed={seed}: {dt.train_time:.0f}s, best step {dt.best_step}; "
                  f"ES99 1d median err {df[(df.model == 'data') & (df.h == 1) & (df.alpha == 0.99)].es_rel.abs().median():.3%}")

    runs = pd.concat([pd.read_csv(p) for p in sorted((out / "runs").glob("N*_s*.csv"))], ignore_index=True)
    runs = runs[runs.N.isin(dc["n_list"]) & runs.seed.isin(dc["seeds"])]
    pd.concat([phys, runs], ignore_index=True).to_csv(out / "risk_table.csv.gz", index=False)
    mp_ = med_err(phys)
    md_ = med_err(runs, ("model", "N", "seed", "h", "alpha"))
    agg = md_.groupby(["model", "N", "h", "alpha"]).es_abs.agg(["median", "min", "max"]).reset_index()
    tails = runs.groupby(["N", "seed"]).first().reset_index()

    # ---- fig02: ES99 error vs N
    fig, axes = plt.subplots(1, len(H), figsize=(11.5, 3.6), sharey=True)
    for ax, h in zip(axes, H):
        for m, key in (("data", "data_nn"), ("mc_same", "mc")):
            a = agg[(agg.model == m) & (agg.h == h) & (agg.alpha == 0.99)]
            lab = "Data-only NN (same network, MLE)" if m == "data" else "Monte Carlo on the same samples"
            ax.plot(a.N, 100 * a["median"], marker=P.METHOD[key]["marker"], ms=6, **P.style(key, label=lab))
            ax.fill_between(a.N, 100 * a["min"], 100 * a["max"], color=P.METHOD[key]["color"], alpha=0.15, lw=0)
        pe = float(mp_[(mp_.model == "physics") & (mp_.h == h) & (mp_.alpha == 0.99)].es_abs.iloc[0])
        fl = float(mp_[(mp_.model == "is_exact") & (mp_.h == h) & (mp_.alpha == 0.99)].es_abs.iloc[0])
        ax.axhline(100 * pe, **P.style("pinn", label="PINN (no samples)"))
        ax.axhline(100 * fl, color=P.MUTED, lw=1, ls=":", label="IS sampling floor")
        ax.set(xscale="log", yscale="log", title=f"{h}-day", xlabel="exact samples N (all times)")
    axes[0].set_ylabel("|ES$_{99}$ error| (%), median over portfolios")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    P.panel_labels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    P.savefig(fig, STAGE, "fig02_es_vs_samples")

    # ---- fig03: density slices along beta
    show_n = [n for n in dict.fromkeys((dc["n_list"][1], dc["n_list"][-1])) if (out / "runs" / f"N{n}_s{dc['seeds'][0]}.pt").exists()]
    nets = {}
    for n in show_n:
        net = NDPINN(mp, t0, T, NDConfig(**pc0)).to(dev)
        net.load_state_dict(torch.load(out / "runs" / f"N{n}_s{dc['seeds'][0]}.pt", map_location=dev)["state"])
        nets[n] = net
    beta_hat = mp.beta / np.linalg.norm(mp.beta)
    s = np.linspace(-0.6, 0.2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    data_ls = [(0, (1, 1)), (0, (4, 2))]
    for ax, h in zip(axes, (1, 10)):
        t = h * DAY
        Xl = mp.a * t + s[:, None] * beta_hat
        le = multi_merton_logpdf(Xl, t, mp) / np.log(10)
        keep = le > le.max() - 12
        Xt = torch.as_tensor(Xl[keep], dtype=torch.float32, device=dev)
        ax.plot(100 * s[keep], le[keep], **P.style("exact", lw=1.4))
        ax.plot(100 * s[keep], logp(tr.model, Xt, t).cpu().numpy() / np.log(10), **P.style("pinn", label="PINN (no samples)"))
        for (n, net), ls in zip(nets.items(), data_ls):
            ax.plot(100 * s[keep], logp(net, Xt, t).cpu().numpy() / np.log(10), **P.style("data_nn", ls=ls, label=f"Data-only, N = {n:,}"))
        ax.set(title=f"{h}-day, along the jump direction $\\beta$", xlabel="distance from the drift point (%)")
    axes[0].set_ylabel(r"$\log_{10}$ joint density")
    axes[0].legend(loc="lower center")
    P.panel_labels(axes)
    fig.tight_layout()
    P.savefig(fig, STAGE, "fig03_density_slices")

    # ---- fig04: tail log-density error vs N
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for h in H:
        g = tails.groupby("N")[f"tail_mae_{h}d"].median()
        ax.plot(g.index, g.values, marker="*", ms=7, color=H_COL[h], ls=":", lw=1.6, label=f"data-only, {h}-day")
        ax.axhline(phys_dens[f"{h}d"]["tail_mae_log"], color=H_COL[h], lw=1.6, label=f"PINN, {h}-day")
    ax.set(xscale="log", yscale="log", xlabel="exact samples N", ylabel="mean |log-density error|, tail",
           title="Tail density error (worst 1% of the equal-weight book)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    P.savefig(fig, STAGE, "fig04_tail_density_error")

    # N at which data-only matches the PINN (log-log interpolation of the median curve)
    match = {}
    for h in H:
        a = agg[(agg.model == "data") & (agg.h == h) & (agg.alpha == 0.99)].sort_values("N")
        pe = float(mp_[(mp_.model == "physics") & (mp_.h == h) & (mp_.alpha == 0.99)].es_abs.iloc[0])
        e, n = np.log(a["median"].values), np.log(a.N.values.astype(float))
        k = np.where(e <= np.log(pe))[0]
        if not len(k):
            match[f"{h}d"] = None                    # data-only never reaches the PINN in the N range
        elif k[0] == 0:
            match[f"{h}d"] = float(a.N.values[0])    # already at the smallest N
        else:                                        # log-log interpolation between the bracketing N
            i = k[0]
            match[f"{h}d"] = float(np.exp(n[i - 1] + (np.log(pe) - e[i - 1]) * (n[i] - n[i - 1]) / (e[i] - e[i - 1])))
    save_results(STAGE, "metrics", dict(
        d=d, device=dev, pinn_config=pc0, data_only=dc, times_days=(times / DAY).tolist(),
        physics_train_time_s=tr.train_time, physics_es=mp_.to_dict(orient="records"), physics_density_error=phys_dens,
        physics_is=diag, data_only_es=agg.to_dict(orient="records"),
        data_only_train_s_median=float(runs.groupby(["N", "seed"]).train_s.first().median()),
        n_to_match_physics_es99=match))
    print(agg[agg.alpha == 0.99].pivot_table(index=["model", "N"], columns="h", values="median").map(lambda v: f"{v:.2%}"))
    print("PINN:", {h: f"{float(mp_[(mp_.model == 'physics') & (mp_.h == h) & (mp_.alpha == 0.99)].es_abs.iloc[0]):.2%}" for h in H})
    print("N to match the PINN:", match)


# ================================================================ part 2: failure cases
def part_failure(case: str):
    over = c7["failure"][case]
    STAGE = f"stage7_failure_{case}{SUFFIX}"
    out = RESULTS / STAGE
    out.mkdir(parents=True, exist_ok=True)
    d = 2
    mp, W = family(d, over.get("lam")), portfolios(d)
    exact = exact_table(mp, W)
    pc = dict(pc0)
    if "k_max" in over:
        pc["k_max"] = over["k_max"]
    t0, T = over.get("t0_days", dom["t0_days"]) * DAY, dom["T_days"] * DAY
    tr, hist = physics_trainer(mp, t0, T, pc, out, W, exact)
    F.plot_training(hist, STAGE)
    rows, diag = evaluate({"physics": tr.model}, mp, tr.q_is, W, exact, ev["n_is"])
    rng = np.random.default_rng(3)
    for h in H:
        Z = simulate_multi_merton(ev["n_is"], h * DAY, mp, rng) @ W.T
        rows += risk_rows("mc", Z, None, h, exact)
    risk = pd.DataFrame(rows)
    risk.to_csv(out / "risk_table.csv", index=False)
    med = med_err(risk)
    dens = density_error(tr.model, mp, W, exact, ev["n_density"])
    wide = med.pivot_table(index=["h", "alpha"], columns="model", values=["es_abs", "var_abs"])
    wide.columns = [f"{'es' if a == 'es_abs' else 'var'}_rel_{'pinn' if m == 'physics' else m}" for a, m in wide.columns]
    F.plot_var_es_error(wide.reset_index(), H, ALPHAS, STAGE, methods=("pinn", "mc"),
                        ylabel="{} error| vs exact (%), median over portfolios", name="fig02_var_es_error")
    p_tail = {f"{h}d": float(stats.poisson.sf(pc["k_max"], mp.lam * h * DAY)) for h in H}
    save_results(STAGE, "metrics", dict(
        case=case, override=over, d=d, device=dev, pinn_config=pc, t0_days=t0 / DAY, lam=mp.lam,
        train_time_s=tr.train_time, es=med.to_dict(orient="records"), importance_sampling=diag, density_error=dens,
        p_more_jumps_than_proposal=p_tail))
    print(med[med.alpha == 0.99].pivot_table(index="model", columns="h", values="es_abs").map(lambda v: f"{v:.2%}"))
    print("P(more jumps than the proposal covers):", p_tail)


# ================================================================ part 3: summary
def _load(stage):
    p = RESULTS / stage / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _es99(recs, model, h, key="es_abs"):
    for r in recs:
        if r["model"] == model and r["h"] == h and r["alpha"] == 0.99:
            return r[key]
    return np.nan


def part_summary():
    STAGE = f"stage7_final{SUFFIX}"
    (RESULTS / STAGE).mkdir(parents=True, exist_ok=True)
    rows = []

    def add(stage, setting, h, pinn, base_name=None, base_err=np.nan):
        rows.append(dict(stage=stage, setting=setting, horizon=h, pinn=pinn, baseline=base_name, baseline_err=base_err))

    m = _load("stage2_pinn1d")
    if m:
        for r in m["risk"]:
            if r["alpha"] == 0.99:
                add("2", "1-D Merton", r["h"], abs(r["es_rel_pinn"]), "fd", abs(r["es_rel_fd"]))
    for d in (1, 2, 5, 10):
        m = _load(f"stage4_pinn_nd_d{d}")
        if m:
            for r in m["median_abs_rel"]:
                if r["alpha"] == 0.99:
                    add("4", f"d = {d}", r["h"], r["es_rel_pinn"], "mc", r["es_rel_mc"])
    for d in (2, 5):
        m = _load(f"stage5_amortised_d{d}")
        if m:
            for k, v in m["es99_median_over_grid"].items():
                add("5", f"amortised d = {d}", int(k[:-1]), v["es_rel_pinn"], "mc", v["es_rel_mc"])
    for mk, name in (("nifty", "NIFTY basket"), ("sp", "S&P sector basket")):
        m = _load(f"stage6_realdata_{mk}")
        if m:
            add("6", f"{name} (vs exact model, daily)", 1, m["pinn_vs_exact"]["linear_0.99"]["median_abs"])
    for d in (2, 5):
        m = _load(f"stage7_physics_vs_data_d{d}{SUFFIX}")
        if m:
            nmax = max(r["N"] for r in m["data_only_es"])
            for h in H:
                de = [r["median"] for r in m["data_only_es"] if r["model"] == "data" and r["N"] == nmax and r["h"] == h and r["alpha"] == 0.99]
                add("7", f"d = {d}, physics vs data-only (N = {nmax:,})", h, _es99(m["physics_es"], "physics", h), "data_nn", de[0])
    sc = pd.DataFrame(rows)
    sc.to_csv(RESULTS / STAGE / "scorecard.csv", index=False)

    # ---- fig01: scorecard
    BASE = {"fd": "Finite difference", "mc": "Monte Carlo", "data_nn": "Data-only NN"}
    settings = list(dict.fromkeys(f"S{r.stage}: {r.setting}" for r in sc.itertuples()))
    ypos = {s: i for i, s in enumerate(reversed(settings))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(settings) + 1.6), sharey=True)
    for ax, h in zip(axes, (1, 10)):
        seen = set()
        for r in sc[sc.horizon == h].itertuples():
            y = ypos[f"S{r.stage}: {r.setting}"]
            if r.baseline and np.isfinite(r.baseline_err):
                lab = BASE[r.baseline] if r.baseline not in seen else None
                seen.add(r.baseline)
                ax.plot([100 * r.pinn, 100 * r.baseline_err], [y, y], color=P.GRID, lw=2, zorder=1)
                ax.plot(100 * r.baseline_err, y, marker=P.METHOD[r.baseline]["marker"], ms=7, ls="", color=P.METHOD[r.baseline]["color"], label=lab, zorder=2)
            ax.plot(100 * r.pinn, y, marker="o", ms=7, ls="", color=P.METHOD["pinn"]["color"], label="PINN" if "pinn" not in seen else None, zorder=3)
            seen.add("pinn")
        ax.axvline(1.0, color=P.AXIS, lw=0.8)
        ax.set(xscale="log", title=f"{h}-day ES$_{{99}}$", xlabel="|error| vs exact (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set(yticks=list(ypos.values()), yticklabels=list(ypos.keys()))
    hs, ls_ = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls_:
                hs.append(hh); ls_.append(ll)
    fig.legend(hs, ls_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    P.panel_labels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.savefig(fig, STAGE, "fig01_scorecard")

    # ---- fig02: failure cases (a) and far-tail extrapolation (b)
    cases = []
    m = _load("stage4_pinn_nd_d2")
    if m:
        cases.append(("d = 2 (Stage 4 reference)", {r["h"]: r["es_rel_pinn"] for r in m["median_abs_rel"] if r["alpha"] == 0.99}))
    for case, lab in (("high_lam", "λ = 50/yr, proposal k ≤ 3"), ("high_lam_kfix", "λ = 50/yr, proposal k ≤ 10"),
                      ("tiny_t0", "t$_0$ = 0.05 day")):
        m = _load(f"stage7_failure_{case}{SUFFIX}")
        if m:
            cases.append((lab, {h: _es99(m["es"], "physics", h) for h in H}))
    m = _load("stage5_amortised_d5")
    if m:
        cases.append(("d = 5 amortised, worst θ (Stage 5)", {int(k[:-1]): v["es_rel_pinn"] for k, v in m["es99_worst_theta"].items()}))
    m = _load("stage4_pinn_nd_d10")
    if m:
        cases.append(("d = 10 (Stage 4)", {r["h"]: r["es_rel_pinn"] for r in m["median_abs_rel"] if r["alpha"] == 0.99}))

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * max(len(cases), 4) + 1.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    for i, (lab, e) in enumerate(reversed(cases)):
        for h in H:
            if h in e and np.isfinite(e[h]):
                ax.plot(100 * e[h], i + (h - 5) * 0.04, marker="o", ms=7, ls="", color=H_COL[h], label=f"{h}-day" if i == 0 else None)
    ax.axvline(1.0, color=P.AXIS, lw=0.8)
    ax.set(xscale="log", yticks=range(len(cases)), yticklabels=[c[0] for c in reversed(cases)],
           xlabel="|ES$_{99}$ error| vs exact (%), median over portfolios", title="Where the PINN breaks")
    ax.grid(axis="y", visible=False)
    ax.legend(title="horizon", loc="lower right")

    ax = axes[1]
    phys_pt = RESULTS / f"stage7_physics_vs_data_d2{SUFFIX}" / "model.pt"
    if phys_pt.exists():
        mp = family(2)
        tr = NDTrainer(mp, dom["t0_days"] * DAY, dom["T_days"] * DAY, NDConfig(**pc0))
        tr.model.load_state_dict(torch.load(phys_pt, map_location=tr.device, weights_only=False)["state"])
        g = torch.Generator().manual_seed(0)
        u = torch.randn(ev["n_rays"], 2, generator=g)
        u = (u / u.norm(dim=1, keepdim=True)).to(dev)
        far = {}
        for h, ls in ((1, "-"), (10, (0, (4, 2)))):
            t = h * DAY
            tv = torch.full((1,), t, device=dev)
            res = []
            for r in ev["ray_radii"]:
                x = tr.a * t + (r * np.sqrt(t)) * u @ tr.L.T
                lq = tr.q_is.log_prob(x, tv)
                lp = logp(tr.model, x, t)
                le = torch.as_tensor(multi_merton_logpdf(x.cpu().numpy().astype(float), t, mp), device=dev)
                res.append((float((lp - lq).max()), float((le - lq).max())))
            res = np.array(res)
            far[f"{h}d"] = dict(radii=ev["ray_radii"], pinn=res[:, 0].tolist(), exact=res[:, 1].tolist())
            ax.plot(ev["ray_radii"], res[:, 0], **P.style("pinn", ls=ls, label=f"PINN, {h}-day"))
            ax.plot(ev["ray_radii"], res[:, 1], **P.style("exact", lw=1.3, ls=ls if h == 1 else (0, (1, 1)), label=f"exact, {h}-day"))
        ax.axhline(0, color=P.AXIS, lw=0.8)
        ax.set(xscale="log", xlabel="radius r (whitened diffusion units, $\\sqrt{t}\\,L$)",
               ylabel="max over directions of log p − log q", title="Far tail: PINN density vs IS proposal")
        ax.text(ev["ray_radii"][0], 2, "above 0: a single draw can carry all the IS weight", ha="left", va="bottom", fontsize=8, color=P.INK_2)
        ax.legend(fontsize=8, loc="lower left")
        json.dump(far, open(RESULTS / STAGE / "far_tail_rays.json", "w"), indent=2)
    else:
        ax.text(0.5, 0.5, "run --part data --d 2 first", transform=ax.transAxes, ha="center")
    P.panel_labels(axes)
    fig.tight_layout()
    P.savefig(fig, STAGE, "fig02_failure_cases")

    pd.DataFrame([dict(case=c, **{f"{h}d": e.get(h, np.nan) for h in H}) for c, e in cases]).to_csv(RESULTS / STAGE / "failure_cases.csv", index=False)
    print(sc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


tic = time.time()
if args.part == "data":
    part_data(args.d)
elif args.part == "failure":
    if not args.case:
        ap.error("--part failure needs --case")
    part_failure(args.case)
else:
    part_summary()
print(f"done in {(time.time() - tic) / 60:.1f} min")
