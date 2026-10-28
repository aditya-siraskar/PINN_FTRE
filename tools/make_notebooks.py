"""Generate the Colab notebooks in notebooks/. Run with any Python (stdlib only):

    python tools/make_notebooks.py            # writes notebooks/*.ipynb

Edit this file, not the .ipynb files, so all notebooks stay consistent.
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)
REPO = "aditya-siraskar/PINN_FTRE"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def badge(fname):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{fname})")


SETUP = r'''
# @title Setup: repo on Google Drive, install, GPU check  { display-mode: "form" }
USE_DRIVE = True   # keep figures/results across Colab sessions (recommended)
REPO_URL = "https://github.com/aditya-siraskar/PINN_FTRE"

import os, sys
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive"
else:
    BASE = "/content"
ROOT = f"{BASE}/PINN_FTRE"
if not os.path.exists(ROOT):
    !git clone -q {REPO_URL} "{ROOT}"
else:
    !git -C "{ROOT}" pull -q --ff-only
%cd "{ROOT}"
!pip install -q -e .
sys.path.insert(0, ROOT)

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (Runtime > Change runtime type > T4 GPU)")
'''

IMPORTS = r'''
# autoreload picks up library edits without a restart. Colab's IPython build imports the
# removed `imp` module on Python >= 3.12, so it is optional: if it is skipped, use
# Runtime > Restart session after editing pinn_tailrisk/.
try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except ModuleNotFoundError:
    print("autoreload unavailable here; restart the session after editing the library")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pinn_tailrisk import plotting as P
from pinn_tailrisk.notebook import show_figures, load_metrics
P.use_style()
'''


def notebook(fname, cells):
    nb = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    (OUT / fname).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def header(fname, title, phase, goal, outputs):
    return md(f"""
{badge(fname)}

# {title}

**Phase:** {phase}

**Goal:** {goal}

**Outputs:** {outputs}

> Runtime → Change runtime type → **T4 GPU**. Run the setup cell first. With `USE_DRIVE = True` the repo lives in
> `MyDrive/PINN_FTRE`, so figures and results persist. Core logic lives in `pinn_tailrisk/` and `scripts/`;
> this notebook only runs it and shows the results.
""")


# ------------------------------------------------------------------ 00
notebook("00_setup_and_tests.ipynb", [
    header("00_setup_and_tests.ipynb", "00 · Setup and test suite", "all",
           "Check the environment, run the tests of the reference implementations, preview the figure style.",
           "test report; style preview"),
    code(SETUP),
    md("## 1. Tests\nAll six must pass before any stage is trusted."),
    code("!python -m pytest -q"),
    md("## 2. Figure style preview\nEach method keeps one color, linestyle and marker for the whole project."),
    code(IMPORTS),
    code(r'''
x = np.linspace(0, 1, 50)
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (k, s) in enumerate(P.METHOD.items()):
    ax.plot(x, x * 0 + i, **P.style(k), marker=s["marker"] or None, markevery=7)
ax.set(yticks=[], title="Method styles used in every figure", xlabel="")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.05))
plt.show()
'''),
])

# ------------------------------------------------------------------ 01
notebook("01_phase1_stage2a_reference.ipynb", [
    header("01_phase1_stage2a_reference.ipynb", "01 · Stage 2a: ground truth (exact vs FD vs Monte Carlo)",
           "Phase 1: validation (1-D)",
           "Make sure the three reference routes agree before any PINN is trained (Box A, the maths check).",
           "`figures/stage2_reference/fig01–fig05`, `results/stage2_reference/metrics.json`"),
    code(SETUP),
    code(IMPORTS),
    md("## 1. Configuration\nThe ground truth shared by every stage (`configs/base.yaml`)."),
    code("print(open('configs/base.yaml').read())"),
    md("## 2. Run the stage\nTakes a few minutes (fig05 repeats Monte Carlo 200× per N)."),
    code("!python scripts/stage2_reference.py"),
    md("""
## 3. Figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_target_density | the log-scale panel shows the jump-fattened left tail vs the Gaussian |
| fig02_fd_mc_validation | FD and MC lie on the exact curve down to ~1e-5; FD log-error below 1e-3 where p > 1e-6 |
| fig03_fd_convergence | L¹ and ES₉₉ errors fall with slope ≈ 2 (FD runs on the wide `fd_domain`, so no truncation floor) |
| fig04_var_es_vs_alpha | Gaussian ES₉₉ clearly below the jump-diffusion ES₉₉ |
| fig05_mc_cost | MC error ∝ N^(−1/2); MC cost grows linearly. This is the motivation for RQ5 |
"""),
    code('show_figures("stage2_reference")'),
    md("## 4. Numbers"),
    code(r'''
m = load_metrics("stage2_reference")
rows = [dict(horizon=h, alpha=a, VaR_exact=v["exact"][0], VaR_fd=v["fd"][0], ES_exact=v["exact"][1], ES_fd=v["fd"][1])
        for h, d in m["fd_reference"]["var_es"].items() for a, v in d.items()]
df = pd.DataFrame(rows); df["ES_rel_err_fd"] = (df.ES_fd - df.ES_exact) / df.ES_exact
display(df.style.format(precision=5))
display(pd.DataFrame(m["mc_convergence"]))
'''),
])

# ------------------------------------------------------------------ 02
notebook("02_phase1_stage2b_pinn1d.ipynb", [
    header("02_phase1_stage2b_pinn1d.ipynb", "02 · Stage 2b: 1-D PINN for the jump-diffusion PIDE",
           "Phase 1: validation (1-D)",
           "Train the tail-focused PINN and measure density accuracy (bulk vs tail, RQ1) and VaR/ES accuracy against the exact answer.",
           "`figures/stage2_pinn1d/fig01–fig05`, `results/stage2_pinn1d/{metrics.json, history.csv, risk_table.csv, model.pt}`"),
    code(SETUP),
    code(IMPORTS),
    md("## 1. Hyperparameters (`configs/stage2_pinn1d.yaml`)\n`n_gh = 48` jump-quadrature nodes (24 left the 1-day ES₉₉ error at 1.8%; see notebook 03)."),
    code("print(open('configs/stage2_pinn1d.yaml').read())"),
    md("## 2. Smoke run\n500 steps: checks that the code runs and measures speed. Results go to `stage2_pinn1d_smoke`."),
    code("!python scripts/stage2_pinn1d.py --steps 500 --tag smoke"),
    code(r'''
h = pd.read_csv("results/stage2_pinn1d_smoke/history.csv")
print(f"{h.step.iloc[-1] / h.time.iloc[-1]:.1f} steps/s  ->  full run of 20k steps ≈ {20000 / (h.step.iloc[-1] / h.time.iloc[-1]) / 60:.0f} min")
'''),
    md("## 3. Full training run"),
    code("!python scripts/stage2_pinn1d.py"),
    md("""
## 4. Figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_training | all loss terms decrease; ES₉₉ error ends below 1% at 1 d and 10 d |
| fig02_density | PINN on top of the exact curve inside the worst-1% band |
| fig03_error_map | no systematic error band near the 1% quantile line |
| fig04_bulk_vs_tail | **RQ1**: how much worse the tail is than the bulk, for PINN vs FD |
| fig05_var_es_error | PINN VaR/ES errors at every α, compared with FD |
"""),
    code('show_figures("stage2_pinn1d")'),
    md("_fig01, fig04 and fig05 can be redrawn from saved results without retraining:_ `!python scripts/replot_pinn1d.py`"),
    md("## 5. Numbers"),
    code(r'''
m = load_metrics("stage2_pinn1d")
print(f"train {m['train_time_s']/60:.1f} min | PINN VaR/ES query {m['pinn_query_ms']:.1f} ms | FD solve {m['fd_solve_s']*1e3:.0f} ms")
display(pd.DataFrame(m["density_error"]).T.style.format("{:.2e}"))
r = pd.read_csv("results/stage2_pinn1d/risk_table.csv")
display(r[["h", "alpha", "var_exact", "var_pinn", "es_exact", "es_pinn", "var_rel_pinn", "es_rel_pinn", "es_rel_fd"]].style.format(precision=5))
'''),
])

# ------------------------------------------------------------------ 03
notebook("03_phase1_stage2c_ablations.ipynb", [
    header("03_phase1_stage2c_ablations.ipynb", "03 · Stage 2c: ablations of the PINN design choices",
           "Phase 1: validation (1-D)",
           "Test each tail-specific design choice instead of assuming it helps (log residual, tail sampling, Gauss–Hermite nodes, normalisation).",
           "`configs/ablations/*.yaml`, `results/stage2_pinn1d_<tag>/`, `figures/stage2_ablation/fig01_ablation_summary`"),
    code(SETUP),
    code(IMPORTS),
    md("""
## 1. Variants
Each variant changes **one** setting relative to the baseline `configs/stage2_pinn1d.yaml` (now `n_gh = 48`),
and each runs with **3 seeds**, so differences can be told apart from seed noise.
18 runs × ~9 min ≈ **2.7 h on a T4**. Set `SEEDS = [0]` and/or `STEPS = 5000` for a quick pass.
"""),
    code(r'''
import yaml, copy, pathlib
STEPS = 20000
SEEDS = [0, 1, 2]
base = yaml.safe_load(open("configs/stage2_pinn1d.yaml"))
VARIANTS = {                       # tag: change vs baseline
    "baseline":       {},
    "linear_resid":   {"residual": "linear"},
    "no_tail_sampl":  {"tail_share": 0.0},
    "gh8":            {"n_gh": 8},
    "gh24":           {"n_gh": 24},
    "no_norm":        {"w_norm": 0.0},
}
pathlib.Path("configs/ablations").mkdir(exist_ok=True)
for tag, change in VARIANTS.items():
    c = copy.deepcopy(base); c["pinn"].update(change)
    yaml.safe_dump(c, open(f"configs/ablations/{tag}.yaml", "w"))
print(list(VARIANTS), "x seeds", SEEDS)
'''),
    md("## 2. Train all runs\nResumable: finished runs are saved to Drive and skipped on rerun, so a Colab disconnect only costs the run in progress."),
    code(r'''
for tag in VARIANTS:
    for seed in SEEDS:
        run = f"ab2_{tag}_s{seed}"
        if pathlib.Path(f"results/stage2_pinn1d_{run}/metrics.json").exists():
            print("skip", run); continue
        !python scripts/stage2_pinn1d.py --config configs/ablations/{tag}.yaml --steps {STEPS} --seed {seed} --tag {run}
'''),
    md("""
## 3. Summary figure
Bars are the mean over seeds; the black whiskers span min–max, and dots are the individual seeds.
A difference counts only if the whiskers do not overlap.
"""),
    code(r'''
rows = []
for tag in VARIANTS:
    for seed in SEEDS:
        run = f"stage2_pinn1d_ab2_{tag}_s{seed}"
        if not pathlib.Path(f"results/{run}/metrics.json").exists():
            continue
        m = load_metrics(run); r = pd.DataFrame(m["risk"])
        for h in (1, 10):
            q = r[(r.h == h) & (r.alpha == 0.99)].iloc[0]
            rows.append(dict(variant=tag, seed=seed, horizon=f"{h}-day", es99_abs_rel=abs(q.es_rel_pinn),
                             var99_abs_rel=abs(q.var_rel_pinn),
                             tail_log_mae=m["density_error"][f"{h}d"]["left_tail_mae_log"], train_min=m["train_time_s"] / 60))
ab = pd.DataFrame(rows)
summary = ab.groupby(["variant", "horizon"])[["es99_abs_rel", "var99_abs_rel", "tail_log_mae", "train_min"]].agg(["mean", "min", "max"])
display(summary.style.format("{:.2e}"))
ab.to_csv("results/stage2_ablation_seeds.csv", index=False)

order = [v for v in VARIANTS if v in set(ab.variant)]
y = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, col, xl, title in [(axes[0], "es99_abs_rel", "|ES₉₉ relative error|", "Risk accuracy"),
                           (axes[1], "tail_log_mae", "left-tail mean |log error|", "Tail density accuracy")]:
    for h, c, dy in [("1-day", P.SEQ_BLUE(0.45), -0.17), ("10-day", P.SEQ_BLUE(0.95), 0.17)]:
        g = ab[ab.horizon == h].groupby("variant")[col]
        mean, lo, hi = g.mean().loc[order], g.min().loc[order], g.max().loc[order]
        ax.barh(y + dy, mean.values, 0.3, color=c, label=h)
        ax.errorbar(mean.values, y + dy, xerr=[mean - lo, hi - mean], fmt="none", ecolor=P.INK, elinewidth=0.9, capsize=2)
        for i, v in enumerate(order):
            pts = ab[(ab.horizon == h) & (ab.variant == v)][col]
            ax.plot(pts, np.full(len(pts), y[i] + dy), lw=0, marker="o", ms=2.5, color=P.INK)
    ax.set(xscale="log", xlabel=xl, title=title); ax.grid(axis="y", visible=False)
axes[0].axvline(0.01, color=P.AXIS, lw=0.8)
axes[0].set(yticks=y, yticklabels=order); axes[0].invert_yaxis()
axes[0].legend(title="horizon", loc="lower right"); P.panel_labels(axes); fig.tight_layout()
P.savefig(fig, "stage2_ablation", "fig01_ablation_summary")
show_figures("stage2_ablation")
'''),
])

# ------------------------------------------------------------------ 04
notebook("04_phase2_stage3_multiasset_reference.ipynb", [
    header("04_phase2_stage3_multiasset_reference.ipynb", "04 · Stage 3: multi-asset ground truth",
           "Phase 2: multi-asset",
           "Exact 2-asset common-jump density vs a 2-D finite-difference solver vs Monte Carlo; how grid cost explodes "
           "with dimension (RQ4 motivation) and how Monte Carlo behaves as d grows.",
           "`figures/stage3_multiasset_reference/fig01–fig04`, `results/stage3_multiasset_reference/metrics.json`"),
    code(SETUP),
    code(IMPORTS),
    md("## 1. Configuration (`configs/stage3_multiasset.yaml`)\nJump parameters, t0 and T come from `configs/base.yaml`."),
    code("print(open('configs/stage3_multiasset.yaml').read())"),
    md("## 2. Run the stage\nCPU only (no GPU needed), about 5–10 min: the 2-D grid convergence study dominates."),
    code("!python scripts/stage3_multiasset_reference.py"),
    md("""
## 3. Figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_density_2d | FD contours (orange) sit on the exact contours (dashed); the common-jump ridge is visible; FD log-error small |
| fig02_fd_cost | (a) L¹ falls with slope ≈ 2; (b) wall time grows with the number of unknowns; (c) one grid exceeds GPU memory beyond d = 3 |
| fig03_mc_vs_d | MC error at fixed N roughly flat in d; time per query grows with d. **The PINN has to win on latency, not accuracy** |
| fig04_portfolio_es | FD and MC agree with the exact 1-D projection for every weight; the ES curve shows the diversification benefit |
"""),
    code('show_figures("stage3_multiasset_reference")'),
    md("## 4. Numbers"),
    code(r'''
m = load_metrics("stage3_multiasset_reference")
print("reference 2-D FD:", m["fd2d_reference"])
display(pd.DataFrame(m["fd2d_convergence"]).style.format(precision=3))
display(pd.DataFrame(m["mc_vs_d"]).style.format(precision=4))
w = pd.DataFrame(m["portfolio_weights"])
w["fd_rel"] = (w.es_fd - w.es_exact) / w.es_exact; w["mc_rel"] = (w.es_mc - w.es_exact) / w.es_exact
display(w.style.format(precision=5))
'''),
])

# ------------------------------------------------------------------ 05
notebook("05_phase2_stage4_pinn_nd.ipynb", [
    header("05_phase2_stage4_pinn_nd.ipynb", "05 · Stage 4: multi-asset PINN and scaling study",
           "Phase 2: multi-asset",
           "PINN for the d-dimensional common-jump PIDE at d = 1, 2, 5, 10: joint-density and portfolio VaR/ES accuracy "
           "against the exact mixture, Monte Carlo at the same budget, and training time and memory vs d (RQ1, RQ4).",
           "`figures/stage4_pinn_nd_d<d>/fig01–fig04`, `figures/stage4_summary/`, `results/stage4_*`"),
    code(SETUP),
    code(IMPORTS),
    md("""
## 1. Method in one paragraph
Same recipe as the 1-D PINN (log density, log residual × t, start at t0 from the exact density, 48 Gauss–Hermite
nodes for the jump integral, which stays 1-D because the jump is common). New for d dimensions: inputs whitened by
the Cholesky factor of Σt; the trace term tr(Σ∇²u) is exact for d ≤ 4 and a **Hutchinson** estimate (4 Rademacher'''
probes) above; no box domain: collocation, the normalisation integral and portfolio VaR/ES all use one importance
proposal (Student-t mixture over 0–3 jumps, built from the PIDE coefficients only). Tail-adaptive sampling is
dropped (Stage 2c: no gain). Every portfolio is checked against its **exact** 1-D law.

Each run also reports **IS with the exact density** on the same draws: the sampling floor. PINN error above that
floor is density error.
"""),
    code("print(open('configs/stage4_pinn_nd.yaml').read())"),
    md("""
## 2. Train one run per dimension
Resumable: finished runs are skipped, so a disconnect only costs the run in progress.
Measured on a T4: 9–14 min per run, **about 1 h for all five**. For a quick pass set `STEPS = 5000`.
"""),
    code(r'''
import pathlib
STEPS = 20000
RUNS = [(1, ""), (2, ""), (5, ""), (10, ""), (5, "exact")]   # (d, tag); "exact" forces the exact trace
for d, tag in RUNS:
    run = f"stage4_pinn_nd_d{d}" + (f"_{tag}" if tag else "")
    if pathlib.Path(f"results/{run}/metrics.json").exists():
        print("skip", run); continue
    extra = f"--trace exact --tag {tag}" if tag else ""
    !python scripts/stage4_pinn_nd.py --d {d} --steps {STEPS} {extra}
'''),
    md("""
## 3. Per-dimension figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_training | all loss terms fall; tracked ES₉₉ error of the equal-weight portfolio drops below 1% |
| fig02_density_slices | PINN (blue) on the exact (dashed) along the jump direction β, including the jump shoulder, and across it |
| fig03_portfolio_es | PINN points near the grey IS floor and below 1% for every portfolio; MC (aqua) for scale |
| fig04_var_es_error | median VaR/ES error over portfolios below 1% at every α and horizon |
"""),
    code(r'''
for d in (1, 2, 5, 10):
    print(f"======== d = {d}")
    show_figures(f"stage4_pinn_nd_d{d}")
'''),
    md("## 4. Scaling summary (accuracy, time and memory vs d; Hutchinson vs exact trace)"),
    code(r'''
!python scripts/stage4_summary.py
show_figures("stage4_summary")
'''),
    md("## 5. Numbers"),
    code(r'''
display(pd.read_csv("results/stage4_summary.csv").style.format(precision=4))
for d in (1, 2, 5, 10):
    m = load_metrics(f"stage4_pinn_nd_d{d}")
    print(d, m["trace"], f'{m["train_time_s"]/60:.1f} min', f'{m["peak_mem_mb"]:.0f} MB', m["importance_sampling"], m["timing"]["1d"])
'''),
    md("""
## 6. d = 10: reported as the breakdown point (not pursued)
At d = 10 the ES₉₉ error is 5% at 1 day and 62% at 10 days (PDE loss stalls at 0.76 vs 0.14 at d = 5; density too wide
along β). Following the plan's fallback, the project stops at d = 5 and reports d = 10 as where the method breaks down.
The cell below is a commented skeleton of the follow-up runs (exact trace; then width 256 + 40k steps) for future work.
"""),
    code(r'''
# Future work, not run in this project. Uncomment to try the d = 10 follow-up (~75 min on a T4).
# FOLLOWUP = [("exact", "--trace exact"),                                     # A: exact trace only
#             ("exact_w256_40k", "--trace exact --width 256 --steps 40000")]  # B: + capacity and steps
# for tag, flags in FOLLOWUP:
#     if pathlib.Path(f"results/stage4_pinn_nd_d10_{tag}/metrics.json").exists():
#         print("skip", tag); continue
#     !python scripts/stage4_pinn_nd.py --d 10 --tag {tag} {flags}
# !python scripts/stage4_summary.py
'''),
])

# ------------------------------------------------------------------ planned stages
PLANNED = [
    ("06_phase3_stage5_realtime_amortised.ipynb", "06 · Stage 5: amortised, real-time risk (headline)",
     "Phase 3: real-time",
     "One network over the parameter space θ (and portfolio weights w); latency per VaR/ES query vs Monte Carlo at equal tail error (RQ5).",
     "`figures/stage5_realtime/`",
     """
**To build:** θ = (σ's, ρ, λ, μ_J, σ_J) and optionally w as network inputs (or a DeepONet-style operator), with ranges taken from the rolling calibrations.

**Benchmark protocol:** fixed GPU (T4), warm-up, median of 100 queries, batch sizes 1 and 1024; MC paths chosen to match the PINN's ES₉₉ error.

**Planned figures:**
- fig01 accuracy across θ (error heatmap)
- fig02 latency vs MC at matched error'''
- fig03 break-even query count (training cost amortised)
- fig04 Euler risk contributions ∂ES/∂wᵢ via autodiff vs finite-difference bumps
""", None),
    ("07_phase4_stage6_real_data.ipynb", "07 · Stage 6: real data, non-linear portfolio and backtests",
     "Phase 4: empirical",
     "Rolling one-day-ahead VaR/ES on a NIFTY basket and an S&P sector-ETF basket (2005–), including an index-option position; benchmarks and backtests (RQ2).",
     "`data/raw/`, `figures/stage6_empirical/`",
     """
**To build:** `pinn_tailrisk/data.py` (download and cache), `calibrate.py` (rolling maximum-likelihood fit of the Merton mixture), `benchmarks.py` (Gaussian, HS, FHS, GARCH-t via `arch`, DCC-GARCH-t, EVT-POT), `backtest.py` (Kupiec, Christoffersen, FZ0 score, Diebold–Mariano / MCS).

Split: develop 2005–2017, out-of-sample 2018– (COVID included). No look-ahead: refit using past data only.

**Planned figures:**
- fig01 return series and rolling vol
- fig02 tail log-histogram vs normal
- fig03 rolling calibrated parameters
- fig04 VaR/ES forecasts with breaches
- fig05 backtest table and score comparison
""",
     r'''
# Data preview: download and cache daily closes, then the first EDA figure
import yfinance as yf, pathlib
TICKERS = {"NIFTY": ["^NSEI", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS"],
           "SP500": ["^GSPC", "XLK", "XLF", "XLE", "XLV", "XLY"]}
pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
for name, tick in TICKERS.items():
    f = pathlib.Path(f"data/raw/{name}.csv")
    if not f.exists():
        yf.download(tick, start="2005-01-01", auto_adjust=True, progress=False)["Close"].to_csv(f)
px = pd.read_csv("data/raw/NIFTY.csv", index_col=0, parse_dates=True)
r = np.log(px["^NSEI"]).diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(r.index, r * 100, color=P.INK_2, lw=0.5); axes[0].set(title="NIFTY 50 daily log return", ylabel="%")
z = (r - r.mean()) / r.std(); bins = np.linspace(-10, 6, 80)
h, e = np.histogram(z, bins, density=True); c = 0.5 * (e[1:] + e[:-1])
axes[1].semilogy(c[h > 0], h[h > 0], lw=0, marker="o", ms=3.5, color=P.INK, label="NIFTY 50")
from scipy import stats
axes[1].semilogy(c, stats.norm.pdf(c), **P.style("gaussian")); axes[1].set(title="Standardised returns vs Gaussian", xlabel="z")
axes[1].legend(); P.panel_labels(axes); fig.tight_layout()
P.savefig(fig, "stage6_empirical", "fig01_nifty_eda"); plt.show()
'''),
    ("08_phase4_stage7_ablation_release.ipynb", "08 · Stage 7: physics vs data, failure cases, release",
     "Phase 4: empirical and thesis",
     "Same network trained on samples only (maximum likelihood) vs physics-informed, across data sizes (RQ3); failure cases; final thesis figures.",
     "`figures/stage7_final/`",
     """
**To build:** a data-only density network (same architecture, trained on samples by maximum likelihood) and a loop over sample sizes; collect all stage metrics into the final thesis tables.

**Planned figures:**
- fig01 ES₉₉ error vs number of samples, physics vs data-only
- fig02 failure cases (high λ, tiny t0, very wide domain)
- fig03 summary scorecard across stages
""", None),
]

for fname, title, phase, goal, outputs, plan, live in PLANNED:
    cells = [header(fname, title, phase, goal, outputs), code(SETUP), code(IMPORTS),
             md("## Plan for this stage\n" + plan.strip())]
    if live:
        cells += [md("## Preview with the current library"), code(live)]
    cells += [md("## Run the stage\n_Placeholder: the stage script is not written yet._"),
              code(f'# !python scripts/{fname.split("_", 2)[2].replace(".ipynb", "")}.py\n# show_figures("...")')]
    notebook(fname, cells)

print("\n".join(sorted(p.name for p in OUT.glob("*.ipynb"))))
