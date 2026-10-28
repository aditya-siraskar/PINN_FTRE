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
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{fname}) "
            f"[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
            f"(https://kaggle.com/kernels/welcome?src=https://github.com/{REPO}/blob/main/notebooks/{fname})")


SETUP = r'''
# @title Setup: repo on Google Drive (Colab) or /kaggle/working (Kaggle), install, GPU check  { display-mode: "form" }
USE_DRIVE = True   # Colab: keep figures/results across sessions (recommended). Ignored on Kaggle.
REPO_URL = "https://github.com/aditya-siraskar/PINN_FTRE"

import os, sys
ON_KAGGLE = os.path.exists("/kaggle/working")   # needs Settings > Internet on, Accelerator > GPU T4 x2
if ON_KAGGLE:
    BASE = "/kaggle/working"   # kept with the notebook's output when run via Save Version
elif USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive"
else:
    BASE = "/content"
ROOT = f"{BASE}/PINN_FTRE"
if not os.path.exists(ROOT):
    !git clone -q {REPO_URL} "{ROOT}"
    if not os.path.exists(ROOT):
        raise RuntimeError("git clone failed: no internet. Kaggle: Settings > Internet on (phone-verified account), "
                           "then Run > Restart & clear outputs and run again.")
else:
    rc = os.system(f'git -C "{ROOT}" pull -q --ff-only')
    if rc != 0:   # usually: result files on Drive that have since been committed to GitHub
        print("WARNING: git pull failed; the Drive copy is out of date. To sync (keeps untracked files), run:")
        print(f'!git -C "{ROOT}" fetch -q origin && git -C "{ROOT}" reset -q --hard origin/main')
%cd "{ROOT}"
!pip install -q -e .
sys.path.insert(0, ROOT)

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else
      "none (Colab: Runtime > Change runtime type > T4 GPU; Kaggle: Settings > Accelerator > GPU T4 x2)")
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
the Cholesky factor of Σt; the trace term tr(Σ∇²u) is exact for d ≤ 4 and a **Hutchinson** estimate (4 Rademacher
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

# ------------------------------------------------------------------ 06
notebook("06_phase3_stage5_realtime_amortised.ipynb", [
    header("06_phase3_stage5_realtime_amortised.ipynb", "06 · Stage 5: amortised, real-time risk (headline)",
           "Phase 3: real-time",
           "One network over a box of market parameters θ = (vol level s, jump intensity λ, mean jump μ_J) at d = 2 and 5: "
           "accuracy across θ, latency per VaR/ES query vs Monte Carlo at matched tail error, break-even query count, "
           "Euler risk contributions and what-if curves (RQ5).",
           "`figures/stage5_amortised_d<d>/fig01–fig05`, `results/stage5_amortised_d<d>/`"),
    code(SETUP),
    code(IMPORTS),
    md("""
## 1. Method in one paragraph
Stage 4 trained one network per market. Here θ = (s, λ, μ_J) are extra network inputs (s scales every vol;
λ is sampled log-uniformly; correlations, loadings and σ_J stay fixed), and every collocation, initial-condition
and normalisation point gets its own θ drawn from the training box. The PIDE coefficients, the exact initial
density at t0 and the importance proposal are all evaluated at each point's θ. After one training run, a **new
market state is one forward pass**: draw from the proposal at θ, weight by the network density, sort.
Any portfolio w reuses the same draws.

Scope: d ≤ 5 (Stage 4 found d = 10 is where the method breaks down). Truth for every (θ, w, horizon) is the
closed-form portfolio law. MC runs on the same GPU, so latency is like-for-like.
"""),
    code("print(open('configs/stage5_amortised.yaml').read())"),
    md("""
## 2. Train and evaluate (d = 2, 5)
Resumable: finished runs are skipped. Estimated (not yet measured) **about 1 h on a T4** for both, including
evaluation. A crash after training can be re-evaluated without retraining: add `--eval-only`.
"""),
    code(r'''
import pathlib
STEPS = 30000
for d in (2, 5):
    run = f"stage5_amortised_d{d}"
    if pathlib.Path(f"results/{run}/metrics.json").exists():
        print("skip", run); continue
    !python scripts/stage5_amortised.py --d {d} --steps {STEPS}

# d = 10: not pursued (Stage 4 breakdown point). Skeleton for future work:
# !python scripts/stage5_amortised.py --d 10 --steps 60000
'''),
    md("""
## 3. Figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_training | all loss terms fall; tracked ES₉₉ error at the base market drops to about 1% |
| fig02_accuracy_theta | median ES₉₉ error ≲ 1–2% over most of the (s, λ) box, at the level of MC with the same N; edges of the box may be worse |
| fig03_latency | error-vs-latency curves on the same GPU; the title states the speed-up at the PINN's error level |
| fig04_breakeven | queries after which training time is paid back (or "no break-even" if MC is faster per query) |
| fig05_euler_whatif | PINN Euler contributions match the exact bars; PINN what-if curves track the exact ES smoothly across λ and s |
"""),
    code(r'''
for d in (2, 5):
    print(f"======== d = {d}")
    show_figures(f"stage5_amortised_d{d}")
'''),
    md("## 4. Numbers"),
    code(r'''
rows = []
for d in (2, 5):
    m = load_metrics(f"stage5_amortised_d{d}")
    for h in ("1d", "10d"):
        g, b = m["es99_median_over_grid"][h], m["breakeven"][h]
        rows.append(dict(d=d, horizon=h, es99_pinn=g["es_rel_pinn"], es99_mc=g["es_rel_mc"], es99_is_floor=g["es_rel_is_exact"],
                         pinn_ms=b["pinn_ms"], mc_ms_matched=b["mc_ms"], speedup=b["speedup"],
                         breakeven_queries=b["breakeven_queries"], train_min=m["train_time_s"] / 60))
display(pd.DataFrame(rows).style.format(precision=4))
for d in (2, 5):
    print(d, "worst theta:", load_metrics(f"stage5_amortised_d{d}")["es99_worst_theta"])
'''),
])

# ------------------------------------------------------------------ 07
notebook("07_phase4_stage6_real_data.ipynb", [
    header("07_phase4_stage6_real_data.ipynb", "07 · Stage 6: real data, non-linear portfolio and backtests",
           "Phase 4: empirical",
           "Rolling one-day-ahead VaR/ES on a 5-stock NIFTY basket and a 5-ETF S&P sector basket (2006–2026; test from 2018), "
           "for a linear and a non-linear (basket + short put) book; the Stage 5 amortised PINN against Gaussian, HS, "
           "filtered HS, GARCH-t, GARCH-EVT, DCC-GARCH-t and the unfiltered Merton model; Kupiec, Christoffersen, FZ0, "
           "Diebold–Mariano (RQ2).",
           "`figures/stage6_realdata_<market>/fig01–fig05`, `results/stage6_realdata_<market>/`"),
    code(SETUP),
    code(IMPORTS),
    md("""
## 1. Method in one paragraph
**Model.** The 5-asset common-jump Merton model with a daily vol scale s_t (EWMA of whitened returns, data up to
t−1): covariance s_t²Σ, jumps λ, μ_J, σ_J with equal loadings (a market-wide jump). This is exactly the density the
Stage 5 PINN learns, with θ_t = (s_t, λ_t, μ_J,t).
**Calibration.** Maximum likelihood on 2006–2017 (standard errors from the Hessian); over the test period (λ, μ_J)
are refitted monthly on a 1000-day window with shrinkage to the development fit.
**PINN.** Trained once per market on the development fit, over a θ box set from development data only. Every
test day is one query: proposal draws at θ_t, PINN weights, VaR/ES of both books. The same draws weighted by the
exact density give "Merton, filtered (exact)", so the gap between the two is the PINN error alone.
**Books.** Linear: equal-weight basket. Non-linear: basket + short one 3%-OTM one-month put on the basket
(Black–Scholes, implied vol = EWMA basket vol, rolled monthly), revalued in full on every scenario.

Data: `data/processed/<market>_returns.csv` (committed snapshot of Yahoo prices), so reruns give the same numbers.
"""),
    code("print(open('configs/stage6_realdata.yaml').read())"),
    md("""
## 2. Calibrate, train and backtest (NIFTY, S&P)
Resumable: finished markets are skipped. Estimated (not yet measured) **about 30–40 min per market on a T4**
(PINN training ≈ 25 min; calibration, GARCH/DCC filters and ~2200 daily forecasts the rest).
A crash after training can be re-evaluated without retraining: add `--eval-only`.
"""),
    code(r"""
import pathlib
STEPS = 30000
for market in ("nifty", "sp"):
    run = f"stage6_realdata_{market}"
    if pathlib.Path(f"results/{run}/metrics.json").exists():
        print("skip", run); continue
    !python scripts/stage6_realdata.py --market {market} --steps {STEPS}

# d = 10 basket: not pursued (Stage 4 breakdown point). Skeleton for future work:
# add a 10-ticker market to configs/stage6_realdata.yaml, then
# !python scripts/stage6_realdata.py --market nifty10 --steps 60000
"""),
    md("""
## 3. Figures and what to check

| Figure | Pass criterion |
|---|---|
| fig01_training | loss terms fall; tracked 1-day ES₉₉ error at the development market ≲ 1% |
| fig02_data_calibration | s_t stays inside the PINN box (clipped days are counted in metrics); rolling λ, μ_J stable, near the development fit |
| fig03_var_paths | VaR₉₉ follows volatility (filtered models) through COVID; unfiltered models react slowly and breach in clusters |
| fig04_backtests | breach rates inside the Kupiec region; no ×; FZ0 bars show which models beat the PINN (negative) and whether significantly |
| fig05_pinn_vs_exact | daily PINN/exact ES₉₉ gap ≲ 1–2% on both books (the PINN reproduces its own model on real states) |
"""),
    code(r"""
for market in ("nifty", "sp"):
    print(f"======== {market}")
    show_figures(f"stage6_realdata_{market}")
"""),
    md("## 4. Numbers"),
    code(r"""
for market in ("nifty", "sp"):
    m = load_metrics(f"stage6_realdata_{market}")
    print(f"== {market}: test {m['test']} ({m['n_test']} days); dev fit lam={m['dev_fit']['lam']:.1f} "
          f"mu_J={m['dev_fit']['mu_j']:.4f} sigma_J={m['dev_fit']['sigma_j']:.4f}; clipped days {m['n_clipped']}; "
          f"train {m['train_time_s'] / 60:.0f} min")
    print("   PINN vs exact ES:", {k: round(100 * v["median_abs"], 2) for k, v in m["pinn_vs_exact"].items()}, "% (median |gap|)")
    bt = pd.read_csv(f"results/stage6_realdata_{market}/backtest.csv")
    cols = ["model", "breaches", "rate", "p_uc", "p_cc", "fz0", "dm_p_vs_pinn", "es_ratio"]
    for book in ("linear", "nonlinear"):
        for a in (0.99, 0.975):
            print(f"   {book}, alpha = {a}")
            display(bt[(bt.book == book) & (bt.alpha == a)][cols].sort_values("fz0").style.format(precision=4).hide(axis="index"))
"""),
])

# ------------------------------------------------------------------ 08
notebook("08_phase4_stage7_ablation_release.ipynb", [
    header("08_phase4_stage7_ablation_release.ipynb", "08 · Stage 7: physics vs data, failure cases, final scorecard",
           "Phase 4: empirical and thesis",
           "Does the physics help (RQ3)? The Stage 4 PINN, trained on no samples, against the same network fitted to N exact "
           "samples by maximum likelihood (N = 10³…10⁶) and against Monte Carlo on those samples. Then the cases where the "
           "method breaks, and one scorecard over Stages 2–7 for the thesis.",
           "`figures/stage7_physics_vs_data_d{2,5}/`, `figures/stage7_failure_<case>/`, `figures/stage7_final/`; same under `results/`"),
    md("""
**On Kaggle** (free GPU, 30 h/week): open with the Kaggle badge. Right panel → Settings: **Accelerator = GPU T4 x2**
(not P100: recent PyTorch builds no longer support it), **Internet = on** (needs a phone-verified account).
Then **Save Version → Save & Run All (Commit)**: it runs in the background for up to 12 h, so closing the
browser does not stop it. When it finishes, open the version → **Output** → download `stage7_outputs.zip`.
"""),
    code(SETUP),
    code(IMPORTS),
    md("""
## 1. Method in one paragraph
**Physics vs data-only.** Same d-asset family and network as Stage 4 (d = 2 and 5). The PINN learns from the PIDE and
the exact density at t₀ only. The **data-only NN** is the identical network, optimiser, schedule, step count and
normaliser estimator, but its loss is the negative log-likelihood of N exact samples spread over 16 times in
[t₀, T] (13 log-spaced + the 1/5/10-day horizons), with 20% held out for early stopping. N = 10³, 10⁴, 10⁵, 10⁶,
three seeds each. **Monte Carlo on the same samples** uses the N/16 samples at each horizon directly. All models are
weighted on the *same* importance draws, so the gaps are model gaps. The output is the N at which data alone matches
the PINN.

**Failure cases** (d = 2, one change each): λ = 50/yr with the default proposal (k ≤ 3 jumps), the same with the
proposal widened to k ≤ 10, and t₀ = 0.05 day. Collected from earlier stages: d = 10 (Stage 4) and the worst θ of the
d = 5 amortised model (Stage 5). The far-tail probe shows why Stage 6 needed the r_max trust region.
"""),
    code("print(open('configs/stage7_ablation.yaml').read())"),
    md("""
## 2. Run (resumable)
Estimated, not yet measured, on a T4: **≈ 30 min for d = 2, ≈ 35 min for d = 5** (PINN 10–13 min + 12 data-only
fits of ~1.5 min), **≈ 11 min per failure case**, summary < 1 min; about 1.7 h in total. Finished parts are skipped;
inside the data part, finished data-only fits (`results/stage7_physics_vs_data_d<d>/runs/`) and the trained PINN
(`model.pt`) are reused, so a disconnect loses at most one fit.
"""),
    code(r"""
import pathlib
for d in (2, 5):
    if pathlib.Path(f"results/stage7_physics_vs_data_d{d}/metrics.json").exists():
        print("skip physics vs data, d =", d); continue
    !python scripts/stage7_ablation.py --part data --d {d}
for case in ("high_lam", "high_lam_kfix", "tiny_t0"):
    if pathlib.Path(f"results/stage7_failure_{case}/metrics.json").exists():
        print("skip", case); continue
    !python scripts/stage7_ablation.py --part failure --case {case}
!python scripts/stage7_ablation.py --part summary

# d = 10 physics vs data: not pursued (Stage 4 breakdown point). Skeleton:
# !python scripts/stage7_ablation.py --part data --d 10 --steps 40000
"""),
    md("""
## 3. Figures and what to check

| Figure | Pass criterion |
|---|---|
| physics_vs_data fig02_es_vs_samples | the blue PINN line (no samples) sits below the red data-only curve up to a large N; Monte Carlo on the same samples is the honest data baseline |
| physics_vs_data fig03_density_slices | data-only density follows the exact one where samples exist and departs in the far tail; the PINN follows it throughout |
| physics_vs_data fig04_tail_density_error | data-only tail error falls with N; the PINN's (horizontal) is reached only at large N |
| failure_high_lam fig02 | large ES error at 5–10 days (k ≤ 3 proposal misses the 4+ jump region); `high_lam_kfix` restores ≲ 1–2% |
| failure_tiny_t0 fig02 | whatever happens is reported: errors at 1 day show whether the sharper start hurts |
| final fig01_scorecard | one row per stage/setting; PINN (blue) vs its baseline; 1% reference line |
| final fig02_failure_cases | (a) where the error leaves the 1% band; (b) PINN log p − log q rises above 0 in the far tail, the exact density does not |
"""),
    code(r"""
for st in ["stage7_physics_vs_data_d2", "stage7_physics_vs_data_d5", "stage7_failure_high_lam",
           "stage7_failure_high_lam_kfix", "stage7_failure_tiny_t0", "stage7_final"]:
    print("========", st)
    show_figures(st)
"""),
    md("## 4. Numbers"),
    code(r"""
for d in (2, 5):
    m = load_metrics(f"stage7_physics_vs_data_d{d}")
    print(f"== d = {d}: PINN trained {m['physics_train_time_s'] / 60:.0f} min; data-only fit ~{m['data_only_train_s_median']:.0f} s")
    print("   N at which data-only matches the PINN's ES99 error:", m["n_to_match_physics_es99"])
    a = pd.DataFrame(m["data_only_es"]); a = a[a.alpha == 0.99]
    display(a.pivot_table(index=["model", "N"], columns="h", values="median").style.format("{:.2%}"))
    p = pd.DataFrame(m["physics_es"]); p = p[p.alpha == 0.99]
    display(p.pivot_table(index="model", columns="h", values="es_abs").style.format("{:.2%}"))
print("== failure cases (ES99 error, median over portfolios)")
display(pd.read_csv("results/stage7_final/failure_cases.csv").style.format(precision=4))
print("== scorecard")
display(pd.read_csv("results/stage7_final/scorecard.csv").style.format(precision=4))
"""),
    md("""
## 5. Package the outputs
Stage 7 figures and results (without model weights) in one zip: on Kaggle it appears under **Output**; on Colab it is written to the repo folder.
"""),
    code(r"""
import pathlib, zipfile
zpath = pathlib.Path("/kaggle/working" if ON_KAGGLE else ".") / "stage7_outputs.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("figures", "results"):
        for f in sorted(pathlib.Path(top).glob("stage7_*/**/*")):
            if f.is_file() and f.suffix != ".pt" and "_quick" not in str(f):
                z.write(f)
print(zpath.resolve(), f"{zpath.stat().st_size / 1e6:.1f} MB,", len(zipfile.ZipFile(zpath).namelist()), "files")
"""),
    md("""
## 6. Release checklist
- [ ] all notebooks 00–08 run top to bottom from a fresh Colab
- [ ] `figures/` and `results/` committed (zip from Colab → local → push)
- [ ] README reproduce table and PROJECT_HANDOFF status current
- [ ] tag the thesis version: `git tag -a v1.0 -m "thesis results" && git push --tags`
"""),
])

PLANNED = []

for fname, title, phase, goal, outputs, plan, live in PLANNED:
    cells = [header(fname, title, phase, goal, outputs), code(SETUP), code(IMPORTS),
             md("## Plan for this stage\n" + plan.strip())]
    if live:
        cells += [md("## Preview with the current library"), code(live)]
    cells += [md("## Run the stage\n_Placeholder: the stage script is not written yet._"),
              code(f'# !python scripts/{fname.split("_", 2)[2].replace(".ipynb", "")}.py\n# show_figures("...")')]
    notebook(fname, cells)

print("\n".join(sorted(p.name for p in OUT.glob("*.ipynb"))))
