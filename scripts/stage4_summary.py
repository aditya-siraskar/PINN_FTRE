"""Stage 4 summary across dimensions: accuracy and cost of the d-asset PINN vs d.

Reads results/stage4_pinn_nd_d<d>/metrics.json (and the optional _exact-trace run at d = 5)
and writes figures/stage4_summary/fig01_scaling, fig02_trace (if the comparison run exists)
and results/stage4_summary.csv. Needs no torch.
Run:  python scripts/stage4_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS  # noqa: E402

STAGE = "stage4_summary"
P.use_style()


def load(run: str) -> dict | None:
    f = RESULTS / run / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


rows = []
for f in sorted(RESULTS.glob("stage4_pinn_nd_d*/metrics.json")):
    run = f.parent.name
    tag = run.split("_d", 1)[1]
    d, _, tag = tag.partition("_")
    if tag not in ("", "exact"):
        continue
    m = json.loads(f.read_text())
    med = pd.DataFrame(m["median_abs_rel"])
    for h in (1, 10):
        r = med[(med.h == h) & (med.alpha == 0.99)].iloc[0]
        rows.append(dict(d=int(d), run=tag or "default", trace=m["trace"], horizon=f"{h}-day",
                         es99_pinn=r.es_rel_pinn, es99_is_floor=r.es_rel_is_exact, es99_mc=r.es_rel_mc,
                         tail_mae_log=m["density_error"][f"{h}d"]["tail_mae_log"],
                         ess_frac=m["importance_sampling"][f"{h}d"]["ess_frac_pinn"],
                         train_min=m["train_time_s"] / 60, peak_mem_mb=m["peak_mem_mb"]))
if not rows:
    sys.exit("no Stage 4 results found; run scripts/stage4_pinn_nd.py first")
S = pd.DataFrame(rows).sort_values(["run", "d", "horizon"])
S.to_csv(RESULTS / "stage4_summary.csv", index=False)
print(S.to_string(index=False))
D = S[S.run == "default"]
HC = {"1-day": P.SEQ_BLUE(0.45), "10-day": P.SEQ_BLUE(0.95)}

# ---------------------------------------------------------------- fig01: accuracy and cost vs d
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
ax = axes[0]
for hz, ls in (("1-day", "-"), ("10-day", "--")):
    g = D[D.horizon == hz]
    ax.plot(g.d, 100 * g.es99_pinn, marker="o", **P.style("pinn", ls=ls, label=f"PINN, {hz}"))
    ax.plot(g.d, 100 * g.es99_mc, marker="^", **P.style("mc", ls=ls, label=f"Monte Carlo, {hz}"))
    ax.plot(g.d, 100 * g.es99_is_floor, marker="d", color=P.MUTED, ls=ls, lw=1.2, label=f"IS floor, {hz}")
ax.axhline(1.0, color=P.AXIS, lw=0.8)
ax.set(yscale="log", xlabel="number of assets d", ylabel="median |ES$_{99}$ error| (%)", title="Portfolio ES accuracy")
ax.legend(fontsize=7, ncol=1, loc="upper left")
ax = axes[1]
for hz in HC:
    g = D[D.horizon == hz]
    ax.plot(g.d, g.tail_mae_log, marker="o", color=HC[hz], label=hz)
ax.set(yscale="log", xlabel="number of assets d", ylabel="mean |log-density error|", title="Joint density, worst 1%")
ax.legend(title="horizon", title_fontsize=8.5)
g = D[D.horizon == "1-day"]
axes[2].plot(g.d, g.train_min, marker="o", **P.style("pinn"))
axes[2].set(xlabel="number of assets d", ylabel="training time (min, T4)", title="Training cost", ylim=(0, None))
axes[3].plot(g.d, g.peak_mem_mb / 1024, marker="o", **P.style("pinn"))
axes[3].axhline(4, color=P.AXIS, lw=0.8)
axes[3].text(g.d.max(), 4.1, "GTX 1650 (4 GB)", ha="right", va="bottom", fontsize=8, color=P.INK_2)
axes[3].set(xlabel="number of assets d", ylabel="peak GPU memory (GB)", title="Memory", ylim=(0, None))
for ax in axes:
    ax.set_xticks(sorted(D.d.unique()))
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig01_scaling")

# ---------------------------------------------------------------- fig02: Hutchinson vs exact trace at the same d
pair = S[S.d.isin(S[S.run == "exact"].d)]
if (pair.run == "exact").any():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    lab = {"default": "Hutchinson estimator", "exact": "exact trace"}
    col = {"default": P.BLUE, "exact": P.INK_2}
    xi = np.arange(2)
    for k, (run, g) in enumerate(pair.groupby("run")):
        g = g.set_index("horizon").loc[["1-day", "10-day"]]
        axes[0].bar(xi + (k - 0.5) * 0.36, 100 * g.es99_pinn, 0.34, color=col[run], label=lab[run])
        axes[1].bar(k, g.train_min.iloc[0], 0.6, color=col[run])
    axes[0].set(xticks=xi, xticklabels=["1-day", "10-day"], yscale="log", ylabel="median |ES$_{99}$ error| (%)",
                title=f"Accuracy, d = {pair.d.iloc[0]}")
    axes[0].legend()
    axes[1].set(xticks=[0, 1], xticklabels=[lab[r] for r in sorted(pair.run.unique())], ylabel="training time (min)",
                title="Cost")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    P.panel_labels(axes)
    fig.tight_layout()
    P.savefig(fig, STAGE, "fig02_trace")
