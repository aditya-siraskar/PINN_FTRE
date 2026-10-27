"""Stage 2b figures that need only saved results (history.csv, risk_table.csv, metrics.json).

Kept out of the training script so figures can be regenerated without retraining:
    python scripts/replot_pinn1d.py --tag <tag>
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P

HORIZON_COLORS = [P.SEQ_BLUE(v) for v in (0.35, 0.65, 0.95)]


def plot_training(hist: pd.DataFrame, stage: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for k, c, lab in [("pde", P.VIOLET, "PIDE residual"), ("ic", P.MAGENTA, "initial condition"),
                      ("norm", P.GREEN, "normalisation"), ("bc", P.YELLOW, "boundary decay")]:
        pos = hist[k] > 0
        if pos.sum() < 2:   # a term that is exactly zero has no place on a log axis
            last = int(hist.step[pos].max()) if pos.any() else 0
            lab += f" (= 0 after step {last})"
        ax.semilogy(hist.step[pos], hist[k][pos], color=c, lw=1.5, label=lab)
    ax.set(xlabel="training step", ylabel="loss term", title="Loss components")
    ax.legend(loc="upper right")
    ax = axes[1]
    for h, c in [(1, HORIZON_COLORS[0]), (10, HORIZON_COLORS[2])]:
        ax.semilogy(hist.step, hist[f"es99_rel_{h}d"], color=c, lw=1.5, label=f"{h}-day")
    ax.axhline(0.01, color=P.AXIS, lw=0.8)
    ax.text(hist.step.iloc[-1], 0.0105, "1%", ha="right", va="bottom", fontsize=8, color=P.INK_2)
    ax.set(xlabel="training step", ylabel="ES$_{99}$ relative error", title="Tail accuracy during training")
    ax.legend(title="horizon", title_fontsize=8.5)
    P.panel_labels(axes)
    fig.tight_layout()
    return P.savefig(fig, stage, "fig01_training")


def plot_bulk_vs_tail(dens_err: dict, horizons, stage: str):
    """dens_err: {h: {...}} with h an int horizon in days (or '1d' style keys)."""
    get = lambda h: dens_err.get(h, dens_err.get(f"{h}d"))  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    xi = np.arange(len(horizons))
    wbar = 0.36
    for ax, (kp, kf, ylab, title) in zip(axes, [
            ("bulk_rel_l1", "fd_bulk_rel_l1", r"relative $L^1$ error", "Bulk (1%–99% quantiles)"),
            ("left_tail_mae_log", "fd_left_tail_mae_log", "mean |log-density error|", "Left tail (worst 1%)")]):
        vp = [get(h)[kp] for h in horizons]
        vf = [get(h)[kf] for h in horizons]
        b1 = ax.bar(xi - wbar / 2 - 0.01, vp, wbar, color=P.METHOD["pinn"]["color"], label="PINN")
        b2 = ax.bar(xi + wbar / 2 + 0.01, vf, wbar, color=P.METHOD["fd"]["color"], label="Finite difference")
        for b in (*b1, *b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1e}",
                    ha="center", va="bottom", fontsize=7.5, color=P.INK_2)
        ax.set(xticks=xi, xticklabels=[f"{h}-day" for h in horizons], ylabel=ylab, title=title, yscale="log")
        ax.set_ylim(top=ax.get_ylim()[1] * 3)   # headroom for the value labels
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    P.panel_labels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return P.savefig(fig, stage, "fig04_bulk_vs_tail")


def plot_var_es_error(risk: pd.DataFrame, horizons, alphas, stage: str):
    """Rows VaR / ES, columns horizons; method identity stays in color + marker."""
    fig, axes = plt.subplots(2, len(horizons), figsize=(11, 5.6), sharey=True, sharex=True)
    ai = np.arange(len(alphas))
    for i, (meas, name) in enumerate([("var", "VaR"), ("es", "ES")]):
        for j, h in enumerate(horizons):
            ax, r = axes[i, j], risk[risk.h == h]
            for m in ("pinn", "fd"):
                ax.plot(ai, 100 * r[f"{meas}_rel_{m}"].abs().clip(lower=1e-7), marker=P.METHOD[m]["marker"],
                        ms=6, **P.style(m, lw=1.2))
            ax.axhline(1.0, color=P.AXIS, lw=0.8)
            ax.set(yscale="log", title=f"{name}, {h}-day")
            if i == 1:
                ax.set(xticks=ai, xticklabels=[f"{a*100:g}%" for a in alphas], xlabel=r"confidence level $\alpha$")
        axes[i, 0].set_ylabel(f"|{name} error| vs exact (%)")
    axes[0, -1].text(ai[-1], 1.15, "1%", ha="right", va="bottom", fontsize=8, color=P.INK_2)
    axes[0, 0].legend(loc="lower left")
    P.panel_labels(axes.ravel())
    fig.tight_layout()
    return P.savefig(fig, stage, "fig05_var_es_error")
