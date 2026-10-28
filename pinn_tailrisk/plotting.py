"""Shared figure style for every figure in the project.

Every method owns one color, linestyle and marker for the whole project, so a
reader can follow "PINN" or "Monte Carlo" from Stage 2 to the thesis without
re-reading legends. Colors come from a categorical palette validated for
color-vision deficiency; the linestyle/marker give a second, color-free channel
(needed in grayscale print and for the three slots below 3:1 contrast).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

# ---- ink & chrome --------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TAIL_FILL = "#f0efec"

# ---- categorical slots (fixed order; never cycled) -----------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"

# One style per method, used everywhere.  "exact" is the ground truth: ink, dashed.
METHOD = {
    "exact":    dict(color=INK,     ls=(0, (5, 3)), marker="",  label="Exact (Merton)"),
    "pinn":     dict(color=BLUE,    ls="-",         marker="o", label="PINN"),
    "fd":       dict(color=ORANGE,  ls="-",         marker="s", label="Finite difference"),
    "mc":       dict(color=AQUA,    ls="-",         marker="^", label="Monte Carlo"),
    "gaussian": dict(color=YELLOW,  ls=(0, (1, 1)), marker="D", label="Gaussian"),
    "hs":       dict(color=MAGENTA, ls="-.",        marker="v", label="Historical sim."),
    "garch_t":  dict(color=GREEN,   ls="-",         marker="P", label="GARCH-t"),
    "evt":      dict(color=VIOLET,  ls="--",        marker="X", label="EVT"),
    "data_nn":  dict(color=RED,     ls=":",         marker="*", label="Data-only NN"),
    # Stage 6 variants keep their family's color with a dashed line
    "fhs":      dict(color=MAGENTA, ls=(0, (4, 2)), marker="<", label="Filtered HS"),
    "dcc":      dict(color=GREEN,   ls=(0, (4, 2)), marker="h", label="DCC-GARCH-t"),
    "merton_unf": dict(color=MUTED, ls=(0, (1, 2)), marker=".", label="Merton, unfiltered"),
}

# Sequential (magnitude): one hue, light -> dark.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Diverging (signed error): blue <-> neutral gray <-> red.
DIV_BR = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f4b4b3", "#e34948", "#9c2222"])


def style(method: str, **override) -> dict:
    """Matplotlib kwargs for a method's line: ``ax.plot(x, y, **style("pinn"))``."""
    s = METHOD[method]
    kw = dict(color=s["color"], ls=s["ls"], label=s["label"], lw=1.8)
    kw.update(override)
    return kw


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]),
        "image.cmap": "seq_blue",
    })
    for cmap in (SEQ_BLUE, DIV_BR):
        if cmap.name not in mpl.colormaps:
            mpl.colormaps.register(cmap)


def panel_labels(axes, start: str = "a") -> None:
    """Bold (a), (b), ... in the top-left corner of each panel."""
    for i, ax in enumerate(axes):
        ax.text(-0.02, 1.04, f"({chr(ord(start) + i)})", transform=ax.transAxes,
                fontweight="bold", fontsize=10.5, ha="right", va="bottom", color=INK)


def shade_tail(ax, x_lo, x_hi, label: str | None = None) -> None:
    """Neutral band marking a tail region (not a status color)."""
    ax.axvspan(x_lo, x_hi, color=TAIL_FILL, zorder=0, lw=0)
    if label:
        ax.text((x_lo + x_hi) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=INK_2)


def savefig(fig, stage: str, name: str) -> Path:
    """Write ``figures/<stage>/<name>.{pdf,png}``. PDF for the thesis, PNG for slides."""
    out = FIG_DIR / stage
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    return out / f"{name}.png"
