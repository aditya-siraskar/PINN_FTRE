"""Regenerate the result-only Stage 2b figures (fig01, fig04, fig05) from saved results.

No training and no torch needed:  python scripts/replot_pinn1d.py [--tag smoke]
fig02/fig03 need the trained model and are only made by stage2_pinn1d.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import figs_pinn1d as F  # noqa: E402
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config  # noqa: E402
from pinn_tailrisk.notebook import load_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
STAGE = "stage2_pinn1d" + (f"_{args.tag}" if args.tag else "")
ev = load_config("configs/base.yaml")["eval"]
P.use_style()
m = load_metrics(STAGE)
print(F.plot_training(pd.read_csv(RESULTS / STAGE / "history.csv"), STAGE))
print(F.plot_bulk_vs_tail(m["density_error"], ev["horizons_days"], STAGE))
print(F.plot_var_es_error(pd.read_csv(RESULTS / STAGE / "risk_table.csv"), ev["horizons_days"], ev["alphas"], STAGE))
