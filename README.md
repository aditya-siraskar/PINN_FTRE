# PINN_FTRE
PINN for Financial Tail Risk Estimation

Physics-informed neural networks for VaR / Expected Shortfall under jump-diffusion,
from a validated 1-D study up to real-time, multi-asset tail risk. See [`PLAN.md`](PLAN.md).

## Layout

```
pinn_tailrisk/     library (imported by scripts and notebooks)
  jumpdiff.py      Merton model: exact density, VaR/ES, simulation (1-D and d-asset common-jump)
  fd.py            finite-difference PIDE solver (CN + AB2 IMEX, FFT jump convolution)
  pinn.py          physics-informed network, losses, tail-adaptive training
  risk.py          VaR/ES from density grids, samples, Gaussian
  plotting.py      project-wide figure style; one fixed color + linestyle per method
  config.py        YAML configs, results stamping (git hash, device)
configs/           one YAML per experiment (base.yaml = shared ground truth)
scripts/           one script per stage; each writes figures/ and results/
notebooks/         exploration and thesis figures (import the library, no core logic)
figures/<stage>/   PDF (thesis) + PNG (slides) for every figure
results/<stage>/   metrics.json, CSV tables, checkpoints
tests/             pytest checks of the reference implementations
```

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -e .[dev]
.venv/Scripts/python -m pytest
```

## Reproduce

| Stage | Command | Output |
|---|---|---|
| 2a Ground truth (exact, FD, MC) | `python scripts/stage2_reference.py` | `figures/stage2_reference/` |
| 2b 1-D PINN | `python scripts/stage2_pinn1d.py` | `figures/stage2_pinn1d/` |

## Figure conventions

One categorical palette (checked for color-vision deficiency), with each method fixed for the whole project:
Exact = black dashed, PINN = blue, Finite difference = orange, Monte Carlo = aqua,
Gaussian = yellow dotted, Historical sim. = magenta, GARCH-t = green, EVT = violet, Data-only NN = red.
Every method also has its own linestyle/marker, so figures stay readable in grayscale print.
Horizons use a single-hue blue ramp; signed errors use a blue–gray–red diverging map.
