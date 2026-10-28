# Updated Research Plan: Real-Time Multi-Asset Tail Risk with PINNs

Supersedes the roadmap in `PINN_TailRisk_Review2_Script.md` (slides 6, 11, 12). The Review 2 work becomes Phase 1, which validates the method. Phases 2–3 produce the headline result.

## Headline claim (what we want to show)

> One trained physics-informed network gives **portfolio VaR/ES for a multi-asset jump-diffusion in milliseconds**, at a scale where grid solvers cannot run and Monte Carlo is too slow for real-time use.

## Revised research questions

- **RQ1 (tail vs. bulk):** unchanged, but now measured in *d* = 1, 2, 5, 10 assets.
- **RQ2 (risk accuracy):** VaR/ES calibration for multi-asset portfolios on synthetic data and real data.
- **RQ3 (physics vs. data-only):** unchanged.
- **RQ4 (scaling):** How do accuracy and training cost grow with dimension *d*, compared with finite differences (feasible only up to about 3 dimensions) and Monte Carlo?
- **RQ5 (real-time):** What is the latency per query of the amortised model compared with Monte Carlo *at equal tail error*? What does training cost, and after how many queries does it pay for itself?

## Model

- *d* correlated assets: diffusion with covariance Σ, plus jumps.
- Jumps: one **common market-jump factor** (crash days hit all assets) plus optional idiosyncratic jumps. This keeps the jump integral at 1D per factor, so Gauss–Hermite quadrature stays cheap. A full *d*-dimensional integral would cost nᵈ points.
- **Why the high-dimensional density is actually needed:** if the portfolio is linear in log-returns and coefficients are constant, the portfolio return reduces to a 1D jump-diffusion, and reviewers will catch that. The project therefore must include at least one of:
  - a **non-linear portfolio**, i.e. one holding index options (the main choice),
  - **stochastic volatility per asset**, which doubles the state dimension,
  - **any-weights queries:** one learned joint density serves every weight vector *w*, plus Euler risk contributions ∂ES/∂wᵢ via autodiff.

## Architecture changes

| Component | Phase 1 (1D) | Phases 2–3 (multi-asset, amortised) |
|---|---|---|
| Input | x, t, θ | x ∈ ℝᵈ, t, θ (vols, correlations, jump params) |
| Network | MLP → log p | MLP / DeepONet-style operator → log p |
| Second derivatives | exact autodiff | Hutchinson trace estimator when *d* ≥ 5 |
| Normalisation | 1D quadrature | Quasi-Monte Carlo with importance sampling (Student-t proposal) |
| VaR/ES from density | quantile / integral | self-normalised importance sampling: one batched forward pass |
| Reference solution | Merton closed form + finite differences | multivariate Merton closed form (a Gaussian mixture) + large Monte Carlo runs |

Keep from Phase 1: log-density output, tail-focused adaptive sampling, a small start time t₀, the normalisation penalty, and ablations of each.

## Benchmarks (added)

- **Monte Carlo** at matched tail error, the main speed competitor
- Filtered historical simulation and DCC-GARCH-t, the multi-asset versions of the risk models
- Finite differences for *d* ≤ 3, which shows where grids stop being feasible

Metrics are unchanged (Kupiec, Christoffersen, Fissler–Ziegel), plus **latency (ms/query), throughput and break-even query count**.

## Data

- Synthetic: *d* = 1, 2, 5, 10 with known ground truth
- Real: a 5–10-stock NIFTY basket (e.g. the largest constituents) plus the NIFTY index option for the non-linear portfolio; an S&P 500 sector-ETF basket for transfer
- Split unchanged: develop on 2005–2017, test from 2018 with rolling one-day-ahead forecasts

## Roadmap

| Stage | Work | Output |
|---|---|---|
| 1 | Literature review: add multi-asset PINNs, neural operators and Hutchinson-trace PINNs | confirmed gap |
| 2 | 1D reference solver and 1D PINN (original plan) | validated pipeline |
| 3 | *d*-dimensional Merton closed form and Monte Carlo reference; 2D finite differences | ground truth for scaling |
| 4 | Multi-asset PINN with the common jump factor; scaling study for *d* = 2 → 10 | RQ1, RQ4 |
| 5 | Amortise over θ (and *w*); latency benchmark against Monte Carlo | RQ5 (headline) |
| 6 | Non-linear option portfolio; empirical backtests on the NIFTY and S&P baskets | RQ2 |
| 7 | Physics-vs-data ablation, failure cases, open-source release, thesis | RQ3, final |

**Fallback:** if *d* = 10 fails to train, report the dimension where the method breaks down. That is still a valid scaling result.

**Scope decision (2026-10-28):** Stage 4 trained well up to *d* = 5 (median portfolio ES₉₉ error about 1%) and failed at *d* = 10 (5% at 1 day, 62% at 10 days). The fallback applies: *d* = 10 is reported as the breakdown point, and Stages 5–7 use *d* ≤ 5 (the real-data basket is 5 stocks). *d* = 10 appears only as commented future-work cells.

## Compute

- Development: GTX 1650 (4 GB), FP32, for *d* ≤ 5 with batches of about 4–8k points
- Large runs (*d* = 10, the amortised operator, final latency numbers): Kaggle/Colab T4 (16 GB), free tier
