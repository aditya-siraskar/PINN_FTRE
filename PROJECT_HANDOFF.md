# PROJECT HANDOFF: PINNs for Financial Tail-Risk Estimation

> One file to continue this project on another machine (or in a new Claude/AI session).
> Sections 1–9 explain the context, decisions, status and next steps.
> **Appendix A holds every source file of the repo verbatim**, so the project can be rebuilt from this file alone.
> Written 2026-10-27.

---

## Status update (2026-10-28, Stage 6 built)
New modules: `data.py`, `calibrate.py`, `benchmarks.py`, `backtest.py`. Also `scripts/stage6_realdata.py`, `configs/stage6_realdata.yaml` and notebook 07. 26 tests pass.

Data:
- NIFTY: Reliance, HDFC Bank, Infosys, ICICI Bank, TCS.
- S&P: XLK, XLF, XLE, XLV, XLI.
- 2006-01 to 2026-10-27, test from 2018. Yahoo's 2005 NIFTY data has bad prints.
- Log returns are committed in data/processed/.

Calibration findings (development period, filtered model, equal jump loadings):
- NIFTY: lam ≈ 20/yr, mu_J ≈ 0, sigma_J 2.4%.
- S&P: lam ≈ 100/yr, mu_J −0.2%, sigma_J 0.9%.
- Real-data jumps are small and frequent, a variance mixture rather than rare crashes.
- With free loadings, the NIFTY jump attaches to Infosys alone (idiosyncratic), so loadings are fixed at 1.
- Rolling (lam, mu_J) is weakly identified (the S&P unshrunk fit ranged 0.3–75/yr), so we use MAP shrinkage to the development fit.
- The PINN box is set from development data only. T is 1.5 days and k_max is 8.

Non-linear book: basket + short 3%-OTM 1-month put. Black–Scholes pricing, implied vol = EWMA basket vol, full revaluation.

The "Merton, filtered (exact)" model uses the same IS draws as the PINN, so the gap between the two is the PINN error.

The CPU smoke tests (120 days, 200–300 steps) ran end to end for both markets. The T4 run is pending.

## Status update (2026-10-28, Stage 5 results)
Notebook 06 ran on a T4: training took 18 min (d = 2) and 22 min (d = 5), ≤ 0.45 GB. Median ES99 error over the (s, λ) grid, 1-day / 5-day / 10-day:
- d = 2: PINN 1.3 / 0.8 / 0.7%; MC at the same N 1.8 / 0.9 / 0.8%; exact-IS floor 0.3–0.6%.
- d = 5: PINN 1.3 / 1.8 / 3.4%; MC 1.4 / 0.9 / 0.7%. The worst cells are 10% (10-day, s = 0.6, λ = 15) and 7% (s = 1.6, λ ≤ 2.5).

One amortised network covers the whole θ box. At d = 2 it matches or beats MC; at d = 5 it degrades with horizon.
IS ESS is 0.6–0.8. Euler contributions are within about 5% of the finite differences of the exact ES. ES(λ) follows the exact curve; ES(s) has a bias of up to +1.3 points at mid s.

Latency: MC (0.9–1.6 ms/query) is faster than PINN-IS (4–18 ms) at matched error, so fig04 finds no break-even. As expected for Merton, where exact MC is cheap. The value of the PINN is a smooth, differentiable surrogate in θ (what-if, Euler), not raw speed.

## Status update (2026-10-28, Stage 5 built)
`pinn_tailrisk/pinn_amortised.py`, `scripts/stage5_amortised.py`, `configs/stage5_amortised.yaml`, notebook 06.
θ = (s vol multiplier [0.6, 1.6], λ [1, 15] log-uniform, μ_J [-0.10, -0.02]) as network inputs; fresh θ per point
for PDE, IC (exact mixture at t0 in torch) and normalisation. `Proposal` in pinn_nd now takes optional per-row θ.
Exact trace at d ≤ 5. Width 192 diverged at d = 5 on CPU within 20 steps (PDE loss 1e22); width 128 is stable.
Evaluation: (s, λ) grid + μ_J sweep vs closed form, MC and exact-IS at the same N; torch latency of PINN-IS vs MC
on the same device; break-even from fitted MC error ~ N^slope and latency ~ l0 + l1 N at the PINN's error level;
Euler contributions (IS tail average vs FD of exact ES); what-if ES(λ), ES(s) with common random numbers.
Risk: for Merton, MC sampling is as cheap as proposal sampling, so the PINN's speed edge (if any) comes only from
needing fewer draws (tail-focused proposal); report honestly if MC wins.

## Status update (2026-10-28, Stage 4 results)
Notebook 05 ran on a T4 (9–14 min per run, ~1 h total, ≤ 0.45 GB). Median portfolio ES₉₉ error, 1-day / 10-day:
d = 1: 1.4% / 0.05%; d = 2: 0.7% / 1.0%; d = 5: 1.1% / 0.6% (Hutchinson) and 0.09% / 1.4% (exact trace);
**d = 10: 5.2% / 62% (fails)**. MC at the same N: 0.2–1.6%. At d = 10 the PDE loss stalls at 0.76 (d = 5: 0.14) and the
density is too wide along β, worsening with horizon; across β it is fine. Hutchinson ≈ exact at d = 5.
Decision: d = 10 is not pursued; it is reported as the breakdown point (PLAN fallback). Stages 5–7 use d ≤ 5.
Notebook 05 §6 keeps the follow-up runs as a commented skeleton (exact trace; width 256 + 40k steps).

## Status update (2026-10-28, later): Stage 2 rerun with 3 seeds; Stage 4 built

- Notebooks 01–04 reran on Colab at a46e669. All results and main figures are now in the repo
  (per-run ablation figures are gitignored; their metrics are kept).
- **Stage 2b (n_gh = 48):** every VaR/ES error < 1% at 1/5/10 days; ES ≤ 0.42% across 3 seeds.
- **Stage 2c, 3 seeds (ES₉₉ error, 1-day / 10-day):** baseline 0.18–0.30% / 0.06–0.16%;
  linear residual 1–22% / 57–152%; gh8 12–15%; gh24 ≈ 2%; no normalisation ≈ 2.2% at 1 day.
  **Tail-adaptive sampling gives no gain** (without it: 0.06–0.36% / ≈ 0.05%), so Stage 4 drops it.
  Still to test: α = 99.9% and fat-jump parameters before removing it from the 1-D claim.
- **Stage 3 on Colab** matches the local run (2-D FD second order; portfolio ES within 0.012% FD, 0.3% MC).
- **Stage 4 is built, not yet trained on a GPU:** `pinn_tailrisk/pinn_nd.py`, `scripts/stage4_pinn_nd.py`,
  `scripts/stage4_summary.py`, `configs/stage4_pinn_nd.yaml`, notebook 05, `tests/test_pinn_nd.py`.
  - Residual verified: with the exact density plugged in it is ~0 and falls with quadrature nodes
    (median 5e-4 / 1.6e-5 / 1.5e-7 at 24 / 48 / 96 nodes).
  - Importance proposal (Student-t mixture over 0–3 jumps): ESS 75–80% against the exact density.
  - Normalisation loss is log(mass)²: (mass − 1)² overflowed for the untrained net and gave NaNs.
  - Collocation/IC proposal uses df = 30 (IS keeps df = 5). With df = 5, IC targets reached log p = −7000 at d = 10,
    and on d = 2 (2000 CPU steps) PDE loss was 0.68 vs 0.21, ES₉₉ 10-day error 115% vs 42%.
  - CPU sanity check at 2000 steps: nd code at d = 1 matches the 1-D trainer (ES₉₉ 6.6% / 2.8% vs 10.8% / 3.7%
    at 1 / 10 days), so no implementation gap; d = 2 is harder (12% / 42%). The full 20k-step T4 run decides it.
    If d ≥ 2 stays above ~2%, first try: equal weights on the 0–3 jump components of the collocation proposal
    (more tail points, like the 1-D uniform box) and more steps.
  - Portfolio VaR/ES: 20 portfolios per d (equal weight, 15 long-only, 4 long-short), exact 1-D truth,
    compared with MC and with IS using the exact density (the sampling floor) on the same budget (262k).
- Next: run notebook 05 (≈ 3–3.5 h on a T4, resumable), review, then Stage 5 (amortised over θ and w, latency benchmark).

## Status update (2026-10-28): supersedes §0, §5.1 and §7 where they differ

- Stage 2 has run on Colab (T4). FD/MC match exact (FD ES error ≤ 2e-4). PINN with 24 jump nodes reached 0.18% ES₉₉ at 10 d but 1.8% at 1 d.
  **The default is now `n_gh = 48`** (max VaR/ES error 0.61%, 1-day ES₉₉ 0.30%; training 8.9 min on a T4).
- Ablation findings (single seed, being repeated with 3 seeds in notebook 03):
  - the log-form residual is essential (linear: 20–225% ES errors);
  - jump-quadrature resolution is the short-horizon bottleneck (8 → 24 → 48 nodes: 12% → 1.8% → 0.3%);
  - the normalisation penalty is needed;
  - tail-adaptive sampling is inconclusive;
  - 1-day VaR₉₉ is ill-conditioned (the quantile sits in the density gap between the diffusion peak and the jump hump).
- The FD *reference* now runs on a wider `fd_domain` [−1.0, 0.5] (base.yaml). Its p = 0 edges on [−0.6, 0.3] had distorted the far tail (fake FD tail error 0.4) and set an ES floor of 1e-4.
- Stage 3 is built: `fd2d.py` (Douglas ADI + FFT line-kernel jumps), `scripts/stage3_multiasset_reference.py`, notebook 04.
- Notebooks are generated from `tools/make_notebooks.py`. Edit that file, then run `python tools/make_notebooks.py`.
- Next: rerun notebooks 01–04 on Colab, then Stage 4 (multi-asset PINN).

## 0. TL;DR for whoever picks this up

1. The research direction is fixed: Phase 1 (validated 1-D study), then the headline, **real-time multi-asset VaR/ES with an amortised PINN**. See §2.
2. The **library code and the Stage 2 scripts are written but have never been run**: the old laptop's network could not install scipy/matplotlib. Expect small bugs on the first run (§5).
3. On the new laptop: set up the environment (§6), run `pytest`, then `scripts/stage2_reference.py`, then `scripts/stage2_pinn1d.py`, and inspect every figure (§7).
4. Then continue with Stage 3 onwards (§8).

---

## 1. Project identity

| | |
|---|---|
| Title | Physics-Informed Neural Networks for Financial Tail-Risk Estimation |
| Subtitle | A comparative study of VaR and Expected Shortfall under jump-diffusion dynamics → extended to real-time multi-asset tail risk |
| Type | M.Tech research project (Technical Seminar-II; Review 2 already presented) |
| Team | Aditya Siraskar (D004), Utkarsh Sinha (D019), Aditya Sharma (D005) |
| Mentor | Dr. Jaykrishna Joshi |
| Repo | GitHub repo `PINN_FTRE` (local folder `PINN`), branch `main` |
| Existing deliverables | `PINN_TailRisk_Review2.pptx` (12 slides) and `PINN_TailRisk_Review2_Script.md` (speaker script + prepared Q&A), both for Review 2 and describing the *original* 1-D scope |

**What the user asked for when building:**
- It is a research project: **figures at every step**, with the best possible, consistent color theme.
- It must be **very systematic** and cover **every aspect of `PLAN.md`**.
- Code layout: `.py` library plus scripts, with notebooks only for exploration and figures (decided together; see §4).

---

## 2. Research direction

### 2.1 Original scope (Review 2)
- Log returns follow a Merton jump-diffusion. The density p(x,t) solves the forward **PIDE**
  `p_t = -a p_x + ½σ² p_xx + λ ∫ p(x−y) f_Y(y) dy − λ p`.
- The PINN learns p(x,t) (as log p). VaR = quantile, ES = tail integral.
- Two validations: **Box A, maths** (vs the Merton closed form and finite differences). **Box B, finance** (vs Gaussian, historical simulation, GARCH-t, EVT).
- Data: synthetic, NIFTY 50, S&P 500 from 2005. Develop on 2005–2017, test from 2018 (includes COVID). Rolling one-day-ahead forecasts.
- Metrics: density error (bulk vs tail), Kupiec, Christoffersen, Fissler–Ziegel joint VaR–ES score, compute cost.
- Original RQ1–RQ4: density vs tail, risk accuracy, physics vs data-only, computational break-even.

### 2.2 Upgrade chosen by the user: suggestions #1 + #2
Out of eight suggestions, the user chose **(1) go high-dimensional (multi-asset)** and **(2) real-time amortised risk**.
Rejected or deferred: Hawkes jumps, Bates/rough volatility, option-implied tails, UQ and DM/MCS. These were only "if time". DM/MCS-style statistics are cheap and still recommended in Stage 6.

**Headline claim:** one trained physics-informed network gives **portfolio VaR/ES for a multi-asset jump-diffusion in milliseconds**, where grid solvers cannot run and Monte Carlo is too slow for real-time use.

**Why it matters to quant firms:** FD grids die beyond ~3 dimensions; Monte Carlo tail estimates converge as N^(-1/2); latency per query is the metric desks care about.

**Important caveat (must stay in the design):** with constant coefficients and a *linear* portfolio of log returns, `w·X` is itself a 1-D jump-diffusion (implemented as `MultiMertonParams.portfolio(w)`). A reviewer will catch that a d-dim PINN is then unnecessary. The project must therefore include at least one of:
- a **non-linear portfolio** holding index options (main choice);
- **stochastic volatility per asset**, which doubles the state dimension;
- **any-weights queries:** one joint density serves every w, plus Euler risk contributions ∂ES/∂wᵢ via autodiff.

Note: the exact 1-D projection is also useful as a **free exact ground truth** for linear multi-asset tests.

### 2.3 Research questions (current)
- **RQ1** bulk vs tail accuracy, for d = 1, 2, 5, 10
- **RQ2** VaR/ES calibration, synthetic and real multi-asset
- **RQ3** physics-informed vs data-only network
- **RQ4** scaling with dimension (vs FD ≤ ~3D and Monte Carlo)
- **RQ5** latency per query vs Monte Carlo at equal tail error; training cost; break-even query count

### 2.4 Multi-asset model
- d correlated assets: diffusion covariance Σ, plus **one common market-jump factor**. On a jump, all assets move by β·Y with Y ~ N(μ_J, σ_J²).
- This keeps the jump integral **1-D in any d** (Gauss–Hermite stays cheap; a full d-dim integral would cost n^d).
- The exact density is a Gaussian mixture, giving ground truth in any d (`multi_merton_logpdf`).

### 2.5 Roadmap (from PLAN.md)

| Stage | Work | Output | Status |
|---|---|---|---|
| 1 | Literature review: add multi-asset PINNs, neural operators, Hutchinson-trace PINNs | confirmed gap | in progress (team) |
| 2 | 1-D reference solver + 1-D PINN | validated pipeline | **code written, not run** |
| 3 | d-dim Merton closed form + MC reference; 2-D FD | ground truth for scaling | closed form/MC done in `jumpdiff.py`; 2-D FD not started |
| 4 | Multi-asset PINN (common jump factor), scaling d = 2→10 | RQ1, RQ4 | not started |
| 5 | Amortise over θ (and w); latency benchmark vs MC | RQ5 (headline) | not started |
| 6 | Non-linear option portfolio; empirical backtests (NIFTY + S&P baskets) | RQ2 | not started |
| 7 | Physics-vs-data ablation, failure cases, open-source release, thesis | RQ3, final | not started |

**Fallback:** if d = 10 fails to train, report the dimension where the method breaks down. That is still a valid scaling result.

---

## 3. Compute decisions
- The old laptop has a **GTX 1650 (4 GB, Turing, no tensor cores)**. It is fine for 1-D and d ≤ 5 (FP32, batches of ~4–8k points).
- Use **FP32 only**: there are no tensor cores, and second derivatives are more stable in FP32. FP64 on consumer GPUs is ~1/32 speed.
- Large runs (d = 10, the amortised operator, final latency numbers) go on **Kaggle/Colab T4 (16 GB)**, which is free. **Report final latency from one fixed GPU (the T4)** so comparisons are fair.
- Python **3.11** venv (best PyTorch wheel support). Python 3.14 is also installed on the old laptop but not used.

## 4. Engineering decisions
- **Hybrid layout:** core logic in `pinn_tailrisk/` modules; `scripts/` are config-driven and runnable from the command line; `notebooks/` only import the library (EDA, checks, thesis figures). Use `%load_ext autoreload` and `%autoreload 2`, and clear notebook outputs before commits.
- **One YAML per experiment** in `configs/`. `base.yaml` is the shared ground truth.
- Every script writes `figures/<stage>/<name>.{pdf,png}` (PDF for the thesis, PNG for slides) and `results/<stage>/metrics.json`, stamped with the git hash, Python/torch versions and device (`config.save_results`).
- Figure filenames: `figNN_<what>.{pdf,png}`.
- `.gitignore` excludes `.venv/`, caches, `results/**/*.pt` and `data/raw/`.

### 4.1 Figure and color system (do not change casually; consistency across the thesis matters)
- Categorical palette in fixed slot order (validated with a color-vision-deficiency checker on a white surface):
  blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, yellow `#eda100`, magenta `#e87ba4`, green `#008300`, violet `#6250d6`, red `#e34948`.
  Validator result: all hard checks PASS (worst adjacent CVD ΔE 9.1, normal-vision ΔE 19.6). Contrast WARN for aqua, yellow and magenta (below 3:1), so **every method also has its own linestyle/marker**, and figures stay readable in grayscale print.
- **Each method has one fixed style for the whole project:**

  | Method | Color | Line | Marker |
  |---|---|---|---|
  | Exact (Merton) | ink `#0b0b0b` | dashed | – |
  | PINN | blue | solid | o |
  | Finite difference | orange | solid | s |
  | Monte Carlo | aqua | solid | ^ |
  | Gaussian | yellow | dotted | D |
  | Historical sim. | magenta | dash-dot | v |
  | GARCH-t | green | solid | P |
  | EVT | violet | dashed | X |
  | Data-only NN | red | dotted | * |
- **Loss-component curves** use violet, magenta, green and yellow (PIDE, IC, norm, BC), so they never reuse PINN-blue or FD-orange.
- **Horizons (1/5/10 days)** use a single-hue blue sequential ramp (`SEQ_BLUE` at 0.35/0.65/0.95).
- **Signed errors** use a diverging blue ↔ gray ↔ red map (`div_br`), symmetric limits.
- **Tail regions** use a neutral gray band (`TAIL_FILL #f0efec`), never a status color.
- Rules: never dual y-axes (two measures → two panels); legends whenever there are ≥ 2 series; text in ink colors, not series colors; panel labels (a), (b), …; top/right spines off; light grid `#e1e0d9`.
- Font: Segoe UI / Helvetica / Arial / DejaVu Sans; mathtext `dejavusans`.

## 5. Model and implementation details (what the code does and why)

**Units:** time in years, 252 trading days (`DAY = 1/252`); x is the log return; loss L = −X; α = confidence (0.99 = "99% VaR"); VaR/ES reported as positive losses.

**Ground-truth parameters (`configs/base.yaml`):** a = 0.05, σ = 0.20, λ = 5/yr, μ_J = −0.05, σ_J = 0.07. Domain x ∈ [−0.60, 0.30], t0 = 0.5 day, T = 10 days. Evaluation horizons 1/5/10 days; α ∈ {0.95, 0.975, 0.99, 0.995}; eval grid nx = 4001.
(At 1 day λt = 0.02, so jumps sit right in the 99% tail. That is deliberately interesting.)

**`jumpdiff.py`:** exact Merton mixture pdf/logpdf/cdf (Poisson terms truncated at mass 1e-16); exact VaR/ES (Brent root on the cdf; closed-form partial expectation Σ w_k (m_k Φ(z_k) − s_k φ(z_k))); exact sampling (no time-discretisation error); path simulation for synthetic daily data. Multi-asset: `MultiMertonParams` (a, cov, λ, μ_J, σ_J, β), `equicorrelated()`, `marginal(i)`, `portfolio(w)` (exact 1-D law of w·X), `multi_merton_logpdf`, `simulate_multi_merton`.

**`fd.py`:** 1-D PIDE solver with IMEX: Crank–Nicolson for drift + diffusion − λp, and second-order Adams–Bashforth for the jump convolution, so the error is O(dx² + dt²). The jump convolution uses FFT with a **cell-averaged** kernel (CDF differences, so mass is exact). Dirichlet p = 0 at the edges. The IC is the exact Merton density at t0, **the same IC the PINN gets**.

**`pinn.py`:** design choices, each a config switch for ablation:
1. **Log-density output** u = log p (positivity, tail resolution).
2. **Similarity-variable inputs:** z = (x − a t)/(σ√t) (divided by `z_scale` = 10) and τ = normalised log t. The known Gaussian normaliser −½log(2πσ²t) is added to the output, and the network output is scaled by `out_scale` = 10.
   Reason: at t0 = 0.5 day the bulk is very narrow (std ≈ 0.009). In raw x the log-density curvature is ~10⁴, which is unlearnable without rescaling.
3. **Log-form residual:** r = u_t + a u_x − ½σ²(u_xx + u_x²) − λ(S − 1), with S = Σ_i w_i exp(u(x−y_i) − u(x)) over Gauss–Hermite nodes (n_gh = 24). r is **multiplied by t** to make it dimensionless (u_t is ~1e5 in the tails at small t). The exponent is clamped at 30. **Shifted points outside the domain count as p = 0**, consistent with FD Dirichlet. `residual: linear` multiplies by p/max p (the standard PINN residual, for the ablation).
4. **IC at t0** from the exact Merton density at t0 (MSE in log space over 1024 points).
5. **Normalisation penalty** (∫exp(u)dx − 1)² at 16 random log-uniform times on a 1024-point grid.
6. **Boundary decay:** relu(−u_x) at the left edge and relu(u_x) at the right edge (penalty / 100). No exact boundary data is used, so nothing leaks from the exact solution.
7. **Tail-adaptive sampling:** 40% of collocation points ~ N(q₁%(t), width), where q₁%(t) is the **model's own current** 1% quantile, recomputed every 250 steps on 16 log-spaced times. The other 60% are uniform in x. t is log-uniform (fast dynamics at small t).
8. Training: Adam, lr 2e-3 with cosine decay to 2e-5, 20k steps, grad clip 1.0, batch 4096, tanh MLP 4×64, Xavier init. History is logged every 250 steps with the ES₉₉ relative error at 1d/10d (callback).
- Not yet implemented but planned: θ as an input (amortisation), L-BFGS fine-tuning, residual-based adaptive refinement (RAR), a direct-p-output variant for the ablation, and a d-dim version with the Hutchinson trace estimator.

**`risk.py`:** VaR/ES from a density on a grid (cumulative trapezoid cdf, linear interpolation of the quantile, exact partial last cell, renormalised); from samples; Gaussian closed form; `bulk_tail_masks` (bulk = 1%–99% quantiles).

**Stage 2a figures (`scripts/stage2_reference.py` → `figures/stage2_reference/`):**
- fig01_target_density: density at 1/5/10 d, linear and log10, plus 10-day vs a moment-matched Gaussian with the worst-1% band
- fig02_fd_mc_validation: 10-day exact vs FD vs MC (2M paths) in log scale, plus FD |log10 error| where p > 1e-6
- fig03_fd_convergence: L¹ and ES₉₉ error vs Δx and Δt, with a slope-2 reference
- fig04_var_es_vs_alpha: exact vs Gaussian VaR/ES vs α at 1 d and 10 d, with an annotation of how much Gaussian ES₉₉ underestimates
- fig05_mc_cost: MC ES₉₉ error vs N (median and 10–90% band, N^(−1/2)) and wall time vs N against the FD solve time. **Sets up RQ5.**
- `results/stage2_reference/metrics.json`

**Stage 2b figures (`scripts/stage2_pinn1d.py` → `figures/stage2_pinn1d/`):**
- fig01_training: loss components, plus ES₉₉ relative error during training (1 d, 10 d)
- fig02_density: PINN vs FD vs exact, log10, at 1/5/10 d, with the worst-1% band
- fig03_error_map: signed log10 error over (x, log t), blank where p < 1e-8, with 1%/99% quantile curves
- fig04_bulk_vs_tail: bulk relative L¹ and left-tail mean |log error|, PINN vs FD, per horizon (**RQ1**)
- fig05_var_es_error: |relative error| of VaR and ES vs α for PINN and FD
- Also writes `history.csv`, `risk_table.csv`, `model.pt` (gitignored) and `metrics.json` (train time, PINN query ms, FD solve s)
- CLI: `--config`, `--steps N` (quick smoke runs), `--tag name` (writes to `stage2_pinn1d_<tag>`, used for ablations)

**Tests (`tests/test_reference.py`):** density integrates to 1; exact VaR/ES match the grid and MC; λ = 0 reduces to Gaussian; FD matches exact (L¹ < 1e-3, ES rel 1e-3); multi-asset portfolio projection matches simulation; the d = 1 multi logpdf equals the 1-D pdf.

### 5.1 Things to watch on the first run (code was never executed)
- `pytest` first. If `test_fd_matches_exact` fails, try a finer `dt` or a wider domain before blaming the scheme.
- Stage 2a: the coarse grids in fig03 (nx = 151, dx ≈ 0.6%) under-resolve the 0.5-day IC (std ≈ 0.9%), so the convergence curve may be pre-asymptotic at the coarse end. That's fine, but check that the slope-2 trend is visible at the fine end. fig04 annotation positions may need nudging. fig05 with 200 reps × N = 1e6 takes a few minutes. Reduce `reps` for a quick run.
- Stage 2b: start with a **smoke run** `python scripts/stage2_pinn1d.py --steps 500 --tag smoke`, then check speed (steps/s) on the GPU. Each step evaluates the net at 4096 × 25 points plus second derivatives. If it's too slow, lower `n_colloc` or `n_gh`.
- If training is unstable or the tails are wrong, the knobs in order are: `w_ic`, `w_norm`, `out_scale`, `z_scale`, lr, `tail_share`, domain width, t0.
- Any unused imports (e.g. `merton_cdf` in stage2_reference.py) are harmless.
- The `inside` mask in `pinn.residual` makes S drop near the domain edges. If that biases the far tails, widen the domain.

## 6. Environment setup on the new laptop

```bash
git clone <repo>   # or copy the folder; or rebuild from Appendix A
cd PINN
py -3.11 -m venv .venv                  # Windows; Linux/mac: python3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e .[dev]
#   (= numpy scipy pandas matplotlib pyyaml tqdm arch yfinance pytest jupyterlab ipykernel)
.venv/Scripts/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/Scripts/python -m pytest -q
```
Versions on the old laptop: torch 2.14.1+cu126, numpy 2.4.6 (≥ 2.0 is required for `np.trapezoid`). NVIDIA driver 610.88.
If the new GPU has no CUDA, everything runs on CPU (just slower). For Kaggle/Colab: upload the repo, `pip install -e .`, run the scripts.

## 7. Immediate next steps (in order, with acceptance criteria)
1. `pytest`: all six tests pass.
2. `python scripts/stage2_reference.py`. Check:
   - FD and MC overlay exact in fig02;
   - fig03 shows ~2nd-order convergence;
   - fig04 shows the Gaussian underestimating ES₉₉;
   - fig05 shows the N^(−1/2) error and linear cost.
3. `python scripts/stage2_pinn1d.py --steps 500 --tag smoke` to check the code runs and to time it.
4. Full `python scripts/stage2_pinn1d.py`. **Target:** ES₉₉ relative error < 1% at 1 d and 10 d; left-tail mean |log error| small (< 0.05 nats is a good goal).
5. Ablations (configs + `--tag`): `residual: linear`, `tail_share: 0`, n_gh ∈ {8, 16, 32}, t0 ∈ {0.25, 0.5, 1} day, w_norm = 0. Then a summary figure `figures/stage2_ablation/`.
6. Commit everything (figures and metrics included; checkpoints are gitignored).

## 8. Remaining stages: what to build

**Stage 3: multi-asset ground truth** (`scripts/stage3_reference_nd.py`)
- Already available: `multi_merton_logpdf`, `simulate_multi_merton`, and `portfolio(w)` for an exact linear-portfolio VaR/ES.
- Build: a 2-D FD solver (ADI or Douglas scheme plus a 1-D common-jump convolution along β) to show the grid-cost blow-up; a cost-vs-d chart (grid points n^d vs MC paths).
- Figures: 2-D density contours exact vs FD; FD memory/time vs grid; MC tail error vs d.

**Stage 4: multi-asset PINN** (`pinn_nd.py`, `scripts/stage4_pinn_nd.py`)
- Input x ∈ ℝ^d (whitened by the Cholesky factor of Σt, the d-dim similarity variable), t; output log p.
- Residual: u_t + a·∇u − ½ tr(Σ(∇²u + ∇u∇uᵀ)) − λ(S − 1), with S over 1-D Gauss–Hermite along β.
- For d ≥ 5, estimate tr(Σ∇²u) with the **Hutchinson** trace estimator (Rademacher probes).
- Normalisation: quasi-MC with importance sampling (Student-t proposal).
- VaR/ES of a portfolio: self-normalised importance sampling (draw from the proposal, weight by exp(u)/q, one batched forward pass).
- Scaling study d = 1, 2, 5, 10: tail error, training time, memory. **Ground truth** from the exact mixture and the exact `portfolio(w)` projection.

**Stage 5: amortisation and real-time** (headline, RQ5)
- Add θ = (σ's, ρ, λ, μ_J, σ_J), and optionally w, as inputs (or use a DeepONet-style operator), trained over parameter ranges taken from the rolling calibrations.
- Benchmark: latency per query (ms) vs MC at matched ES₉₉ error; throughput; training cost; **break-even query count**. Euler contributions ∂ES/∂wᵢ via autodiff.
- Run the final latency numbers on the T4.

**Stage 6: real data and non-linear portfolio** (RQ2)
- Data: a 5–10-stock NIFTY basket (largest constituents) plus the NIFTY index; an S&P 500 sector-ETF basket; 2005 onwards via yfinance (cache to `data/raw/`, gitignored). Use the Total Return Index if a consistent series is available.
- Rolling one-day-ahead: maximum-likelihood calibration of the jump-diffusion on rolling windows (Merton mixture likelihood; report parameter uncertainty).
- Benchmarks: Gaussian, historical simulation, filtered HS, GARCH(1,1)-t (`arch` package), DCC-GARCH-t, EVT (peaks over threshold), MC, FD (d ≤ 3), data-only NN.
- Backtests: Kupiec, Christoffersen, Fissler–Ziegel (FZ0) joint VaR–ES score; add Diebold–Mariano / Model Confidence Set.
- Non-linear portfolio: stocks plus index options priced from the same model. This is what makes the d-dim density necessary (§2.2 caveat).

**Stage 7:** physics-vs-data ablation (same network trained by maximum likelihood on samples only, varying sample size: RQ3), failure cases, clean open-source release (README, tests, reproducible configs), thesis.

## 9. Literature pointers already cited in Review 2
Raissi, Perdikaris & Karniadakis (2019) PINNs; Merton (1976) jump-diffusion; PINNs for option pricing incl. jump-diffusion; PINNs for Fokker–Planck/forward densities; Bank of England study on filtered-HS VaR; Basel FRTB (97.5% ES); Kupiec (1995); Christoffersen (1998); Fissler & Ziegel (2016) joint elicitability of VaR/ES. To add in Stage 1: neural operators (DeepONet, FNO), high-dimensional PINNs with Hutchinson/stochastic-trace losses, deep BSDE / high-dim PDE solvers, Hawkes processes (future work).

---

## 10. File structure

```
PINN/
├── AGENTS.md                         # agent memory context (auto-generated, not project code)
├── README.md                         # layout, setup, reproduce table, figure conventions
├── PLAN.md                           # updated research plan (multi-asset + real-time)
├── PROJECT_HANDOFF.md                # this file
├── PINN_TailRisk_Review2.pptx        # Review 2 deck (binary, not reproduced here)
├── PINN_TailRisk_Review2_Script.md   # Review 2 speaker script + Q&A (in Appendix A)
├── pyproject.toml                    # package + pytest config
├── .gitignore
├── configs/
│   ├── base.yaml                     # Merton ground truth, domain, eval settings
│   └── stage2_pinn1d.yaml            # PINN hyperparameters
├── pinn_tailrisk/
│   ├── __init__.py
│   ├── config.py                     # YAML loading, results stamping
│   ├── jumpdiff.py                   # Merton model 1-D + multi-asset (exact + simulation)
│   ├── fd.py                         # 1-D FD PIDE solver (CN + AB2, FFT jumps)
│   ├── pinn.py                       # PINN model, losses, tail-adaptive trainer
│   ├── risk.py                       # VaR/ES utilities
│   └── plotting.py                   # figure style + method color map
├── scripts/
│   ├── stage2_reference.py           # Stage 2a: exact vs FD vs MC (5 figures)
│   └── stage2_pinn1d.py              # Stage 2b: 1-D PINN (5 figures)
├── tests/
│   └── test_reference.py             # 6 tests
├── notebooks/                        # (to create) exploration and thesis figures
├── figures/<stage>/                  # generated (PDF + PNG)
├── results/<stage>/                  # generated (json, csv, model.pt)
└── data/raw/                         # (to create) downloaded market data, gitignored
```

Git state at handoff: last commit `6354525 removed schedule`. **All the new files above are uncommitted** (README.md modified; PLAN.md, configs/, pinn_tailrisk/, scripts/, tests/, pyproject.toml, .gitignore new). Commit and push them, or copy this file, before leaving the old laptop.

---

## Appendix A: every source file, verbatim


### `PLAN.md`

~~~~markdown
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

## Compute

- Development: GTX 1650 (4 GB), FP32, for *d* ≤ 5 with batches of about 4–8k points
- Large runs (*d* = 10, the amortised operator, final latency numbers): Kaggle/Colab T4 (16 GB), free tier

~~~~

### `README.md`

~~~~markdown
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

~~~~

### `pyproject.toml`

~~~~toml
[project]
name = "pinn-tailrisk"
version = "0.1.0"
description = "Physics-informed neural networks for financial tail-risk (VaR/ES) estimation"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "pandas", "matplotlib", "pyyaml", "tqdm", "torch", "arch", "yfinance"]

[project.optional-dependencies]
dev = ["pytest", "jupyterlab", "ipykernel"]

[tool.setuptools]
packages = ["pinn_tailrisk"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

~~~~

### `.gitignore`

~~~~text
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
results/**/*.pt
data/raw/

~~~~

### `configs/base.yaml`

~~~~yaml
# Shared ground-truth setup for Phase 1 (1-D). Time in years, 252 trading days.
merton:            # an equity index with a crash component
  a: 0.05          # drift of log return
  sigma: 0.20      # diffusion vol (20% annual)
  lam: 5.0         # 5 jumps per year on average
  mu_j: -0.05      # mean jump -5%
  sigma_j: 0.07    # jump size std 7%

domain:
  x_min: -0.60     # log-return range covered by the solvers
  x_max: 0.30
  t0_days: 0.5     # start time (avoids the Dirac delta at t = 0)
  T_days: 10.0     # longest horizon (FRTB 10-day)

eval:
  horizons_days: [1, 5, 10]
  alphas: [0.95, 0.975, 0.99, 0.995]
  nx: 4001

~~~~

### `configs/stage2_pinn1d.yaml`

~~~~yaml
# Stage 2b: 1-D PINN on the Merton PIDE (parameters in base.yaml).
pinn:
  width: 64
  depth: 4
  out_scale: 10.0
  residual: log        # log | linear (ablation)
  n_gh: 24
  n_colloc: 4096
  tail_share: 0.4      # 0 disables tail-adaptive sampling (ablation)
  tail_level: 0.01
  adapt_every: 250
  w_pde: 1.0
  w_ic: 10.0
  w_norm: 10.0
  w_bc: 1.0
  steps: 20000
  lr: 0.002
  lr_min: 0.00002
  log_every: 250
  seed: 0

~~~~

### `pinn_tailrisk/__init__.py`

~~~~python
"""PINNs for financial tail-risk estimation."""

~~~~

### `pinn_tailrisk/config.py`

~~~~python
"""Experiment configs (YAML) and result bookkeeping."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_config(path: str | Path) -> dict:
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def save_results(stage: str, name: str, payload: dict) -> Path:
    """results/<stage>/<name>.json, stamped with git hash and machine info."""
    import torch
    out = RESULTS / stage
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(git=git_hash(), python=platform.python_version(), torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    path = out / f"{name}.json"
    path.write_text(json.dumps(dict(meta=meta, **payload), indent=2, default=float))
    return path

~~~~

### `pinn_tailrisk/jumpdiff.py`

~~~~python
"""Merton jump-diffusion for log returns: exact densities and exact simulation.

Model (log return X_t, X_0 = 0):

    dX_t = a dt + sigma dW_t + dJ_t,     J_t = sum_{i<=N_t} Y_i,
    N_t ~ Poisson(lam t),                Y_i ~ N(mu_j, sigma_j^2).

Forward (Fokker-Planck-Kolmogorov) PIDE for the density p(x, t):

    p_t = -a p_x + 0.5 sigma^2 p_xx + lam * int p(x - y, t) f_Y(y) dy - lam p.

Exact solution (Merton 1976): a Poisson-weighted mixture of Gaussians,

    p(x, t) = sum_k Pois(k; lam t) * N(x; a t + k mu_j, sigma^2 t + k sigma_j^2).

The multivariate version uses one *common* jump factor: on a jump all assets
move by beta * Y with scalar Y ~ N(mu_j, sigma_j^2). The jump integral is
therefore 1-D in any dimension d, and the density is again a Gaussian mixture.

Units: time in years (252 trading days), returns as log returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import optimize, stats

DAY = 1.0 / 252.0


@dataclass(frozen=True)
class MertonParams:
    a: float = 0.05          # drift of the log return (per year)
    sigma: float = 0.20      # diffusion volatility (per sqrt year)
    lam: float = 5.0         # jump intensity (jumps per year)
    mu_j: float = -0.05      # mean jump size (log return)
    sigma_j: float = 0.07    # jump-size std

    def as_dict(self) -> dict:
        return asdict(self)

    def variance(self, t):
        return (self.sigma**2 + self.lam * (self.mu_j**2 + self.sigma_j**2)) * t

    def mean(self, t):
        return (self.a + self.lam * self.mu_j) * t


def _n_terms(lam_t: float, tol: float = 1e-16) -> int:
    """Number of Poisson terms so that the neglected mass is < tol."""
    return int(stats.poisson.ppf(1.0 - tol, lam_t)) + 3 if lam_t > 0 else 1


def mixture(p: MertonParams, t: float):
    """Weights, means and stds of the Gaussian mixture at time t."""
    k = np.arange(_n_terms(p.lam * t))
    w = stats.poisson.pmf(k, p.lam * t)
    m = p.a * t + k * p.mu_j
    s = np.sqrt(p.sigma**2 * t + k * p.sigma_j**2)
    return w, m, s


def merton_pdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    return (w * stats.norm.pdf(x[..., None], m, s)).sum(-1)


def merton_logpdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    lw = np.log(w, where=w > 0, out=np.full_like(w, -np.inf))
    return np.logaddexp.reduce(lw + stats.norm.logpdf(x[..., None], m, s), axis=-1)


def merton_cdf(x, t: float, p: MertonParams):
    x = np.asarray(x, dtype=float)
    w, m, s = mixture(p, t)
    return (w * stats.norm.cdf(x[..., None], m, s)).sum(-1)


def merton_var_es(alpha: float, t: float, p: MertonParams) -> tuple[float, float]:
    """Exact VaR and ES of the loss L = -X at level alpha (e.g. 0.99).

    q = F^{-1}(1 - alpha);  VaR = -q;
    ES = -E[X | X <= q] = -(1/(1-alpha)) * sum_k w_k (m_k Phi(z_k) - s_k phi(z_k)),
    with z_k = (q - m_k) / s_k.
    """
    w, m, s = mixture(p, t)
    sd = np.sqrt(p.variance(t))
    lo, hi = p.mean(t) - 50 * sd, p.mean(t) + 50 * sd
    q = optimize.brentq(lambda x: (w * stats.norm.cdf(x, m, s)).sum() - (1 - alpha), lo, hi, xtol=1e-14)
    z = (q - m) / s
    partial = (w * (m * stats.norm.cdf(z) - s * stats.norm.pdf(z))).sum()
    return -q, -partial / (1 - alpha)


def simulate_merton(n: int, t: float, p: MertonParams, rng: np.random.Generator) -> np.ndarray:
    """Exact samples of X_t (no time discretisation error)."""
    k = rng.poisson(p.lam * t, size=n)
    return (p.a * t + p.sigma * np.sqrt(t) * rng.standard_normal(n)
            + k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal(n))


def simulate_merton_paths(n_paths: int, n_steps: int, dt: float, p: MertonParams,
                          rng: np.random.Generator) -> np.ndarray:
    """Daily increments of a path; shape (n_paths, n_steps). Used for synthetic data."""
    k = rng.poisson(p.lam * dt, size=(n_paths, n_steps))
    return (p.a * dt + p.sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
            + k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal((n_paths, n_steps)))


# --------------------------------------------------------------------------- multi-asset
@dataclass(frozen=True)
class MultiMertonParams:
    """d assets: correlated diffusion + one common jump factor beta * Y."""
    a: np.ndarray                 # (d,) drifts
    cov: np.ndarray               # (d, d) diffusion covariance (per year)
    lam: float = 5.0
    mu_j: float = -0.05
    sigma_j: float = 0.07
    beta: np.ndarray = field(default=None)  # (d,) jump loadings; default ones

    def __post_init__(self):
        if self.beta is None:
            object.__setattr__(self, "beta", np.ones(len(self.a)))

    @property
    def d(self) -> int:
        return len(self.a)

    @classmethod
    def equicorrelated(cls, d: int, sigma=0.20, rho=0.5, a=0.05, beta=None, **jump):
        vols = np.full(d, sigma) if np.isscalar(sigma) else np.asarray(sigma)
        corr = np.full((d, d), rho) + (1 - rho) * np.eye(d)
        return cls(a=np.full(d, a), cov=np.outer(vols, vols) * corr,
                   beta=None if beta is None else np.asarray(beta), **jump)

    def marginal(self, i: int) -> MertonParams:
        return MertonParams(a=self.a[i], sigma=float(np.sqrt(self.cov[i, i])), lam=self.lam,
                            mu_j=self.beta[i] * self.mu_j, sigma_j=abs(self.beta[i]) * self.sigma_j)

    def portfolio(self, w: np.ndarray) -> MertonParams:
        """Exact 1-D law of w . X (linear portfolio of log returns)."""
        w = np.asarray(w, float)
        bw = float(w @ self.beta)
        return MertonParams(a=float(w @ self.a), sigma=float(np.sqrt(w @ self.cov @ w)), lam=self.lam,
                            mu_j=bw * self.mu_j, sigma_j=abs(bw) * self.sigma_j)


def multi_mixture(p: MultiMertonParams, t: float):
    k = np.arange(_n_terms(p.lam * t))
    w = stats.poisson.pmf(k, p.lam * t)
    means = p.a[None] * t + k[:, None] * p.mu_j * p.beta[None]
    bb = np.outer(p.beta, p.beta) * p.sigma_j**2
    covs = p.cov[None] * t + k[:, None, None] * bb[None]
    return w, means, covs


def multi_merton_logpdf(x, t: float, p: MultiMertonParams):
    """x: (..., d). Exact log density of the d-dimensional common-jump model."""
    x = np.asarray(x, float)
    w, means, covs = multi_mixture(p, t)
    terms = [np.log(wk) + stats.multivariate_normal.logpdf(x, mk, ck)
             for wk, mk, ck in zip(w, means, covs) if wk > 0]
    return np.logaddexp.reduce(np.stack(terms, -1), axis=-1)


def simulate_multi_merton(n: int, t: float, p: MultiMertonParams, rng: np.random.Generator) -> np.ndarray:
    k = rng.poisson(p.lam * t, size=n)
    diff = rng.multivariate_normal(np.zeros(p.d), p.cov * t, size=n)
    y = k * p.mu_j + np.sqrt(k) * p.sigma_j * rng.standard_normal(n)
    return p.a[None] * t + diff + y[:, None] * p.beta[None]

~~~~

### `pinn_tailrisk/fd.py`

~~~~python
"""Finite-difference solver for the 1-D forward jump-diffusion PIDE.

    p_t = -a p_x + 0.5 sigma^2 p_xx - lam p + lam * (p * f_Y)(x)

Scheme: Crank-Nicolson for the local (drift, diffusion, -lam p) part,
second-order Adams-Bashforth for the nonlocal jump convolution (IMEX CN-AB2),
so the global error is O(dx^2 + dt^2). The convolution uses FFT with a
cell-averaged jump kernel (exact mass on the grid). Boundaries: p = 0
(the domain is chosen wide enough that the mass outside is negligible).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse, stats
from scipy.signal import fftconvolve
from scipy.sparse.linalg import splu

from .jumpdiff import MertonParams, merton_pdf


def jump_kernel(dx: float, n: int, p: MertonParams) -> np.ndarray:
    """Cell-averaged jump density at offsets m*dx, m = -(n-1)..(n-1)."""
    m = np.arange(-(n - 1), n) * dx
    return (stats.norm.cdf(m + dx / 2, p.mu_j, p.sigma_j) - stats.norm.cdf(m - dx / 2, p.mu_j, p.sigma_j))


def solve_pide_1d(p: MertonParams, x_min: float, x_max: float, nx: int, t0: float,
                  times, dt: float, p0=None):
    """Return (x, {t: pdf}) and wall-clock time. Initial condition is p(., t0).

    p0 defaults to the exact Merton density at t0, the same initial condition
    the PINN is given, so both solve exactly the same problem.
    """
    tic = time.perf_counter()
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    u = merton_pdf(x, t0, p) if p0 is None else np.asarray(p0, float).copy()

    # local operator L on interior points (Dirichlet zero at both ends)
    c_xx = 0.5 * p.sigma**2 / dx**2
    c_x = p.a / (2 * dx)
    main = np.full(nx, -2 * c_xx - p.lam)
    lower = np.full(nx - 1, c_xx + c_x)     # coefficient of p_{i-1}
    upper = np.full(nx - 1, c_xx - c_x)     # coefficient of p_{i+1}
    L = sparse.diags([lower, main, upper], [-1, 0, 1], format="csc")
    I = sparse.identity(nx, format="csc")

    kern = jump_kernel(dx, nx, p)

    def jump(v):
        return p.lam * fftconvolve(v, kern, mode="full")[nx - 1: 2 * nx - 1]

    times = sorted(times)
    out, t = {}, t0
    j_prev = None
    lu_cache = {}
    for t_target in times:
        while t < t_target - 1e-15:
            h = min(dt, t_target - t)
            if h not in lu_cache:
                lu_cache[h] = (splu((I - 0.5 * h * L).tocsc()), (I + 0.5 * h * L).tocsr())
            lu, rhs_op = lu_cache[h]
            j_now = jump(u)
            j_ext = j_now if j_prev is None else 1.5 * j_now - 0.5 * j_prev
            u = lu.solve(rhs_op @ u + h * j_ext)
            u[0] = u[-1] = 0.0
            j_prev, t = j_now, t + h
        out[t_target] = u.copy()
    return x, out, time.perf_counter() - tic

~~~~

### `pinn_tailrisk/pinn.py`

~~~~python
"""Physics-informed network for the forward jump-diffusion PIDE (1-D, Phase 1).

Design choices (each is a switch in the config so it can be ablated):

* **Log-density output.** The network returns u = log p, so p > 0 by
  construction and 1e-4-level tail densities are O(1) numbers for the network.
* **Similarity-variable inputs.** x enters as z = (x - a t) / (sigma sqrt t),
  t as normalised log t. The diffusive bulk then has width O(1) at every t,
  instead of collapsing as t -> t0. The known Gaussian normaliser
  -0.5 log(2 pi sigma^2 t) is added to the output.
* **Log-form residual.** Dividing the PIDE by p gives
      r = u_t + a u_x - 0.5 sigma^2 (u_xx + u_x^2) - lam (S - 1),
      S = int exp(u(x - y) - u(x)) f_Y(y) dy,
  which weighs a tail point as much as a bulk point (relative error).
  r is multiplied by t (a dimensionless residual). The "linear" option
  (r * p) is the standard PINN residual for the ablation.
* **Gauss-Hermite quadrature** for the jump integral S (Y Gaussian).
* **Start at t0 > 0** from the known short-time density (no Dirac delta).
* **Normalisation penalty** int p dx = 1, and a decay condition at both ends.
* **Tail-adaptive collocation**: a share of points is drawn around the model's
  own current 1% quantile, refreshed during training.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .jumpdiff import MertonParams, merton_logpdf


@dataclass
class PINNConfig:
    width: int = 64
    depth: int = 4
    out_scale: float = 10.0
    residual: str = "log"            # "log" | "linear"
    n_gh: int = 24                   # Gauss-Hermite nodes for the jump integral
    n_colloc: int = 4096
    tail_share: float = 0.4          # share of collocation points near the tail quantile
    tail_level: float = 0.01
    adapt_every: int = 250
    n_ic: int = 1024
    n_norm_t: int = 16
    n_norm_x: int = 1024
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_norm: float = 10.0
    w_bc: float = 1.0
    steps: int = 20000
    lr: float = 2e-3
    lr_min: float = 2e-5
    log_every: int = 250
    seed: int = 0
    extra: dict = field(default_factory=dict)


class TailPINN(nn.Module):
    def __init__(self, p: MertonParams, t0: float, T: float, cfg: PINNConfig, z_scale: float = 10.0):
        super().__init__()
        self.p, self.t0, self.T, self.cfg, self.z_scale = p, t0, T, cfg, z_scale
        layers, n_in = [], 2
        for _ in range(cfg.depth):
            layers += [nn.Linear(n_in, cfg.width), nn.Tanh()]
            n_in = cfg.width
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p = self.p
        s = p.sigma * torch.sqrt(t)
        z = (x - p.a * t) / s / self.z_scale
        tau = 2 * (torch.log(t) - math.log(self.t0)) / (math.log(self.T) - math.log(self.t0)) - 1
        g = self.net(torch.stack([z, tau], -1)).squeeze(-1)
        return -0.5 * torch.log(2 * math.pi * s**2) + self.cfg.out_scale * g


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class Trainer:
    def __init__(self, p: MertonParams, x_min: float, x_max: float, t0: float, T: float,
                 cfg: PINNConfig, device=None):
        self.p, self.x_min, self.x_max, self.t0, self.T, self.cfg = p, x_min, x_max, t0, T, cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.model = TailPINN(p, t0, T, cfg).to(self.device)

        gh_z, gh_w = np.polynomial.hermite.hermgauss(cfg.n_gh)
        self.jump_y = torch.tensor(p.mu_j + math.sqrt(2) * p.sigma_j * gh_z, dtype=torch.float32, device=self.device)
        self.jump_w = torch.tensor(gh_w / math.sqrt(math.pi), dtype=torch.float32, device=self.device)

        # fixed initial-condition data and normalisation grid
        x_ic = np.linspace(x_min, x_max, cfg.n_ic)
        self.x_ic = self._t(x_ic)
        self.u_ic = self._t(merton_logpdf(x_ic, t0, p))
        self.x_norm = torch.linspace(x_min, x_max, cfg.n_norm_x, device=self.device)
        # tail-quantile table used by the adaptive sampler (log t -> q); start from the bulk scale
        self.q_logt = np.linspace(math.log(t0), math.log(T), 16)
        self.q_tail = np.full(16, p.a * T - 2.3 * p.sigma * math.sqrt(T))
        self.history: list[dict] = []

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)

    # ---- sampling ---------------------------------------------------------
    def sample_collocation(self):
        n, cfg = self.cfg.n_colloc, self.cfg
        logt = self.rng.uniform(math.log(self.t0), math.log(self.T), n)
        x = self.rng.uniform(self.x_min, self.x_max, n)
        n_tail = int(cfg.tail_share * n)
        if n_tail:
            q = np.interp(logt[:n_tail], self.q_logt, self.q_tail)
            width = 0.5 * np.abs(q - self.p.a * np.exp(logt[:n_tail])) + 1e-3
            x[:n_tail] = np.clip(q + width * self.rng.standard_normal(n_tail), self.x_min, self.x_max)
        x, t = self._t(x), self._t(np.exp(logt))
        return x.requires_grad_(True), t.requires_grad_(True)

    @torch.no_grad()
    def update_tail_quantiles(self):
        ts = torch.exp(self._t(self.q_logt))
        X, Tt = torch.meshgrid(self.x_norm, ts, indexing="ij")
        pdf = torch.exp(self.model(X.reshape(-1), Tt.reshape(-1))).reshape(X.shape)
        dx = self.x_norm[1] - self.x_norm[0]
        cdf = torch.cumsum(pdf, 0) * dx
        cdf = cdf / cdf[-1:]
        idx = torch.searchsorted(cdf.T.contiguous(), torch.full((len(ts), 1), self.cfg.tail_level, device=self.device))
        self.q_tail = self.x_norm[idx.squeeze(-1).clamp(max=len(self.x_norm) - 1)].cpu().numpy()

    # ---- losses -------------------------------------------------------------
    def residual(self, x, t):
        p, m = self.p, self.model
        u = m(x, t)
        u_t, u_x = _grad(u, t), _grad(u, x)
        u_xx = _grad(u_x, x)
        xs = (x[:, None] - self.jump_y[None]).reshape(-1)
        ts = t[:, None].expand(-1, len(self.jump_y)).reshape(-1)
        u_shift = m(xs, ts).reshape(len(x), -1)
        # p = 0 outside the domain (same convention as the FD solver's Dirichlet edges)
        inside = ((xs >= self.x_min) & (xs <= self.x_max)).reshape(len(x), -1).float()
        S = (self.jump_w[None] * inside * torch.exp(torch.clamp(u_shift - u[:, None], max=30.0))).sum(-1)
        r = u_t + p.a * u_x - 0.5 * p.sigma**2 * (u_xx + u_x**2) - p.lam * (S - 1.0)
        r = t * r
        if self.cfg.residual == "linear":
            r = r * torch.exp(u - u.detach().max())
        return r

    def losses(self):
        cfg, m = self.cfg, self.model
        x, t = self.sample_collocation()
        l_pde = self.residual(x, t).pow(2).mean()

        l_ic = (m(self.x_ic, torch.full_like(self.x_ic, self.t0)) - self.u_ic).pow(2).mean()

        logt = torch.empty(cfg.n_norm_t, device=self.device).uniform_(math.log(self.t0), math.log(self.T))
        X, Tt = torch.meshgrid(self.x_norm, torch.exp(logt), indexing="ij")
        pdf = torch.exp(m(X.reshape(-1), Tt.reshape(-1))).reshape(X.shape)
        mass = torch.trapezoid(pdf, self.x_norm, dim=0)
        l_norm = (mass - 1.0).pow(2).mean()

        tb = torch.exp(logt).requires_grad_(True)
        xl = torch.full_like(tb, self.x_min).requires_grad_(True)
        xr = torch.full_like(tb, self.x_max).requires_grad_(True)
        ul_x, ur_x = _grad(m(xl, tb), xl), _grad(m(xr, tb), xr)
        l_bc = (torch.relu(-ul_x).pow(2) + torch.relu(ur_x).pow(2)).mean() / 100.0

        total = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_norm * l_norm + cfg.w_bc * l_bc
        return total, dict(pde=l_pde.item(), ic=l_ic.item(), norm=l_norm.item(), bc=l_bc.item())

    # ---- training -----------------------------------------------------------
    def fit(self, callback=None):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr_min)
        tic = time.perf_counter()
        for step in range(1, cfg.steps + 1):
            if cfg.tail_share and step % cfg.adapt_every == 0:
                self.update_tail_quantiles()
            opt.zero_grad(set_to_none=True)
            total, parts = self.losses()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % cfg.log_every == 0 or step == 1:
                rec = dict(step=step, total=total.item(), time=time.perf_counter() - tic, **parts)
                if callback:
                    rec.update(callback(self))
                self.history.append(rec)
        self.train_time = time.perf_counter() - tic
        return self.history

    @torch.no_grad()
    def logpdf(self, x: np.ndarray, t: float) -> np.ndarray:
        xt = self._t(x)
        return self.model(xt, torch.full_like(xt, t)).cpu().numpy().astype(float)

~~~~

### `pinn_tailrisk/risk.py`

~~~~python
"""VaR / ES from a density on a grid, from samples, and for a Gaussian model.

Convention: X is a log return, loss L = -X, alpha is the confidence level
(0.99 means "99% VaR"). VaR and ES are reported as positive loss numbers.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid

ALPHAS = (0.95, 0.975, 0.99, 0.995)


def var_es_from_pdf(x: np.ndarray, pdf: np.ndarray, alpha: float, renormalise: bool = True):
    """VaR/ES from a density sampled on an increasing grid x.

    The CDF is the cumulative trapezoid of pdf; the quantile is found by linear
    interpolation; ES integrates x * pdf up to the quantile (with the partial
    last cell handled exactly under linear interpolation of pdf).
    """
    pdf = np.clip(pdf, 0.0, None)
    cdf = cumulative_trapezoid(pdf, x, initial=0.0)
    if renormalise:
        pdf, cdf = pdf / cdf[-1], cdf / cdf[-1]
    tail = 1.0 - alpha
    i = int(np.searchsorted(cdf, tail))
    if i == 0:
        raise ValueError("quantile lies left of the grid; widen the domain")
    # linear interpolation inside cell [i-1, i]
    frac = (tail - cdf[i - 1]) / max(cdf[i] - cdf[i - 1], 1e-300)
    q = x[i - 1] + frac * (x[i] - x[i - 1])
    xm = np.append(x[:i], q)
    pm = np.append(pdf[:i], np.interp(q, x, pdf))
    partial = np.trapezoid(xm * pm, xm)
    # rescale so the integrated tail mass is exactly `tail`
    mass = np.trapezoid(pm, xm)
    es = -partial / mass if mass > 0 else np.nan
    return -q, es


def var_es_from_samples(x: np.ndarray, alpha: float):
    q = np.quantile(x, 1.0 - alpha)
    return -q, -x[x <= q].mean()


def gaussian_var_es(mean: float, std: float, alpha: float):
    z = stats.norm.ppf(1.0 - alpha)
    return -(mean + std * z), -(mean - std * stats.norm.pdf(z) / (1.0 - alpha))


def bulk_tail_masks(x: np.ndarray, cdf: np.ndarray, tail: float = 0.01):
    """Boolean masks for the bulk [q_tail, q_{1-tail}] and the two tails."""
    lo, hi = np.interp([tail, 1 - tail], cdf, x)
    bulk = (x >= lo) & (x <= hi)
    return bulk, x < lo, x > hi

~~~~

### `pinn_tailrisk/plotting.py`

~~~~python
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

~~~~

### `scripts/stage2_reference.py`

~~~~python
"""Stage 2a: ground truth. Exact Merton density vs finite differences vs Monte Carlo.

Produces figures/stage2_reference/fig01..fig05 and results/stage2_reference/metrics.json.
Run:  python scripts/stage2_reference.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import load_config, save_results  # noqa: E402
from pinn_tailrisk.fd import solve_pide_1d  # noqa: E402
from pinn_tailrisk.jumpdiff import (DAY, MertonParams, merton_cdf, merton_logpdf,  # noqa: E402
                                    merton_pdf, merton_var_es, simulate_merton)
from pinn_tailrisk.risk import gaussian_var_es, var_es_from_pdf, var_es_from_samples  # noqa: E402

STAGE = "stage2_reference"
cfg = load_config("configs/base.yaml")
p = MertonParams(**cfg["merton"])
dom = cfg["domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H = cfg["eval"]["horizons_days"]
ALPHAS = cfg["eval"]["alphas"]
rng = np.random.default_rng(0)
P.use_style()
metrics: dict = {"params": p.as_dict(), "domain": dom}
x = np.linspace(dom["x_min"], dom["x_max"], cfg["eval"]["nx"])
horizon_colors = [P.SEQ_BLUE(v) for v in (0.35, 0.65, 0.95)]
pct = lambda v: v * 100  # noqa: E731  (log return -> %)

# ---------------------------------------------------------------- fig01: the target
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for h, c in zip(H, horizon_colors):
    axes[0].plot(pct(x), merton_pdf(x, h * DAY, p) / 100, color=c, label=f"{h}-day")
    axes[1].plot(pct(x), merton_logpdf(x, h * DAY, p) / np.log(10), color=c, label=f"{h}-day")
axes[0].set(xlim=(-12, 8), xlabel="log return (%)", ylabel="density (per %)", title="Density, linear scale")
axes[1].set(xlim=(-45, 20), ylim=(-8, 2), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
            title="Same densities, log scale")
axes[0].legend(title="horizon", title_fontsize=8.5)

h = 10
sd = np.sqrt(p.variance(h * DAY))
gauss = -0.5 * ((x - p.mean(h * DAY)) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))
axes[2].plot(pct(x), merton_logpdf(x, h * DAY, p) / np.log(10), **P.style("exact"))
axes[2].plot(pct(x), gauss / np.log(10), **P.style("gaussian", label="Gaussian, same mean & variance"))
v99, _ = merton_var_es(0.99, h * DAY, p)
P.shade_tail(axes[2], -45, -pct(v99), "worst 1%")
axes[2].set(xlim=(-45, 20), ylim=(-8, 2), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
            title="10-day: jumps fatten the left tail")
axes[2].legend(loc="lower center")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig01_target_density")

# ---------------------------------------------------------------- fig02: FD and MC agree with exact
nx_fd, dt_fd = 2401, 0.02 * DAY
xf, sol, fd_time = solve_pide_1d(p, dom["x_min"], dom["x_max"], nx_fd, t0, [h * DAY for h in H], dt_fd)
mc = simulate_merton(2_000_000, T, p, rng)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
edges = np.linspace(dom["x_min"], dom["x_max"], 241)
cnt, _ = np.histogram(mc, edges)
dens = cnt / (len(mc) * np.diff(edges))
ok = cnt > 0
mid = 0.5 * (edges[1:] + edges[:-1])
ax.plot(pct(mid[ok]), np.log10(dens[ok]), lw=0, marker=P.METHOD["mc"]["marker"], ms=3.5,
        color=P.METHOD["mc"]["color"], label="Monte Carlo (2M paths)")
ax.plot(pct(xf), np.log10(np.clip(sol[T], 1e-300, None)), **P.style("fd"))
ax.plot(pct(x), merton_logpdf(x, T, p) / np.log(10), **P.style("exact"))
ax.set(xlim=(-50, 20), ylim=(-6.5, 1.5), xlabel="log return (%)", ylabel=r"$\log_{10}$ density",
       title="10-day density: three independent routes")
ax.legend(loc="lower center")
ax = axes[1]
for hh, c in zip(H, horizon_colors):
    ex = merton_pdf(xf, hh * DAY, p)
    m = ex > 1e-6
    ax.plot(pct(xf[m]), np.abs(np.log10(np.clip(sol[hh * DAY][m], 1e-300, None)) - np.log10(ex[m])),
            color=c, lw=1.4, label=f"{hh}-day")
ax.set(yscale="log", xlim=(-50, 20), xlabel="log return (%)",
       ylabel=r"$|\log_{10}p_{FD} - \log_{10}p|$", title="Finite-difference error where p > 1e-6")
ax.legend(title="horizon", title_fontsize=8.5)
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_fd_mc_validation")

metrics["fd_reference"] = dict(nx=nx_fd, dt_days=0.02, wall_s=fd_time, var_es={})
for hh in H:
    row = {}
    for a in ALPHAS:
        ve = merton_var_es(a, hh * DAY, p)
        vf = var_es_from_pdf(xf, sol[hh * DAY], a)
        row[str(a)] = dict(exact=ve, fd=vf)
    metrics["fd_reference"]["var_es"][f"{hh}d"] = row

# ---------------------------------------------------------------- fig03: FD convergence
def fd_errors(nx, dt):
    xg, s, wall = solve_pide_1d(p, dom["x_min"], dom["x_max"], nx, t0, [T], dt)
    l1 = np.trapezoid(np.abs(s[T] - merton_pdf(xg, T, p)), xg)
    v, e = var_es_from_pdf(xg, s[T], 0.99)
    vt, et = merton_var_es(0.99, T, p)
    return l1, abs(e - et) / et, wall

nxs = [151, 301, 601, 1201, 2401]
dx_rows = [fd_errors(n, 0.005 * DAY) for n in nxs]
dts = [1.0, 0.5, 0.25, 0.125, 0.0625]
dt_rows = [fd_errors(4801, d * DAY) for d in dts]
dxv = [(dom["x_max"] - dom["x_min"]) / (n - 1) for n in nxs]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, xs, rows, xl, title in [(axes[0], np.array(dxv) * 100, dx_rows, r"grid step $\Delta x$ (%)", "Space refinement"),
                                (axes[1], np.array(dts), dt_rows, r"time step $\Delta t$ (days)", "Time refinement")]:
    l1 = np.array([r[0] for r in rows])
    es = np.array([r[1] for r in rows])
    ax.loglog(xs, l1, marker="s", **P.style("fd", label=r"$L^1$ density error"))
    ax.loglog(xs, es, marker="o", color=P.INK_2, ls="-", lw=1.4, label="ES$_{99}$ relative error")
    ref = l1[-1] * (xs / xs[-1]) ** 2
    ax.loglog(xs, ref, color=P.MUTED, ls=(0, (2, 2)), lw=1, label="slope 2")
    ax.set(xlabel=xl, ylabel="error at 10 days", title=title)
    ax.legend()
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_fd_convergence")
metrics["fd_convergence"] = dict(
    dx=[dict(nx=n, l1=r[0], es99_rel=r[1], wall_s=r[2]) for n, r in zip(nxs, dx_rows)],
    dt=[dict(dt_days=d, l1=r[0], es99_rel=r[1], wall_s=r[2]) for d, r in zip(dts, dt_rows)])

# ---------------------------------------------------------------- fig04: why the tail matters
al = np.linspace(0.90, 0.999, 120)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, hh in zip(axes, [1, 10]):
    t = hh * DAY
    ve = np.array([merton_var_es(a, t, p) for a in al])
    vg = np.array([gaussian_var_es(p.mean(t), np.sqrt(p.variance(t)), a) for a in al])
    ax.plot(al * 100, pct(ve[:, 0]), **P.style("exact", ls="-", label="VaR, jump-diffusion"))
    ax.plot(al * 100, pct(ve[:, 1]), **P.style("exact", label="ES, jump-diffusion"))
    ax.plot(al * 100, pct(vg[:, 0]), **P.style("gaussian", ls="-", label="VaR, Gaussian"))
    ax.plot(al * 100, pct(vg[:, 1]), **P.style("gaussian", label="ES, Gaussian"))
    for a in (0.99, 0.975):
        ax.axvline(a * 100, color=P.AXIS, lw=0.8, zorder=0)
    ax.set(xlabel=r"confidence level $\alpha$ (%)", title=f"{hh}-day horizon")
    i = np.argmin(np.abs(al - 0.99))
    gap = (ve[i, 1] - vg[i, 1]) / ve[i, 1] * 100
    ax.annotate(f"Gaussian ES$_{{99}}$ is\n{gap:.0f}% too low", xy=(99, pct(vg[i, 1])),
                xytext=(92, pct(ve[-1, 1]) * 0.8), fontsize=8.5, color=P.INK_2,
                arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.8))
axes[0].set_ylabel("loss (% of portfolio)")
axes[0].legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig04_var_es_vs_alpha")

# ---------------------------------------------------------------- fig05: Monte Carlo cost of tail accuracy
ns = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]).astype(int)
_, es_true = merton_var_es(0.99, T, p)
reps = 200
err, wall = [], []
for n in ns:
    tic = time.perf_counter()
    e = [var_es_from_samples(simulate_merton(n, T, p, rng), 0.99)[1] for _ in range(reps)]
    wall.append((time.perf_counter() - tic) / reps)
    err.append(np.abs(np.array(e) - es_true) / es_true)
err = np.array(err)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
med, lo, hi = np.median(err, 1), np.quantile(err, 0.1, 1), np.quantile(err, 0.9, 1)
ax.fill_between(ns, lo, hi, color=P.METHOD["mc"]["color"], alpha=0.18, lw=0, label="10–90% of runs")
ax.loglog(ns, med, marker="^", **P.style("mc", label="median"))
ax.loglog(ns, med[0] * np.sqrt(ns[0] / ns), color=P.MUTED, ls=(0, (2, 2)), lw=1, label=r"$N^{-1/2}$")
ax.axhline(0.01, color=P.AXIS, lw=0.8)
ax.text(ns[0], 0.0105, "1% error", fontsize=8, color=P.INK_2, va="bottom")
ax.set(xlabel="Monte Carlo paths N", ylabel="ES$_{99}$ relative error", title="Tail accuracy improves slowly")
ax.legend(loc="lower left")
ax = axes[1]
ax.loglog(ns, np.array(wall) * 1e3, marker="^", **P.style("mc", label="Monte Carlo"))
ax.axhline(fd_time * 1e3, **P.style("fd", label=f"FD, full density ({fd_time*1e3:.0f} ms)"))
ax.set(xlabel="Monte Carlo paths N", ylabel="wall time per estimate (ms)", title="…and the cost grows linearly")
ax.legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig05_mc_cost")
metrics["mc_convergence"] = [dict(n=int(n), median_rel_err=float(m), wall_ms=w * 1e3)
                             for n, m, w in zip(ns, med, wall)]

print(save_results(STAGE, "metrics", metrics))

~~~~

### `scripts/stage2_pinn1d.py`

~~~~python
"""Stage 2b: train the 1-D PINN on the Merton PIDE and compare it with exact and FD.

Produces figures/stage2_pinn1d/fig01..fig05, results/stage2_pinn1d/{metrics.json, history.csv, model.pt}.
Run:  python scripts/stage2_pinn1d.py [--config configs/stage2_pinn1d.yaml] [--steps N] [--tag name]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pinn_tailrisk import plotting as P  # noqa: E402
from pinn_tailrisk.config import RESULTS, load_config, save_results  # noqa: E402
from pinn_tailrisk.fd import solve_pide_1d  # noqa: E402
from pinn_tailrisk.jumpdiff import DAY, MertonParams, merton_cdf, merton_logpdf, merton_var_es  # noqa: E402
from pinn_tailrisk.pinn import PINNConfig, Trainer  # noqa: E402
from pinn_tailrisk.risk import bulk_tail_masks, var_es_from_pdf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/stage2_pinn1d.yaml")
ap.add_argument("--steps", type=int)
ap.add_argument("--tag", default="")
args = ap.parse_args()

STAGE = "stage2_pinn1d" + (f"_{args.tag}" if args.tag else "")
base = load_config("configs/base.yaml")
pc = load_config(args.config)["pinn"]
if args.steps:
    pc["steps"] = args.steps
cfg = PINNConfig(**pc)
p = MertonParams(**base["merton"])
dom = base["domain"]
t0, T = dom["t0_days"] * DAY, dom["T_days"] * DAY
H, ALPHAS = base["eval"]["horizons_days"], base["eval"]["alphas"]
x = np.linspace(dom["x_min"], dom["x_max"], base["eval"]["nx"])
P.use_style()
pct = lambda v: v * 100  # noqa: E731
horizon_colors = [P.SEQ_BLUE(v) for v in (0.35, 0.65, 0.95)]
exact_es = {h: merton_var_es(0.99, h * DAY, p)[1] for h in (1, 10)}


def track(tr: Trainer) -> dict:
    out = {}
    for h in (1, 10):
        _, es = var_es_from_pdf(x, np.exp(tr.logpdf(x, h * DAY)), 0.99)
        out[f"es99_rel_{h}d"] = abs(es - exact_es[h]) / exact_es[h]
    return out


tr = Trainer(p, dom["x_min"], dom["x_max"], t0, T, cfg)
print(f"device={tr.device}  params={sum(q.numel() for q in tr.model.parameters())}  steps={cfg.steps}")
tr.fit(callback=track)
print(f"trained in {tr.train_time:.0f}s")
hist = pd.DataFrame(tr.history)
(RESULTS / STAGE).mkdir(parents=True, exist_ok=True)
hist.to_csv(RESULTS / STAGE / "history.csv", index=False)
torch.save(dict(state=tr.model.state_dict(), cfg=pc, params=p.as_dict(), domain=dom), RESULTS / STAGE / "model.pt")

# ---------------------------------------------------------------- evaluation
xf, fd_sol, fd_time = solve_pide_1d(p, dom["x_min"], dom["x_max"], 2401, t0, [h * DAY for h in H], 0.02 * DAY)
torch.cuda.synchronize() if torch.cuda.is_available() else None
tic = time.perf_counter()
for h in H:
    _ = var_es_from_pdf(x, np.exp(tr.logpdf(x, h * DAY)), 0.99)
pinn_query_ms = (time.perf_counter() - tic) / len(H) * 1e3

rows, dens_err = [], {}
for h in H:
    t = h * DAY
    lp_ex, lp_pi = merton_logpdf(x, t, p), tr.logpdf(x, t)
    bulk, left, right = bulk_tail_masks(x, merton_cdf(x, t, p), 0.01)
    pex, ppi = np.exp(lp_ex), np.exp(lp_pi)
    lp_fd = np.log(np.clip(np.interp(x, xf, fd_sol[t]), 1e-300, None))
    sig = lp_ex > np.log(1e-8)  # ignore regions with negligible mass
    dens_err[h] = dict(
        bulk_rel_l1=np.trapezoid(np.abs(ppi - pex)[bulk], x[bulk]) / np.trapezoid(pex[bulk], x[bulk]),
        left_tail_mae_log=np.mean(np.abs(lp_pi - lp_ex)[left & sig]),
        right_tail_mae_log=np.mean(np.abs(lp_pi - lp_ex)[right & sig]),
        fd_bulk_rel_l1=np.trapezoid(np.abs(np.exp(lp_fd) - pex)[bulk], x[bulk]) / np.trapezoid(pex[bulk], x[bulk]),
        fd_left_tail_mae_log=np.mean(np.abs(lp_fd - lp_ex)[left & sig]),
    )
    for a in ALPHAS:
        ve, ee = merton_var_es(a, t, p)
        vp, ep = var_es_from_pdf(x, ppi, a)
        vf, ef = var_es_from_pdf(xf, fd_sol[t], a)
        rows.append(dict(h=h, alpha=a, var_exact=ve, es_exact=ee, var_pinn=vp, es_pinn=ep, var_fd=vf, es_fd=ef))
risk = pd.DataFrame(rows)
for m in ("pinn", "fd"):
    risk[f"var_rel_{m}"] = (risk[f"var_{m}"] - risk.var_exact) / risk.var_exact
    risk[f"es_rel_{m}"] = (risk[f"es_{m}"] - risk.es_exact) / risk.es_exact
risk.to_csv(RESULTS / STAGE / "risk_table.csv", index=False)

# ---------------------------------------------------------------- fig01: training dynamics
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
for k, c, lab in [("pde", P.VIOLET, "PIDE residual"), ("ic", P.MAGENTA, "initial condition"),
                  ("norm", P.GREEN, "normalisation"), ("bc", P.YELLOW, "boundary decay")]:
    ax.semilogy(hist.step, hist[k].clip(lower=1e-12), color=c, lw=1.5, label=lab)
ax.set(xlabel="training step", ylabel="loss term", title="Loss components")
ax.legend(ncol=2, loc="upper right")
ax = axes[1]
for h, c in [(1, horizon_colors[0]), (10, horizon_colors[2])]:
    ax.semilogy(hist.step, hist[f"es99_rel_{h}d"], color=c, lw=1.5, label=f"{h}-day")
ax.axhline(0.01, color=P.AXIS, lw=0.8)
ax.text(hist.step.iloc[-1], 0.0105, "1%", ha="right", va="bottom", fontsize=8, color=P.INK_2)
ax.set(xlabel="training step", ylabel="ES$_{99}$ relative error", title="Tail accuracy during training")
ax.legend(title="horizon", title_fontsize=8.5)
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig01_training")

# ---------------------------------------------------------------- fig02: densities
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
for ax, h in zip(axes, H):
    t = h * DAY
    v99, _ = merton_var_es(0.99, t, p)
    P.shade_tail(ax, -60, -pct(v99), "worst 1%")
    ax.plot(pct(xf), np.log10(np.clip(fd_sol[t], 1e-300, None)), **P.style("fd", lw=1.2))
    ax.plot(pct(x), tr.logpdf(x, t) / np.log(10), **P.style("pinn"))
    ax.plot(pct(x), merton_logpdf(x, t, p) / np.log(10), **P.style("exact", lw=1.4))
    ax.set(xlim=(-50, 20), ylim=(-8, 2.2), xlabel="log return (%)", title=f"{h}-day horizon")
axes[0].set_ylabel(r"$\log_{10}$ density")
axes[0].legend(loc="lower center")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig02_density")

# ---------------------------------------------------------------- fig03: error map over (x, t)
tt = np.exp(np.linspace(np.log(t0 * 1.01), np.log(T), 120))
xx = np.linspace(dom["x_min"], dom["x_max"], 600)
E = np.full((len(tt), len(xx)), np.nan)
qlo, qhi = [], []
for i, t in enumerate(tt):
    le = merton_logpdf(xx, t, p)
    e = (tr.logpdf(xx, t) - le) / np.log(10)
    e[le < np.log(1e-8)] = np.nan
    E[i] = e
    c = merton_cdf(xx, t, p)
    qlo.append(np.interp(0.01, c, xx))
    qhi.append(np.interp(0.99, c, xx))
lim = np.nanquantile(np.abs(E), 0.99)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
im = ax.pcolormesh(pct(xx), tt / DAY, E, cmap="div_br", vmin=-lim, vmax=lim, shading="auto", rasterized=True)
ax.plot(pct(np.array(qlo)), tt / DAY, color=P.INK, lw=1, ls=(0, (4, 2)))
ax.plot(pct(np.array(qhi)), tt / DAY, color=P.INK, lw=1, ls=(0, (4, 2)))
ax.text(pct(qlo[-1]) - 1, tt[-1] / DAY, "1% quantile", ha="right", va="top", fontsize=8, color=P.INK)
ax.set(yscale="log", xlim=(-55, 25), xlabel="log return (%)", ylabel="horizon (days)",
       title=r"PINN error $\log_{10}\hat p - \log_{10} p$ (blank where p < 1e-8)")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("decades (signed)", color=P.INK_2)
cb.outline.set_visible(False)
fig.tight_layout()
P.savefig(fig, STAGE, "fig03_error_map")

# ---------------------------------------------------------------- fig04: bulk vs tail (RQ1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
xi = np.arange(len(H))
wbar = 0.36
for ax, (kp, kf, ylab, title) in zip(axes, [
        ("bulk_rel_l1", "fd_bulk_rel_l1", r"relative $L^1$ error", "Bulk (1%–99% quantiles)"),
        ("left_tail_mae_log", "fd_left_tail_mae_log", "mean |log-density error|", "Left tail (worst 1%)")]):
    vp = [dens_err[h][kp] for h in H]
    vf = [dens_err[h][kf] for h in H]
    b1 = ax.bar(xi - wbar / 2 - 0.01, vp, wbar, color=P.METHOD["pinn"]["color"], label="PINN")
    b2 = ax.bar(xi + wbar / 2 + 0.01, vf, wbar, color=P.METHOD["fd"]["color"], label="Finite difference")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1e}",
                ha="center", va="bottom", fontsize=7.5, color=P.INK_2)
    ax.set(xticks=xi, xticklabels=[f"{h}-day" for h in H], ylabel=ylab, title=title, yscale="log")
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig04_bulk_vs_tail")

# ---------------------------------------------------------------- fig05: VaR/ES errors (RQ2, controlled)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, meas in zip(axes, ["var", "es"]):
    for m, off in [("pinn", -0.12), ("fd", 0.12)]:
        for j, h in enumerate(H):
            r = risk[risk.h == h]
            ax.plot(np.arange(len(ALPHAS)) + off + (j - 1) * 0.03, 100 * r[f"{meas}_rel_{m}"].abs(),
                    marker=P.METHOD[m]["marker"], lw=0, ms=6, color=P.METHOD[m]["color"],
                    alpha=[0.55, 0.8, 1.0][j], label=f"{P.METHOD[m]['label']}" if j == 2 else None)
    ax.set(xticks=range(len(ALPHAS)), xticklabels=[f"{a*100:g}%" for a in ALPHAS], yscale="log",
           xlabel=r"confidence level $\alpha$", title=f"{meas.upper()} relative error (all horizons)")
axes[0].set_ylabel("|error| vs exact (%)")
axes[0].legend(loc="upper left")
P.panel_labels(axes)
fig.tight_layout()
P.savefig(fig, STAGE, "fig05_var_es_error")

save_results(STAGE, "metrics", dict(
    pinn_config=pc, params=p.as_dict(), domain=dom,
    train_time_s=tr.train_time, pinn_query_ms=pinn_query_ms, fd_solve_s=fd_time,
    density_error={f"{h}d": v for h, v in dens_err.items()},
    risk=risk.to_dict(orient="records")))
print(risk[["h", "alpha", "es_exact", "es_pinn", "es_rel_pinn", "es_rel_fd"]].to_string(index=False))
print({h: {k: f"{v:.2e}" for k, v in d.items()} for h, d in dens_err.items()})

~~~~

### `tests/test_reference.py`

~~~~python
import numpy as np
import pytest

from pinn_tailrisk.fd import solve_pide_1d
from pinn_tailrisk.jumpdiff import (DAY, MertonParams, MultiMertonParams, merton_cdf, merton_pdf,
                                    merton_var_es, multi_merton_logpdf, simulate_merton,
                                    simulate_multi_merton)
from pinn_tailrisk.risk import gaussian_var_es, var_es_from_pdf, var_es_from_samples

P = MertonParams()
T = 10 * DAY


def test_density_integrates_to_one():
    x = np.linspace(-1, 1, 20001)
    assert np.trapezoid(merton_pdf(x, T, P), x) == pytest.approx(1.0, abs=1e-8)
    assert merton_cdf(np.array([1.0]), T, P)[0] == pytest.approx(1.0, abs=1e-10)


def test_var_es_exact_matches_grid_and_mc():
    x = np.linspace(-1, 0.5, 40001)
    for a in (0.95, 0.99):
        v, e = merton_var_es(a, T, P)
        vg, eg = var_es_from_pdf(x, merton_pdf(x, T, P), a)
        assert vg == pytest.approx(v, rel=1e-5) and eg == pytest.approx(e, rel=1e-5)
        vm, em = var_es_from_samples(simulate_merton(2_000_000, T, P, np.random.default_rng(1)), a)
        assert vm == pytest.approx(v, rel=0.02) and em == pytest.approx(e, rel=0.02)


def test_no_jumps_reduces_to_gaussian():
    q = MertonParams(lam=0.0)
    v, e = merton_var_es(0.99, T, q)
    vg, eg = gaussian_var_es(q.a * T, q.sigma * np.sqrt(T), 0.99)
    assert v == pytest.approx(vg, rel=1e-8) and e == pytest.approx(eg, rel=1e-8)


def test_fd_matches_exact():
    x, sol, _ = solve_pide_1d(P, -0.6, 0.3, 1201, 0.5 * DAY, [T], 0.05 * DAY)
    l1 = np.trapezoid(np.abs(sol[T] - merton_pdf(x, T, P)), x)
    assert l1 < 1e-3
    assert var_es_from_pdf(x, sol[T], 0.99)[1] == pytest.approx(merton_var_es(0.99, T, P)[1], rel=1e-3)


def test_multi_asset_portfolio_projection():
    mp = MultiMertonParams.equicorrelated(3, rho=0.4, beta=[1.0, 0.8, 1.2])
    w = np.array([0.5, 0.3, 0.2])
    xs = simulate_multi_merton(1_000_000, T, mp, np.random.default_rng(2)) @ w
    v, e = merton_var_es(0.99, T, mp.portfolio(w))
    assert var_es_from_samples(xs, 0.99)[1] == pytest.approx(e, rel=0.02)


def test_multi_logpdf_marginal_consistency():
    mp = MultiMertonParams.equicorrelated(1)
    x = np.linspace(-0.3, 0.2, 7)
    np.testing.assert_allclose(multi_merton_logpdf(x[:, None], T, mp),
                               np.log(merton_pdf(x, T, mp.marginal(0))), rtol=1e-8)

~~~~

### `PINN_TailRisk_Review2_Script.md`

~~~~markdown
# Presentation Script: Technical Seminar-II, Review 2

**Physics-Informed Neural Networks for Financial Tail-Risk Estimation**
*A comparative study of Value at Risk and Expected Shortfall under jump-diffusion dynamics*

Team: Aditya Siraskar (D004), Utkarsh Sinha (D019), Aditya Sharma (D005)
Mentor: Dr. Jaykrishna Joshi
Deck: `PINN_TailRisk_Review2.pptx` (12 slides)

**Suggested speaker split (change as you like):**

| Part | Slides | Speaker | Time |
|---|---|---|---|
| Problem and model | 1–4 | Aditya Siraskar | ~6 min |
| Research gap and method | 5–8 | Utkarsh Sinha | ~7 min |
| Data, evaluation, plan | 9–12 | Aditya Sharma | ~6 min |

Total: about 18–20 minutes, plus questions.

---

## Slide 1: Title
**Speaker:** Aditya Siraskar · ~1 min

> Good morning, sir. We are Aditya Siraskar, Utkarsh Sinha and Aditya Sharma, and our project is titled *Physics-Informed Neural Networks for Financial Tail-Risk Estimation*.
>
> In one sentence: we want to find out whether a physics-informed neural network, or PINN, can estimate the extreme tail of a portfolio's loss distribution accurately enough to give better Value at Risk and Expected Shortfall estimates than standard methods, and at what computational cost.
>
> The chart on the right shows the idea. It is a fat-tailed distribution of returns. The orange region on the far left is the worst 1% of outcomes. Where that region starts is the VaR, and the average loss inside it is the Expected Shortfall. Our whole project is about estimating that small orange region correctly.

**Transition:** "Let me start with why the tail matters so much."

---

## Slide 2: Motivation
**Speaker:** Aditya Siraskar · ~1.5 min

> Banks and fund managers mainly worry about rare, large losses. Day-to-day small fluctuations do not threaten a portfolio; the extreme days do.
>
> Financial returns have three properties that make this difficult.
>
> **First, fat tails.** Extreme daily moves happen much more often than a normal distribution predicts. A Gaussian model treats such days as almost impossible, but markets see them regularly.
>
> **Second, volatility clustering.** Calm periods and turbulent periods alternate, so risk is not constant over time.
>
> **Third, jumps.** Crashes like March 2020 are sudden discontinuities, not smooth movements.
>
> The key point, shown at the bottom, is that a model can fit the centre of the distribution well and still get the worst 1% wrong. The worst 1% is exactly what VaR and ES measure. So in our evaluation we keep two questions separate: how well we fit the overall distribution, and how well we fit the tail.

**Transition:** "Let me define the two risk measures precisely."

---

## Slide 3: Value at Risk and Expected Shortfall
**Speaker:** Aditya Siraskar · ~1.5 min

> **Value at Risk** at confidence level alpha is the loss threshold that is exceeded with probability one minus alpha. For example, if a ₹10 lakh portfolio has a 99% one-day VaR of ₹30,000, the model says there is about a 1% chance of losing more than ₹30,000 in a day. That number is only an illustration, not a result.
>
> VaR tells us where the tail begins, but not how bad things get beyond that point.
>
> **Expected Shortfall** answers that. It is the average loss given that the loss has already exceeded VaR, so it captures the severity of the tail. This is why regulators have moved towards ES: the Basel FRTB framework uses 97.5% Expected Shortfall for market-risk capital.
>
> The link to our project is the line at the bottom. If we know the full loss density p(x, t), both measures follow directly: VaR is a quantile of the density, and ES is an integral over its tail. So the question becomes how to get an accurate density.

**Transition:** "To model that density, we use a jump-diffusion process."

---

## Slide 4: Jump-diffusion and its forward equation
**Speaker:** Aditya Siraskar · ~2 min

> We model log returns as a jump-diffusion process. The first line is the stochastic differential equation. It has a drift term *a*, a diffusion term with volatility *sigma* driven by Brownian motion, and a jump term J. J is a compound Poisson process: jumps arrive at rate *lambda*, and their sizes follow a distribution f_Y.
>
> The equation in the dark box describes how the probability density of returns evolves over time. It has three parts:
> - The **drift** term moves the density along.
> - The **diffusion** term spreads it out; this captures ordinary daily moves.
> - The **jump term** is an integral: a convolution of the density with the jump-size distribution. This term produces the heavy tail.
>
> Because of the integral, this is not a plain Fokker–Planck PDE. It is a **partial integro-differential equation**, or PIDE. The integral term is nonlocal: the density at a point depends on the density everywhere else. That makes it harder for a PINN than ordinary PDEs.
>
> Two important notes at the bottom.
> - **Exact reference.** When jumps are Gaussian, which is the Merton 1976 model, the density has a closed form: a Poisson-weighted mixture of Gaussians. So we can check the PINN against an exact answer.
> - **Limitation.** Here drift and volatility are constant, but real markets have time-varying volatility. We handle this with rolling re-estimation, and stochastic volatility is a planned extension.

**Handover:** "Now Utkarsh will explain where our work fits in the literature."

---

## Slide 5: Literature and gap
**Speaker:** Utkarsh Sinha · ~1.5 min

> We want to be upfront: simply applying a PINN to a financial equation is no longer novel.
>
> - PINNs were introduced by Raissi and colleagues in 2019 and are well established for forward and inverse PDE problems.
> - In finance, PINNs have already been used for option pricing, including jump-diffusion models.
> - Using PINNs to solve Fokker–Planck and forward-density equations is an active research area.
> - On the risk side, traditional tail-risk models are mature: historical simulation, GARCH, filtered historical simulation and extreme value theory. The Bank of England, for example, has published a detailed study of filtered historical simulation VaR and its competitors.
>
> The gap we see is this: there is no systematic test of whether PINN-solved densities actually give better VaR and ES. That means looking at tail accuracy, calibration and computational cost, and comparing against both numerical solvers and standard risk models.
>
> Our literature review is still in progress, and we will confirm this novelty claim before we start Stage 2.

**Transition:** "This gap leads to our research questions."

---

## Slide 6: Research questions
**Speaker:** Utkarsh Sinha · ~2 min

> Our central question is: **when does a PINN beat conventional methods for tail-risk estimation, and when does its overhead outweigh the gain?**
>
> Our goal is not to prove that PINNs are better. It is to find out whether, when and why they are. If historical simulation or GARCH turns out to be better, that is still a valid and useful finding, provided the experiment is rigorous.
>
> We split this into four sub-questions:
> - **RQ1, density versus tail.** How accurately does the PINN recover the overall density, and separately its extreme tail? A small average error can hide a large tail error.
> - **RQ2, risk accuracy.** How accurate and well-calibrated are the resulting 95% and 99% VaR and ES?
> - **RQ3, does the physics help?** We compare the PINN with the same neural network trained on data alone, especially when data is scarce. This isolates the value of adding the equation.
> - **RQ4, computational break-even.** For a single one-dimensional problem, a classical solver will almost certainly be faster. But if one PINN is trained across many parameter values, it can answer many risk scenarios instantly. We want to find out whether a break-even point exists.

**Transition:** "Here is the methodology we will use to answer these."

---

## Slide 7: Methodology pipeline
**Speaker:** Utkarsh Sinha · ~1.5 min

> The pipeline has six steps:
> 1. **Data.** Compute daily log returns.
> 2. **Calibrate.** Estimate the model parameters: drift, volatility, jump intensity and the jump-size distribution.
> 3. **Equation.** Set up the forward PIDE with those parameters.
> 4. **PINN.** Train the network to learn the density p(x, t).
> 5. **Risk.** Compute VaR and ES from the learned density.
> 6. **Evaluate.** Backtest and compare with other models.
>
> There are two separate validations, shown in the two boxes below.
> - **Box A checks the maths.** Does the PINN solve the equation correctly? We compare it with the Merton closed-form solution and a finite-difference solver of the same PIDE.
> - **Box B checks the finance.** Are the risk forecasts useful? We compare them with Gaussian, historical simulation, GARCH-t and EVT.
>
> A PINN could pass the first test and fail the second: it could solve the equation perfectly and still forecast risk worse than GARCH, because the equation itself is too simple. That would be an important finding.

**Transition:** "Let me explain how we design the PINN itself."

---

## Slide 8: PINN design
**Speaker:** Utkarsh Sinha · ~2 min

> On the left is the basic setup. The network takes position x, time t and the model parameters theta as inputs, and outputs the **log** of the density. Including theta as an input means one trained network covers many parameter scenarios, which is what RQ4 needs.
>
> The loss has four parts: the PIDE residual, the initial condition, the boundary conditions, and a normalisation term. We use PyTorch, and automatic differentiation gives us the time and space derivatives.
>
> On the right is the main technical challenge. A standard PINN minimises an average residual, and that average is dominated by the centre of the distribution where the density is large. In the 99% tail the density is tiny, around 10 to the minus 4, so tail errors barely affect the loss. But those errors decide VaR and ES. We address this in five ways:
> 1. **Log-density output.** It keeps the density positive and lets the network resolve very small tail values.
> 2. **Adaptive tail sampling.** It places more training points near the 1% quantile.
> 3. **Gauss–Hermite quadrature.** It evaluates the jump integral efficiently at every training point.
> 4. **Starting at a small time t-zero.** At t = 0 the density is a Dirac delta, a spike that a network cannot represent, so we start slightly later using the analytic solution.
> 5. **A normalisation penalty.** It makes the density integrate to one.
>
> We will test each of these in ablation studies rather than assume they help.

**Handover:** "Aditya Sharma will now cover the data, evaluation and our plan."

---

## Slide 9: Data
**Speaker:** Aditya Sharma · ~1.5 min

> We use one controlled dataset and two real markets.
>
> - **Synthetic data.** We simulate returns from a jump-diffusion with known parameters, so the true VaR and ES are known exactly. This is our controlled experiment, and it lets us vary jump intensity and dataset size, including data-scarce settings.
> - **NIFTY 50.** This is our primary real dataset: daily closing values from 2005 to the present, from NSE or Yahoo Finance. If we can get a consistent series, we will use the Total Return Index, which includes dividends.
> - **S&P 500.** Same period and setup. It is a developed market, so it tests whether our conclusions hold beyond one emerging-market index.
>
> The bar at the bottom shows the chronological split. We develop on 2005–2017 and test out of sample from 2018 onwards, which includes the COVID-19 crash. Returns are daily log returns. Forecasts are rolling one-day-ahead: at each date we refit using only past data, forecast the next day's risk, observe the actual return, and move forward. This avoids look-ahead bias.
>
> Synthetic data tells us whether the implementation is correct; real data tells us whether the method is useful. Neither result guarantees the other.

**Transition:** "Here is what we compare the PINN against, and how."

---

## Slide 10: Benchmarks and evaluation
**Speaker:** Aditya Sharma · ~1.5 min

> The table lists the models.
> - **Core:** Gaussian VaR as the simple baseline; historical simulation, to check whether a complex model is needed at all; GARCH(1,1) with Student-t errors, which captures volatility clustering and fat tails; a conventional numerical PIDE solver, to check whether the PINN solves the maths; and our PINN.
> - **Extensions, if time permits:** extreme value theory, which is designed specifically for tails, and a data-only neural network for the physics-versus-data ablation.
>
> On the right are the metrics.
> - **Density:** error reported separately for the bulk and the tail.
> - **VaR:** the Kupiec test checks whether breaches happen about 1% of the time; the Christoffersen test checks whether breaches cluster together.
> - **ES:** it cannot be backtested on its own in the usual way, so we use a joint VaR–ES scoring function from Fissler and Ziegel.
> - **Cost:** training time, prediction time and memory, which answer RQ4.

**Transition:** "Finally, our plan and what we expect to deliver."

---

## Slide 11: Roadmap
**Speaker:** Aditya Sharma · ~1.5 min

> We have organised the work into six stages:
> 1. **Literature review.** In progress now; it fixes the research questions and confirms the gap.
> 2. **Reference solver.** Next: derive the PIDE and implement the Merton closed form and a finite-difference solver.
> 3. **PINN on synthetic data.** Get a working PINN and measure density and tail accuracy.
> 4. **Controlled VaR/ES study.** Compare risk estimates with the known true values.
> 5. **Empirical backtests.** Rolling forecasts on NIFTY 50 and S&P 500.
> 6. **Cost and ablations.** Break-even analysis, failure cases and the thesis write-up.
>
> The order is deliberate: mathematics first. We verify the equation and a reliable numerical solution before working on network architecture. Otherwise, if the PINN gives wrong risk numbers, we could not tell whether the error comes from the maths, the code or the training.
>
> Tools: Python, PyTorch for the PINN, SciPy for the reference solver, and the `arch` package for GARCH models.

**Transition:** "To conclude."

---

## Slide 12: Contributions and scope
**Speaker:** Aditya Sharma · ~1.5 min

> We expect to contribute five things:
> 1. Evidence on how density accuracy differs from tail accuracy.
> 2. A clean ablation of physics-informed versus data-only learning.
> 3. VaR and ES calibration results on two different markets.
> 4. A computational break-even point, if one exists.
> 5. Clear conditions under which simpler models remain preferable.
>
> **Out of scope:** XVA and counterparty credit risk, American-option hedging and multi-asset portfolios. These are separate problems and are left for future work.
>
> **Main risk:** a constant-parameter model may react too slowly to volatility clusters in real data. We address this with rolling re-estimation and, later, a stochastic-volatility extension.
>
> To summarise: our objective is not to prove PINNs are better, but to determine whether they are better for a clearly defined risk-management problem, and to explain why.
>
> Thank you. We would welcome your feedback, especially on whether two markets plus synthetic data is the right scope, and on our choice of benchmarks.

---

## Likely questions: prepared answers

**Q: Why use a PINN if the Merton model has a closed-form solution?**
The closed form exists only for constant coefficients and Gaussian jumps. It is our validation reference, not the end goal. The PINN becomes interesting when (a) one network is amortised over many parameter scenarios, or (b) the model is extended, for example to stochastic volatility, where closed forms disappear. RQ4 tests (a) directly.

**Q: Why not just use GARCH-t, since it already handles fat tails?**
GARCH-t is one of our core benchmarks. If it wins, that is a valid finding. The jump-diffusion model captures discontinuous moves explicitly, which GARCH does not, and we want to measure whether that matters for 99% tail risk.

**Q: How will you estimate jump parameters from real data?**
By maximum likelihood on rolling windows, using the Merton mixture density. We know jump-parameter estimation can be unstable, so we will report parameter uncertainty and compare against simpler calibrations.

**Q: What is the novelty, given that PINNs for jump-diffusion already exist?**
Existing work focuses on option pricing or on solving the equation. Our contribution is a systematic evaluation focused on the tail: density accuracy versus tail accuracy, physics versus data-only learning, calibration on two markets, and a computational break-even analysis. We will confirm the gap as the literature review completes.

**Q: How do you backtest ES?**
ES on its own lacks the property that standard backtesting relies on (elicitability), but VaR and ES are jointly elicitable. So we use a joint VaR–ES scoring function (Fissler–Ziegel), alongside Kupiec and Christoffersen tests for VaR.

**Q: Why add the S&P 500?**
NIFTY 50 alone could give market-specific conclusions. The S&P 500 is a deep, developed market, so it tests whether the findings transfer.

**Q: What computing resources do you need?**
The one-dimensional controlled experiments run on a standard laptop GPU or even a CPU. Only the parameterised PINN and later extensions may need more compute.

~~~~
