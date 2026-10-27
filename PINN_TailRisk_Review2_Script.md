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
