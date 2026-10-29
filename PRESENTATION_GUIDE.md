# Presentation Guide: PINNs for Financial Tail-Risk Estimation

A plain-language walkthrough of the whole project, for preparing the presentation. It uses almost no maths: every
idea is explained in words first, and each section ends with **"Say it in one line"** for your slides or speech.

---

## 1. The project in 30 seconds

Banks need to know **how much they could lose on a very bad day**. We built a type of AI model, a
**physics-informed neural network (PINN)**, that learns this from the *equations of how prices move* rather than from
historical data alone. We then tested it carefully against the methods banks use today: on simulated markets where we
know the exact answer, and on 8 years of real Indian and US market data.

**What we found:**
- The PINN is **very accurate**: within about 1% of the true answer for portfolios of up to 5 assets.
- It needs **no historical data to learn**. Without the equations, the same AI would need between 100,000 and
  1,000,000 examples to be as good.
- It is **not faster** than the standard simulation method for the model we used.
- On real data it is **only as good as the market model it solves**. The best traditional methods still beat it there.
- It **breaks** at 10 assets, and if asked about situations it never saw during training.

> **Say it in one line:** *We taught an AI the physics of market crashes, and it learned to measure extreme risk
> accurately without needing historical data. We also show honestly where it falls short.*

---

## 2. Key ideas explained simply

### What is "tail risk"?
Picture all possible daily returns of a portfolio as a hill-shaped curve. Most days sit in the middle of the hill:
small gains, small losses. The **"tails"** are the thin edges of the hill: the rare, extreme days such as the 2008
crash or COVID in March 2020.

**Tail risk** is the risk of those rare, extreme losses. A bank does not fail because of normal days. It fails because
of the tail.

### The two numbers banks use

| Measure | Plain meaning | Example |
|---|---|---|
| **Value at Risk (VaR)** | "On 99 days out of 100, we won't lose more than this." | 99% VaR = ₹30,000 means a loss bigger than ₹30,000 happens only about 1 day in 100. |
| **Expected Shortfall (ES)** | "On that 1 bad day in 100, *how much do we lose on average*?" | ES = ₹55,000 means that when things go really wrong, the average loss is ₹55,000. |

**Why ES matters more:** VaR tells you where the danger zone *starts*; ES tells you *how deep* it goes. Since 2019
the international banking rules (**Basel FRTB**) use **97.5% ES** to decide how much capital a bank must hold.

> **Say it in one line:** *Tail risk is about the rare disaster days. VaR says where they begin; Expected Shortfall
> says how bad they get.*

### Why is this hard?
Real market returns don't follow the neat "bell curve" that textbooks assume:
1. **Fat tails:** extreme days happen far more often than a bell curve predicts.
2. **Volatility clustering:** calm periods and stormy periods come in streaks.
3. **Jumps:** crashes are sudden leaps, not smooth slides.

A model can get the normal days right and still badly misjudge the extreme ones. That is the core difficulty.

### What is a "jump-diffusion" model?
It is the market model we used, and it describes price movement as two things added together:
- **Diffusion:** the everyday small random wiggles, like a leaf drifting on water.
- **Jumps:** occasional sudden shocks, like a stone dropped in the water (a crash, bad news).

This model, from Merton (1976), is a standard way to capture crashes. It comes with an **equation** that describes how
the probabilities of all possible returns spread out over time. If you can solve that equation, you can read off
VaR and ES.

### What is a PINN?
A normal neural network learns by looking at **lots of data**. A **physics-informed neural network** learns by being
told the **rules** (the equation) and is penalised every time its answer breaks them.

**Analogy:** to teach someone to predict where a thrown ball lands, you could show them 10,000 videos of balls
(data-driven AI), or teach them the law of gravity and let them work it out (physics-informed). The second needs far
less data, and it works for throws they have never seen.

> **Say it in one line:** *A PINN is an AI that learns from the laws of the system rather than from examples.*

### How do people solve this equation today?

| Method | How it works | Strength | Weakness |
|---|---|---|---|
| **Finite differences (grid)** | Chop the space into a fine grid and solve step by step | Very accurate | Cost explodes with more assets: 2 assets take 55 s; 10 assets are impossible |
| **Monte Carlo (simulation)** | Simulate millions of random market scenarios and count the bad ones | Works for any number of assets | Needs a huge number of scenarios for rare events, and must be rerun whenever the market changes |
| **PINN (ours)** | A neural network trained to satisfy the equation | No grid, handles many assets, can learn many market conditions at once | Training takes time, and it can be wrong in regions it never saw |

---

## 3. Main objective: what were we trying to find out?

**Main question:** *When does a PINN measure tail risk better than traditional methods, and when is it not worth
it?*

**Important:** our aim was *not* to prove that the PINN wins. It was a fair, honest test. A finding like "the old
method is better here" is a valid research result.

### The five research questions (RQs)

| RQ | Simple version |
|---|---|
| **RQ1** | Does the PINN get the *extreme* cases right, not just the normal ones? |
| **RQ2** | Are its risk numbers accurate, and do they hold up on real market data? |
| **RQ3** | Does teaching it the equation actually help, compared with the same AI trained on data only? |
| **RQ4** | Does it still work when the portfolio has more assets (1, 2, 5, 10)? |
| **RQ5** | Is it faster than simulation once trained? |

### What were we optimising?
When training, the PINN minimises **one combined error score** with three parts:
1. **Equation error:** how much the network's answer breaks the market equation.
2. **Starting-point error:** how far it is from the known answer at the very start (half a day).
3. **Total-probability error:** all probabilities must add up to 100%.

Training means adjusting the network, step by step (20,000 steps), to push this combined score as close to zero as
possible.

**How we judged success:** the **error in Expected Shortfall at 99%**, compared with the *exact* answer, measured at
**1 day, 5 days and 10 days** ahead and averaged over 20 different portfolios. Below 1% error is very good.

> **Say it in one line:** *We trained the AI to obey the market's equation, and judged it by how close its
> worst-case loss estimate came to the true value.*

---

## 4. Data used

### Simulated (synthetic) markets: for testing accuracy
We generated markets from the jump-diffusion model with realistic settings: 20% annual volatility, about 5 crashes per
year on average, each about −5%.

**Why simulated data?** For this model we can compute the **exact true answer** with a formula. So every error we
report is a real, measured error, not a guess. This is what makes the study rigorous.

### Real market data: for real-world testing

| Basket | Assets | Why |
|---|---|---|
| **NIFTY (India)** | Reliance, HDFC Bank, Infosys, ICICI Bank, TCS | 5 of the largest Indian companies |
| **S&P sectors (USA)** | Tech (XLK), Financials (XLF), Energy (XLE), Health care (XLV), Industrials (XLI) | 5 major US sectors |

- **Source:** Yahoo Finance daily prices, 2006 to October 2026.
- **Training period:** 2006–2017, used to fit the models.
- **Test period:** 2018–2026, about 2,200 trading days the models never saw, **including the COVID crash**.
- We tested two portfolios: a simple **equal-weight basket**, and the basket plus a **sold put option**, a
  non-linear position like those real trading desks hold.

---

## 5. Models compared

| Model | Plain description |
|---|---|
| **PINN (ours)** | AI that solves the market equation |
| **Exact formula** | The mathematically perfect answer; our "answer key" on simulated data |
| **Finite differences** | Grid-based equation solver |
| **Monte Carlo** | Simulate many scenarios |
| **Data-only neural network** | *Same AI as ours*, but trained only on data, without the equation (tests whether the physics helps) |
| **Gaussian** | Assumes a bell curve; the simplest baseline |
| **Historical simulation (HS)** | "The future will look like the last 2 years" |
| **Filtered HS (FHS)** | HS adjusted for whether markets are currently calm or stormy |
| **GARCH-t** | Classic model of changing volatility with fat tails |
| **GARCH-EVT** | GARCH plus Extreme Value Theory, a specialised model for extreme events |
| **DCC-GARCH** | GARCH for several assets, with changing correlations between them |

---

## 6. How we built the PINN, step by step

### The design choices that made it work
1. **Predict the log of the probability, not the probability itself.** Tail probabilities are tiny numbers like
   0.0001, and neural networks handle such numbers badly. Taking the log turns them into ordinary numbers like −9.
   *Analogy: measuring earthquakes on the Richter scale instead of in raw energy.*
2. **Measure errors in relative terms.** A normal PINN treats a small mistake in a rare region as unimportant, because
   the probabilities there are tiny. We rewrote the equation so that **a 10% error counts the same everywhere**,
   whether on a normal day or a crash day. **This was the single most important choice.**
3. **Calculate the crash term carefully.** The jump part of the equation needs an average over all possible crash
   sizes. We use a precise averaging technique with 48 points; using only 8 points made 1-day errors 40 times worse.
4. **Start slightly after time zero.** At the very first instant the answer is a single sharp spike, which is
   impossible to learn, so we start from the known answer at half a day.
5. **Force probabilities to add up to 100%.** The equation alone doesn't guarantee this, so we added a penalty.
6. **No grid for many assets.** Instead of a grid, we pick training points smartly, concentrating them where crashes
   could put the portfolio, using only the model's own settings.

### The seven project stages

| Stage | What we did | Result |
|---|---|---|
| **1** | Literature review | Found the gap: no one had systematically tested PINN *densities* for *tail risk* |
| **2** | One-asset version: built the answer key, built the PINN, tested which design choices matter | Within 1%; the design choices above proven |
| **3** | Two-asset answer key with a grid solver | Showed grids become impractical fast |
| **4** | Scaled the PINN to 1, 2, 5 and 10 assets | Works up to 5, breaks at 10 |
| **5** | One PINN covering *many* market conditions at once ("amortised") | Works, but simulation is faster |
| **6** | Real data: NIFTY and S&P, 2018–2026 | As good as its model; the best traditional models win |
| **7** | Physics vs data-only test, and stress tests to find failure points | Physics is worth about 1 million data points; documented the failures |

---

## 7. Results: where we did well, and where we didn't

### ✅ Stage 2: one asset, very accurate
- All risk numbers were **within 1%** of the exact answer. ES at 99% was off by only **0.3%** for 1 day and **0.16%**
  for 10 days.
- **Testing each design choice:**
  - Without the "relative error" trick, 10-day errors rose to **57–152%**. That one trick is the difference between
    useless and excellent.
  - With too few points for the crash term (8 instead of 48), 1-day errors rose to **12–15%**.
  - A popular technique, "tail-adaptive sampling" (focusing training on the tail), gave **no benefit**. A useful
    negative finding.
- **Honest note:** for a single asset, the traditional grid method is faster *and* more accurate. Stage 2 proves the
  PINN works; it doesn't show it is needed.

### ✅ Stage 4: more assets, works up to 5
| Assets | PINN error (1 day / 10 days) | Simulation error (same budget) |
|---|---|---|
| 1 | 1.4% / 0.05% | 1.1% / 0.22% |
| 2 | 0.7% / 1.0% | 1.4% / 1.6% |
| 5 | 1.1% / 0.6% | 0.2% / 0.4% |
| 10 | ❌ 5.2% / **62%** | 0.6% / 0.8% |

- **Good:** about 1% error up to 5 assets, and training time barely grows (9 → 13 minutes). The grid method can't
  even run here.
- **Bad:** at **10 assets it breaks**. The network gets the spread of crashes wrong and the error grows over time. We
  report this honestly as the method's limit.
- **Why it breaks:** with 10 assets the problem is much harder, and the same network size and training time were not
  enough to learn it.

### ⚠️ Stage 5: one model for all market conditions, accurate but not faster
- We trained **one PINN** that takes the market condition (volatility level, crash frequency, crash size) as input.
  One training run (about 20 minutes) then covers a whole range of markets with no retraining.
- **Good:** accuracy matched simulation for 2 assets (1.3% vs 1.8% at 1 day). It can also tell you **how much each
  asset contributes** to the risk, almost for free.
- **Bad:** **simulation was faster**: about 1 ms per question, against 4–18 ms for the PINN. So we found **no
  break-even point**.
- **Why:** our market model is very cheap to simulate. A PINN's speed advantage would only appear for more complex
  models that are slow to simulate. We say this openly.

### ⚠️ Stage 6: real data, as good as its model but not the best

**Backtests: how did each model do over 2018–2026 (99% level)?**

*"Breaches" = days the actual loss exceeded the predicted VaR. Expected: about 22. The score (FZ0) is a fair combined
rating of VaR and ES; lower is better.*

| Model | NIFTY breaches | NIFTY score | S&P breaches | S&P score |
|---|---|---|---|---|
| DCC-GARCH | 28 | **−3.30 (best)** | 33 | −3.29 |
| Filtered HS | 19 | −3.28 | 20 | −3.24 |
| GARCH-EVT | 26 | −3.20 | 20 | **−3.35 (best)** |
| **PINN (ours)** | 15 | −3.16 | 27 | −3.05 |
| Historical sim. | 26 | −3.02 | 30 | −2.73 |
| Gaussian | 38 | −2.88 (worst) | 59 | −2.30 (worst) |

- ✅ **The PINN reproduces its model almost perfectly:** its risk numbers differed from the exact model by only
  **0.2% (S&P)** and **2.8% (NIFTY)**, with the same breach days.
- ✅ **It beat** the Gaussian (bell-curve) model, historical simulation and the plain jump model.
- ✅ It **passed** the main accuracy test at 99% on both markets.
- ❌ **It lost to** the volatility-based models (DCC, Filtered HS, GARCH-EVT).
- ❌ At 97.5% it was **too cautious**, predicting bigger losses than actually happened.
- **Why it lost:** the PINN solves the market model faithfully, but the model itself is simpler than real markets.
  Real markets have strongly changing volatility, which the GARCH-type models capture better. **The weakness is the
  market model, not the AI.** Better AI can't fix a simpler model.
- **A real failure we found and fixed:** on one day (16 June 2022, S&P) the PINN gave a nonsense answer, because a
  rare random scenario landed far outside the region it was trained on. We added a **"trust region"** that ignores
  such far-away scenarios. This removes essentially zero real probability and fixed the problem.

### ✅ Stage 7: does the physics help? Yes, hugely
We compared our PINN with the **exact same AI trained only on data** (no equation).

| Training data | Data-only AI error (1 day) | PINN error (1 day) |
|---|---|---|
| 10,000 examples | 13% | **1.3%, with zero examples** |
| 100,000 examples | 5% | |
| 1,000,000 examples | 1.3% | |

- The equation is **worth about 100,000 to 1,000,000 examples of data**.
- Real markets give only a few thousand days of data, and at that size the data-only AI is **5–60% wrong**.
- **Honest caveat:** if you already know the model, you could simulate unlimited data. The PINN's real advantage is a
  smooth, ready-to-use risk curve for all times and conditions, at no extra cost per question.

### Stress tests: where it breaks

| Test | Result | Verdict |
|---|---|---|
| Crisis-level crash frequency (50 per year) | 0.8% error | ✅ Survived (we expected it to fail) |
| Same, with wider training coverage | 0.1% error | ✅ Fixed by a setting |
| Starting very close to time zero | 5% error at 1 day | ❌ Fails for very short horizons |
| Extreme market conditions (5 assets) | up to 10% | ⚠️ Gets worse at longer horizons |
| 10 assets | up to 62% | ❌ Breaks |
| Far outside the trained region | Nonsense values | ❌ Must not be trusted there; the trust region is needed |

---

## 8. Summary scorecard: when the PINN is better or worse

| Compared with | PINN better? | Why |
|---|---|---|
| Grid methods, 1 asset | ❌ No | Grids are faster and more accurate in 1D |
| Grid methods, 3+ assets | ✅ Yes | Grids can't run; the PINN can |
| Monte Carlo, accuracy (≤ 5 assets) | ≈ Similar | Both around 1% |
| Monte Carlo, speed | ❌ No | Our model is cheap to simulate |
| Monte Carlo, many market conditions | ✅ Partly | One model covers all conditions and gives risk contributions for free |
| Data-only AI | ✅ Yes, strongly | The equation is worth about 1 million data points |
| Gaussian / historical simulation (real data) | ✅ Yes | Captures crashes and fat tails |
| GARCH / EVT / DCC / Filtered HS (real data) | ❌ No | They model changing volatility better; our *market model* is the limit |
| 10 assets | ❌ No | Breaks down |

---

## 9. Key takeaways (for your conclusion slide)

1. **Design beats size.** Small, clever choices (log probabilities, relative error) matter more than making the AI
   bigger.
2. **Physics replaces data.** Teaching the AI the equation saves about a million data points. That matters most when
   data are scarce, as they always are for rare crashes.
3. **Scales to 5 assets.** It works where grid methods can't, but stops at 10.
4. **Not faster for simple models.** The PINN's value is a smooth, flexible risk tool, not raw speed. Speed gains
   would need more complex market models.
5. **The model matters more than the AI.** On real data the PINN is only as good as the market model it learns. To
   predict better, improve the model.
6. **Know the limits.** An AI must not be trusted outside what it was trained on. We found a real case of this and
   built a fix.

---

## 10. Limitations and future work

**Limitations:**
- All assets share one "market crash" factor; company-specific crashes weren't modelled.
- Volatility changes enter only as a daily adjustment, not as a full model.
- It doesn't work at 10 or more assets yet.
- It was tested on two baskets and one period (2018–2026).

**Future work:**
- Richer market models with changing volatility (the Bates model) and crashes that trigger more crashes (Hawkes
  processes). These are slow to simulate, so this is where a PINN could finally win on speed.
- Better network designs to go beyond 10 assets.
- Adding "confidence bars" so the AI can flag when it is unsure.
- Using option prices to learn what the market expects about future crashes.

---

## 11. Likely questions and simple answers

**Q: Why not just use historical data?**
Crashes are rare. Even 20 years of data contain only a handful of true disasters, so data-only methods have very
little to learn from. The PINN learns from the equation, which describes crashes even when the data has few of them.

**Q: If Monte Carlo is faster, why use a PINN?**
For our simple model, Monte Carlo *is* the better tool for speed, and we say so. The PINN gives a smooth, complete
picture that answers many questions at once (any portfolio, any market condition, each asset's contribution). For
complex models that are slow to simulate, it could also win on speed.

**Q: Your model lost to GARCH on real data. Isn't the project a failure?**
No. We showed the PINN solves its market model almost perfectly (0.2–2.8% difference). It lost because the
*jump-diffusion model* is simpler than real markets, not because the AI is poor. That tells future work exactly what
to improve: the model, not the AI.

**Q: How do you know your errors are real?**
For the simulated tests we have the **exact mathematical answer**, so every error is measured, not estimated. We also
compared every method on the same random scenarios, so the differences come from the methods themselves and not from
luck.

**Q: Why does it fail at 10 assets?**
The problem gets much harder with more dimensions, and we kept the same network size and training time for a fair
comparison. Bigger networks and longer training might fix it. That is future work.

**Q: What is new about your work?**
PINNs have been used for option pricing before, but no one had systematically tested whether PINN-learned
probability distributions give accurate **tail-risk** numbers, against an exact answer, against traditional risk
models on real data, and against the same AI without physics. We did all of that, including documenting where it
fails.

**Q: What hardware did you use?**
A single free cloud GPU (NVIDIA T4, on Google Colab and Kaggle). Each model trained in 10–25 minutes, using less than
0.5 GB of memory.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Tail** | The extreme, rare outcomes at the edge of the distribution |
| **VaR (Value at Risk)** | The loss level exceeded only on the worst X% of days |
| **ES (Expected Shortfall)** | The average loss on those worst days |
| **99% / 97.5%** | The confidence level: the worst 1% or 2.5% of days |
| **Jump-diffusion** | A price model with small random wiggles plus sudden crashes |
| **PINN** | A neural network trained to obey an equation |
| **Monte Carlo** | Estimating by simulating many random scenarios |
| **Finite differences** | Solving an equation on a grid of points |
| **Backtest** | Checking past predictions against what actually happened |
| **Breach** | A day when the actual loss exceeded the predicted VaR |
| **Amortised model** | One model trained once that works for many market conditions |
| **Data-only model** | The same AI trained on examples instead of on the equation |
| **Trust region** | A safety rule that ignores scenarios far outside what the model was trained on |
| **GARCH** | A classic model in which volatility changes over time |
| **EVT** | Extreme Value Theory: statistics focused on extreme events |
