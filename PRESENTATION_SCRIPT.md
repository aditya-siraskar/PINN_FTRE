# Presentation Script: PINNs for Financial Tail-Risk Estimation

Slide-by-slide script for `PINN_TailRisk_Interview.pptx` (15 slides, about 20 minutes).
Read it a few times, then say it in your own words; don't read it out word for word.
**Bold** marks the points to stress. *Next* is the line that leads into the following slide.

---

## Slide 1: Title (~30 s)

> Good morning. My name is *[your name]*, and I'll present our M.Tech research project, done by Aditya Siraskar,
> Utkarsh Sinha and Aditya Sharma in the Department of Data Science at MPSTME.
>
> The project is about **measuring extreme financial risk with AI**. More specifically, it uses a kind of AI that
> learns from the **equations of how markets move**, not just from past data. We then tested it honestly against the
> methods banks use today, on simulated markets where we know the exact answer and on eight years of real Indian and
> US market data.

*Next:* "Let me start with why extreme risk matters."

---

## Slide 2: Banks fail on the worst days, not average ones (~1.5 min)

> Banks and funds are not threatened by ordinary days. Small ups and downs are normal. What sinks them are the
> **rare, extreme days**, like 2008 or the COVID crash in March 2020. The risk of those days is called **tail risk**,
> because those days sit at the thin edges, the "tails", of the distribution of returns.
>
> The industry uses two numbers for it.
>
> **Value at Risk** answers: *where does the danger zone start?* A 99% VaR says, "on 99 days out of 100, we won't
> lose more than this amount."
>
> But VaR says nothing about how bad that one remaining day can be. **Expected Shortfall** answers that: *on the 1 bad
> day in 100, what is the average loss?* That's why the Basel banking rules now use **97.5% Expected Shortfall** to
> decide how much capital a bank must hold.
>
> The key point is at the bottom: if we know the **full probability curve** of returns, both numbers can be read
> straight off it. So the real problem is getting that curve right, **especially at its edges**.

*Next:* "And getting the edges right is hard, because real markets don't behave like textbooks."

---

## Slide 3: Real markets crash, so the bell curve fails (~1.5 min)

> Textbooks often assume returns follow a bell curve. Real markets break that in three ways:
>
> - **Fat tails:** extreme days happen far more often than a bell curve predicts.
> - **Volatility clusters:** calm periods and stormy periods come in streaks.
> - **Sudden jumps:** crashes are leaps, not smooth slides.
>
> So a model can get the normal days right and still badly misjudge the extreme ones.
>
> We used a well-known model called **jump-diffusion**, shown on the right. It describes price movement as two things
> together: small everyday random **wiggles**, plus occasional sudden **jumps**, which are the crashes.
>
> This model comes with an **equation** that describes how the probability of every possible return spreads out over
> time. If we can solve that equation, VaR and ES follow. And for this model we can also compute the **exact answer**,
> which matters a lot for testing, as you'll see.

*Next:* "So how do people solve this equation today?"

---

## Slide 4: Three ways to solve the market equation (~1.5 min)

> There are three main options.
>
> **Grid solvers** chop the space into a fine grid and solve step by step. They're very accurate, but the cost
> **explodes with the number of assets**: 0.3 seconds for one asset, 55 seconds for two, and impossible for ten.
>
> **Monte Carlo** simulates millions of random market scenarios and counts the bad ones. It works for any number of
> assets, but rare events need **huge numbers of scenarios**, and everything must be rerun whenever market
> conditions change.
>
> The third option is ours, a **physics-informed neural network, or PINN**. It's an AI that learns from the equation
> itself.
>
> **The analogy:** to teach someone where a thrown ball lands, you could show them ten thousand videos, which is the
> data-driven approach. Or you could teach them the law of gravity. The second needs far less data and works for
> throws they've never seen. A PINN is the second approach.
>
> Its weakness: training takes time, and it can be wrong in regions it never saw during training.

*Next:* "That brings us to what we actually set out to test."

---

## Slide 5: Objective: an honest test, not a sales pitch (~1.5 min)

> Our main question was: **when does a PINN measure tail risk better than traditional methods, and when is it not
> worth it?**
>
> I want to stress that our goal was **not to prove the PINN wins**. It was a fair test. If the old method turns out
> better in some setting, that is still a valid research result, as long as the experiment is rigorous.
>
> We broke this into five research questions:
> - **RQ1:** does it get the *extreme* cases right, not just the normal ones?
> - **RQ2:** are its risk numbers accurate, and do they hold up on **real data**?
> - **RQ3:** does teaching it the equation actually help, compared with the **same AI trained on data only**?
> - **RQ4:** does it still work with **more assets**: one, two, five, ten?
> - **RQ5:** once trained, is it **faster** than simulation?

*Next:* "To answer these, we used two kinds of data."

---

## Slide 6: Data: simulated markets and two real baskets (~1.5 min)

> First, **simulated markets**: realistic settings with 20% yearly volatility and around five crashes a year, each
> about minus five percent, for one up to ten assets.
>
> Why simulate? Because for this model **the exact answer is known**. So every error we report is a **real, measured
> error**, not a guess. That's what makes the study rigorous.
>
> Second, **real data** from Yahoo Finance, two baskets of five assets each:
> - an Indian basket of **NIFTY** stocks: Reliance, HDFC Bank, Infosys, ICICI Bank and TCS;
> - a US basket of **S&P sector funds**: technology, financials, energy, health care and industrials.
>
> We fitted the models on **2006 to 2017** and tested on **2018 to 2026**: about 2,200 days the models had never seen,
> **including the COVID crash**.

*Next:* "Now, how does the PINN actually learn?"

---

## Slide 7: How the PINN learns: obey the equation (~2 min)

> Training means adjusting the network, over 20,000 small steps, to push **one combined error to zero**. It has three
> parts:
> - **Equation error:** how much its answer breaks the market equation.
> - **Starting-point error:** how far it is from the known answer at the very beginning, half a day in.
> - **Total probability:** all probabilities must add up to 100%.
>
> A plain PINN did not work well for tail risk, so the **design choices** at the bottom are where our contribution
> lies:
>
> 1. **We predict the log of the probability.** Tail probabilities are tiny numbers like 0.0001, and neural networks
>    handle those badly. Taking the log turns them into ordinary numbers like minus nine, much like the Richter scale
>    for earthquakes.
> 2. **We count errors in relative terms.** A normal PINN treats a mistake in the tail as unimportant, because the
>    probabilities there are tiny. We rewrote the equation so a 10% miss on a crash day counts **as much as** on a
>    normal day. **This turned out to be the single most important choice.**
> 3. We average over crash sizes precisely, with 48 points, and place training points where crashes can land, so we
>    **don't need a grid at all**.

*Next:* "We built and tested all this in seven stages."

---

## Slide 8: Seven stages, each checked before the next (~1 min)

> We moved from simple to hard, and never moved on until the current stage was checked against the exact answer:
>
> 1. literature review to confirm the gap;
> 2. one asset, to build and test the design;
> 3. a two-asset answer key using a grid solver;
> 4. scaling from one to ten assets;
> 5. one model covering many market conditions, to test speed;
> 6. real data backtests;
> 7. physics versus data, plus stress tests to deliberately break it.
>
> Everything is a tested Python library, and every model trained on **a single free cloud GPU in 10 to 25 minutes**.
> Every figure and number can be reproduced from the code.

*Next:* "Let me go through the results, starting with one asset."

---

## Slide 9: 1 asset: within 1%, and one choice is critical (~1.5 min)

> With one asset, **every risk number was within 1% of the exact answer**: Expected Shortfall was off by only
> **0.3%** at one day.
>
> Then we tested each design choice by **removing it**, which is what the chart shows. Note the log scale.
> - Full design: under 0.1% error at ten days.
> - Without the probability penalty: still small.
> - Too few points for the crash term: errors grow.
> - But with the **standard, non-relative error**, the 10-day error explodes to **57 to 152 percent**.
>
> So that one choice is the difference between **useless and excellent**.
>
> We also found that a popular trick, focusing training on the tail, gave **no gain**. That's a useful negative
> finding.
>
> And to be honest: for one asset, the grid solver is faster and more accurate. This stage proves the PINN works; it
> doesn't show it's needed. **It becomes useful with more assets.**

*Next:* "So what happens with more assets?"

---

## Slide 10: More assets: works up to 5, breaks at 10 (~1.5 min)

> We kept the **same network and training time** for one, two, five and ten assets. The chart compares the PINN in
> blue with Monte Carlo in green at ten days, again on a log scale.
>
> **Up to five assets**, the error stays around 1%, close to Monte Carlo, and training time barely grows, from 9 to 13
> minutes. Grid solvers **can't even run** at this size.
>
> **At ten assets it breaks**: the error jumps to **62%**. The same network size and training budget simply weren't
> enough for a problem that much harder.
>
> We **report that honestly as the method's limit** rather than tuning around it. Finding where a method breaks is
> part of the result.

*Next:* "Next, can one model handle many market conditions, and is it fast?"

---

## Slide 11: One model, many markets: accurate, not faster (~1.5 min)

> Here we trained **one network** that takes the market condition, meaning volatility, crash frequency and crash size,
> as an input. It is trained **once, in about 20 minutes**, and then answers for any condition without retraining.
>
> - **Accuracy is good:** 1.3% error, against 1.8% for Monte Carlo with the same budget.
> - It also gives **each asset's share of the risk**, within about 5%, almost for free.
> - But on **speed, Monte Carlo wins**: about 1 millisecond per question, against 4 to 18 for the PINN.
>
> **Why?** Our market model is very **cheap to simulate**. A PINN's speed advantage would only appear for more complex
> models that are slow to simulate. We say this openly; it tells us where the method is actually worth using.

*Next:* "Now the most practical test: real market data."

---

## Slide 12: Real data: as good as its model, not the best (~2 min)

> This table shows how each method did over 2018 to 2026 at the 99% level.
>
> Two columns per market:
> - **Breaches** are the days the actual loss exceeded the forecast. At 99% we'd expect about 22.
> - The **score** rates VaR and ES together; **lower is better**.
>
> What we found:
> - The PINN **reproduces its own market model almost perfectly**: within 0.2 to 2.8%, with the same breach days.
> - It **beats** the bell-curve model and historical simulation, and passes the main 99% test on both markets.
> - But it **loses to the volatility-based models**: DCC-GARCH, filtered historical simulation and GARCH-EVT.
>
> **Why?** The PINN solves its market model faithfully, but that model is **simpler than real markets**, which have
> strongly changing volatility that GARCH-type models capture better. So **the limit is the market model, not the AI**.
> A better AI can't fix a simpler model.
>
> At the bottom: we also **found and fixed a real failure**. On one day in June 2022, a single far-off random scenario
> took over the estimate, because the network had never been trained that far out. We added a **trust region** that
> ignores such points, which fixed it.

*Next:* "The next result is the most important one: does the physics actually help?"

---

## Slide 13: The physics is worth ~1 million data points (~2 min)

> To answer this, we took the **exact same AI** and trained it on data only, without the equation, giving it from one
> thousand up to one million examples.
>
> In the chart:
> - the **red** line is the data-only AI;
> - the **green** line is plain Monte Carlo on the same data;
> - the flat **blue** line is our PINN, which used **zero** examples.
>
> The data-only AI needs **between a hundred thousand and a million examples** just to catch up with the PINN.
>
> Real markets give us **a few thousand daily observations**, and at that size the data-only AI is **5 to 60 percent
> wrong**. Crashes are rare by definition, so data will always be scarce exactly where it matters.
>
> One honest caveat: if you already know the model, you could simulate unlimited data. The PINN's real edge is that it
> gives a **smooth, ready-made risk curve** for all times and conditions, at no extra cost per question.

*Next:* "Let me pull everything together."

---

## Slide 14: Where the PINN wins, and where it breaks (~1.5 min)

> **It's better than:**
> - grid solvers at three or more assets, which can't run there at all;
> - the same AI without physics, by about a million samples;
> - bell-curve and historical models on real data.
> - It even **survived a crisis-level crash rate** of fifty a year, which we expected to break it.
>
> **It's worse, or it breaks:**
> - grid solvers win at one asset, and Monte Carlo wins on speed;
> - volatility models win on real data;
> - it **breaks at ten assets**;
> - it fails for **very short horizons**;
> - and it **can't be trusted far outside its training region**.
>
> **The bottom line:** excellent where the equation is the right model, and limited by the model, by size, and by
> where it was trained.

*Next:* "So what are the key takeaways?"

---

## Slide 15: Key takeaways (~1.5 min)

> Four takeaways.
>
> 1. **Design beats size.** Small, clever choices, like log probabilities and relative errors, mattered far more than
>    making the network bigger.
> 2. **Physics replaces data.** The equation is worth about a million examples, which matters most for rare events
>    like crashes.
> 3. **It scales to five assets**, where grid solvers can't run, but stops at ten.
> 4. **The model matters more than the AI.** On real data, the market model limits accuracy, which tells future work
>    exactly what to improve.
>
> As **next steps**: richer market models with changing volatility and crashes that trigger more crashes. Those are
> slow to simulate, which is exactly where a PINN could also win on speed. Also bigger networks to go beyond ten
> assets, and confidence bars so the AI can flag when it's unsure.
>
> All our code, data and results are on GitHub at the link shown.
>
> **Thank you. I'm happy to take questions.**

---

## If you are short on time

| Time available | What to do |
|---|---|
| 20–25 min | Full script |
| ~15 min | Skip slides 3 and 8; shorten 4 to one sentence per method |
| ~10 min | Slides 1, 2, 5, 7, 10, 12, 13, 15 only |

## Quick answers to likely questions

**"Why not just use historical data?"**
Crashes are rare; even 20 years of data have only a handful. Without the equation, the same AI needs about a million
examples to match the PINN.

**"If Monte Carlo is faster, why use a PINN?"**
For our simple model Monte Carlo *is* better for speed, and we say so. The PINN gives a smooth, complete picture
(any portfolio, any condition, each asset's share of risk) and could also win on speed for complex models that are
slow to simulate.

**"It lost to GARCH on real data. Isn't that a failure?"**
No. The PINN matched its own model within 0.2–2.8%. It lost because the jump-diffusion model is simpler than real
markets. That tells us to improve the model, not the AI.

**"How do you know your errors are real?"**
On simulated data we have the exact mathematical answer, and every method was compared on the same random
scenarios, so differences come from the methods, not luck.

**"Why does it fail at 10 assets?"**
The problem gets much harder, and we deliberately kept the same network and training time for a fair comparison.
Bigger networks and longer training are the next step.

**"What's new in your work?"**
PINNs have been used for option pricing, but no one had systematically tested whether PINN-learned probability
curves give accurate tail risk, against an exact answer, against bank risk models on real data, and against the
same AI without physics. We did all three, and documented where it fails.

**"What was your own contribution?"**
*(Prepare this yourself: the parts you personally built, tested or analysed.)*
