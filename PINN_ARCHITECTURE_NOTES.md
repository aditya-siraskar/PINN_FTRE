# My Part: The PINN Architecture

My personal prep notes on the part of the project I owned: how the PINN is designed, why each choice was made, and
how to explain it in an interview. Code: `pinn_tailrisk/pinn.py` (1 asset), `pinn_nd.py` (many assets),
`pinn_amortised.py` (many market conditions).

---

## 1. My 60-second pitch

> I was responsible for the PINN architecture. My main idea was that **a standard PINN is built to be accurate on
> average, but tail risk lives where probabilities are tiny**, so a standard PINN ignores exactly the region we care
> about. I redesigned it around that problem.
>
> First, the network outputs the **log of the probability** instead of the probability, so a tail value like 0.0001
> becomes an ordinary number like −9. Second, I rewrote the market equation in **log form**, which makes the training
> error **relative**: a 10% miss in the tail counts as much as a 10% miss in the middle. Third, I fed the network
> **rescaled inputs** so the shape it has to learn looks similar at every horizon.
>
> The log-form error turned out to be the single most important choice: without it, the 10-day error goes from
> **under 0.1% to 57–152%**. I then extended the same design from one asset to five, and to a single network that covers
> many market conditions at once.

---

## 2. The core problem I was solving

- The PINN has to learn the **probability curve of returns** (how likely each gain or loss is) at every horizon
  from half a day to 10 days.
- VaR and ES come from the **far left edge** of that curve, where probabilities are 1,000 to 100,000 times smaller
  than in the middle.
- A standard PINN minimises the **absolute** equation error. In the tail, all the numbers are tiny, so the errors are
  tiny too. The network can be **100% wrong in the tail** and the loss barely notices.

**My guiding principle:** *make the network care about the tail as much as the middle, without hand-tuned
weights.* Almost every design choice below follows from that.

---

## 3. The architecture, piece by piece

### 3.1 The network itself (deliberately simple)

| Item | 1 asset | 2–10 assets |
|---|---|---|
| Type | Fully connected network (MLP) | Same |
| Size | 4 hidden layers × 64 neurons (~13k weights) | 4 × 128 (~50k weights at 5 assets) |
| Activation | tanh | tanh |
| Inputs | position (return) + time | d returns + time (+3 market parameters in Stage 5) |
| Output | 1 number: log-probability | same |
| Initialisation | Xavier normal, zero bias | same |

**Why tanh?** The equation needs **second derivatives** of the network output (the curvature of the curve). tanh is
smooth, so those derivatives exist and behave well. ReLU's second derivative is zero almost everywhere, which would
break the physics term.

**Why so small?** I wanted to show that **design beats size**. The network stayed the same size from one to five assets.

### 3.2 Output: log-probability, not probability

- The network outputs **u = log p**, and the probability is p = exp(u).
- **Probabilities are then always positive**, automatically. A plain network can output negative probabilities.
- A tail value of 0.0001 becomes about −9, a normal-sized number that a network handles easily (the Richter-scale
  analogy).
- I also **add the known bell-curve shape** to the output, so the network only learns the *correction* to it,
  mainly the jump/crash part. The raw output is multiplied by 10 so the network's small internal values can cover
  the full range of log-probabilities.

### 3.3 Inputs: rescaled so every horizon looks alike

- The curve is very **narrow at half a day and wide at 10 days**. A network fed raw inputs would have to learn very
  sharp shapes early on and smooth shapes later.
- So I feed a **rescaled return**, (return − drift) ÷ (volatility × √time), which keeps the curve's width at
  **about 1 at every horizon**. Time enters as **log-time**, scaled to [−1, 1].
- With many assets, the inputs are also **"whitened"**: rotated and rescaled to remove the correlation between
  assets, so the network sees roughly uncorrelated inputs.

### 3.4 The training error (loss): what the network is pushed to satisfy

| Part | Meaning | Weight |
|---|---|---|
| **Equation error** | How much the curve breaks the jump-diffusion equation, at 4,096 random points per step | 1 |
| **Starting-point error** | Match the exact known curve at **half a day** | 10 |
| **Total probability** | The area under the curve must equal 1 (100%) | 10 |
| **Edge condition** (1 asset only) | The curve must fall off at both ends | 1 |

The key changes I made:

1. **Log-form equation (the crucial one).** I divided the whole equation by the probability. Mathematically it's
   the same equation, but now every point's error is a **relative** error, so tail points count fully.
2. **Multiplied by time.** This makes the error a pure number of similar size at all horizons, so the short
   horizons don't dominate.
3. **Start at half a day, not at zero.** At time zero the "curve" is an infinitely thin spike, which no network can
   represent. At half a day the exact curve is known and smooth, so I start there.
4. **Total-probability penalty.** The log-form equation only fixes the shape *up to a constant*, so this term pins
   the level down.

### 3.5 The crash (jump) term: Gauss–Hermite quadrature

- The jump part of the equation is an **average over all possible crash sizes**: an integral.
- Instead of random sampling, I use **Gauss–Hermite quadrature**: a fixed set of 48 crash sizes with optimal
  weights, which is exact for bell-shaped crash sizes.
- **Why 48?** I tested it. With 8 points the 1-day ES error is 12–15%; with 24 it is 1.8%; with 48 it is **0.3%**.
- Because these 48 points can be evaluated anywhere, the method needs **no grid** at all.

### 3.6 Training recipe

- **Adam** optimiser, learning rate 0.002, decayed smoothly to 0.00002 (cosine schedule), for **20,000 steps**.
- **Gradient clipping** at 1.0 so a few extreme tail points can't destabilise a step.
- **Fresh random points every step** rather than a fixed set, so the network can't memorise particular points.
- Fixed random seeds, so every run is reproducible. Training took 10–25 minutes on one T4 GPU, using under 0.5 GB.

---

## 4. Scaling it up: what I changed for many assets (Stages 4–5)

| Problem at higher dimension | My solution |
|---|---|
| Curvature in d dimensions costs d extra derivative passes | Exact up to 4 assets; above that, the **Hutchinson trick**, which estimates it with a few random probes, at a cost that doesn't grow with d |
| A box-shaped domain wastes almost all points in empty space | **No box.** Points are drawn from a "smart sampler": a mixture of 0, 1, 2 or 3 crashes, each a heavy-tailed bell shape built only from the equation's coefficients, never from the answer |
| Crashes are common to all assets | The jump integral stays **one-dimensional** (one common crash shared by all assets), so the 48-point quadrature carries over unchanged |
| Answer needed for any portfolio | Sample the curve once, then weight the samples (**importance sampling**). Any portfolio's VaR/ES then costs one dot product and a sort |

**Stage 5, one network for many markets:** the market parameters (volatility level, crash frequency, crash size)
become **three extra inputs**, each scaled to [−1, 1], with crash frequency on a log scale. New parameters are drawn
for every training point, so one 20-minute training run covers the whole range.

---

## 5. Things I paid close attention to (lessons I can talk about)

1. **Relative vs absolute error.** Getting this right was the main idea. It's backed by the ablation:
   **< 0.1% vs 57–152%** at 10 days.
2. **Don't trust "obvious" tricks; test them.** Focusing 40% of training points on the tail sounded sensible but gave
   **no gain**, so I dropped it for the multi-asset version. It's a useful negative result.
3. **Sampler tails must be moderate.** My first multi-asset sampler used very heavy tails. At 10 assets it put
   training points so far out that their errors dominated the whole loss (target log-probabilities around −7000). I
   switched to a near-bell-shaped sampler for training and kept the heavy-tailed one only for computing risk.
4. **A network is only trustworthy where it was trained.** On real data (16 June 2022) one far-out sample dominated
   the ES estimate, because the network had never seen that region. I added a **trust region**: samples beyond 6
   "standard distances" from the training region are ignored. Real probability mass there is below 10⁻¹⁵, so nothing
   real is lost.
5. **Every choice is a switch.** Each design choice is a config option, so it could be removed and measured
   (ablation). That's how I can defend each one with numbers.
6. **Always compare against an exact answer.** The test model has a known exact solution, so every error is real,
   not a guess.

---

## 6. Where my architecture breaks (be upfront)

| Case | Result | Why |
|---|---|---|
| 10 assets | 62% ES error at 10 days | Same network size and training budget as at 5 assets; the problem is much harder |
| Start 10× closer to zero (0.05 day) | 5.2% at 1 day | The starting curve is nearly a spike, hard to represent |
| Far outside the training region | Unreliable | No equation was enforced there, hence the trust region |
| Crisis-level crash rate (50/yr) | **Survived**: 0.8%, 0.1% with a wider sampler | Shows the design is robust |

---

## 7. Likely interview questions on my part

**"What was your main idea?"**
Make the PINN tail-aware by working in log space: log-probability output plus a log-form equation, so errors are
relative and tail points count as much as the middle.

**"Why not just put a bigger weight on tail points?"**
Weights need tuning and would differ by horizon and asset. The log form gives the right balance automatically, and
it's the same equation, just rewritten.

**"Why did you choose that network size?"**
Small on purpose. The ablations showed the design choices mattered far more than size, and a fixed size made the
1-to-10-asset comparison fair.

**"How do you know each choice helps?"**
Ablation: I removed one choice at a time and measured the error against the exact answer. Log residual:
< 0.1% → 57–152%. Quadrature 48 → 8 points: 0.3% → 12–15%. Tail-focused sampling: no gain, so it was dropped.

**"How does it handle the crash term without a grid?"**
48-point Gauss–Hermite quadrature: the network is evaluated at 48 shifted points, and their weighted sum gives the
crash average exactly for bell-shaped crash sizes.

**"How did you make it work in 5 dimensions?"**
Whitened inputs, the Hutchinson trick for curvature, a smart sampler instead of a grid, a one-dimensional common-jump
integral, and importance sampling for portfolio risk.

**"Why does it fail at 10 assets, and how would you fix it?"**
I kept the network and budget fixed for a fair test. Next steps: a wider or deeper network, more training steps,
and possibly a structure that separates the shared crash factor from asset-specific noise.

**"What would you do differently?"**
Add uncertainty estimates (e.g. an ensemble of networks) so the model can flag when it's outside the region it can
be trusted in, rather than relying on a fixed trust region.
