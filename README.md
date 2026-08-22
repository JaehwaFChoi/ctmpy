# cogtraitmodel

[![DOI](https://zenodo.org/badge/1340220989.svg)](https://doi.org/10.5281/zenodo.22031040)
[![tests](https://github.com/JaehwaFChoi/ctmpy/actions/workflows/ci.yml/badge.svg)](https://github.com/JaehwaFChoi/ctmpy/actions/workflows/ci.yml)

Bounded-trait psychometrics in Python: the **Anchored Logistic Family (ALF)** —
the Cognitive Trait Model of Choi (2022) and its generalisation to an arbitrary
mastery anchor — with deterministic, quadrature-based estimation.

A trait level of `0.62` means *62% of mastery of a stated task domain*, not
"0.62 standard deviations above whoever happened to be calibrated".

```bash
pip install cogtraitmodel
```

## Why

Item response theory frees the scale from the items but leaves it pinned to the
calibration sample: `θ = 0` is a statement about other examinees. For formative
assessment, learning progressions and adaptive instruction, the question is not
*who is ahead* but *how far along is this learner*. Placing θ on `[0, 1]` with
0 and 1 defined by the task domain answers that question directly.

The package makes the model usable inside a response loop:

- **No MCMC.** Item parameters come from Bayes modal EM on a Gauss–Legendre
  grid — 35× faster than a converged random-walk sampler and 73× faster than
  NUTS, agreeing with both to a θ correlation of 0.9999.
- **A single adaptive update costs about 5 µs**, because the grid is fixed on
  `[0, 1]` for the lifetime of an item bank and the item-by-node log-probability
  matrix is precomputed once.
- **Honest uncertainty.** Scoring reports the posterior SD, not the
  Fisher-information SEM, which collapses to zero at the mastery boundary.

## Quick start

```python
import numpy as np
import cogtraitmodel as ctm      # the paper's abbreviation, kept short

# simulate
theta = ctm.gen_theta(1000, rng=np.random.default_rng(0))
alpha = np.full(20, 8.0)
beta  = np.linspace(0.1, 0.9, 20)
Y = ctm.gen_responses(theta, alpha, beta, rng=np.random.default_rng(1))

# calibrate items
fit = ctm.bayes_modal_em(Y)

# score persons — MAP point estimate with posterior SD
out = ctm.score(Y, fit["alpha"], fit["beta"])
out["theta"]   # 0.62  ->  "62% of mastery"
out["sd"]      # report this alongside; it decides when to stop testing
```

Three-parameter (with guessing), information functions, and the empirical
dataset used in the paper:

```python
fit3 = ctm.bayes_modal_em(Y, three_p=True)
ctm.tif(np.linspace(0.02, 0.98, 25), alpha, beta)   # test information
Y_lsat = ctm.datasets.build()                       # LSAT Section 6, (1000, 5)
```

### Tracking a learner over time

Bayesian updating alone is not enough when the quantity being measured is
changing: carrying the posterior forward asserts "this learner is where they
were last time", and it is **worse than having no memory at all** (RMSE 0.230
against 0.122) while reporting a *smaller* standard deviation. What is needed
is the prediction step of a state-space model — a transition that moves the
distribution, not just widens it.

```python
from cogtraitmodel import growth

out = growth.sequential_score(
    responses,      # list of (n, K_t) arrays, one per occasion
    items,          # list of item indices administered at each occasion
    alpha, beta,    # anchored item parameters
    memory="hctm",  # "none" for the memoryless baseline
)
out["theta"]        # (T, n) posterior means
out["sd"]           # (T, n) posterior SDs — these stay honest
```

The growth curve is the `L → ∞` member on the time axis, so no horizon constant
is needed. Its four parameters read as *entry level*, *level converged to*,
*rate*, and *when growth happens*; setting `delta < gamma` expresses forgetting
in the same form. Non-monotone trajectories cannot be represented at any
parameter setting — there, memoryless scoring is better.

**The degrees-of-freedom discipline is enforced, not optional.** Releasing
parameters as fast as the history grows lets the curve interpolate it exactly,
the predictive variance collapses, and the next prior becomes a spike; that
condition produced the worst results in the study (RMSE 0.21 with a reported SD
seven times too small). `fit` therefore keeps `k ≤ H − 1` and floors the
residual variance at the measurement error implied by the posterior SDs.

### An arbitrary mastery anchor

Finite `L` needs no separate estimator, because

```
P_L(θ; α, β)  ==  P_1(θ/L; αL, β/L)      # exact to machine precision
```

so `fit_L` and `score_L` are rescaling wrappers around the `L = 1` path:

```python
fit = ctm.fit_L(Y, L=3.0)                      # item parameters on the L scale
out = ctm.score_L(Y, fit["alpha"], fit["beta"], L=3.0)   # θ ∈ [0, 3]
```

One consequence is worth stating: the same response data fit equally well at
every finite `L`, so **`L` is a definitional choice rather than a parameter the
data identify**. Moving the unit is a rescaling; only the origin is structural.
`L → ∞` is the exception — it does not reduce this way, and it is used as a
growth model rather than a measurement model.

## The family

One parameter, the **mastery anchor** `L`, indexes the family. It says where the
mark "1" falls on the ruler; the members share a kernel and differ by a
normalising constant.

| | support | anchors fixed | reading of θ |
|---|---|---|---|
| `L = 1` — CTM | `[0, 1]` | origin **and** unit | proportion of mastery |
| `L → ∞` — HCTM | `[0, ∞)` | origin only | ratio scale, no ceiling |

**Origin before unit.** An origin alone already licenses ratio statements; the
unit is a further commitment — that the tasks in hand exhaust the domain. Use
`L = 1` for measurement, and `L → ∞` (`cogtraitmodel.hctm`) as a growth curve on the
time axis, where no finite horizon should be imposed.

## What to watch out for

- **Boundary solutions are more dangerous here than in IRT.** A perfect scorer
  gets `θ̂ = +∞` under a logistic IRT model, which announces itself; here they
  get `θ̂ = 1`, a finite number that reads as "all tasks mastered". Use MAP
  (the default), which cannot reach the boundary under a Beta prior. `score()`
  returns an `at_bound` flag; MLE is a diagnostic only.
- **Wright maps do not transfer.** `P(β) = ½` only when `β = L/2`; otherwise
  `P(β)` depends on α.
- **Only α > 0 is representable**, so reverse-keyed items cannot be detected.
  Pre-screen before calibration. On the time axis this makes growth curves
  monotone: a rise followed by a fall is outside the family.
- **The prior, not the task domain, holds the scale under marginal estimation.**
  Replacing the item domain with its easier or harder half moves θ̂ by under
  1.5%; moving the population shifts a fixed examinee by 69% of the score
  spread. Anchor item parameters when comparing cohorts.

## Correctness

`p1_naive`, `p2_naive` and `p3_naive` are literal transcriptions of Eqs.
(12)–(14) of Choi (2022) and are kept permanently as the correctness oracle;
every optimised path is pinned by tests asserting agreement with them to 1e-12
over `α ∈ [0.1, 100]`. The bundled LSAT data verifies itself against the
published response-pattern frequencies on load.

```bash
pytest -q     # 127 tests
```
## In the browser

The same estimator runs as a web page — paste a response matrix, calibrate,
and score, with nothing installed and nothing uploaded:
[jaehwafchoi.github.io/cogtraitmodel-js](https://jaehwafchoi.github.io/cogtraitmodel-js/)
([doi:10.5281/zenodo.22050655](https://doi.org/10.5281/zenodo.22050655)).
The JavaScript kernel is a port of this package and is held to it by a parity
test on every commit.

## Citation

If you use this package, please cite both the software and the model.

```bibtex
@software{choi_cogtraitmodel,
  author  = {Choi, Jaehwa},
  title   = {cogtraitmodel: Bounded-trait psychometrics with the Anchored
             Logistic Family},
  version = {0.2.2},
  year    = {2026},
  doi     = {10.5281/zenodo.22031040},   % concept DOI — resolves to the latest version
  url     = {https://github.com/JaehwaFChoi/ctmpy}
}

@article{choi2022ctm,
  author  = {Choi, Jaehwa},
  title   = {Cognitive Trait Model: Measurement Model for Mastery Level and
             Progression of Learning},
  journal = {Mathematics},
  volume  = {10},
  number  = {15},
  pages   = {2651},
  year    = {2022},
  doi     = {10.3390/math10152651}
}
```

## License

MIT
