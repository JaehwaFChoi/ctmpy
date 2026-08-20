"""cogtraitmodel.growth — HCTM on the time axis: growth-curve fitting and
sequential prior updating.

Bayesian updating alone is not enough
-------------------------------------
Bayes' theorem accumulates evidence about a quantity that is **fixed**. A
learner's theta is not fixed — that is the reason for measuring repeatedly.
Carrying the posterior forward unchanged asserts as prior information that
"this person is where they were last time", and the estimator then spends the
new evidence on overcoming its own prior. Empirically the carry-forward was
**worse than having no memory at all** (RMSE 0.2295 against 0.1215) — and it
reported a *smaller* SD, claiming a quarter of the actual error.

What is needed is the **prediction step** of a state-space model: a transition
that moves the distribution rather than merely widening it. This module
supplies that transition.

Why HCTM
--------
A growth curve needs a floor but must not impose a ceiling in advance. The
L -> inf member has an absolute zero and no upper limit, so no time-horizon
constant H is required — theta(t) = gamma + (delta - gamma) *
P_hctm(t; alpha, beta) takes t in its natural units. The four parameters carry
meanings a teacher would recognise:

    gamma  entry level          delta  level converged to
    alpha  rate of learning     beta   when growth happens

Setting delta < gamma expresses decay (forgetting) within the same four
parameters. The curve is monotone, so **a trajectory that rises and then falls
cannot be represented at any parameter setting** — a failure that does not go
away as data accumulate, and where memoryless scoring is the better choice.

The degrees-of-freedom discipline (enforced)
--------------------------------------------
Releasing parameters as fast as the history grows lets the curve interpolate
that history exactly: the residual variance goes to zero, the predictive
variance collapses, and the next prior becomes a spike. That was the worst
condition observed in this study (RMSE 0.21, with a reported SD one seventh of
the actual error). The unlock schedule therefore always keeps **k <= H - 1**,
and the residual variance is floored at the measurement error implied by the
posterior SDs. This discipline is fixed inside `fit` rather than exposed as an
option.

    H <= 2 : gamma                      (k=1)
    H == 3 : gamma, delta               (k=2)
    H == 4 : + alpha                    (k=3)
    H >= 5 : + beta                     (k=4)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from . import core
from .hctm import p2_hctm

__all__ = [
    "curve", "fit", "predict", "to_prior", "sequential_score",
    "PROCESS_SD", "COHORT_DEFAULT",
]

PROCESS_SD = 0.35          # process-noise scale (the floor on prior width)
COHORT_DEFAULT = (1.2, 2.0, 0.95)   # (alpha, beta, delta) cohort starting values


def curve(t, alpha, beta, gamma, delta):
    """Four-parameter HCTM growth curve.

    Parameters
    ----------
    t : array_like
        Time, in its natural units (no rescaling needed — there is no horizon
        constant).
    alpha : float   rate of learning
    beta : float    when growth happens (location of the inflection)
    gamma : float   entry level, theta(0)
    delta : float   level converged to. ``delta < gamma`` gives a decay curve.

    Returns
    -------
    ndarray
    """
    return gamma + (delta - gamma) * p2_hctm(np.asarray(t, dtype=float),
                                             alpha, beta)


def _schedule(H):
    """How many parameters to release given a history of length H.

    Always k <= H - 1 for H >= 2.
    """
    if H <= 2:
        return 1
    if H == 3:
        return 2
    if H == 4:
        return 3
    return 4


def fit(times, theta, sd, cohort=None):
    """Fit a growth curve per learner, with the dof discipline built in.

    Parameters
    ----------
    times : (H,) array_like        observation times
    theta : (n, H) array_like      theta point estimates by occasion
                                   (posterior means recommended)
    sd : (n, H) array_like         posterior SDs by occasion. These weight the
                                   fit and also floor the residual variance.
    cohort : tuple, optional       (alpha, beta, delta) cohort reference values,
                                   used to fill in parameters that a short
                                   history cannot identify per person.

    Returns
    -------
    dict
        ``params``     (n, 4) — columns are (alpha, beta, gamma, delta)
        ``k``          how many parameters were actually released in this fit
        ``resid_var``  (n,) dof-corrected residual variance, after the
                       measurement-error floor is applied
        ``ok``         (n,) whether the optimizer converged. Where False, the
                       last observation is carried instead.
    """
    ts = np.asarray(times, dtype=float)
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    s = np.atleast_2d(np.asarray(sd, dtype=float))
    n, H = th.shape
    if s.shape != th.shape:
        raise ValueError(f"sd shape {s.shape} != theta shape {th.shape}")
    if ts.shape[0] != H:
        raise ValueError(f"times has length {ts.shape[0]} != history length {H}")

    a0, b0, d0 = cohort or COHORT_DEFAULT
    k = _schedule(H)
    params = np.zeros((n, 4))
    rvar = np.zeros(n)
    ok = np.ones(n, dtype=bool)

    for i in range(n):
        y, si = th[i], np.maximum(s[i], 0.02)
        g_init = float(np.clip(y[0], 0.01, 0.95))
        d_init = float(np.clip(y[-1] + 0.2, 0.05, 0.99))

        if H == 1:
            params[i] = (a0, b0, y[0], d0)
            rvar[i] = float(si[0] ** 2)
            continue

        if k == 1:                       # gamma
            def f(p, tt, _a=a0, _b=b0, _d=d0):
                return curve(tt, _a, _b, p[0], _d)
            x0, lo, hi = [g_init], [0.001], [0.98]
        elif k == 2:                     # + delta
            def f(p, tt, _a=a0, _b=b0):
                return curve(tt, _a, _b, p[0], p[1])
            x0 = [g_init, d_init]
            lo, hi = [0.001, 0.01], [0.98, 0.999]
        elif k == 3:                     # + alpha
            def f(p, tt, _b=b0):
                return curve(tt, p[2], _b, p[0], p[1])
            x0 = [g_init, d_init, a0]
            lo, hi = [0.001, 0.01, 0.05], [0.98, 0.999, 20.0]
        else:                            # + beta
            def f(p, tt):
                return curve(tt, p[2], p[3], p[0], p[1])
            x0 = [g_init, d_init, a0, b0]
            lo, hi = [0.001, 0.01, 0.05, 0.05], [0.98, 0.999, 20.0, 30.0]

        try:
            r = least_squares(lambda p: (f(p, ts) - y) / si,
                              x0, bounds=(lo, hi), max_nfev=300)
            full = {1: (a0, b0, r.x[0], d0),
                    2: (a0, b0, r.x[0], r.x[1]),
                    3: (r.x[2], b0, r.x[0], r.x[1]),
                    4: (r.x[2], r.x[3], r.x[0], r.x[1])}[k]
            params[i] = full
            resid = f(r.x, ts) - y
            # dof correction: H - k >= 1 is guaranteed, so this cannot divide by zero
            rv = float((resid ** 2).sum()) / max(H - k, 1)
            floor = float((si ** 2).mean()) / H       # measurement-error floor
            rvar[i] = max(rv, floor)
        except Exception:                 # no convergence -> keep the last observation
            params[i] = (a0, b0, y[-1], y[-1])
            rvar[i] = float(si[-1] ** 2)
            ok[i] = False

    return {"params": params, "k": k, "resid_var": rvar, "ok": ok}


def predict(times, theta, sd, t_next, cohort=None):
    """Predictive mean and SD for the next occasion — the prediction step.

    The predictive SD sums three terms: the dof-corrected residual variance
    (inflated by 1 + k/H for the number of parameters released), a term
    proportional to the extrapolation distance, and process noise. Using the
    fit error alone would understate the predictive variance.

    Returns
    -------
    (mean, sd) : each an (n,) ndarray
    """
    ts = np.asarray(times, dtype=float)
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    s = np.atleast_2d(np.asarray(sd, dtype=float))
    n, H = th.shape

    res = fit(ts, th, s, cohort)
    P, k, rv = res["params"], res["k"], res["resid_var"]

    mean = np.array([curve(t_next, P[i, 0], P[i, 1], P[i, 2], P[i, 3])
                     for i in range(n)], dtype=float)

    if H == 1:
        psd = np.sqrt(s[:, 0] ** 2 + PROCESS_SD ** 2 * 0.25)
    else:
        gap = float(t_next - ts[-1])
        psd = np.sqrt(rv * (1.0 + k / H)
                      + (0.03 * gap) ** 2
                      + (PROCESS_SD * 0.15) ** 2)
        # where the fit failed, carry the last observation with extra slack
        bad = ~res["ok"]
        if bad.any():
            mean[bad] = th[bad, -1]
            psd[bad] = s[bad, -1] + 0.05

    return np.clip(mean, 0.005, 0.995), np.clip(psd, 0.02, 0.5)


def to_prior(mean, sd, nodes, quad_w):
    """Turn a predicted (mean, SD) on the theta scale into prior weights on the
    quadrature grid.

    A logit-normal is used: theta is confined to [0,1], so laying a normal
    directly on it leaks mass past the boundaries.

    Parameters
    ----------
    mean, sd : (n,) array_like    predictive mean and SD on the theta scale
    nodes : (K,) ndarray          nodes from `core.make_grid`
    quad_w : (K,) ndarray         Gauss-Legendre weights, without the prior

    Returns
    -------
    (n, K) ndarray — each row sums to 1
    """
    m = np.clip(np.asarray(mean, dtype=float), 2e-3, 1 - 2e-3)
    s = np.asarray(sd, dtype=float)
    lg_nodes = np.log(nodes / (1.0 - nodes))
    s_lg = np.clip(s / (m * (1.0 - m)), 0.05, 5.0)
    m_lg = np.log(m / (1.0 - m))
    z = (lg_nodes[None, :] - m_lg[:, None]) / s_lg[:, None]
    dens = norm.pdf(z) / (s_lg[:, None] * nodes * (1.0 - nodes))
    W = dens * quad_w
    return W / W.sum(axis=1, keepdims=True)


def sequential_score(responses, items, alpha, beta, gamma=None,
                     times=None, memory="hctm", n_nodes=41,
                     prior=(2.0, 2.0), cohort=None):
    """Score responses occasion by occasion, including the prediction step.

    Item parameters are **assumed to be anchored**. Recalibrating at every
    occasion lets the scale move along with the learners, which makes growth
    indistinguishable from drift.

    Parameters
    ----------
    responses : list of (n, K_t) arrays    0/1 responses per occasion
    items : list of (K_t,) integer arrays  bank indices administered per occasion
    alpha, beta : (J,) array_like          anchored item parameters (whole bank)
    gamma : (J,) array_like, optional      for the three-parameter model
    times : (T,) array_like, optional      occasion times. Defaults to 0, 1, ..., T-1
    memory : {"hctm", "none"}              "none" uses the population prior at
                                           every occasion (the memoryless baseline)

    Returns
    -------
    dict
        ``theta``  (T, n) posterior means by occasion
        ``sd``     (T, n) posterior SDs by occasion
        ``k``      (T,) parameters released by the growth model at each
                   occasion (0 when memory="none")
    """
    if memory not in ("hctm", "none"):
        raise ValueError("memory must be 'hctm' or 'none'")

    T = len(responses)
    if len(items) != T:
        raise ValueError("responses and items have different lengths")
    ts = np.arange(T, dtype=float) if times is None \
        else np.asarray(times, dtype=float)

    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gam = None if gamma is None else np.asarray(gamma, dtype=float)

    nodes, w_pop = core.make_grid(n_nodes, *prior)
    quad_w = 0.5 * np.polynomial.legendre.leggauss(n_nodes)[1]

    n = np.asarray(responses[0]).shape[0]
    cur = np.tile(w_pop, (n, 1))
    TH, SD, KS = [], [], []

    for t in range(T):
        idx = np.asarray(items[t], dtype=int)
        Yt = np.asarray(responses[t], dtype=float)
        g_t = None if gam is None else gam[idx]
        P = np.clip(core.p2_naive(nodes[:, None], alpha[idx], beta[idx])
                    if g_t is None else
                    core.p3_naive(nodes[:, None], alpha[idx], beta[idx], g_t),
                    1e-12, 1 - 1e-12)
        ll = Yt @ np.log(P).T + (1.0 - Yt) @ np.log(1.0 - P).T
        ll -= ll.max(axis=1, keepdims=True)
        post = np.exp(ll) * cur
        post /= post.sum(axis=1, keepdims=True)

        m = post @ nodes
        v = post @ (nodes ** 2) - m ** 2
        s = np.sqrt(np.maximum(v, 1e-12))
        TH.append(m)
        SD.append(s)

        if t == T - 1:
            KS.append(0)
            break

        if memory == "none":
            cur = np.tile(w_pop, (n, 1))
            KS.append(0)
        else:
            mn, sn = predict(ts[:t + 1], np.array(TH).T, np.array(SD).T,
                             float(ts[t + 1]), cohort)
            cur = to_prior(mn, sn, nodes, quad_w)
            KS.append(_schedule(t + 1))

    return {"theta": np.array(TH), "sd": np.array(SD), "k": np.array(KS)}
