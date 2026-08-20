"""Half-truncated CTM (HCTM) — support on [0, inf).

The published Cognitive Trait Model imposes two boundary conditions, P(0)=0
and P(1)=1, which together fix two constants. Truncating only at zero leaves a
single condition, P(0)=0, and lets the upper end keep the asymptote of 1 that
the logistic function already has. One constant then suffices.

    P(theta) = [sigma(a(theta-b)) - sigma(-a b)] / sigma(a b)

which simplifies to

    P(theta) = expm1(a*theta) / [exp(a*b) * (1 + exp(a*(theta-b)))]

This shares its kernel with Eq. (13) of the source article and differs only in
the normalizing constant:

    general form with mastery anchor L:
        expm1(a th)/(1+exp(a(th-b))) * (1+exp(a(L-b)))/expm1(a L)
      L = 1    -> the published CTM
      L -> inf -> exp(-a b)  = this model

The published CTM is therefore the case L = 1 of this family.

Properties:
  P(0) = 0,  P(inf) = 1,  monotone increasing in theta
  P(b) = (1 - exp(-a b)) / 2   -> approaches 0.5 as a*b grows
  theta has an absolute zero (the ignorance level) but no ceiling, which is
  what licenses a ratio-scale reading.
"""

from __future__ import annotations

import numpy as np

__all__ = ["p2_hctm", "p3_hctm", "p2_ctm_L", "dp_hctm"]


def _log_expm1(u):
    """Stable log(exp(u)-1). For large u, u + log1p(-exp(-u))."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    big = u > 30.0
    out[big] = u[big] + np.log1p(-np.exp(-u[big]))
    sm = ~big
    out[sm] = np.log(np.expm1(np.maximum(u[sm], 1e-300)))
    return out


def _softplus(u):
    return np.logaddexp(0.0, u)


def p2_hctm(theta, alpha, beta):
    """Two-parameter half-truncated CTM. theta in [0, inf).

        P = expm1(a*th) / [exp(a*b) * (1 + exp(a*(th-b)))]

    Evaluated in log space to avoid overflow:
        log P = log_expm1(a*th) - a*b - softplus(a*(th-b))
    """
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    u = alpha * theta
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = _log_expm1(u) - alpha * beta - _softplus(alpha * (theta - beta))
    P = np.exp(lp)
    return np.where(theta <= 0.0, 0.0, np.clip(P, 0.0, 1.0))


def p3_hctm(theta, alpha, beta, gamma):
    """Three-parameter half-truncated CTM. gamma is P(correct) at theta = 0."""
    gamma = np.asarray(gamma, dtype=float)
    return gamma + (1.0 - gamma) * p2_hctm(theta, alpha, beta)


def p2_ctm_L(theta, alpha, beta, L):
    """General form with the mastery anchor L stated explicitly.

    L = 1 gives the published CTM; L -> inf gives HCTM.

    Caution: this expression contains exp(a*L) and therefore overflows once
    a*L > 709. It is intended for derivation and exposition; for computation
    use core.p2_naive when L = 1 and p2_hctm when L is infinite.
    """
    theta = np.asarray(theta, dtype=float)
    core_num = np.expm1(alpha * theta)
    core_den = 1.0 + np.exp(alpha * (theta - beta))
    const = (1.0 + np.exp(alpha * (L - beta))) / np.expm1(alpha * L)
    return core_num / core_den * const


def dp_hctm(theta, alpha, beta):
    """dP/dtheta — the analytic derivative, for information and Newton steps.

    Since log P = log(expm1(a th)) - a b - softplus(a(th-b)),
        dlogP/dth = a*exp(a th)/expm1(a th) - a*sigmoid(a(th-b))

    Note: the normalized form P' = (a/D) s(1-s) derived in Section 2 has no
    singularity at theta = 0. This function is based on the expanded form and
    so branches on theta <= 0.
    """
    theta = np.asarray(theta, dtype=float)
    u = alpha * theta
    P = p2_hctm(theta, alpha, beta)
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = alpha / (-np.expm1(-u))          # a*exp(u)/expm1(u) = a/(1-exp(-u))
    term2 = alpha / (1.0 + np.exp(-alpha * (theta - beta)))
    return np.where(theta <= 0.0, 0.0, P * (term1 - term2))


if __name__ == "__main__":
    from cogtraitmodel import core as ctm

    print("[1] Identities — P(0)=0, P(inf)=1, monotone increasing")
    A = [0.5, 1.0, 3.0, 10.0]
    B = [0.2, 0.5, 1.0, 3.0]
    th = np.linspace(0, 60, 20001)
    ok0 = ok1 = mono = True
    for a in A:
        for b in B:
            p = p2_hctm(th, a, b)
            ok0 &= abs(p[0]) < 1e-15
            ok1 &= abs(p[-1] - 1.0) < 1e-9
            mono &= np.diff(p).min() >= -1e-13
    print(f"  P(0)=0 {ok0}   P(large theta)->1 {ok1}   monotone {mono}")

    print("\n[2] Does P(beta) = (1 - exp(-a*b))/2 hold?")
    for a, b in [(1.0, 0.5), (3.0, 1.0), (10.0, 2.0), (0.5, 0.2)]:
        lhs = float(p2_hctm(b, a, b))
        rhs = (1 - np.exp(-a * b)) / 2
        print(f"  a={a:5.1f} b={b:4.1f}   P(b)={lhs:.10f}   formula={rhs:.10f}   "
              f"diff={abs(lhs-rhs):.2e}")

    print("\n[3] Limits of the general form — L=1 is CTM, L->inf is HCTM")
    a, b = 5.0, 0.4
    t = np.array([0.1, 0.3, 0.5, 0.8])
    d1 = np.abs(p2_ctm_L(t, a, b, 1.0) - ctm.p2_naive(t, a, b)).max()
    print(f"  L=1 vs published p2_naive : max diff {d1:.3e}")
    for L in [5.0, 20.0, 50.0, 100.0]:
        d = np.abs(p2_ctm_L(t, a, b, L) - p2_hctm(t, a, b)).max()
        print(f"  L={L:6.0f} vs HCTM         : max diff {d:.3e}")

    print("\n[4] Derivative check — against numerical differentiation")
    h = 1e-6
    for a, b in [(1.0, 0.5), (3.0, 1.0), (10.0, 2.0)]:
        for t0 in [0.2, 1.0, 3.0]:
            num = (p2_hctm(t0 + h, a, b) - p2_hctm(t0 - h, a, b)) / (2 * h)
            ana = float(dp_hctm(t0, a, b))
            print(f"  a={a:5.1f} b={b:4.1f} th={t0:4.1f}  analytic {ana:.8f}  "
                  f"numeric {float(num):.8f}  rel {abs(ana-num)/max(abs(num),1e-12):.2e}")

    print("\n[5] Numerical stability — no overflow at large alpha*theta")
    for a, t0 in [(10.0, 100.0), (50.0, 200.0), (100.0, 500.0)]:
        v = float(p2_hctm(t0, a, 1.0))
        print(f"  a={a:6.1f} theta={t0:6.1f}  P={v:.12f}  finite={np.isfinite(v)}")
