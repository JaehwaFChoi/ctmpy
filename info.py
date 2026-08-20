"""Information functions — item information, TIF and SEM for the ALF family.

These are the quantities the source article names as future work (Discussion:
TCC, IIF, TIF, SEM).

Derivation
  log P2 = log(expm1(a*th)) - softplus(a*(th-b)) + const
  The normalizing constant does not depend on theta and so drops out under
  differentiation. Consequently L = 1 (CTM) and L = inf (HCTM) **share the
  same derivative formula**.

      dlogP2/dth = a/(1 - exp(-a*th)) - a*sigmoid(a*(th-b))
      P2'        = P2 * a * [1/(1-exp(-a*th)) - sigmoid(a*(th-b))]
      P3'        = (1-gamma) * P2'

  Fisher information for a dichotomous response
      I(th) = [P'(th)]^2 / (P(th) * (1-P(th)))

[Reflecting the Section 2 revision] Written in the normalized form
P = [s(th) - s(-ab)] / D, the derivative is simply

      P' = (a/D) * s * (1-s),   s = sigmoid(a(th-b))

which has **no singularity at theta = 0**. This module is based on the
expanded form and therefore branches through _dp_at_zero, but the two forms
agree numerically to eight digits.

Boundary behaviour (the central result of this module)
  The numerator s(1-s) is at most 1/4, so P' is bounded. **The divergence is
  therefore attributable to the denominator P(1-P) alone.**
  th -> L : I ~ P'(L) / (L - th) — the rate is exactly 1 and the residue is P'(L)
  th -> 0 : the 2P form diverges as I ~ P'(0)/th; the 3P form stays finite
            because P3(0) = gamma > 0
  So gamma removes the divergence at the lower boundary but not at the upper
  one, since P(L) = 1 is fixed. Because the rate is exactly (L-th)^{-1},
  integrating the information function diverges logarithmically — the total
  information is not finite. Report uncertainty as the posterior SD, and read
  the information function on [0.02, 0.98] for item analysis.

[Result of the CAT experiment] The boundary divergence does not bite in
practice when the interim estimate is EAP or MAP: truncation never once
triggered. Combined with MLE it is actively harmful. See study_cat.py.
"""

from __future__ import annotations

import numpy as np

from . import core as ctm
from . import hctm

__all__ = ["dp2_ctm", "dp3_ctm", "item_info", "tif", "sem", "info_table"]


def _dlogp_core(theta, alpha, beta):
    """The shared kernel of the derivative. L = 1 and L = inf both use it."""
    theta = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t1 = alpha / (-np.expm1(-alpha * theta))          # a/(1-exp(-a*th))
    t2 = alpha / (1.0 + np.exp(-alpha * (theta - beta)))  # a*sigmoid
    return t1 - t2


def dp2_ctm(theta, alpha, beta, family="ctm"):
    """dP/dtheta for the two-parameter link.

    family='ctm' selects L = 1; family='hctm' selects L = inf.
    """
    theta = np.asarray(theta, dtype=float)
    P = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
        else hctm.p2_hctm(theta, alpha, beta)
    d = P * _dlogp_core(theta, alpha, beta)
    return np.where(theta <= 0.0, _dp_at_zero(alpha, beta, family), d)


def _dp_at_zero(alpha, beta, family):
    """The limit at theta = 0. Since P2 ~ C*B(0)*a*th, P2'(0) = C*B(0)*a."""
    B0 = 1.0 / (1.0 + np.exp(-alpha * beta))
    if family == "ctm":
        C = (1.0 + np.exp(alpha * (1.0 - beta))) / (-np.expm1(alpha))
        return -alpha * C * B0          # C < 0, so the sign is corrected here
    return alpha * np.exp(-alpha * beta) * B0


def dp3_ctm(theta, alpha, beta, gamma, family="ctm"):
    """dP/dtheta for the three-parameter link = (1-gamma) * dP2/dtheta."""
    return (1.0 - np.asarray(gamma, dtype=float)) * dp2_ctm(theta, alpha, beta, family)


def item_info(theta, alpha, beta, gamma=None, family="ctm", eps=1e-12):
    """Item information function I(theta) = (P')^2 / (P(1-P))."""
    if gamma is None:
        P = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
            else hctm.p2_hctm(theta, alpha, beta)
        d = dp2_ctm(theta, alpha, beta, family)
    else:
        P2 = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
            else hctm.p2_hctm(theta, alpha, beta)
        P = gamma + (1.0 - gamma) * P2
        d = dp3_ctm(theta, alpha, beta, gamma, family)
    den = np.clip(P * (1.0 - P), eps, None)
    return d ** 2 / den


def tif(theta, alpha, beta, gamma=None, family="ctm"):
    """Test information function — the sum of item information under local
    independence."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    g = None if gamma is None else np.atleast_1d(gamma)
    I = item_info(theta[:, None], np.atleast_1d(alpha), np.atleast_1d(beta),
                  None if g is None else g, family)
    return I.sum(axis=1)


def sem(theta, alpha, beta, gamma=None, family="ctm"):
    """Standard error from Fisher information.

    This collapses to zero at the boundaries and is therefore not used for
    reporting; see the design decision posterior_sd_not_fisher.
    """
    return 1.0 / np.sqrt(np.maximum(tif(theta, alpha, beta, gamma, family), 1e-300))


def info_table(alpha, beta, gamma=None, family="ctm", lo=0.02, hi=0.98, n=25):
    """Information table for item analysis, with the boundaries trimmed off."""
    th = np.linspace(lo, hi, n)
    return th, tif(th, alpha, beta, gamma, family)


if __name__ == "__main__":
    print("[1] Derivative check against numerical differentiation (CTM, L=1)")
    h = 1e-6
    for a, b in [(1.0, 0.3), (5.0, 0.5), (10.0, 0.7), (25.0, 0.5)]:
        errs = []
        for t0 in [0.05, 0.2, 0.5, 0.8, 0.95]:
            num = (ctm.p2_naive(t0 + h, a, b) - ctm.p2_naive(t0 - h, a, b)) / (2 * h)
            ana = float(dp2_ctm(t0, a, b))
            errs.append(abs(ana - num) / max(abs(num), 1e-12))
        print(f"  a={a:5.1f} b={b:.1f}  max relative diff {max(errs):.2e}")

    print("\n[2] Derivative check — HCTM (L=inf)")
    for a, b in [(1.0, 1.0), (3.0, 2.0), (0.5, 0.5)]:
        errs = []
        for t0 in [0.1, 0.5, 1.5, 3.0]:
            num = (hctm.p2_hctm(t0 + h, a, b) - hctm.p2_hctm(t0 - h, a, b)) / (2 * h)
            ana = float(dp2_ctm(t0, a, b, family="hctm"))
            errs.append(abs(ana - num) / max(abs(num), 1e-12))
        print(f"  a={a:5.1f} b={b:.1f}  max relative diff {max(errs):.2e}")

    print("\n[3] The limit at theta = 0 (CTM)")
    for a, b in [(5.0, 0.3), (10.0, 0.5), (20.0, 0.8)]:
        lim = float(_dp_at_zero(a, b, "ctm"))
        num = (ctm.p2_naive(1e-7, a, b) - 0.0) / 1e-7
        print(f"  a={a:5.1f} b={b:.1f}  analytic {lim:.6f}  numeric {num:.6f}  "
              f"relative {abs(lim-num)/max(abs(num),1e-12):.2e}")

    print("\n[4] Divergence of information at the boundaries — 2P vs 3P")
    a, b = 10.0, 0.5
    print(f"{'theta':>8}{'2P I':>14}{'3P I (g=0.2)':>16}")
    for t0 in [1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999]:
        i2 = float(item_info(t0, a, b))
        i3 = float(item_info(t0, a, b, 0.2))
        print(f"{t0:>8.4f}{i2:>14.2f}{i3:>16.2f}")
    print("  -> 2P diverges at both boundaries, 3P only at the upper one")

    print("\n[5] Rate of divergence — estimating k in I ~ (1-theta)^(-k)")
    print("    (the finite-difference regression is biased; the analytic "
          "value is exactly k=1 with residue P'(L))")
    for a in [5.0, 10.0, 20.0]:
        ts = 1 - np.array([1e-2, 1e-3, 1e-4, 1e-5])
        Is = np.array([float(item_info(t, a, 0.5)) for t in ts])
        k = np.polyfit(np.log(1 - ts), np.log(Is), 1)[0]
        print(f"  a={a:5.1f}  k = {-k:.3f}")
