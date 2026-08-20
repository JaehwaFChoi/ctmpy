"""cogtraitmodel.scale — calibration and scoring at an arbitrary mastery anchor L.

Finite L needs no separate estimator
------------------------------------
In the normalized form the sigmoid argument rearranges as
alpha*(theta - beta) = (alpha*L)*(theta/L - beta/L), and the denominator
undergoes the same transformation, so the following identity holds:

    P_L(theta; alpha, beta) == P_1(theta/L; alpha*L, beta/L)

Verified numerically: with (alpha, beta, L) set to (1, 0.5, 2), (5, 0.4, 3),
(2, 1.5, 5) and (0.8, 2, 4), the maximum difference over 300 points of
theta in (0, 0.99L] ranges from 2.2e-16 to 2.8e-15.

Estimation on the L scale is therefore exactly the same operation as
compressing the data by theta/L, running the L = 1 estimator, and mapping the
parameters back. This module hides that change of variable.

One consequence deserves to be stated: the same response data fit equally well
at every finite L, so **L is a definitional choice rather than a parameter the
data identify**. Where the unit is placed is a rescaling; only the origin is
structural.

L -> inf is the exception. Since theta/L -> 0 the reduction fails, and in that
case HCTM is used not as a measurement model but as a growth model on the time
axis (see `cogtraitmodel.growth`).
"""

from __future__ import annotations

import numpy as np

from . import core

__all__ = ["fit_L", "score_L"]


def _check(L):
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError(
            f"L must be a finite positive number (received {L}). L -> inf is "
            "a growth model rather than a measurement model; use "
            "cogtraitmodel.growth instead."
        )
    return L


def fit_L(Y, L, **kwargs):
    """Calibrate item parameters on a scale whose mastery anchor is L.

    Internally this runs the L = 1 estimator and maps the parameters back to
    the L scale.

    Parameters
    ----------
    Y : (n, J) array of 0/1 responses
    L : float                 mastery anchor (finite and positive)
    **kwargs                  passed through to `core.bayes_modal_em`

    Returns
    -------
    dict — the same structure as `bayes_modal_em`, with alpha and beta on the
           L scale and an additional ``L`` key.
    """
    L = _check(L)
    fit = core.bayes_modal_em(Y, **kwargs)
    out = dict(fit)
    out["alpha"] = np.asarray(fit["alpha"], dtype=float) / L
    out["beta"] = np.asarray(fit["beta"], dtype=float) * L
    out["L"] = L
    return out


def score_L(Y, alpha, beta, L, gamma=None, **kwargs):
    """Score with item parameters on the L scale. theta is returned in [0, L].

    Returns
    -------
    dict — the same keys as `cogtraitmodel.score`, with ``theta``, ``sd`` and
           ``eap`` on the L scale.
    """
    from . import score

    L = _check(L)
    a1 = np.asarray(alpha, dtype=float) * L
    b1 = np.asarray(beta, dtype=float) / L
    out = score(Y, a1, b1, gamma, **kwargs)
    return {"theta": out["theta"] * L, "sd": out["sd"] * L,
            "eap": out["eap"] * L, "at_bound": out["at_bound"], "L": L}
