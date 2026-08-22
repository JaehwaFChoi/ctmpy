"""cogtraitmodel — estimation, scoring and information for the CTM family.

This package implements the **Anchored Logistic Family (ALF)**, which places a
cognitive trait theta on the bounded continuum [0, L]. The mastery anchor L
indexes the family:

    L = 1     Cognitive Trait Model (Choi, 2022) — theta reads as "% of mastery"
    L -> inf  Half-truncated CTM (HCTM) — an absolute zero but no ceiling

Design
------
* Estimation is **Bayes modal EM on a Gauss-Legendre grid**, not MCMC.
  Forty-one nodes reach the double-precision floor, and the result agrees with
  a converged random-walk sampler to a theta correlation of 0.9999 while
  running 35 times faster (73 times faster than NUTS).
* The default scoring output is the **MAP point estimate with the EAP
  posterior SD**. The Fisher-information SEM collapses to zero at the
  boundaries and is not used for reporting.
* MLE is a diagnostic only, and its boundary solutions (theta_hat = 0 or L)
  are never reported as trait levels — being finite and readable as "100% of
  mastery", they are more dangerous than the divergence to infinity in IRT.

Quick start
-----------
    >>> import numpy as np, cogtraitmodel as ctm
    >>> theta = ctm.gen_theta(500, rng=np.random.default_rng(0))
    >>> alpha = np.full(20, 8.0); beta = np.linspace(0.1, 0.9, 20)
    >>> Y = ctm.gen_responses(theta, alpha, beta,
    ...                         rng=np.random.default_rng(1))
    >>> fit = ctm.bayes_modal_em(Y)          # calibrate item parameters
    >>> out = ctm.score(Y, fit["alpha"], fit["beta"])
    >>> out["theta"], out["sd"]        # MAP estimate + posterior SD

An arbitrary finite anchor L is handled by `fit_L` / `score_L`; these are not
a separate estimator, since finite L reduces to L = 1 by rescaling. Growth and
sequential updating on the time axis live in `cogtraitmodel.growth`, which is
where the L -> inf member is used.

Correctness standard
--------------------
`p1_naive` / `p2_naive` / `p3_naive` are literal transcriptions of Eqs.
(12)-(14) of Choi (2022) and are kept permanently. Every optimised
implementation is pinned by tests asserting agreement with them to within
1e-12 over alpha in [0.1, 100].

Reference
---------
Choi, J. (2022). Cognitive Trait Model: Measurement model for mastery level
and progression of learning. *Mathematics*, 10(15), 2651.
doi:10.3390/math10152651
"""

from __future__ import annotations

import numpy as np

from . import datasets, growth, hctm, info, irt, scale
from .core import (
    bayes_modal_em,
    eap,
    eap_batch,
    gen_responses,
    gen_theta,
    make_grid,
    map_theta,
    mle_theta,
    p1_naive,
    p2_naive,
    p3_naive,
    posterior,
)
from .hctm import dp_hctm, p2_ctm_L, p2_hctm, p3_hctm
from .info import info_table, item_info, sem, tif
from .scale import fit_L, score_L

__version__ = "0.2.2"

__all__ = [
    # link functions (L = 1; the literal transcriptions are the oracle)
    "p1_naive", "p2_naive", "p3_naive",
    # link functions (general L, and L -> inf)
    "p2_ctm_L", "p2_hctm", "p3_hctm", "dp_hctm",
    # quadrature and posterior
    "make_grid", "posterior", "posterior_sd",
    # scoring
    "eap", "eap_batch", "map_theta", "mle_theta", "score",
    # calibration
    "bayes_modal_em",
    # arbitrary finite anchor L — rescaling wrappers
    "fit_L", "score_L",
    # information functions
    "item_info", "tif", "sem", "info_table",
    # simulation
    "gen_theta", "gen_responses",
    # submodules
    "hctm", "info", "irt", "datasets", "growth", "scale",
    "__version__",
]


def posterior_sd(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """Posterior standard deviation on the quadrature grid.

    This is the recommended measure of uncertainty. The Fisher-information SEM
    (`sem`) collapses to zero as theta approaches a boundary, which is an
    artefact of holding P(L) = 1 fixed rather than a statement about precision.
    The posterior SD has no such pathology.

    Returns
    -------
    ndarray, shape (n,)
    """
    _, sd = eap(Y, alpha, beta, gamma, n_nodes, prior)
    return sd


def score(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """The recommended scoring path — MAP point estimate with posterior SD.

    Both quantities share the same quadrature grid, so obtaining the pair
    costs essentially the same as obtaining either one alone.

    Returns
    -------
    dict
        ``theta``    MAP point estimate (the value to report)
        ``sd``       posterior standard deviation (the uncertainty to report)
        ``eap``      EAP point estimate (for reference; shrinks more)
        ``at_bound`` examinees whose mode was found at the edge of the grid
                     (bool). Under MAP these are normally all False; any True
                     warrants investigation.
    """
    m, sd = eap(Y, alpha, beta, gamma, n_nodes, prior)
    theta, at_bound = map_theta(Y, alpha, beta, gamma, prior)
    return {"theta": theta, "sd": sd, "eap": m, "at_bound": at_bound}
