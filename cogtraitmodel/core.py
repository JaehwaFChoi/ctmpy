"""
Cognitive Trait Model (CTM) — v0.4
==================================================
Choi, J. (2022). Cognitive Trait Model: Measurement Model for Mastery Level
and Progression of Learning. Mathematics, 10, 2651.

Sections:
  link      direct implementation of Eqs. (12)/(13)/(14) (naive, the oracle)
  quad      Gauss-Legendre grid on [0,1] weighted by a Beta prior
  score     EAP, posterior SD, MAP, MLE
  calib     Bayes modal EM (MMLE with priors)
  simulate  response generation

The link section is a reference implementation that transcribes the published
equations **literally**. A numerically stabilised version (rewritten with
expm1/softplus) is added separately in v0.5, at which point this naive
implementation serves as the oracle for correctness.

Within the range covered by the article (alpha in [1, 100], theta in [0, 1])
this implementation is exact. It breaks in exactly two places, which are left
as they are deliberately:
  - alpha -> 0  : 1 - exp(0) = 0 gives 0/0 -> NaN
                  (the actual limits are P2 -> theta and
                   P3 -> gamma + (1-gamma)*theta. The expression
                   'theta + gamma' in the article's text is incorrect.)
  - alpha > 709 : exp overflows
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

__all__ = [
    "p2_naive", "p3_naive", "p1_naive",
    "make_grid", "posterior", "eap", "eap_batch", "map_theta", "mle_theta",
    "bayes_modal_em",
    "gen_theta", "gen_responses",
]


def p2_naive(theta, alpha, beta):
    """Two-parameter CTM — Eq. (13) of the source article.

    P = (1 - exp(a*th)) / (1 + exp(a*(th - b))) * (1 + exp(a*(1 - b))) / (1 - exp(a))

    Both (1 - exp(a*th)) and (1 - exp(a)) are negative for a > 0, so their
    ratio is positive.
    """
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    num = 1.0 - np.exp(alpha * theta)
    den = 1.0 + np.exp(alpha * (theta - beta))
    scale = (1.0 + np.exp(alpha * (1.0 - beta))) / (1.0 - np.exp(alpha))
    return num / den * scale


def p3_naive(theta, alpha, beta, gamma):
    """Three-parameter CTM — Eq. (12) of the source article.

    P = g + (1 - g) * P2(theta; a, b)

    gamma is not the lower asymptote of the conventional 3PL. It is the
    probability of a correct response for a person located **at the lower
    boundary**, Pr(y = 1 | theta = 0). See Table 1 of the article.
    """
    gamma = np.asarray(gamma, dtype=float)
    return gamma + (1.0 - gamma) * p2_naive(theta, alpha, beta)


def p1_naive(theta, beta, alpha=10.0):
    """One-parameter CTM — Eq. (14): the 2P CTM with alpha held constant."""
    return p2_naive(theta, alpha, beta)


# =====================================================================
# quad — Gauss-Legendre quadrature grid on [0,1]
# =====================================================================

def make_grid(n_nodes=61, a=2.0, b=2.0):
    """Build Gauss-Legendre nodes on [0,1] with Beta(a,b) prior weights.

    Conventional IRT has an unbounded theta, so it uses Gauss-Hermite and must
    fix a truncation range (typically +-4) by fiat. Here theta is bounded on
    [0,1], so Gauss-Legendre applies directly and no truncation error exists.

    Result R3: forty-one nodes converge to the double-precision floor
    (4e-15 against a 61-node grid).
    """
    x, gw = np.polynomial.legendre.leggauss(n_nodes)
    nodes = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
    w = 0.5 * gw * beta_dist.pdf(nodes, a, b)
    return nodes, w / w.sum()


# =====================================================================
# score — EAP / posterior SD / MAP / MLE
# =====================================================================

def _loglik(Y, nodes, alpha, beta, gamma=None):
    """Build the (n, K) log-likelihood matrix over persons by quadrature points.

    Y : (n, J) matrix of 0/1 responses. Missing entries left as np.nan are
        excluded item by item.
    """
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        P = p2_naive(nodes[:, None], alpha, beta)  # (K, J)
    else:
        P = p3_naive(nodes[:, None], alpha, beta, gamma)
    P = np.clip(P, 1e-12, 1 - 1e-12)  # guard against log(0)

    obs = ~np.isnan(Y)
    Y0 = np.where(obs, Y, 0.0)
    return Y0 @ np.log(P).T + (obs & (Y0 == 0)) @ np.log(1 - P).T


def posterior(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """Return the nodes and the normalized (n, K) posterior on the grid."""
    nodes, w = make_grid(n_nodes, *prior)
    ll = _loglik(Y, nodes, alpha, beta, gamma)
    ll -= ll.max(axis=1, keepdims=True)  # guard against overflow
    post = np.exp(ll) * w
    return nodes, post / post.sum(axis=1, keepdims=True)


def eap(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """Return the EAP estimate together with the posterior SD.

    Why the posterior SD: in CTM, P is pinned to 1 at theta = 1, so the
    denominator of the Fisher information I = (P')^2/(P(1-P)) goes to zero and
    the SEM collapses. The Beta prior has zero density at the boundaries, so
    the width of the posterior does not collapse.
    """
    nodes, post = posterior(Y, alpha, beta, gamma, n_nodes, prior)
    m = post @ nodes
    v = post @ (nodes ** 2) - m ** 2
    return m, np.sqrt(np.maximum(v, 0.0))


def eap_batch(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """Alias for eap."""
    return eap(Y, alpha, beta, gamma, n_nodes, prior)


def _mode_on_grid(Y, alpha, beta, gamma, prior, n_grid, eps):
    """Locate the mode on a dense grid, refining between nodes by a parabolic fit.

    prior=None gives MLE; prior=(a,b) gives MAP.
    No iterative optimizer is used — a grid is the safer choice when solutions
    can sit against a boundary.

    Returns
    -------
    theta_hat : (n,) ndarray
    at_bound  : (n,) bool  the maximum was attained at an end of the grid
    """
    grid = np.linspace(eps, 1.0 - eps, n_grid)
    obj = _loglik(Y, grid, alpha, beta, gamma)  # (n, G)
    if prior is not None:
        obj = obj + np.log(beta_dist.pdf(grid, *prior))

    idx = obj.argmax(axis=1)
    at_bound = (idx == 0) | (idx == n_grid - 1)

    hat = grid[idx].copy()
    inner = ~at_bound
    if inner.any():
        i = idx[inner]
        r = np.arange(len(idx))[inner]
        y1, y2, y3 = obj[r, i - 1], obj[r, i], obj[r, i + 1]
        denom = y1 - 2 * y2 + y3
        shift = np.where(np.abs(denom) > 1e-300, 0.5 * (y1 - y3) / denom, 0.0)
        hat[inner] = grid[i] + np.clip(shift, -1.0, 1.0) * (grid[1] - grid[0])
    return np.clip(hat, 0.0, 1.0), at_bound


def map_theta(Y, alpha, beta, gamma=None, prior=(2.0, 2.0), n_grid=4001, eps=1e-6):
    """MAP: maximize the log-likelihood plus the log Beta prior.

    Beta(2,2) has zero density at the boundaries, so the MAP stays strictly
    inside (0,1). Result R2: recommended as the default point estimate — it
    shrinks half as much as EAP and produces no boundary solutions.
    """
    return _mode_on_grid(Y, alpha, beta, gamma, prior, n_grid, eps)


def mle_theta(Y, alpha, beta, gamma=None, n_grid=4001, eps=1e-6):
    """MLE: maximize the log-likelihood alone. Diagnostic use only.

    A perfect score lands on theta = 1, and under the 2P model a zero score
    lands on theta = 0. In CTM these boundary values are not infinities but
    finite, interpretable numbers — 'the mastery level' and 'the ignorance
    level' — which makes them more dangerous, not less. They are always
    flagged through at_bound. At J = 5 this affects roughly 14.5% of examinees.
    """
    return _mode_on_grid(Y, alpha, beta, gamma, None, n_grid, eps)


# =====================================================================
# calib — Bayes modal EM (MMLE with priors)
# =====================================================================

def _logit(p):
    return np.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pack(alpha, beta, gamma):
    """Map constrained parameters to an unconstrained space.

    alpha > 0, and beta, gamma in (0,1).
    """
    z = [np.log(alpha), _logit(beta)]
    if gamma is not None:
        z.append(_logit(gamma))
    return np.asarray(z, dtype=float)


def _unpack(z, three_p):
    alpha = np.exp(z[0])
    beta = _expit(z[1])
    gamma = _expit(z[2]) if three_p else None
    return alpha, beta, gamma


def _item_neg_obj(z, nodes, Nk, rk, three_p, pa, pb, pg):
    """Negative (expected log-likelihood + log prior) for one item.

    This is the M-step objective.

    Nk : (K,) expected number of persons at each quadrature point
    rk : (K,) expected number of correct responses at each quadrature point
    """
    alpha, beta, gamma = _unpack(z, three_p)
    if not np.isfinite(alpha) or alpha <= 0 or alpha > 1e4:
        return 1e10

    P = p2_naive(nodes, alpha, beta) if not three_p \
        else p3_naive(nodes, alpha, beta, gamma)
    P = np.clip(P, 1e-12, 1 - 1e-12)

    ll = np.sum(rk * np.log(P) + (Nk - rk) * np.log(1 - P))

    # priors: alpha ~ LogNormal(mu, sd), beta and gamma ~ Beta
    ll += -0.5 * ((np.log(alpha) - pa[0]) / pa[1]) ** 2 - np.log(alpha)
    ll += (pb[0] - 1) * np.log(beta) + (pb[1] - 1) * np.log(1 - beta)
    if three_p:
        ll += (pg[0] - 1) * np.log(gamma) + (pg[1] - 1) * np.log(1 - gamma)
    return -ll if np.isfinite(ll) else 1e10


def bayes_modal_em(Y, three_p=False, n_nodes=61, prior_theta=(2.0, 2.0),
                   prior_alpha=(np.log(10.0), 1.0), prior_beta=(2.0, 2.0),
                   prior_gamma=(7.0, 25.0), max_iter=300, tol=1e-6,
                   init=None, verbose=False):
    """Estimate item parameters by MMLE with priors (Bayes modal).

    The E-step is the posterior on the [0,1] quadrature grid; the M-step is an
    unconstrained optimization per item. Items are independent, so the problem
    splits into J separate two- or three-dimensional problems.

    prior_alpha is (mu, sd), specifying a normal distribution for log(alpha).
    Note: dlnorm(7.5, 1) in the WinBUGS code of the source article is in
    precision notation, so it means log(alpha) ~ N(7.5, 1) — a very strong
    prior with a median alpha of about 1808. That is far from the generating
    values of 5 to 10, and is examined separately under R6.

    Returns
    -------
    dict : alpha, beta, gamma, n_iter, converged, loglik
    """
    from scipy.optimize import minimize

    Y = np.asarray(Y, dtype=float)
    n, J = Y.shape
    nodes, w = make_grid(n_nodes, *prior_theta)

    if init is None:
        alpha = np.full(J, 10.0)
        beta = np.clip(1.0 - Y.mean(axis=0), 0.05, 0.95)  # high p correct = easy item
        gamma = np.full(J, 0.2) if three_p else None
    else:
        alpha, beta, gamma = init

    hist = []
    for it in range(max_iter):
        # ---- E-step: posterior and expected counts ----
        ll = _loglik(Y, nodes, alpha, beta, gamma)
        mx = ll.max(axis=1, keepdims=True)
        post = np.exp(ll - mx) * w
        norm = post.sum(axis=1, keepdims=True)
        post /= norm
        hist.append(float((np.log(norm[:, 0]) + mx[:, 0]).sum()))

        Nk = post.sum(axis=0)  # (K,)
        rk = post.T @ Y        # (K, J)

        # ---- M-step: independent optimization per item ----
        a_new, b_new = alpha.copy(), beta.copy()
        g_new = gamma.copy() if three_p else None
        for j in range(J):
            z0 = _pack(alpha[j], beta[j], gamma[j] if three_p else None)
            res = minimize(_item_neg_obj, z0, method="L-BFGS-B",
                           args=(nodes, Nk, rk[:, j], three_p,
                                 prior_alpha, prior_beta, prior_gamma))
            aj, bj, gj = _unpack(res.x, three_p)
            a_new[j], b_new[j] = aj, bj
            if three_p:
                g_new[j] = gj

        delta = max(np.abs(np.log(a_new) - np.log(alpha)).max(),
                    np.abs(b_new - beta).max(),
                    np.abs(g_new - gamma).max() if three_p else 0.0)
        alpha, beta, gamma = a_new, b_new, g_new
        if verbose and it % 10 == 0:
            print(f"  iter {it:3d}  loglik={hist[-1]:.2f}  delta={delta:.2e}")
        if delta < tol:
            return dict(alpha=alpha, beta=beta, gamma=gamma,
                        n_iter=it + 1, converged=True, loglik=hist)

    return dict(alpha=alpha, beta=beta, gamma=gamma,
                n_iter=max_iter, converged=False, loglik=hist)


# =====================================================================
# simulate — response generation
# =====================================================================

def gen_theta(n, a=2.0, b=2.0, rng=None):
    """Draw true theta from Beta(a,b).

    The simulation in the source article uses Beta(2,2) with n = 1000.
    """
    rng = np.random.default_rng(rng)
    return rng.beta(a, b, size=n)


def gen_responses(theta, alpha, beta, gamma=None, rng=None):
    """Eq. (22) of the source article: 1 if P > Uni(0,1), else 0."""
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta, dtype=float)
    if gamma is None:
        P = p2_naive(theta[:, None], alpha, beta)
    else:
        P = p3_naive(theta[:, None], alpha, beta, gamma)
    return (P > rng.random(P.shape)).astype(int)
