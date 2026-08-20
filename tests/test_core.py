"""Regression tests for cogtraitmodel.

Design principle: the literal transcriptions of the published equations
(`p*_naive`) are the oracle for correctness, and any optimised path must agree
with them to within 1e-12 (Decisions: naive_is_oracle).
"""

import numpy as np
import pytest

import cogtraitmodel as ctm
from cogtraitmodel import core, hctm, info

RNG = np.random.default_rng(20260820)
ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 100.0]
BETAS = [0.05, 0.3, 0.5, 0.7, 0.95]


# ── link function: boundaries and monotonicity ──────────────────────
@pytest.mark.parametrize("a", ALPHAS)
@pytest.mark.parametrize("b", BETAS)
def test_boundaries_l1(a, b):
    """P(0) = 0 and P(1) = 1 — are both anchors held exactly?"""
    assert abs(float(ctm.p2_naive(0.0, a, b))) < 1e-12
    assert abs(float(ctm.p2_naive(1.0, a, b)) - 1.0) < 1e-12


@pytest.mark.parametrize("a", ALPHAS)
def test_monotone(a):
    th = np.linspace(0, 1, 2001)
    p = ctm.p2_naive(th, a, 0.5)
    assert np.diff(p).min() >= -1e-13


def test_three_parameter_boundaries():
    """P3(0) = gamma, P3(1) = 1."""
    g = 0.2
    assert abs(float(ctm.p3_naive(0.0, 10.0, 0.5, g)) - g) < 1e-12
    assert abs(float(ctm.p3_naive(1.0, 10.0, 0.5, g)) - 1.0) < 1e-12


# ── normalized form == transcription (pinning the oracle) ───────────
@pytest.mark.parametrize("a", ALPHAS)
@pytest.mark.parametrize("b", BETAS)
def test_general_L_equals_naive_at_L1(a, b):
    """Does the general form at L = 1 reproduce Eq. (13), to within 1e-12?"""
    th = np.linspace(0.01, 0.99, 401)
    d = np.abs(hctm.p2_ctm_L(th, a, b, 1.0) - ctm.p2_naive(th, a, b)).max()
    assert d < 1e-12, f"alpha={a} beta={b} max diff {d:.2e}"


@pytest.mark.parametrize("a,b", [(1.0, 0.5), (5.0, 0.4), (10.0, 0.7)])
def test_hctm_is_large_L_limit(a, b):
    """Does the general form converge to HCTM as L grows?

    Convergence is governed by alpha*(L-beta), not by L alone (Section 2.3).
    """
    th = np.linspace(0.01, 0.99, 201)
    L = max(50.0, 700.0 / a)          # alpha*L < 709 (overflow limit of the expanded form)
    d = np.abs(hctm.p2_ctm_L(th, a, b, L) - hctm.p2_hctm(th, a, b)).max()
    assert d < 1e-9, f"alpha={a} beta={b} L={L} max diff {d:.2e}"


def test_hctm_no_overflow_at_large_alpha_theta():
    """Does the log-space HCTM stay finite where the expanded form overflows?"""
    for a, t in [(10.0, 100.0), (50.0, 200.0), (100.0, 500.0)]:
        v = float(hctm.p2_hctm(t, a, 1.0))
        assert np.isfinite(v) and 0.0 <= v <= 1.0


# ── properties (Section 2.5) ────────────────────────────────────────
@pytest.mark.parametrize("a", ALPHAS)
def test_median_exception_at_beta_half(a):
    """P(beta) = 1/2 holds exactly only when beta = L/2 (Prop. 2.5)."""
    assert abs(float(ctm.p2_naive(0.5, a, 0.5)) - 0.5) < 1e-14


@pytest.mark.parametrize("a", [1.0, 5.0, 25.0])
def test_p_at_beta_depends_on_alpha(a):
    """For beta != L/2, P(beta) depends on alpha — Wright maps do not transfer."""
    v = float(ctm.p2_naive(0.1, a, 0.1))
    assert abs(v - 0.5) > 1e-6 or a > 50


def test_alpha_to_zero_limit():
    """The alpha -> 0 limit is gamma + (1-gamma)*theta (correction C1)."""
    g, th = 0.2, 0.5
    v = float(ctm.p3_naive(th, 1e-6, 0.5, g))
    assert abs(v - (g + (1 - g) * th)) < 1e-5
    assert abs(v - (th + g)) > 0.05          # disagrees with the published expression


# ── quadrature and scoring ──────────────────────────────────────────
def test_grid_integrates_prior_exactly():
    """Beta(2,2) is quadratic, so Gauss-Legendre integrates it exactly."""
    nodes, w = ctm.make_grid(21, 2.0, 2.0)
    assert abs(float(w @ nodes) - 0.5) < 1e-12
    var = float(w @ nodes ** 2) - 0.25
    assert abs(var - 0.05) < 1e-12


def test_41_nodes_reach_precision_floor():
    """Do 41 nodes reach the double-precision floor against 61 (Section 3.1)?"""
    th = ctm.gen_theta(200, rng=np.random.default_rng(1))
    alpha = np.full(20, 8.0)
    beta = np.linspace(0.1, 0.9, 20)
    Y = ctm.gen_responses(th, alpha, beta, rng=np.random.default_rng(2))
    m41, _ = ctm.eap(Y, alpha, beta, n_nodes=41)
    m61, _ = ctm.eap(Y, alpha, beta, n_nodes=61)
    assert np.abs(m41 - m61).max() < 1e-12


def test_map_has_no_boundary_solutions():
    """The Beta prior keeps MAP strictly inside (0,1) (Section 3.3)."""
    th = ctm.gen_theta(500, rng=np.random.default_rng(3))
    alpha = np.full(10, 6.0)
    beta = np.linspace(0.15, 0.85, 10)
    Y = ctm.gen_responses(th, alpha, beta, rng=np.random.default_rng(4))
    out = ctm.score(Y, alpha, beta)
    assert not out["at_bound"].any()
    assert (out["theta"] > 0).all() and (out["theta"] < 1).all()


def test_mle_produces_boundary_solutions_on_perfect_patterns():
    """A perfect scorer goes to theta = 1 under MLE — a finite, dangerous failure."""
    Y = np.ones((3, 10), dtype=int)
    alpha = np.full(10, 6.0)
    beta = np.linspace(0.15, 0.85, 10)
    hat, at_bound = ctm.mle_theta(Y, alpha, beta)
    assert at_bound.all()
    assert np.allclose(hat, 1.0, atol=1e-3)


def test_score_sd_is_positive_and_finite():
    th = ctm.gen_theta(100, rng=np.random.default_rng(5))
    alpha = np.full(12, 7.0)
    beta = np.linspace(0.1, 0.9, 12)
    Y = ctm.gen_responses(th, alpha, beta, rng=np.random.default_rng(6))
    out = ctm.score(Y, alpha, beta)
    assert np.isfinite(out["sd"]).all() and (out["sd"] > 0).all()


# ── calibration ─────────────────────────────────────────────────────
def test_em_recovers_parameters():
    """Does Bayes modal EM recover the generating parameters (loose tolerance)?"""
    th = ctm.gen_theta(1000, rng=np.random.default_rng(7))
    alpha = np.full(20, 8.0)
    beta = np.linspace(0.1, 0.9, 20)
    Y = ctm.gen_responses(th, alpha, beta, rng=np.random.default_rng(8))
    fit = ctm.bayes_modal_em(Y)
    assert np.corrcoef(beta, fit["beta"])[0, 1] > 0.95
    assert 0.5 < np.median(fit["alpha"]) / 8.0 < 2.0


# ── information functions (Section 3.5) ─────────────────────────────
def test_information_diverges_at_mastery_boundary():
    """2P diverges at both boundaries; 3P only at the upper one."""
    a, b = 10.0, 0.5
    i2_lo = float(info.item_info(1e-4, a, b))
    i3_lo = float(info.item_info(1e-4, a, b, 0.2))
    i2_hi = float(info.item_info(1 - 1e-4, a, b))
    i3_hi = float(info.item_info(1 - 1e-4, a, b, 0.2))
    assert i2_lo > 100 and i3_lo < 1.0        # gamma removes the lower divergence
    assert i2_hi > 100 and i3_hi > 100        # gamma cannot stop the upper one


def test_information_divergence_rate_is_one():
    """(L - theta) * I(theta) -> P'(L) — the rate is exactly first order."""
    a, b = 10.0, 0.5
    eps = 1e-7
    lhs = eps * float(info.item_info(1.0 - eps, a, b))
    dp = (float(ctm.p2_naive(1.0, a, b))
          - float(ctm.p2_naive(1.0 - 1e-6, a, b))) / 1e-6
    assert abs(lhs - dp) / dp < 1e-3


# ── data ────────────────────────────────────────────────────────────
def test_lsat_dataset_matches_published_values():
    """Does the LSAT Section 6 data match the published values (self-checking)?"""
    Y = ctm.datasets.build(verify=True)
    assert Y.shape == (1000, 5)
    p = Y.mean(axis=0)
    assert np.allclose(p, [0.924, 0.709, 0.553, 0.763, 0.870], atol=5e-4)
    assert int((Y.sum(1) == 5).sum()) == 298
    assert int((Y.sum(1) == 0).sum()) == 3


# ── IRT comparison baseline ─────────────────────────────────────────
def test_irt_2pl_is_half_at_difficulty():
    from cogtraitmodel import irt
    assert abs(float(irt.p2pl(0.0, 1.0, 0.0)) - 0.5) < 1e-14
