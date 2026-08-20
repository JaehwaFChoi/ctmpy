"""Tests for cogtraitmodel.growth and cogtraitmodel.scale.

Three claims are pinned here:
  (1) finite L reduces to L = 1 by rescaling, to machine precision
  (2) the dof discipline k <= H-1 always holds — violating it produced the
      worst result in the study
  (3) the growth prior beats memoryless scoring at later occasions
"""

import numpy as np
import pytest

import cogtraitmodel as ctm
from cogtraitmodel import growth, scale
from cogtraitmodel.hctm import p2_ctm_L


# ── (1) the rescaling identity ──────────────────────────────────────
@pytest.mark.parametrize("a,b,L", [(1.0, 0.5, 2.0), (5.0, 0.4, 3.0),
                                   (2.0, 1.5, 5.0), (0.8, 2.0, 4.0)])
def test_finite_L_reduces_to_unit_scale(a, b, L):
    """P_L(th; a, b) == P_1(th/L; a*L, b/L) — holds to machine precision."""
    th = np.linspace(0.01, L * 0.99, 300)
    lhs = p2_ctm_L(th, a, b, L)
    rhs = ctm.p2_naive(th / L, a * L, b / L)
    assert np.abs(lhs - rhs).max() < 1e-13


def test_score_L_returns_theta_on_L_scale():
    L = 3.0
    th = ctm.gen_theta(300, rng=np.random.default_rng(0)) * L
    a = np.full(20, 8.0) / L
    b = np.linspace(0.1, 0.9, 20) * L
    P = p2_ctm_L(th[:, None], a, b, L)
    Y = (np.random.default_rng(1).random(P.shape) < P).astype(int)
    out = scale.score_L(Y, a, b, L)
    assert (out["theta"] > 0).all() and (out["theta"] < L).all()
    assert np.corrcoef(th, out["theta"])[0, 1] > 0.9


def test_fit_L_recovers_difficulty_on_L_scale():
    L = 2.5
    th = ctm.gen_theta(600, rng=np.random.default_rng(2)) * L
    a = np.full(20, 8.0) / L
    b = np.linspace(0.1, 0.9, 20) * L
    P = p2_ctm_L(th[:, None], a, b, L)
    Y = (np.random.default_rng(3).random(P.shape) < P).astype(int)
    fit = scale.fit_L(Y, L)
    assert fit["L"] == L
    assert np.corrcoef(b, fit["beta"])[0, 1] > 0.95


def test_infinite_L_is_rejected_with_pointer_to_growth():
    """L -> inf is not a measurement model — rejected, with a pointer to growth."""
    Y = np.ones((2, 5), dtype=int)
    with pytest.raises(ValueError, match="growth"):
        scale.fit_L(Y, np.inf)
    with pytest.raises(ValueError):
        scale.fit_L(Y, 0.0)


# ── (2) shape of the growth curve ───────────────────────────────────
def test_growth_curve_boundaries():
    """theta(0) = gamma, and theta converges to delta as t -> inf."""
    g, d = 0.2, 0.9
    assert abs(float(growth.curve(0.0, 1.2, 2.0, g, d)) - g) < 1e-12
    assert abs(float(growth.curve(500.0, 1.2, 2.0, g, d)) - d) < 1e-6


def test_growth_curve_represents_decline():
    """delta < gamma gives decay (forgetting) — no separate model needed."""
    y = growth.curve(np.linspace(0, 10, 50), 1.0, 3.0, 0.8, 0.3)
    assert np.diff(y).max() <= 1e-12
    assert y[0] > y[-1]


def test_growth_curve_is_monotone_always():
    """Monotonicity is structural — non-monotone paths are unrepresentable."""
    rng = np.random.default_rng(11)
    t = np.linspace(0, 12, 200)
    for _ in range(20):
        a = rng.uniform(0.1, 15.0)
        b = rng.uniform(0.1, 20.0)
        g, d = rng.uniform(0, 1, 2)
        y = growth.curve(t, a, b, g, d)
        assert np.diff(y).min() >= -1e-12 or np.diff(y).max() <= 1e-12


# ── (3) the degrees-of-freedom discipline ───────────────────────────
@pytest.mark.parametrize("H", [2, 3, 4, 5, 6, 8])
def test_dof_margin_is_enforced(H):
    """Always k <= H - 1 — violating it collapses the predictive variance."""
    ts = np.arange(H, dtype=float)
    th = np.linspace(0.2, 0.8, H)[None, :]
    sd = np.full((1, H), 0.08)
    res = growth.fit(ts, th, sd)
    assert res["k"] <= H - 1, f"at H={H}, k={res['k']} — no dof left in reserve"


def test_residual_variance_never_collapses():
    """Even on a perfectly fitting history the residual variance keeps its floor."""
    ts = np.arange(6, dtype=float)
    th = growth.curve(ts, 1.2, 2.0, 0.2, 0.9)[None, :]   # data the model fits exactly
    sd = np.full((1, 6), 0.07)
    res = growth.fit(ts, th, sd)
    assert res["resid_var"][0] > 0.0
    _, psd = growth.predict(ts, th, sd, 6.0)
    assert psd[0] >= 0.02


def test_predict_shapes_and_bounds():
    n, H = 5, 4
    ts = np.arange(H, dtype=float)
    th = np.tile(np.linspace(0.3, 0.7, H), (n, 1))
    sd = np.full((n, H), 0.09)
    m, s = growth.predict(ts, th, sd, float(H))
    assert m.shape == (n,) and s.shape == (n,)
    assert (m > 0).all() and (m < 1).all() and (s > 0).all()


def test_fit_rejects_mismatched_history():
    ts = np.arange(4, dtype=float)
    with pytest.raises(ValueError):
        growth.fit(ts, np.zeros((3, 5)), np.ones((3, 5)) * 0.1)


# ── conversion to a prior ───────────────────────────────────────────
def test_to_prior_normalises_and_centres():
    nodes, _ = ctm.make_grid(41, 2.0, 2.0)
    qw = 0.5 * np.polynomial.legendre.leggauss(41)[1]
    W = growth.to_prior(np.array([0.3, 0.7]), np.array([0.1, 0.1]), nodes, qw)
    assert np.allclose(W.sum(axis=1), 1.0)
    m = W @ nodes
    assert abs(m[0] - 0.3) < 0.05 and abs(m[1] - 0.7) < 0.05


# ── sequential scoring ──────────────────────────────────────────────
def _sequential_setup(n=250, JB=40, T=6, K=5, seed=6260):
    rng = np.random.default_rng(seed)
    ab = np.full(JB, 8.0)
    bb = np.linspace(0.05, 0.95, JB)
    th0 = rng.beta(2.0, 5.0, n)
    rate = rng.uniform(0.10, 0.60, n)
    TH = np.array([th0 + (1 - th0) * (1 - np.exp(-rate * t)) for t in range(T)])
    items = [rng.choice(JB, K, replace=False) for _ in range(T)]
    resp = [ctm.gen_responses(TH[t], ab[items[t]], bb[items[t]], rng=rng)
            for t in range(T)]
    return TH, items, resp, ab, bb


def test_sequential_beats_no_memory_at_later_occasions():
    """The growth prior is more accurate than memoryless scoring later on."""
    TH, items, resp, ab, bb = _sequential_setup()
    r_none = growth.sequential_score(resp, items, ab, bb, memory="none")
    r_hctm = growth.sequential_score(resp, items, ab, bb, memory="hctm")
    last = TH.shape[0] - 1
    e_none = np.sqrt(((r_none["theta"][last] - TH[last]) ** 2).mean())
    e_hctm = np.sqrt(((r_hctm["theta"][last] - TH[last]) ** 2).mean())
    assert e_hctm < e_none


def test_sequential_unlock_schedule_respects_margin():
    """During a sequential run, k at each occasion stays below the history length."""
    _, items, resp, ab, bb = _sequential_setup()
    r = growth.sequential_score(resp, items, ab, bb, memory="hctm")
    for t, k in enumerate(r["k"][:-1]):
        H = t + 1
        assert k <= max(H - 1, 1)


def test_sequential_shapes_and_finiteness():
    TH, items, resp, ab, bb = _sequential_setup(n=100, T=4)
    r = growth.sequential_score(resp, items, ab, bb, memory="hctm")
    assert r["theta"].shape == (4, 100) and r["sd"].shape == (4, 100)
    assert np.isfinite(r["theta"]).all() and (r["sd"] > 0).all()


def test_sequential_rejects_bad_memory_argument():
    _, items, resp, ab, bb = _sequential_setup(n=50, T=3)
    with pytest.raises(ValueError):
        growth.sequential_score(resp, items, ab, bb, memory="carry_forward")
