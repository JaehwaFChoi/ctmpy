"""ctmpy.growth / ctmpy.scale 테스트.

핵심 주장 세 가지를 고정한다:
  (1) 유한 L 은 척도 재조정으로 L=1 에 환원된다 (기계정밀도).
  (2) 자유도 규율 k <= H-1 이 항상 지켜진다 — 위반은 이 연구 최악의 결과.
  (3) 성장 사전분포가 기억 없는 채점보다 낫다 (후반 회차에서).
"""

import numpy as np
import pytest

import ctmpy
from ctmpy import growth, scale
from ctmpy.hctm import p2_ctm_L


# ── (1) 척도 환원 항등식 ────────────────────────────────────────────
@pytest.mark.parametrize("a,b,L", [(1.0, 0.5, 2.0), (5.0, 0.4, 3.0),
                                   (2.0, 1.5, 5.0), (0.8, 2.0, 4.0)])
def test_finite_L_reduces_to_unit_scale(a, b, L):
    """P_L(th; a, b) == P_1(th/L; a*L, b/L) — 기계정밀도로 성립."""
    th = np.linspace(0.01, L * 0.99, 300)
    lhs = p2_ctm_L(th, a, b, L)
    rhs = ctmpy.p2_naive(th / L, a * L, b / L)
    assert np.abs(lhs - rhs).max() < 1e-13


def test_score_L_returns_theta_on_L_scale():
    L = 3.0
    th = ctmpy.gen_theta(300, rng=np.random.default_rng(0)) * L
    a = np.full(20, 8.0) / L
    b = np.linspace(0.1, 0.9, 20) * L
    P = p2_ctm_L(th[:, None], a, b, L)
    Y = (np.random.default_rng(1).random(P.shape) < P).astype(int)
    out = scale.score_L(Y, a, b, L)
    assert (out["theta"] > 0).all() and (out["theta"] < L).all()
    assert np.corrcoef(th, out["theta"])[0, 1] > 0.9


def test_fit_L_recovers_difficulty_on_L_scale():
    L = 2.5
    th = ctmpy.gen_theta(600, rng=np.random.default_rng(2)) * L
    a = np.full(20, 8.0) / L
    b = np.linspace(0.1, 0.9, 20) * L
    P = p2_ctm_L(th[:, None], a, b, L)
    Y = (np.random.default_rng(3).random(P.shape) < P).astype(int)
    fit = scale.fit_L(Y, L)
    assert fit["L"] == L
    assert np.corrcoef(b, fit["beta"])[0, 1] > 0.95


def test_infinite_L_is_rejected_with_pointer_to_growth():
    """L -> inf 는 측정 모형이 아니다 — 명시적으로 거부하고 growth 를 가리킨다."""
    Y = np.ones((2, 5), dtype=int)
    with pytest.raises(ValueError, match="growth"):
        scale.fit_L(Y, np.inf)
    with pytest.raises(ValueError):
        scale.fit_L(Y, 0.0)


# ── (2) 성장곡선의 형태 ─────────────────────────────────────────────
def test_growth_curve_boundaries():
    """theta(0) = gamma, t -> inf 에서 delta 로 수렴."""
    g, d = 0.2, 0.9
    assert abs(float(growth.curve(0.0, 1.2, 2.0, g, d)) - g) < 1e-12
    assert abs(float(growth.curve(500.0, 1.2, 2.0, g, d)) - d) < 1e-6


def test_growth_curve_represents_decline():
    """delta < gamma 면 감쇠(망각) — 별도 모형이 필요 없다."""
    y = growth.curve(np.linspace(0, 10, 50), 1.0, 3.0, 0.8, 0.3)
    assert np.diff(y).max() <= 1e-12
    assert y[0] > y[-1]


def test_growth_curve_is_monotone_always():
    """단조성은 구조적 — 비단조 궤적은 어떤 모수값으로도 표현 불가."""
    rng = np.random.default_rng(11)
    t = np.linspace(0, 12, 200)
    for _ in range(20):
        a = rng.uniform(0.1, 15.0)
        b = rng.uniform(0.1, 20.0)
        g, d = rng.uniform(0, 1, 2)
        y = growth.curve(t, a, b, g, d)
        assert np.diff(y).min() >= -1e-12 or np.diff(y).max() <= 1e-12


# ── (3) 자유도 규율 ─────────────────────────────────────────────────
@pytest.mark.parametrize("H", [2, 3, 4, 5, 6, 8])
def test_dof_margin_is_enforced(H):
    """항상 k <= H - 1 — 위반하면 예측분산이 붕괴한다 (§5.6)."""
    ts = np.arange(H, dtype=float)
    th = np.linspace(0.2, 0.8, H)[None, :]
    sd = np.full((1, H), 0.08)
    res = growth.fit(ts, th, sd)
    assert res["k"] <= H - 1, f"H={H} 에서 k={res['k']} — 자유도 여유 없음"


def test_residual_variance_never_collapses():
    """완벽히 적합되는 이력에서도 잔차분산이 측정오차 하한을 지킨다."""
    ts = np.arange(6, dtype=float)
    th = growth.curve(ts, 1.2, 2.0, 0.2, 0.9)[None, :]   # 모형이 정확한 자료
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


# ── 사전분포 변환 ───────────────────────────────────────────────────
def test_to_prior_normalises_and_centres():
    nodes, _ = ctmpy.make_grid(41, 2.0, 2.0)
    qw = 0.5 * np.polynomial.legendre.leggauss(41)[1]
    W = growth.to_prior(np.array([0.3, 0.7]), np.array([0.1, 0.1]), nodes, qw)
    assert np.allclose(W.sum(axis=1), 1.0)
    m = W @ nodes
    assert abs(m[0] - 0.3) < 0.05 and abs(m[1] - 0.7) < 0.05


# ── 순차 채점 ───────────────────────────────────────────────────────
def _sequential_setup(n=250, JB=40, T=6, K=5, seed=6260):
    rng = np.random.default_rng(seed)
    ab = np.full(JB, 8.0)
    bb = np.linspace(0.05, 0.95, JB)
    th0 = rng.beta(2.0, 5.0, n)
    rate = rng.uniform(0.10, 0.60, n)
    TH = np.array([th0 + (1 - th0) * (1 - np.exp(-rate * t)) for t in range(T)])
    items = [rng.choice(JB, K, replace=False) for _ in range(T)]
    resp = [ctmpy.gen_responses(TH[t], ab[items[t]], bb[items[t]], rng=rng)
            for t in range(T)]
    return TH, items, resp, ab, bb


def test_sequential_beats_no_memory_at_later_occasions():
    """성장 사전분포가 기억 없는 채점보다 후반에 정확하다."""
    TH, items, resp, ab, bb = _sequential_setup()
    r_none = growth.sequential_score(resp, items, ab, bb, memory="none")
    r_hctm = growth.sequential_score(resp, items, ab, bb, memory="hctm")
    last = TH.shape[0] - 1
    e_none = np.sqrt(((r_none["theta"][last] - TH[last]) ** 2).mean())
    e_hctm = np.sqrt(((r_hctm["theta"][last] - TH[last]) ** 2).mean())
    assert e_hctm < e_none


def test_sequential_unlock_schedule_respects_margin():
    """순차 실행 중 매 회차의 k 가 이력 길이보다 작다."""
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
