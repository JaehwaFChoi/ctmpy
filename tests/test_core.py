"""ctmpy 회귀 테스트.

설계 원칙: 논문 전사판(`p*_naive`)이 정확성 기준(oracle)이며, 어떤 최적화도
이 기준과 1e-12 이내로 일치해야 한다(Decisions: naive_is_oracle).
"""

import numpy as np
import pytest

import ctmpy
from ctmpy import core, hctm, info

RNG = np.random.default_rng(20260820)
ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 100.0]
BETAS = [0.05, 0.3, 0.5, 0.7, 0.95]


# ── 링크함수: 경계·단조성 ────────────────────────────────────────────
@pytest.mark.parametrize("a", ALPHAS)
@pytest.mark.parametrize("b", BETAS)
def test_boundaries_l1(a, b):
    """P(0) = 0, P(1) = 1 — 두 앵커가 정확히 유지되는가."""
    assert abs(float(ctmpy.p2_naive(0.0, a, b))) < 1e-12
    assert abs(float(ctmpy.p2_naive(1.0, a, b)) - 1.0) < 1e-12


@pytest.mark.parametrize("a", ALPHAS)
def test_monotone(a):
    th = np.linspace(0, 1, 2001)
    p = ctmpy.p2_naive(th, a, 0.5)
    assert np.diff(p).min() >= -1e-13


def test_three_parameter_boundaries():
    """P3(0) = gamma, P3(1) = 1."""
    g = 0.2
    assert abs(float(ctmpy.p3_naive(0.0, 10.0, 0.5, g)) - g) < 1e-12
    assert abs(float(ctmpy.p3_naive(1.0, 10.0, 0.5, g)) - 1.0) < 1e-12


# ── 정규화형 = 전사판 (오라클 고정) ──────────────────────────────────
@pytest.mark.parametrize("a", ALPHAS)
@pytest.mark.parametrize("b", BETAS)
def test_general_L_equals_naive_at_L1(a, b):
    """일반형 L=1 이 논문 식 (13) 을 재현하는가 — 1e-12 이내."""
    th = np.linspace(0.01, 0.99, 401)
    d = np.abs(hctm.p2_ctm_L(th, a, b, 1.0) - ctmpy.p2_naive(th, a, b)).max()
    assert d < 1e-12, f"alpha={a} beta={b} 최대차 {d:.2e}"


@pytest.mark.parametrize("a,b", [(1.0, 0.5), (5.0, 0.4), (10.0, 0.7)])
def test_hctm_is_large_L_limit(a, b):
    """L 을 키우면 일반형이 HCTM 으로 수렴하는가.

    수렴은 L 이 아니라 alpha*(L-beta) 가 지배한다(논문 §2.3).
    """
    th = np.linspace(0.01, 0.99, 201)
    L = max(50.0, 700.0 / a)          # alpha*L < 709 (전개형 오버플로우 한계)
    d = np.abs(hctm.p2_ctm_L(th, a, b, L) - hctm.p2_hctm(th, a, b)).max()
    assert d < 1e-9, f"alpha={a} beta={b} L={L} 최대차 {d:.2e}"


def test_hctm_no_overflow_at_large_alpha_theta():
    """전개형이 넘치는 영역에서도 로그공간 HCTM 은 유한한가."""
    for a, t in [(10.0, 100.0), (50.0, 200.0), (100.0, 500.0)]:
        v = float(hctm.p2_hctm(t, a, 1.0))
        assert np.isfinite(v) and 0.0 <= v <= 1.0


# ── 성질 (논문 §2.5) ────────────────────────────────────────────────
@pytest.mark.parametrize("a", ALPHAS)
def test_median_exception_at_beta_half(a):
    """beta = L/2 일 때만 P(beta) = 1/2 가 정확히 성립 (Prop 2.5)."""
    assert abs(float(ctmpy.p2_naive(0.5, a, 0.5)) - 0.5) < 1e-14


@pytest.mark.parametrize("a", [1.0, 5.0, 25.0])
def test_p_at_beta_depends_on_alpha(a):
    """beta != L/2 에서는 P(beta) 가 alpha 에 의존한다 — Wright map 비이식성."""
    v = float(ctmpy.p2_naive(0.1, a, 0.1))
    assert abs(v - 0.5) > 1e-6 or a > 50


def test_alpha_to_zero_limit():
    """alpha -> 0 극한은 gamma + (1-gamma)*theta (원 논문 정정 C1)."""
    g, th = 0.2, 0.5
    v = float(ctmpy.p3_naive(th, 1e-6, 0.5, g))
    assert abs(v - (g + (1 - g) * th)) < 1e-5
    assert abs(v - (th + g)) > 0.05          # 원문 표기와는 불일치


# ── 구적·채점 ───────────────────────────────────────────────────────
def test_grid_integrates_prior_exactly():
    """Beta(2,2) 는 이차 다항식이므로 Gauss-Legendre 가 정확히 적분한다."""
    nodes, w = ctmpy.make_grid(21, 2.0, 2.0)
    assert abs(float(w @ nodes) - 0.5) < 1e-12
    var = float(w @ nodes ** 2) - 0.25
    assert abs(var - 0.05) < 1e-12


def test_41_nodes_reach_precision_floor():
    """41노드가 61노드 기준 배정밀도 한계에 도달하는가 (논문 §3.1)."""
    th = ctmpy.gen_theta(200, rng=np.random.default_rng(1))
    alpha = np.full(20, 8.0)
    beta = np.linspace(0.1, 0.9, 20)
    Y = ctmpy.gen_responses(th, alpha, beta, rng=np.random.default_rng(2))
    m41, _ = ctmpy.eap(Y, alpha, beta, n_nodes=41)
    m61, _ = ctmpy.eap(Y, alpha, beta, n_nodes=61)
    assert np.abs(m41 - m61).max() < 1e-12


def test_map_has_no_boundary_solutions():
    """MAP 은 Beta 사전분포 때문에 (0,1) 내부에 머문다 (논문 §3.3)."""
    th = ctmpy.gen_theta(500, rng=np.random.default_rng(3))
    alpha = np.full(10, 6.0)
    beta = np.linspace(0.15, 0.85, 10)
    Y = ctmpy.gen_responses(th, alpha, beta, rng=np.random.default_rng(4))
    out = ctmpy.score(Y, alpha, beta)
    assert not out["at_bound"].any()
    assert (out["theta"] > 0).all() and (out["theta"] < 1).all()


def test_mle_produces_boundary_solutions_on_perfect_patterns():
    """만점 응답자는 MLE 에서 theta=1 로 간다 — 유한하고 위험한 실패."""
    Y = np.ones((3, 10), dtype=int)
    alpha = np.full(10, 6.0)
    beta = np.linspace(0.15, 0.85, 10)
    hat, at_bound = ctmpy.mle_theta(Y, alpha, beta)
    assert at_bound.all()
    assert np.allclose(hat, 1.0, atol=1e-3)


def test_score_sd_is_positive_and_finite():
    th = ctmpy.gen_theta(100, rng=np.random.default_rng(5))
    alpha = np.full(12, 7.0)
    beta = np.linspace(0.1, 0.9, 12)
    Y = ctmpy.gen_responses(th, alpha, beta, rng=np.random.default_rng(6))
    out = ctmpy.score(Y, alpha, beta)
    assert np.isfinite(out["sd"]).all() and (out["sd"] > 0).all()


# ── 캘리브레이션 ────────────────────────────────────────────────────
def test_em_recovers_parameters():
    """Bayes modal EM 이 생성 모수를 회복하는가 (느슨한 허용오차)."""
    th = ctmpy.gen_theta(1000, rng=np.random.default_rng(7))
    alpha = np.full(20, 8.0)
    beta = np.linspace(0.1, 0.9, 20)
    Y = ctmpy.gen_responses(th, alpha, beta, rng=np.random.default_rng(8))
    fit = ctmpy.bayes_modal_em(Y)
    assert np.corrcoef(beta, fit["beta"])[0, 1] > 0.95
    assert 0.5 < np.median(fit["alpha"]) / 8.0 < 2.0


# ── 정보함수 (논문 §3.5) ────────────────────────────────────────────
def test_information_diverges_at_mastery_boundary():
    """2P 는 양쪽 경계에서, 3P 는 위쪽에서만 발산한다."""
    a, b = 10.0, 0.5
    i2_lo = float(info.item_info(1e-4, a, b))
    i3_lo = float(info.item_info(1e-4, a, b, 0.2))
    i2_hi = float(info.item_info(1 - 1e-4, a, b))
    i3_hi = float(info.item_info(1 - 1e-4, a, b, 0.2))
    assert i2_lo > 100 and i3_lo < 1.0        # gamma 가 하단 발산을 제거
    assert i2_hi > 100 and i3_hi > 100        # 상단은 gamma 로도 못 막는다


def test_information_divergence_rate_is_one():
    """(L - theta) * I(theta) -> P'(L) — 발산 속도가 정확히 1차."""
    a, b = 10.0, 0.5
    eps = 1e-7
    lhs = eps * float(info.item_info(1.0 - eps, a, b))
    dp = (float(ctmpy.p2_naive(1.0, a, b))
          - float(ctmpy.p2_naive(1.0 - 1e-6, a, b))) / 1e-6
    assert abs(lhs - dp) / dp < 1e-3


# ── 자료 ────────────────────────────────────────────────────────────
def test_lsat_dataset_matches_published_values():
    """LSAT Section 6 자료가 문헌값과 일치하는가 (자체 검증 포함)."""
    Y = ctmpy.datasets.build(verify=True)
    assert Y.shape == (1000, 5)
    p = Y.mean(axis=0)
    assert np.allclose(p, [0.924, 0.709, 0.553, 0.763, 0.870], atol=5e-4)
    assert int((Y.sum(1) == 5).sum()) == 298
    assert int((Y.sum(1) == 0).sum()) == 3


# ── IRT 비교 기준 ───────────────────────────────────────────────────
def test_irt_2pl_is_half_at_difficulty():
    from ctmpy import irt
    assert abs(float(irt.p2pl(0.0, 1.0, 0.0)) - 0.5) < 1e-14
