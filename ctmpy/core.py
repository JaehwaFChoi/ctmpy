"""
Cognitive Trait Model (CTM) — v0.4
==================================================
Choi, J. (2022). Cognitive Trait Model: Measurement Model for Mastery Level
and Progression of Learning. Mathematics, 10, 2651.

구획:
  link      논문 식 (12)/(13)/(14) 직접 구현 (naive, 오라클)
  quad      [0,1] Gauss-Legendre 구적 격자 + Beta 사전분포 가중
  score     EAP, 사후 SD, MAP, MLE
  calib     Bayes modal EM (MMLE + 사전분포)
  simulate  응답 생성

link 구획은 논문의 식을 **문자 그대로** 옮긴 참조 구현이다.
수치 안정화판(expm1/softplus 재작성)은 v0.5 에서 별도로 추가하며,
그때 이 naive 구현이 정확성의 오라클(oracle) 역할을 한다.

논문 범위(alpha in [1, 100], theta in [0, 1])에서 이 구현은 정확히 동작한다.
깨지는 지점은 두 곳뿐이며, 의도적으로 그대로 둔다:
  - alpha -> 0  : 1 - exp(0) = 0 이 되어 0/0 -> NaN
                  (실제 극한은 P2 -> theta, P3 -> gamma + (1-gamma)*theta.
                   논문 본문의 'theta + gamma' 표현은 부정확하다.)
  - alpha > 709 : exp 오버플로우
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
    """2-Parameter CTM — 논문 식 (13).

    P = (1 - exp(a*th)) / (1 + exp(a*(th - b))) * (1 + exp(a*(1 - b))) / (1 - exp(a))

    분자 (1 - exp(a*th)) 와 (1 - exp(a)) 는 a > 0 일 때 모두 음수이므로
    비율은 양수가 된다.
    """
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)

    num = 1.0 - np.exp(alpha * theta)
    den = 1.0 + np.exp(alpha * (theta - beta))
    scale = (1.0 + np.exp(alpha * (1.0 - beta))) / (1.0 - np.exp(alpha))
    return num / den * scale


def p3_naive(theta, alpha, beta, gamma):
    """3-Parameter CTM — 논문 식 (12).

    P = g + (1 - g) * P2(theta; a, b)

    gamma 는 전통 3PL 의 하한 점근선이 아니라, 사람이 **최하 경계에 있을 때**의
    정답 확률 Pr(y = 1 | theta = 0) 이다. 논문 Table 1 참조.
    """
    gamma = np.asarray(gamma, dtype=float)
    return gamma + (1.0 - gamma) * p2_naive(theta, alpha, beta)


def p1_naive(theta, beta, alpha=10.0):
    """1-Parameter CTM — 논문 식 (14). alpha 를 상수로 고정한 2P CTM."""
    return p2_naive(theta, alpha, beta)


# =====================================================================
# quad — [0,1] Gauss-Legendre 구적 격자
# =====================================================================

def make_grid(n_nodes=61, a=2.0, b=2.0):
    """[0,1] 위의 Gauss-Legendre 노드와 Beta(a,b) 사전분포 가중치를 만든다.

    전통 IRT 는 theta 가 무계라 Gauss-Hermite 를 쓰고 절단 범위(보통 +-4)를
    임의로 정해야 한다. CTM 은 theta 가 [0,1] 유계이므로 Gauss-Legendre 를
    그대로 쓸 수 있고 절단오차가 아예 존재하지 않는다.

    R3 결과: 41노드면 배정밀도 한계까지 수렴한다 (61 대비 4e-15).
    """
    x, gw = np.polynomial.legendre.leggauss(n_nodes)
    nodes = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
    w = 0.5 * gw * beta_dist.pdf(nodes, a, b)
    return nodes, w / w.sum()


# =====================================================================
# score — EAP / 사후 SD / MAP / MLE
# =====================================================================

def _loglik(Y, nodes, alpha, beta, gamma=None):
    """각 응답자 x 각 구적점의 로그우도 행렬 (n, K) 를 만든다.

    Y : (n, J) 0/1 응답행렬. 결측은 np.nan 으로 두면 해당 문항이 제외된다.
    """
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        P = p2_naive(nodes[:, None], alpha, beta)  # (K, J)
    else:
        P = p3_naive(nodes[:, None], alpha, beta, gamma)
    P = np.clip(P, 1e-12, 1 - 1e-12)  # log(0) 방지

    obs = ~np.isnan(Y)
    Y0 = np.where(obs, Y, 0.0)
    return Y0 @ np.log(P).T + (obs & (Y0 == 0)) @ np.log(1 - P).T


def posterior(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """구적 격자 위의 정규화된 사후분포 (n, K) 와 노드를 돌려준다."""
    nodes, w = make_grid(n_nodes, *prior)
    ll = _loglik(Y, nodes, alpha, beta, gamma)
    ll -= ll.max(axis=1, keepdims=True)  # 오버플로우 방지
    post = np.exp(ll) * w
    return nodes, post / post.sum(axis=1, keepdims=True)


def eap(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """EAP 추정치와 사후 SD 를 돌려준다.

    사후 SD 를 쓰는 이유: CTM 은 theta=1 에서 P=1 로 고정되어 Fisher 정보량
    I=(P')^2/(P(1-P)) 의 분모가 0 으로 가고 SEM 이 0 으로 붕괴한다.
    Beta 사전분포는 경계에서 밀도가 0 이므로 사후분포 폭은 붕괴하지 않는다.
    """
    nodes, post = posterior(Y, alpha, beta, gamma, n_nodes, prior)
    m = post @ nodes
    v = post @ (nodes ** 2) - m ** 2
    return m, np.sqrt(np.maximum(v, 0.0))


def eap_batch(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """eap 의 별칭."""
    return eap(Y, alpha, beta, gamma, n_nodes, prior)


def _mode_on_grid(Y, alpha, beta, gamma, prior, n_grid, eps):
    """조밀 격자에서 최빈값을 찾고 포물선 보간으로 격자 사이를 메운다.

    prior=None 이면 MLE, prior=(a,b) 면 MAP.
    반복 최적화를 쓰지 않는다 — 경계에 붙는 해를 다루기에 격자 방식이 안전하다.

    Returns
    -------
    theta_hat : (n,) ndarray
    at_bound  : (n,) bool  격자 양끝에서 최대가 잡힌 경우 (경계해)
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
    """MAP: 로그우도 + 로그 Beta 사전분포를 최대화한다.

    Beta(2,2) 는 경계에서 밀도가 0 이므로 MAP 은 (0,1) 내부에 머문다.
    R2 결과: 기본 점추정으로 권장. 축소가 EAP 의 절반이면서 경계해가 없다.
    """
    return _mode_on_grid(Y, alpha, beta, gamma, prior, n_grid, eps)


def mle_theta(Y, alpha, beta, gamma=None, n_grid=4001, eps=1e-6):
    """MLE: 로그우도만 최대화한다. 진단용.

    전부 정답이면 theta=1, 2P 에서 전부 오답이면 theta=0 에 붙는다.
    CTM 에서 이 경계값은 무한대가 아니라 '숙달 수준'/'무지 수준'이라는
    해석 가능한 유한값이므로 오히려 위험하다. at_bound 로 반드시 표시한다.
    J=5 에서는 약 14.5% 에게 발생한다.
    """
    return _mode_on_grid(Y, alpha, beta, gamma, None, n_grid, eps)


# =====================================================================
# calib — Bayes modal EM (MMLE + 사전분포)
# =====================================================================

def _logit(p):
    return np.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pack(alpha, beta, gamma):
    """제약 있는 모수를 무제약 공간으로. alpha>0, beta/gamma in (0,1)."""
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
    """한 문항의 음의 (기대로그우도 + 로그사전분포). M-step 목적함수.

    Nk : (K,) 각 구적점의 기대 인원
    rk : (K,) 각 구적점의 기대 정답 수
    """
    alpha, beta, gamma = _unpack(z, three_p)
    if not np.isfinite(alpha) or alpha <= 0 or alpha > 1e4:
        return 1e10

    P = p2_naive(nodes, alpha, beta) if not three_p \
        else p3_naive(nodes, alpha, beta, gamma)
    P = np.clip(P, 1e-12, 1 - 1e-12)

    ll = np.sum(rk * np.log(P) + (Nk - rk) * np.log(1 - P))

    # 사전분포: alpha ~ LogNormal(mu, sd), beta/gamma ~ Beta
    ll += -0.5 * ((np.log(alpha) - pa[0]) / pa[1]) ** 2 - np.log(alpha)
    ll += (pb[0] - 1) * np.log(beta) + (pb[1] - 1) * np.log(1 - beta)
    if three_p:
        ll += (pg[0] - 1) * np.log(gamma) + (pg[1] - 1) * np.log(1 - gamma)
    return -ll if np.isfinite(ll) else 1e10


def bayes_modal_em(Y, three_p=False, n_nodes=61, prior_theta=(2.0, 2.0),
                   prior_alpha=(np.log(10.0), 1.0), prior_beta=(2.0, 2.0),
                   prior_gamma=(7.0, 25.0), max_iter=300, tol=1e-6,
                   init=None, verbose=False):
    """문항모수를 MMLE + 사전분포(Bayes modal)로 추정한다.

    E-step 은 [0,1] 구적 격자 위의 사후분포, M-step 은 문항별 무제약 최적화다.
    문항끼리 독립이므로 J 개의 2~3차원 문제로 쪼개진다.

    prior_alpha 는 (mu, sd) 로 log(alpha) 의 정규분포를 지정한다.
    주의: 논문 WinBUGS 코드의 dlnorm(7.5, 1) 은 정밀도 표기이므로
    log(alpha) ~ N(7.5, 1), 즉 alpha 중앙값 약 1808 이라는 매우 강한
    사전분포가 된다. 참값 5~10 과 크게 어긋나므로 R6 에서 따로 검토한다.

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
        beta = np.clip(1.0 - Y.mean(axis=0), 0.05, 0.95)  # 정답률 높으면 쉬운 문항
        gamma = np.full(J, 0.2) if three_p else None
    else:
        alpha, beta, gamma = init

    hist = []
    for it in range(max_iter):
        # ---- E-step: 사후분포와 기대 카운트 ----
        ll = _loglik(Y, nodes, alpha, beta, gamma)
        mx = ll.max(axis=1, keepdims=True)
        post = np.exp(ll - mx) * w
        norm = post.sum(axis=1, keepdims=True)
        post /= norm
        hist.append(float((np.log(norm[:, 0]) + mx[:, 0]).sum()))

        Nk = post.sum(axis=0)  # (K,)
        rk = post.T @ Y        # (K, J)

        # ---- M-step: 문항별 독립 최적화 ----
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
# simulate — 응답 생성
# =====================================================================

def gen_theta(n, a=2.0, b=2.0, rng=None):
    """Beta(a,b) 에서 참 theta 를 뽑는다. 논문 시뮬레이션은 Beta(2,2), n=1000."""
    rng = np.random.default_rng(rng)
    return rng.beta(a, b, size=n)


def gen_responses(theta, alpha, beta, gamma=None, rng=None):
    """논문 식 (22): P > Uni(0,1) 이면 1, 아니면 0."""
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta, dtype=float)
    if gamma is None:
        P = p2_naive(theta[:, None], alpha, beta)
    else:
        P = p3_naive(theta[:, None], alpha, beta, gamma)
    return (P > rng.random(P.shape)).astype(int)
