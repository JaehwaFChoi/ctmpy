"""cogtraitmodel.growth — 시간축 위의 HCTM: 성장곡선 적합과 순차 사전분포 갱신.

베이즈 갱신만으로는 부족하다
----------------------------
베이즈 갱신은 **고정된** 양에 증거를 쌓는 정리다. 학습자의 theta 는 고정이
아니며, 그것이 반복 측정의 이유다. 사후분포를 그대로 이월하면 "이 사람은
지난번 그 자리에 있다"를 사전정보로 주장하게 되고, 추정기는 새 증거를 자기
사전분포를 이기는 데 소모한다. 실측상 이월은 **기억이 없는 것보다 나빴다**
(RMSE 0.2295 대 0.1215) — 게다가 보고 SD 는 오히려 줄어 오차의 4분의 1을
주장한다.

필요한 것은 상태공간 모형의 **예측 단계**다: 분포를 넓히기만 하는 것이
아니라 옮기는 전이. 이 모듈이 그 전이를 제공한다.

왜 HCTM 인가
------------
성장곡선은 바닥은 필요하되 천장을 미리 부과해서는 안 된다. L -> inf 구성원은
절대영점을 갖고 상한이 없으므로 시간 지평 상수 H 가 필요 없다 —
theta(t) = gamma + (delta - gamma) * P_hctm(t; alpha, beta) 는 t 를 자연
단위 그대로 받는다. 네 모수는 교사가 알아볼 뜻을 가진다:

    gamma  진입 수준          delta  수렴 수준
    alpha  학습 속도          beta   성장이 일어나는 시기

delta < gamma 면 감쇠(망각)가 같은 네 모수로 표현된다. 단조 곡선이므로
**올랐다 내려가는 궤적은 어떤 모수값으로도 표현되지 않는다** — 이 실패는
자료가 늘어도 사라지지 않으며, 그런 경우 기억 없는 채점이 더 낫다.

자유도 규율 (강제됨)
--------------------
이력이 길어지는 속도와 같은 속도로 모수를 풀면 곡선이 이력을 완벽 보간해
잔차분산이 0 이 되고, 예측분산이 붕괴해 사전분포가 뾰족해진다. 그 결과는
이 연구에서 관측된 최악의 조건이었다(RMSE 0.21, 보고 SD 가 실제 오차의
1/7). 따라서 해제 일정은 **항상 k <= H - 1** 을 지키며, 잔차분산은 사후
SD 가 함의하는 측정오차를 하한으로 갖는다. 이 규율은 선택 인자가 아니라
`fit` 내부에 고정되어 있다.

    H <= 2 : gamma                      (k=1)
    H == 3 : gamma, delta               (k=2)
    H == 4 : + alpha                    (k=3)
    H >= 5 : + beta                     (k=4)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from . import core
from .hctm import p2_hctm

__all__ = [
    "curve", "fit", "predict", "to_prior", "sequential_score",
    "PROCESS_SD", "COHORT_DEFAULT",
]

PROCESS_SD = 0.35          # 과정 잡음 규모 (사전분포 폭의 기저)
COHORT_DEFAULT = (1.2, 2.0, 0.95)   # (alpha, beta, delta) 코호트 초기값


def curve(t, alpha, beta, gamma, delta):
    """4모수 HCTM 성장곡선.

    Parameters
    ----------
    t : array_like
        시간. 자연 단위 그대로 (정규화 불필요 — 지평 상수가 없다).
    alpha : float   학습 속도
    beta : float    성장 시기 (변곡 위치)
    gamma : float   진입 수준 theta(0)
    delta : float   수렴 수준. ``delta < gamma`` 면 감쇠 곡선.

    Returns
    -------
    ndarray
    """
    return gamma + (delta - gamma) * p2_hctm(np.asarray(t, dtype=float),
                                             alpha, beta)


def _schedule(H):
    """이력 길이 H 에서 풀어줄 모수 개수. 항상 k <= H - 1 (H >= 2)."""
    if H <= 2:
        return 1
    if H == 3:
        return 2
    if H == 4:
        return 3
    return 4


def fit(times, theta, sd, cohort=None):
    """학습자별 성장곡선 적합 — 자유도 규율이 내장되어 있다.

    Parameters
    ----------
    times : (H,) array_like        관측 시점
    theta : (n, H) array_like      회차별 theta 점추정 (사후평균 권장)
    sd : (n, H) array_like         회차별 사후 SD. 적합의 가중치이자
                                   잔차분산의 하한으로 쓰인다.
    cohort : tuple, optional       (alpha, beta, delta) 코호트 기준값.
                                   이력이 짧아 개인별로 못 푸는 모수를 메운다.

    Returns
    -------
    dict
        ``params``     (n, 4) — 열 순서 (alpha, beta, gamma, delta)
        ``k``          이번 적합에서 실제로 푼 모수 개수
        ``resid_var``  (n,) 자유도 보정 잔차분산 (측정오차 하한 적용 후)
        ``ok``         (n,) 최적화 수렴 여부. False 면 마지막 관측으로 대체됨.
    """
    ts = np.asarray(times, dtype=float)
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    s = np.atleast_2d(np.asarray(sd, dtype=float))
    n, H = th.shape
    if s.shape != th.shape:
        raise ValueError(f"sd shape {s.shape} != theta shape {th.shape}")
    if ts.shape[0] != H:
        raise ValueError(f"times 길이 {ts.shape[0]} != 이력 길이 {H}")

    a0, b0, d0 = cohort or COHORT_DEFAULT
    k = _schedule(H)
    params = np.zeros((n, 4))
    rvar = np.zeros(n)
    ok = np.ones(n, dtype=bool)

    for i in range(n):
        y, si = th[i], np.maximum(s[i], 0.02)
        g_init = float(np.clip(y[0], 0.01, 0.95))
        d_init = float(np.clip(y[-1] + 0.2, 0.05, 0.99))

        if H == 1:
            params[i] = (a0, b0, y[0], d0)
            rvar[i] = float(si[0] ** 2)
            continue

        if k == 1:                       # gamma
            def f(p, tt, _a=a0, _b=b0, _d=d0):
                return curve(tt, _a, _b, p[0], _d)
            x0, lo, hi = [g_init], [0.001], [0.98]
        elif k == 2:                     # + delta
            def f(p, tt, _a=a0, _b=b0):
                return curve(tt, _a, _b, p[0], p[1])
            x0 = [g_init, d_init]
            lo, hi = [0.001, 0.01], [0.98, 0.999]
        elif k == 3:                     # + alpha
            def f(p, tt, _b=b0):
                return curve(tt, p[2], _b, p[0], p[1])
            x0 = [g_init, d_init, a0]
            lo, hi = [0.001, 0.01, 0.05], [0.98, 0.999, 20.0]
        else:                            # + beta
            def f(p, tt):
                return curve(tt, p[2], p[3], p[0], p[1])
            x0 = [g_init, d_init, a0, b0]
            lo, hi = [0.001, 0.01, 0.05, 0.05], [0.98, 0.999, 20.0, 30.0]

        try:
            r = least_squares(lambda p: (f(p, ts) - y) / si,
                              x0, bounds=(lo, hi), max_nfev=300)
            full = {1: (a0, b0, r.x[0], d0),
                    2: (a0, b0, r.x[0], r.x[1]),
                    3: (r.x[2], b0, r.x[0], r.x[1]),
                    4: (r.x[2], r.x[3], r.x[0], r.x[1])}[k]
            params[i] = full
            resid = f(r.x, ts) - y
            # 자유도 보정: H - k >= 1 이 보장되므로 0 으로 나뉘지 않는다
            rv = float((resid ** 2).sum()) / max(H - k, 1)
            floor = float((si ** 2).mean()) / H       # 측정오차 하한
            rvar[i] = max(rv, floor)
        except Exception:                 # 수렴 실패 → 마지막 관측 유지
            params[i] = (a0, b0, y[-1], y[-1])
            rvar[i] = float(si[-1] ** 2)
            ok[i] = False

    return {"params": params, "k": k, "resid_var": rvar, "ok": ok}


def predict(times, theta, sd, t_next, cohort=None):
    """다음 회차의 예측 평균과 예측 SD — 상태공간의 예측 단계.

    예측 SD 는 세 항의 합이다: 자유도 보정 잔차분산(모수 수에 따른 팽창
    계수 1 + k/H 포함), 외삽 거리에 비례하는 항, 과정 잡음. 순수한 적합
    오차만 쓰면 예측분산이 과소평가된다.

    Returns
    -------
    (mean, sd) : 각각 (n,) ndarray
    """
    ts = np.asarray(times, dtype=float)
    th = np.atleast_2d(np.asarray(theta, dtype=float))
    s = np.atleast_2d(np.asarray(sd, dtype=float))
    n, H = th.shape

    res = fit(ts, th, s, cohort)
    P, k, rv = res["params"], res["k"], res["resid_var"]

    mean = np.array([curve(t_next, P[i, 0], P[i, 1], P[i, 2], P[i, 3])
                     for i in range(n)], dtype=float)

    if H == 1:
        psd = np.sqrt(s[:, 0] ** 2 + PROCESS_SD ** 2 * 0.25)
    else:
        gap = float(t_next - ts[-1])
        psd = np.sqrt(rv * (1.0 + k / H)
                      + (0.03 * gap) ** 2
                      + (PROCESS_SD * 0.15) ** 2)
        # 수렴 실패자는 마지막 관측 + 여유
        bad = ~res["ok"]
        if bad.any():
            mean[bad] = th[bad, -1]
            psd[bad] = s[bad, -1] + 0.05

    return np.clip(mean, 0.005, 0.995), np.clip(psd, 0.02, 0.5)


def to_prior(mean, sd, nodes, quad_w):
    """theta 척도의 (평균, SD) 예측을 구적 격자 위 사전분포 가중치로 옮긴다.

    logit 정규분포를 쓴다 — theta 가 [0,1] 에 갇혀 있으므로 정규분포를
    직접 얹으면 경계에서 질량이 새어나간다.

    Parameters
    ----------
    mean, sd : (n,) array_like    예측 평균과 SD (theta 척도)
    nodes : (K,) ndarray          `core.make_grid` 의 노드
    quad_w : (K,) ndarray         Gauss-Legendre 가중치(사전분포 미포함)

    Returns
    -------
    (n, K) ndarray — 행마다 합이 1
    """
    m = np.clip(np.asarray(mean, dtype=float), 2e-3, 1 - 2e-3)
    s = np.asarray(sd, dtype=float)
    lg_nodes = np.log(nodes / (1.0 - nodes))
    s_lg = np.clip(s / (m * (1.0 - m)), 0.05, 5.0)
    m_lg = np.log(m / (1.0 - m))
    z = (lg_nodes[None, :] - m_lg[:, None]) / s_lg[:, None]
    dens = norm.pdf(z) / (s_lg[:, None] * nodes * (1.0 - nodes))
    W = dens * quad_w
    return W / W.sum(axis=1, keepdims=True)


def sequential_score(responses, items, alpha, beta, gamma=None,
                     times=None, memory="hctm", n_nodes=41,
                     prior=(2.0, 2.0), cohort=None):
    """회차별 응답을 순차적으로 채점한다 — 예측 단계 포함.

    문항모수는 **앵커되어 있다고 가정한다**. 회차마다 재캘리브레이션하면
    척도가 함께 움직여 성장과 구별되지 않는다.

    Parameters
    ----------
    responses : list of (n, K_t) 배열     회차별 0/1 응답
    items : list of (K_t,) 정수배열       회차별 투입 문항의 뱅크 색인
    alpha, beta : (J,) array_like         앵커된 문항모수 (뱅크 전체)
    gamma : (J,) array_like, optional     3모수인 경우
    times : (T,) array_like, optional     시점. 기본값 0, 1, ..., T-1
    memory : {"hctm", "none"}             "none" 이면 매 회차 모집단 사전분포
                                          (기억 없음 기준선)
    Returns
    -------
    dict
        ``theta``  (T, n) 회차별 사후평균
        ``sd``     (T, n) 회차별 사후 SD
        ``k``      (T,) 각 회차에서 성장모형이 푼 모수 개수 (memory="none"이면 0)
    """
    if memory not in ("hctm", "none"):
        raise ValueError("memory 는 'hctm' 또는 'none'")

    T = len(responses)
    if len(items) != T:
        raise ValueError("responses 와 items 의 길이가 다르다")
    ts = np.arange(T, dtype=float) if times is None \
        else np.asarray(times, dtype=float)

    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gam = None if gamma is None else np.asarray(gamma, dtype=float)

    nodes, w_pop = core.make_grid(n_nodes, *prior)
    quad_w = 0.5 * np.polynomial.legendre.leggauss(n_nodes)[1]

    n = np.asarray(responses[0]).shape[0]
    cur = np.tile(w_pop, (n, 1))
    TH, SD, KS = [], [], []

    for t in range(T):
        idx = np.asarray(items[t], dtype=int)
        Yt = np.asarray(responses[t], dtype=float)
        g_t = None if gam is None else gam[idx]
        P = np.clip(core.p2_naive(nodes[:, None], alpha[idx], beta[idx])
                    if g_t is None else
                    core.p3_naive(nodes[:, None], alpha[idx], beta[idx], g_t),
                    1e-12, 1 - 1e-12)
        ll = Yt @ np.log(P).T + (1.0 - Yt) @ np.log(1.0 - P).T
        ll -= ll.max(axis=1, keepdims=True)
        post = np.exp(ll) * cur
        post /= post.sum(axis=1, keepdims=True)

        m = post @ nodes
        v = post @ (nodes ** 2) - m ** 2
        s = np.sqrt(np.maximum(v, 1e-12))
        TH.append(m)
        SD.append(s)

        if t == T - 1:
            KS.append(0)
            break

        if memory == "none":
            cur = np.tile(w_pop, (n, 1))
            KS.append(0)
        else:
            mn, sn = predict(ts[:t + 1], np.array(TH).T, np.array(SD).T,
                             float(ts[t + 1]), cohort)
            cur = to_prior(mn, sn, nodes, quad_w)
            KS.append(_schedule(t + 1))

    return {"theta": np.array(TH), "sd": np.array(SD), "k": np.array(KS)}
