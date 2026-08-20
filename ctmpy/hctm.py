"""반절단 CTM (Half-truncated CTM, HCTM) — 지지집합 [0, inf).

논문 CTM 은 P(0)=0 과 P(1)=1 두 조건을 걸어 두 상수를 정했다.
0 에서만 자르면 조건이 하나(P(0)=0)로 줄고, 위쪽은 로지스틱이 원래 갖는
점근선 1 을 그대로 쓴다. 필요한 상수도 하나로 줄어든다.

    P(theta) = [sigma(a(theta-b)) - sigma(-a b)] / sigma(a b)

정리하면
    P(theta) = expm1(a*theta) / [exp(a*b) * (1 + exp(a*(theta-b)))]

논문 식 (13) 과 몸통이 같고 정규화 상수만 다르다:
    숙달점 L 일반형 :  expm1(a th)/(1+exp(a(th-b))) * (1+exp(a(L-b)))/expm1(a L)
      L = 1   -> 논문 CTM
      L -> inf-> exp(-a b)  = 본 모형

즉 논문 CTM 은 이 함수족의 L=1 인 특수 경우다.

성질:
  P(0) = 0,  P(inf) = 1,  theta 에 대해 단조증가
  P(b) = (1 - exp(-a b)) / 2   -> a*b 가 크면 0.5 에 접근
  theta 는 절대영점(무지 수준)을 갖되 상한이 없다 = 비율척도적 해석
"""

from __future__ import annotations

import numpy as np

__all__ = ["p2_hctm", "p3_hctm", "p2_ctm_L", "dp_hctm"]


def _log_expm1(u):
    """log(exp(u)-1) 을 안정적쳸으로. u 가 크면 u + log1p(-exp(-u))."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    big = u > 30.0
    out[big] = u[big] + np.log1p(-np.exp(-u[big]))
    sm = ~big
    out[sm] = np.log(np.expm1(np.maximum(u[sm], 1e-300)))
    return out


def _softplus(u):
    return np.logaddexp(0.0, u)


def p2_hctm(theta, alpha, beta):
    """2모수 반절단 CTM. theta in [0, inf).

        P = expm1(a*th) / [exp(a*b) * (1 + exp(a*(th-b)))]

    로그공간에서 계산해 오버플로우를 피한다:
        log P = log_expm1(a*th) - a*b - softplus(a*(th-b))
    """
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    u = alpha * theta
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = _log_expm1(u) - alpha * beta - _softplus(alpha * (theta - beta))
    P = np.exp(lp)
    return np.where(theta <= 0.0, 0.0, np.clip(P, 0.0, 1.0))


def p3_hctm(theta, alpha, beta, gamma):
    """3모수 반절단 CTM. gamma 는 theta=0 에서의 정답확률."""
    gamma = np.asarray(gamma, dtype=float)
    return gamma + (1.0 - gamma) * p2_hctm(theta, alpha, beta)


def p2_ctm_L(theta, alpha, beta, L):
    """숙달점 L 을 명시한 일반형. L=1 이면 논문 CTM, L->inf 면 HCTM.

    주의: exp(a*L) 을 포함하므로 a*L > 709 에서 오버플로우한다.
    유도·서술용이며 실제 계산은 L=1 이면 ctm.p2_naive, L=inf 면 p2_hctm 을 쓴다.
    """
    theta = np.asarray(theta, dtype=float)
    core_num = np.expm1(alpha * theta)
    core_den = 1.0 + np.exp(alpha * (theta - beta))
    const = (1.0 + np.exp(alpha * (L - beta))) / np.expm1(alpha * L)
    return core_num / core_den * const


def dp_hctm(theta, alpha, beta):
    """dP/dtheta — 정보함수와 뉴턴 갱신용 해석적 도함수.

    log P = log(expm1(a th)) - a b - softplus(a(th-b)) 이므로
    dlogP/dth = a*exp(a th)/expm1(a th) - a*sigmoid(a(th-b))

    참고: 2장 개정에서 정규화 형태 P' = (a/D) s(1-s) 를 얻었으며 그 쪽이
    theta=0 특이점이 없다. 본 함수는 전개형 기반이라 theta<=0 분기를 둔다.
    """
    theta = np.asarray(theta, dtype=float)
    u = alpha * theta
    P = p2_hctm(theta, alpha, beta)
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = alpha / (-np.expm1(-u))          # a*exp(u)/expm1(u) = a/(1-exp(-u))
    term2 = alpha / (1.0 + np.exp(-alpha * (theta - beta)))
    return np.where(theta <= 0.0, 0.0, P * (term1 - term2))


if __name__ == "__main__":
    from ctmpy import core as ctm

    print("[1] 항등식 — P(0)=0, P(inf)=1, 단조증가")
    A = [0.5, 1.0, 3.0, 10.0]
    B = [0.2, 0.5, 1.0, 3.0]
    th = np.linspace(0, 60, 20001)
    ok0 = ok1 = mono = True
    for a in A:
        for b in B:
            p = p2_hctm(th, a, b)
            ok0 &= abs(p[0]) < 1e-15
            ok1 &= abs(p[-1] - 1.0) < 1e-9
            mono &= np.diff(p).min() >= -1e-13
    print(f"  P(0)=0 {ok0}   P(큰 theta)->1 {ok1}   단조증가 {mono}")

    print("\n[2] P(beta) = (1 - exp(-a*b))/2 인가")
    for a, b in [(1.0, 0.5), (3.0, 1.0), (10.0, 2.0), (0.5, 0.2)]:
        lhs = float(p2_hctm(b, a, b))
        rhs = (1 - np.exp(-a * b)) / 2
        print(f"  a={a:5.1f} b={b:4.1f}   P(b)={lhs:.10f}   공식={rhs:.10f}   "
              f"차이={abs(lhs-rhs):.2e}")

    print("\n[3] 일반형 L 극한 — L=1 이 논문 CTM, L->inf 가 HCTM 인가")
    a, b = 5.0, 0.4
    t = np.array([0.1, 0.3, 0.5, 0.8])
    d1 = np.abs(p2_ctm_L(t, a, b, 1.0) - ctm.p2_naive(t, a, b)).max()
    print(f"  L=1 vs 논문 p2_naive : 최대차 {d1:.3e}")
    for L in [5.0, 20.0, 50.0, 100.0]:
        d = np.abs(p2_ctm_L(t, a, b, L) - p2_hctm(t, a, b)).max()
        print(f"  L={L:6.0f} vs HCTM     : 최대차 {d:.3e}")

    print("\n[4] 도함수 검증 — 수치미분과 대조")
    h = 1e-6
    for a, b in [(1.0, 0.5), (3.0, 1.0), (10.0, 2.0)]:
        for t0 in [0.2, 1.0, 3.0]:
            num = (p2_hctm(t0 + h, a, b) - p2_hctm(t0 - h, a, b)) / (2 * h)
            ana = float(dp_hctm(t0, a, b))
            print(f"  a={a:5.1f} b={b:4.1f} th={t0:4.1f}  해석 {ana:.8f}  "
                  f"수치 {float(num):.8f}  상대차 {abs(ana-num)/max(abs(num),1e-12):.2e}")

    print("\n[5] 수치 안정성 — 큰 alpha*theta 에서 오버플로우 없는가")
    for a, t0 in [(10.0, 100.0), (50.0, 200.0), (100.0, 500.0)]:
        v = float(p2_hctm(t0, a, 1.0))
        print(f"  a={a:6.1f} theta={t0:6.1f}  P={v:.12f}  유한={np.isfinite(v)}")
