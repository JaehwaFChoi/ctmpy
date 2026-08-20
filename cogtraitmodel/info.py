"""정보함수 — 앵커드 로지스틱 족의 문항정보·TIF·SEM.

원 논문이 향후 과제로 명시한 부분(Discussion: TCC, IIF, TIF, SEM).

유도
  log P2 = log(expm1(a*th)) - softplus(a*(th-b)) + const
  정규화 상수는 theta 에 무관하므로 미분에서 사라진다. 따라서 L=1(CTM)과
  L=inf(HCTM)이 **같은 도함수 공식**을 공유한다.

      dlogP2/dth = a/(1 - exp(-a*th)) - a*sigmoid(a*(th-b))
      P2'        = P2 * a * [1/(1-exp(-a*th)) - sigmoid(a*(th-b))]
      P3'        = (1-gamma) * P2'

  이분 반응의 Fisher 정보량
      I(th) = [P'(th)]^2 / (P(th) * (1-P(th)))

[2장 개정 반영] 정규화 형태 P = [s(th) - s(-ab)] / D 를 쓰면
      P' = (a/D) * s * (1-s),   s = sigmoid(a(th-b))
로 더 간단하며 **theta=0 특이점이 없다**. 본 모듈은 전개형 기반이라
_dp_at_zero 분기를 두지만, 두 형태는 수치적으로 이치한다(8자리).

경계 거동 (본 모듈의 핵심 결과)
  분자 s(1-s) <= 1/4 이므로 P' 는 유계다. 따라서 **발산의 원인은 분모 P(1-P)
  하나로 특정된다.**
  th -> L : I ~ P'(L) / (L - th)  — 발산 속도 정확히 1, 잔여항은 P'(L)
  th -> 0 : 2P 는 I ~ P'(0)/th 로 발산, 3P 는 P3(0)=gamma>0 이라 유한
  즉 gamma 는 아래쪽 발산을 막지만 위쪽은 막지 못한다(P(L)=1 이 고정이므로).
  속도가 정확히 (L-th)^{-1} 이므로 정보함수를 적분하면 로그발산 — 총 정보량이
  유한하지 않다. 보고용 오차는 사후 SD 를 쓰고, 정보함수는 문항분석용으로
  [0.02, 0.98] 에서 본다.

[CAT 실험 결과] 경계 발산은 중간 추정치를 EAP/MAP 으로 쓰면 실무에서 물지
않는다(절단이 한 번도 발동하지 않음). MLE 와 결합하면 오히려 유해하다.
study_cat.py 참조.
"""

from __future__ import annotations

import numpy as np

from . import core as ctm
from . import hctm

__all__ = ["dp2_ctm", "dp3_ctm", "item_info", "tif", "sem", "info_table"]


def _dlogp_core(theta, alpha, beta):
    """공통 도함수 핵심항. L=1 과 L=inf 가 공유한다."""
    theta = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t1 = alpha / (-np.expm1(-alpha * theta))          # a/(1-exp(-a*th))
    t2 = alpha / (1.0 + np.exp(-alpha * (theta - beta)))  # a*sigmoid
    return t1 - t2


def dp2_ctm(theta, alpha, beta, family="ctm"):
    """2모수 링크의 dP/dtheta. family='ctm'(L=1) 또는 'hctm'(L=inf)."""
    theta = np.asarray(theta, dtype=float)
    P = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
        else hctm.p2_hctm(theta, alpha, beta)
    d = P * _dlogp_core(theta, alpha, beta)
    return np.where(theta <= 0.0, _dp_at_zero(alpha, beta, family), d)


def _dp_at_zero(alpha, beta, family):
    """theta=0 에서의 극한. P2 ~ C*B(0)*a*th 이므로 P2'(0) = C*B(0)*a."""
    B0 = 1.0 / (1.0 + np.exp(-alpha * beta))
    if family == "ctm":
        C = (1.0 + np.exp(alpha * (1.0 - beta))) / (-np.expm1(alpha))
        return -alpha * C * B0          # C<0 이므로 부호 보정
    return alpha * np.exp(-alpha * beta) * B0


def dp3_ctm(theta, alpha, beta, gamma, family="ctm"):
    """3모수 링크의 dP/dtheta = (1-gamma) * dP2/dtheta."""
    return (1.0 - np.asarray(gamma, dtype=float)) * dp2_ctm(theta, alpha, beta, family)


def item_info(theta, alpha, beta, gamma=None, family="ctm", eps=1e-12):
    """문항정보함수 I(theta) = (P')^2 / (P(1-P))."""
    if gamma is None:
        P = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
            else hctm.p2_hctm(theta, alpha, beta)
        d = dp2_ctm(theta, alpha, beta, family)
    else:
        P2 = ctm.p2_naive(theta, alpha, beta) if family == "ctm" \
            else hctm.p2_hctm(theta, alpha, beta)
        P = gamma + (1.0 - gamma) * P2
        d = dp3_ctm(theta, alpha, beta, gamma, family)
    den = np.clip(P * (1.0 - P), eps, None)
    return d ** 2 / den


def tif(theta, alpha, beta, gamma=None, family="ctm"):
    """검사정보함수 — 문항정보의 합 (국소독립 가정)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    g = None if gamma is None else np.atleast_1d(gamma)
    I = item_info(theta[:, None], np.atleast_1d(alpha), np.atleast_1d(beta),
                  None if g is None else g, family)
    return I.sum(axis=1)


def sem(theta, alpha, beta, gamma=None, family="ctm"):
    """Fisher 정보 기반 표준오차. 경계에서 0 으로 붕괴하므로 보고용으로 쓰지 않는다."""
    return 1.0 / np.sqrt(np.maximum(tif(theta, alpha, beta, gamma, family), 1e-300))


def info_table(alpha, beta, gamma=None, family="ctm", lo=0.02, hi=0.98, n=25):
    """문항분석용 정보표 — 경계를 잘라 표시한다 (설계 결정 posterior_sd_not_fisher)."""
    th = np.linspace(lo, hi, n)
    return th, tif(th, alpha, beta, gamma, family)


if __name__ == "__main__":
    print("[1] 도함수 검증 — 수치미분 대조 (CTM, L=1)")
    h = 1e-6
    for a, b in [(1.0, 0.3), (5.0, 0.5), (10.0, 0.7), (25.0, 0.5)]:
        errs = []
        for t0 in [0.05, 0.2, 0.5, 0.8, 0.95]:
            num = (ctm.p2_naive(t0 + h, a, b) - ctm.p2_naive(t0 - h, a, b)) / (2 * h)
            ana = float(dp2_ctm(t0, a, b))
            errs.append(abs(ana - num) / max(abs(num), 1e-12))
        print(f"  a={a:5.1f} b={b:.1f}  최대 상대차 {max(errs):.2e}")

    print("\n[2] 도함수 검증 — HCTM (L=inf)")
    for a, b in [(1.0, 1.0), (3.0, 2.0), (0.5, 0.5)]:
        errs = []
        for t0 in [0.1, 0.5, 1.5, 3.0]:
            num = (hctm.p2_hctm(t0 + h, a, b) - hctm.p2_hctm(t0 - h, a, b)) / (2 * h)
            ana = float(dp2_ctm(t0, a, b, family="hctm"))
            errs.append(abs(ana - num) / max(abs(num), 1e-12))
        print(f"  a={a:5.1f} b={b:.1f}  최대 상대차 {max(errs):.2e}")

    print("\n[3] theta=0 극한값 검증 (CTM)")
    for a, b in [(5.0, 0.3), (10.0, 0.5), (20.0, 0.8)]:
        lim = float(_dp_at_zero(a, b, "ctm"))
        num = (ctm.p2_naive(1e-7, a, b) - 0.0) / 1e-7
        print(f"  a={a:5.1f} b={b:.1f}  해석 {lim:.6f}  수치 {num:.6f}  "
              f"상대차 {abs(lim-num)/max(abs(num),1e-12):.2e}")

    print("\n[4] 경계에서의 정보량 발산 — 2P vs 3P")
    a, b = 10.0, 0.5
    print(f"{'theta':>8}{'2P I':>14}{'3P I (g=0.2)':>16}")
    for t0 in [1e-4, 1e-3, 0.01, 0.5, 0.99, 0.999, 0.9999]:
        i2 = float(item_info(t0, a, b))
        i3 = float(item_info(t0, a, b, 0.2))
        print(f"{t0:>8.4f}{i2:>14.2f}{i3:>16.2f}")
    print("  -> 2P 는 양쪽 경계에서, 3P 는 위쪽 경계에서만 발산한다")

    print("\n[5] 발산 속도 — I ~ (1-theta)^(-k) 의 k 추정")
    print("    (유한차분 회귀는 편향됨. 해석적 참값은 정확히 k=1, 잔여항 P'(L))")
    for a in [5.0, 10.0, 20.0]:
        ts = 1 - np.array([1e-2, 1e-3, 1e-4, 1e-5])
        Is = np.array([float(item_info(t, a, 0.5)) for t in ts])
        k = np.polyfit(np.log(1 - ts), np.log(Is), 1)[0]
        print(f"  a={a:5.1f}  k = {-k:.3f}")
