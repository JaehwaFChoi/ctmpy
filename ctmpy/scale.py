"""ctmpy.scale — 임의의 숙달 앵커 L 에서의 캘리브레이션과 채점.

유한 L 은 별도 추정기가 필요 없다
----------------------------------
정규화형에서 시그모이드 인자가 alpha*(theta - beta) = (alpha*L)*(theta/L -
beta/L) 로 재배열되고 분모도 같은 변환을 받으므로, 다음 항등식이 성립한다:

    P_L(theta; alpha, beta) == P_1(theta/L; alpha*L, beta/L)

수치 확인: (alpha, beta, L) 을 (1, 0.5, 2), (5, 0.4, 3), (2, 1.5, 5),
(0.8, 2, 4) 로 두고 theta in (0, 0.99L] 300점에서 최대차 2.2e-16 ~ 2.8e-15.

따라서 L 척도의 추정은 **자료를 theta/L 로 압축해 L=1 추정기에 넣고 나온
모수를 되돌리는 것**과 정확히 같다. 이 모듈은 그 변환을 감춘다.

함의 하나를 명시해 둘 가치가 있다: 같은 응답자료가 모든 유한 L 에서 동등하게
적합되므로, **L 은 자료가 식별하는 모수가 아니라 정의적 선택**이다. 단위를
어디에 두느냐는 재척도화이고, 구조적으로 다른 것은 원점뿐이다.

L -> inf 는 예외다. theta/L -> 0 이므로 환원되지 않으며, 그 경우 HCTM 은
측정 모형이 아니라 시간축 위의 성장 모형으로 쓴다(`ctmpy.growth`).
"""

from __future__ import annotations

import numpy as np

from . import core

__all__ = ["fit_L", "score_L"]


def _check(L):
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError(
            f"L 은 유한 양수여야 한다 (받은 값 {L}). L -> inf 는 측정 모형이 "
            "아니라 성장 모형이며 ctmpy.growth 를 쓴다."
        )
    return L


def fit_L(Y, L, **kwargs):
    """숙달 앵커가 L 인 척도에서 문항모수를 캘리브레이션한다.

    내부적으로 L=1 추정기를 쓰고 모수를 L 척도로 되돌린다.

    Parameters
    ----------
    Y : (n, J) 0/1 응답행렬
    L : float                 숙달 앵커 (유한 양수)
    **kwargs                  `core.bayes_modal_em` 에 그대로 전달

    Returns
    -------
    dict — `bayes_modal_em` 과 같은 구조이되 alpha, beta 가 L 척도.
           ``L`` 키가 추가된다.
    """
    L = _check(L)
    fit = core.bayes_modal_em(Y, **kwargs)
    out = dict(fit)
    out["alpha"] = np.asarray(fit["alpha"], dtype=float) / L
    out["beta"] = np.asarray(fit["beta"], dtype=float) * L
    out["L"] = L
    return out


def score_L(Y, alpha, beta, L, gamma=None, **kwargs):
    """L 척도의 문항모수로 채점한다. theta 는 [0, L] 로 돌아온다.

    Returns
    -------
    dict — `ctmpy.score` 와 같은 키. ``theta``, ``sd``, ``eap`` 가 L 척도.
    """
    from . import score

    L = _check(L)
    a1 = np.asarray(alpha, dtype=float) * L
    b1 = np.asarray(beta, dtype=float) / L
    out = score(Y, a1, b1, gamma, **kwargs)
    return {"theta": out["theta"] * L, "sd": out["sd"] * L,
            "eap": out["eap"] * L, "at_bound": out["at_bound"], "L": L}
