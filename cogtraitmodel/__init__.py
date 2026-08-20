"""cogtraitmodel — Cognitive Trait Model 계열의 추정·채점·정보함수 구현.

인지적 특성 theta 를 유계 [0, L] 연속체에 놓는 **Anchored Logistic Family (ALF)**
를 다룬다. 숙달 앵커 L 이 가족을 지표화한다:

    L = 1     Cognitive Trait Model (Choi, 2022) — theta 는 "숙달의 몇 %"
    L -> inf  Half-truncated CTM (HCTM) — 절대영점은 있고 천장은 없음

핵심 설계
---------
* 추정은 MCMC 가 아니라 **Gauss-Legendre 구적 위의 Bayes modal EM**.
  41노드면 배정밀도 한계에 도달하며, 수렴한 랜덤워크 표본기 대비 35배,
  NUTS 대비 73배 빠르면서 theta 상관 0.9999 로 일치한다.
* 채점 기본값은 **MAP 점추정 + EAP 사후 SD**. Fisher 정보 기반 SEM 은
  경계에서 0 으로 붕괴하므로 보고용으로 쓰지 않는다.
* MLE 는 진단용이며 경계해(theta_hat = 0 또는 L)는 특성 수준으로 보고하지
  않는다 — 유한하고 "숙달 100%"로 읽히기 때문에 IRT 의 무한대 발산보다
  위험하다.

빠른 사용
---------
    >>> import numpy as np, cogtraitmodel
    >>> theta = cogtraitmodel.gen_theta(500, rng=np.random.default_rng(0))
    >>> alpha = np.full(20, 8.0); beta = np.linspace(0.1, 0.9, 20)
    >>> Y = cogtraitmodel.gen_responses(theta, alpha, beta,
    ...                         rng=np.random.default_rng(1))
    >>> fit = cogtraitmodel.bayes_modal_em(Y)          # 문항모수 캘리브레이션
    >>> out = cogtraitmodel.score(Y, fit["alpha"], fit["beta"])
    >>> out["theta"], out["sd"]        # MAP 점추정 + 사후 SD

임의의 유한 앵커 L 은 `fit_L` / `score_L` 로 다룬다 (척도 재조정으로 L=1 에
환원되므로 별도 추정기가 아니다). 시간축 위의 성장·순차 갱신은
`cogtraitmodel.growth` — L -> inf 구성원이 그곳에서 쓰인다.

정확성 기준
-----------
`p1_naive` / `p2_naive` / `p3_naive` 는 Choi (2022) 식 (12)-(14) 의 문자
그대로의 전사판이며 영구 보존된다. 최적화된 구현은 alpha in [0.1, 100] 에서
이들과 1e-12 이내로 일치함을 단언하는 테스트로 고정된다.

참고문헌
--------
Choi, J. (2022). Cognitive Trait Model: Measurement model for mastery level
and progression of learning. *Mathematics*, 10(15), 2651.
doi:10.3390/math10152651
"""

from __future__ import annotations

import numpy as np

from . import datasets, growth, hctm, info, irt, scale
from .core import (
    bayes_modal_em,
    eap,
    eap_batch,
    gen_responses,
    gen_theta,
    make_grid,
    map_theta,
    mle_theta,
    p1_naive,
    p2_naive,
    p3_naive,
    posterior,
)
from .hctm import dp_hctm, p2_ctm_L, p2_hctm, p3_hctm
from .info import info_table, item_info, sem, tif
from .scale import fit_L, score_L

__version__ = "0.1.1"

__all__ = [
    # 링크함수 (L = 1, 문헌 전사판 = 정확성 기준)
    "p1_naive", "p2_naive", "p3_naive",
    # 링크함수 (일반 L, L -> inf)
    "p2_ctm_L", "p2_hctm", "p3_hctm", "dp_hctm",
    # 구적·사후분포
    "make_grid", "posterior", "posterior_sd",
    # 채점
    "eap", "eap_batch", "map_theta", "mle_theta", "score",
    # 캘리브레이션
    "bayes_modal_em",
    # 임의 앵커 L (유한) — 척도 재조정 래퍼
    "fit_L", "score_L",
    # 정보함수
    "item_info", "tif", "sem", "info_table",
    # 모의생성
    "gen_theta", "gen_responses",
    # 하위 모듈
    "hctm", "info", "irt", "datasets", "growth", "scale",
    "__version__",
]


def posterior_sd(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """구적 격자에서 계산한 사후표준편차 — 권장 불확실성 지표.

    Fisher 정보 기반 SEM(`sem`)은 theta 가 경계에 가까울 때 0 으로 붕괴하는데,
    이는 P(L) = 1 로 고정한 데서 오는 인공물이지 정밀도에 대한 진술이 아니다.
    사후 SD 는 그런 병리가 없다.

    Returns
    -------
    ndarray, shape (n,)
    """
    _, sd = eap(Y, alpha, beta, gamma, n_nodes, prior)
    return sd


def score(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """권장 채점 경로 — MAP 점추정과 EAP 사후 SD 를 함께 돌려준다.

    두 값이 같은 구적 격자를 공유하므로 둘 중 하나만 구하는 것과 비용이
    사실상 같다.

    Returns
    -------
    dict
        ``theta``    MAP 점추정 (권장 보고값)
        ``sd``       사후 표준편차 (권장 불확실성)
        ``eap``      EAP 점추정 (참고용, 축소가 더 큼)
        ``at_bound`` 최빈값이 격자 끝에서 잡힌 응답자 (bool). MAP 에서는
                     통상 전부 False 이며, True 가 있으면 조사가 필요하다.
    """
    m, sd = eap(Y, alpha, beta, gamma, n_nodes, prior)
    theta, at_bound = map_theta(Y, alpha, beta, gamma, prior)
    return {"theta": theta, "sd": sd, "eap": m, "at_bound": at_bound}
