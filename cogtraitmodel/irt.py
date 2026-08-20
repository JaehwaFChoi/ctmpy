"""전통 IRT (2PL / 3PL) — CTM 과의 비교용 최소 구현.

논문 Table 6 (LSAT 자료의 theta 추정치 상관표) 재현을 위해 필요하다.
구조는 ctm.py 와 의도적으로 동일하게 맞췄다. 다른 점은 두 가지뿐:
  - theta 가 무계이므로 Gauss-Hermite 구적을 쓰고 절단 범위가 생긴다
  - 링크 함수가 순수 로지스틱이다

이 차이 자체가 CTM 의 논지이므로, 비교 대상으로 남겨둔다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["p2pl", "p3pl", "make_grid_normal", "eap_irt", "mmle_em_irt"]


def p2pl(theta, a, b):
    """2PL — 논문 식 (3). P = 1 / (1 + exp(-a(th - b)))"""
    z = np.asarray(a, dtype=float) * (np.asarray(theta, dtype=float)
                                      - np.asarray(b, dtype=float))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def p3pl(theta, a, b, c):
    """3PL — 논문 식 (2). P = c + (1 - c) * 2PL"""
    c = np.asarray(c, dtype=float)
    return c + (1.0 - c) * p2pl(theta, a, b)


def make_grid_normal(n_nodes=61, mu=0.0, sd=1.0):
    """N(mu, sd^2) 에 대한 Gauss-Hermite 구적 격자.

    CTM 과 달리 theta 가 무계라 노드가 유한 범위로 잘린다.
    n_nodes=61 이면 대략 +-14.5 까지 덮지만, 그 바깥은 원리적으로 버려진다.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    nodes = mu + sd * x
    return nodes, w / w.sum()


def _loglik_irt(Y, nodes, a, b, c=None):
    Y = np.asarray(Y, dtype=float)
    P = p2pl(nodes[:, None], a, b) if c is None else p3pl(nodes[:, None], a, b, c)
    P = np.clip(P, 1e-12, 1 - 1e-12)
    obs = ~np.isnan(Y)
    Y0 = np.where(obs, Y, 0.0)
    return Y0 @ np.log(P).T + (obs & (Y0 == 0)) @ np.log(1 - P).T


def eap_irt(Y, a, b, c=None, n_nodes=61):
    """EAP 추정치와 사후 SD."""
    nodes, w = make_grid_normal(n_nodes)
    ll = _loglik_irt(Y, nodes, a, b, c)
    ll -= ll.max(axis=1, keepdims=True)
    post = np.exp(ll) * w
    post /= post.sum(axis=1, keepdims=True)
    m = post @ nodes
    v = post @ (nodes ** 2) - m ** 2
    return m, np.sqrt(np.maximum(v, 0.0))


def _item_neg_obj_irt(z, nodes, Nk, rk, three_p, pa, pb, pc):
    a = np.exp(z[0])
    b = z[1]
    c = 1.0 / (1.0 + np.exp(-z[2])) if three_p else None
    if not np.isfinite(a) or a <= 0 or a > 1e3:
        return 1e10

    P = p2pl(nodes, a, b) if not three_p else p3pl(nodes, a, b, c)
    P = np.clip(P, 1e-12, 1 - 1e-12)
    ll = np.sum(rk * np.log(P) + (Nk - rk) * np.log(1 - P))

    ll += -0.5 * ((np.log(a) - pa[0]) / pa[1]) ** 2 - np.log(a)  # a ~ LN
    ll += -0.5 * ((b - pb[0]) / pb[1]) ** 2                      # b ~ N
    if three_p:
        ll += (pc[0] - 1) * np.log(c) + (pc[1] - 1) * np.log(1 - c)
    return -ll if np.isfinite(ll) else 1e10


def mmle_em_irt(Y, three_p=False, n_nodes=61, prior_a=(0.0, 0.5),
                prior_b=(0.0, 2.0), prior_c=(7.0, 25.0),
                max_iter=300, tol=1e-6):
    """2PL/3PL 문항모수를 MMLE + 사전분포로 추정한다. ctm.bayes_modal_em 과 같은 구조."""
    from scipy.optimize import minimize

    Y = np.asarray(Y, dtype=float)
    n, J = Y.shape
    nodes, w = make_grid_normal(n_nodes)

    a = np.ones(J)
    p = np.clip(Y.mean(axis=0), 0.02, 0.98)
    b = -np.log(p / (1 - p))  # 정답률 높으면 쉬운 문항 = 낮은 b
    c = np.full(J, 0.2) if three_p else None

    for it in range(max_iter):
        ll = _loglik_irt(Y, nodes, a, b, c)
        ll -= ll.max(axis=1, keepdims=True)
        post = np.exp(ll) * w
        post /= post.sum(axis=1, keepdims=True)
        Nk = post.sum(axis=0)
        rk = post.T @ Y

        a_new, b_new = a.copy(), b.copy()
        c_new = c.copy() if three_p else None
        for j in range(J):
            z0 = [np.log(a[j]), b[j]]
            if three_p:
                z0.append(np.log(c[j] / (1 - c[j])))
            res = minimize(_item_neg_obj_irt, np.array(z0), method="L-BFGS-B",
                           args=(nodes, Nk, rk[:, j], three_p,
                                 prior_a, prior_b, prior_c))
            a_new[j] = np.exp(res.x[0])
            b_new[j] = res.x[1]
            if three_p:
                c_new[j] = 1.0 / (1.0 + np.exp(-res.x[2]))

        delta = max(np.abs(np.log(a_new) - np.log(a)).max(),
                    np.abs(b_new - b).max(),
                    np.abs(c_new - c).max() if three_p else 0.0)
        a, b, c = a_new, b_new, c_new
        if delta < tol:
            return dict(a=a, b=b, c=c, n_iter=it + 1, converged=True)
    return dict(a=a, b=b, c=c, n_iter=max_iter, converged=False)
