"""LSAT Section 6 자료 (Bock & Lieberman, 1970) — 논문 3.2 절의 실증 데이터.

1000명 x 5문항 이분 응답. 32개 응답패턴과 빈도로 주어지는 고전 자료로,
R 의 ltm 패키지 등에 LSAT 로 수록되어 있다.

이 모듈은 패턴표를 펼쳐 (1000, 5) 행렬을 만든다.
build() 는 다음 두 가지를 강제 검증한다:
  - 총 빈도 1000
  - 문항별 정답률이 알려진 값 (0.924, 0.709, 0.553, 0.763, 0.870) 과 이치

[2026-08-10 확정] 권위 있는 32-패턴 빈도표와 **전 항목 대조 완료**.
32개 패턴의 빈도가 모두 정확히 일치함(총합 1000, 패턴 수 32).
아래 PATTERNS 는 재구성본이 아니라 검증된 자료다.
"""

import numpy as np

# (응답패턴, 빈도). 패턴은 문항 1~5 순서.
# 출처: Bock & Lieberman (1970), Psychometrika 35, 179-197, Table 1.
#       R 패키지 ltm 의 LSAT 데이터셋과 동일. 2026-08-10 전 항목 대조 완료.
PATTERNS = [
    ("00000", 3), ("00001", 6), ("00010", 2), ("00011", 11),
    ("00100", 1), ("00101", 1), ("00110", 3), ("00111", 4),
    ("01000", 1), ("01001", 8), ("01010", 0), ("01011", 16),
    ("01100", 0), ("01101", 3), ("01110", 2), ("01111", 15),
    ("10000", 10), ("10001", 29), ("10010", 14), ("10011", 81),
    ("10100", 3), ("10101", 28), ("10110", 15), ("10111", 80),
    ("11000", 16), ("11001", 56), ("11010", 21), ("11011", 173),
    ("11100", 11), ("11101", 61), ("11110", 28), ("11111", 298),
]

# 문헌에 보고된 문항별 정답률 (검증용)
KNOWN_P = np.array([0.924, 0.709, 0.553, 0.763, 0.870])


def build(verify=True):
    """(1000, 5) 0/1 응답행렬을 만든다. verify=True 면 검증에 실패하면 예외."""
    rows = []
    for pat, freq in PATTERNS:
        if freq:
            rows.extend([[int(c) for c in pat]] * freq)
    Y = np.array(rows, dtype=int)

    if verify:
        n = Y.shape[0]
        if n != 1000:
            raise ValueError(f"총 빈도가 1000 이 아니라 {n} 이다")
        p = Y.mean(axis=0)
        if not np.allclose(p, KNOWN_P, atol=5e-4):
            raise ValueError(f"문항 정답률 불일치: {np.round(p, 4)} vs {KNOWN_P}")
    return Y


if __name__ == "__main__":
    Y = build()
    print(f"LSAT 자료 {Y.shape}")
    print(f"  문항별 정답률 {np.round(Y.mean(axis=0), 4)}")
    print(f"  문헌 보고값   {KNOWN_P}")
    print(f"  총점 분포 {np.bincount(Y.sum(axis=1), minlength=6)}")
    print(f"  만점 {(Y.sum(1)==5).sum()}명, 영점 {(Y.sum(1)==0).sum()}명")
