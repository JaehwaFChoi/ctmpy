"""LSAT Section 6 data (Bock & Lieberman, 1970) — the empirical illustration.

1000 examinees by 5 dichotomous items. A classic dataset given as 32 response
patterns with their frequencies, distributed as LSAT in the ltm package for R
among others.

This module expands the pattern table into a (1000, 5) matrix. build()
enforces two checks:
  - the frequencies total 1000
  - the item proportions correct match the published values
    (0.924, 0.709, 0.553, 0.763, 0.870)

[Confirmed 2026-08-10] Checked entry by entry against an authoritative
32-pattern frequency table. All 32 frequencies agree exactly (total 1000,
32 patterns). PATTERNS below is verified data, not a reconstruction.
"""

import numpy as np

# (response pattern, frequency). Patterns run over items 1 to 5.
# Source: Bock & Lieberman (1970), Psychometrika 35, 179-197, Table 1.
#         Identical to the LSAT dataset in the R package ltm.
#         Checked entry by entry on 2026-08-10.
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

# Published item proportions correct, used for verification
KNOWN_P = np.array([0.924, 0.709, 0.553, 0.763, 0.870])


def build(verify=True):
    """Build the (1000, 5) matrix of 0/1 responses.

    With verify=True a failed check raises rather than returning bad data.
    """
    rows = []
    for pat, freq in PATTERNS:
        if freq:
            rows.extend([[int(c) for c in pat]] * freq)
    Y = np.array(rows, dtype=int)

    if verify:
        n = Y.shape[0]
        if n != 1000:
            raise ValueError(f"total frequency is {n} rather than 1000")
        p = Y.mean(axis=0)
        if not np.allclose(p, KNOWN_P, atol=5e-4):
            raise ValueError(f"item proportions disagree: {np.round(p, 4)} vs {KNOWN_P}")
    return Y


if __name__ == "__main__":
    Y = build()
    print(f"LSAT data {Y.shape}")
    print(f"  proportion correct  {np.round(Y.mean(axis=0), 4)}")
    print(f"  published values    {KNOWN_P}")
    print(f"  total-score distribution {np.bincount(Y.sum(axis=1), minlength=6)}")
    print(f"  {(Y.sum(1)==5).sum()} perfect scores, {(Y.sum(1)==0).sum()} zero scores")
