"""
tables.py - precomputed lookup tables shared by the runtime and offline tools.

The key structure is T2: a (630 x 7140) uint16 table giving the hand score
(dense rank, higher = better) of "hole pair p + board triple t". Any Omaha
hand value is then max over 10 hole pairs x 10 board triples, and - more
importantly - for a FIXED 5-card board we can compute the best value of
every one of the 630 possible hole pairs in one numpy gather. An opponent's
5-card holding is then just max over its 10 pairs. This makes thousands of
range-vs-hand evaluations per millisecond possible in pure numpy.
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

NC = 36

# ---- pair / triple ids ----------------------------------------------------
PAIRS = np.array(list(combinations(range(NC), 2)), dtype=np.int16)        # (630, 2)
TRIPLES = np.array(list(combinations(range(NC), 3)), dtype=np.int16)      # (7140, 3)
PID = np.full((NC, NC), -1, dtype=np.int32)
for _i, (_a, _b) in enumerate(PAIRS):
    PID[_a, _b] = _i
    PID[_b, _a] = _i
TID = np.full((NC, NC, NC), -1, dtype=np.int32)
for _i, (_a, _b, _c) in enumerate(TRIPLES):
    for _x, _y, _z in ((_a, _b, _c), (_a, _c, _b), (_b, _a, _c), (_b, _c, _a), (_c, _a, _b), (_c, _b, _a)):
        TID[_x, _y, _z] = _i

# The 10 (2-of-5) and 10 (3-of-5) index patterns.
P25 = np.array(list(combinations(range(5), 2)), dtype=np.int64)   # (10, 2)
T35 = np.array(list(combinations(range(5), 3)), dtype=np.int64)   # (10, 3)

# Binomial table for colex ranking of 5-card sets.
BINOM = np.zeros((NC + 1, 6), dtype=np.int64)
for _n in range(NC + 1):
    BINOM[_n, 0] = 1
    for _k in range(1, 6):
        BINOM[_n, _k] = 0 if _n == 0 else BINOM[_n - 1, _k - 1] + BINOM[_n - 1, _k]
N5 = int(BINOM[NC, 5])  # 376,992


def colex5(sorted_cards: np.ndarray) -> np.ndarray:
    """Colex index of sorted 5-card sets, shape (..., 5) -> (...)."""
    s = sorted_cards.astype(np.int64)
    return (BINOM[s[..., 0], 1] + BINOM[s[..., 1], 2] + BINOM[s[..., 2], 3]
            + BINOM[s[..., 3], 4] + BINOM[s[..., 4], 5])


def _build_t2() -> np.ndarray:
    """Build T2 from the official evaluator (reference scoring)."""
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from evaluator import get_table  # official scores
    table = get_table()
    # dense-rank all distinct scores
    scores = np.array(sorted(set(table.values())), dtype=np.int64)
    dense = {int(s): i + 1 for i, s in enumerate(scores)}  # 0 reserved = invalid
    # score per colex index
    s5 = np.zeros(N5, dtype=np.uint16)
    for mask, sc in table.items():
        cards = [c for c in range(NC) if mask >> c & 1]
        idx = sum(int(BINOM[c, i + 1]) for i, c in enumerate(cards))
        s5[idx] = dense[sc]
    # T2[p, t]
    pa = PAIRS.astype(np.int64)
    t2 = np.zeros((len(PAIRS), len(TRIPLES)), dtype=np.uint16)
    tr = TRIPLES.astype(np.int64)
    for p in range(len(PAIRS)):
        five = np.concatenate([np.broadcast_to(pa[p], (len(tr), 2)), tr], axis=1)
        five.sort(axis=1)
        ok = (five[:, 1:] != five[:, :-1]).all(axis=1)
        idx = np.where(ok, colex5(five), 0)
        row = np.where(ok, s5[idx], 0)
        t2[p] = row
    return t2, s5


def load_t2() -> np.ndarray:
    path = os.path.join(DATA, "t2.npy")
    if os.path.exists(path):
        try:
            t2 = np.load(path)
            if t2.shape == (len(PAIRS), len(TRIPLES)):
                return t2
        except Exception:
            pass
    t2, _ = _build_t2()
    return t2


if __name__ == "__main__":
    import time
    t = time.time()
    t2, s5 = _build_t2()
    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "t2.npy"), t2)
    np.save(os.path.join(DATA, "s5.npy"), s5)
    print("built", t2.shape, t2.max(), "in", time.time() - t)
