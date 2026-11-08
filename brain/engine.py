"""
engine.py - fast, vectorised equity estimation for Short Deck PLO5.

Core idea ("range matrix"): sample N opponent holdings and M board runouts
once per street, and compute the full N x M matrix of showdown outcomes with
a handful of numpy gathers through the T2 (hole-pair x board-triple) table.
From that one matrix we get, for free:
  * our equity against every sampled opponent hand,
  * every sampled opponent hand's own strength (its equity vs a random hand
    on the same runouts),
so the opponent's RANGE can be re-weighted (by what their actions say about
their strength) without re-simulating anything.
"""

from __future__ import annotations

import os

import numpy as np

from .tables import DATA, N5, P25, PID, T35, TID, colex5, load_t2

T2 = load_t2()                                   # (630, 7140) uint16
PID64 = PID.astype(np.int64)
TID64 = TID.astype(np.int64)
P25_0, P25_1 = P25[:, 0], P25[:, 1]
T35_0, T35_1, T35_2 = T35[:, 0], T35[:, 1], T35[:, 2]
ALL_CARDS = np.arange(36, dtype=np.int64)

# ---- preflop tables ----------------------------------------------------------
_PRE_PCT = None
_PRE_EQK = None
_PRE_KS = None


def _load_pre():
    global _PRE_PCT, _PRE_EQK, _PRE_KS
    path = os.path.join(DATA, "pre.npz")
    z = np.load(path)
    _PRE_PCT = z["pct"].astype(np.float32) / 65535.0       # (N5,)
    _PRE_EQK = z["eqk"].astype(np.float32)                 # (N5, K)
    _PRE_KS = z["ks"].astype(np.float64)                   # (K,)


try:
    _load_pre()
except Exception:  # pragma: no cover - tables missing: degrade gracefully
    _PRE_PCT = None


def have_preflop_tables() -> bool:
    return _PRE_PCT is not None


def hand_index(cards) -> int:
    s = sorted(int(c) for c in cards)
    return int(colex5(np.array(s, dtype=np.int64)))


def preflop_pct(cards) -> float:
    """Strength percentile (0..1) of a 5-card holding, vs all holdings."""
    if _PRE_PCT is None:
        return 0.5
    return float(_PRE_PCT[hand_index(cards)])


def preflop_pct_many(hands_sorted: np.ndarray) -> np.ndarray:
    if _PRE_PCT is None:
        return np.full(len(hands_sorted), 0.5, dtype=np.float32)
    return _PRE_PCT[colex5(hands_sorted)]


def preflop_equity(cards, k: float) -> float:
    """Equity of our holding vs an opponent range weighted by pct**k."""
    if _PRE_PCT is None:
        return 0.5
    row = _PRE_EQK[hand_index(cards)]
    ks = _PRE_KS
    k = float(min(max(k, ks[0]), ks[-1]))
    return float(np.interp(k, ks, row))


# ---- the range matrix ---------------------------------------------------------

class RangeMatrix:
    """Outcome matrix for one decision point (one street of one hand).

    Attributes
      res    (N, M) float32  our result vs opp hand n on runout m (1/0.5/0)
      valid  (N, M) bool     opp hand and runout don't share cards
      eq_n   (N,)   float32  our equity vs each opp hand (over valid runouts)
      s_n    (N,)   float32  opp hand strength = its equity vs a random hand
      pre_n  (N,)   float32  opp hand preflop percentile
    """

    __slots__ = ("res", "valid", "eq_n", "s_n", "pre_n", "N", "M", "opp_hands")

    def equity(self, w: np.ndarray | None = None) -> float:
        if w is None:
            return float(self.eq_n.mean())
        sw = float(w.sum())
        if sw <= 1e-12:
            return float(self.eq_n.mean())
        return float((w * self.eq_n).sum() / sw)


def build_matrix(hole, board, n_opp: int, n_run: int, rng: np.random.Generator) -> RangeMatrix:
    hole = np.array(sorted(int(c) for c in hole), dtype=np.int64)
    board = np.array([int(c) for c in board], dtype=np.int64)
    nb = len(board)
    dead = np.zeros(36, dtype=bool)
    dead[hole] = True
    dead[board] = True
    live = ALL_CARDS[~dead]
    L = len(live)
    r = 5 - nb

    # --- runouts
    if r == 0:
        runs = np.zeros((1, 0), dtype=np.int64)
    elif r == 1:
        runs = live.reshape(-1, 1)                         # exact: all rivers
        if n_run < L:
            runs = runs[rng.choice(L, n_run, replace=False)]
    else:
        idx = np.argpartition(rng.random((n_run, L)), r, axis=1)[:, :r]
        runs = live[idx]
    M = len(runs)
    boards = np.concatenate([np.broadcast_to(board, (M, nb)), runs], axis=1)  # (M,5)

    # --- opponent hands
    idx = np.argpartition(rng.random((n_opp, L)), 5, axis=1)[:, :5]
    opp = np.sort(live[idx], axis=1)                       # (N,5)
    N = n_opp

    # validity masks
    bits = np.left_shift(np.int64(1), ALL_CARDS)
    omask = bits[opp].sum(axis=1)
    if r:
        rmask = bits[runs].sum(axis=1)
        valid = (omask[:, None] & rmask[None, :]) == 0
    else:
        valid = np.ones((N, 1), dtype=bool)

    # --- board triples -> per-runout pair table over the pairs we need
    tids = TID64[boards[:, T35_0], boards[:, T35_1], boards[:, T35_2]]       # (M,10)
    our_p = PID64[hole[P25_0], hole[P25_1]]                                   # (10,)
    opp_p = PID64[opp[:, P25_0], opp[:, P25_1]]                               # (N,10)
    need = np.unique(np.concatenate([our_p, opp_p.ravel()]))
    inv = np.zeros(630, dtype=np.int64)
    inv[need] = np.arange(len(need))
    pt = T2[need[:, None, None], tids[None, :, :]].max(axis=2)               # (R, M)
    ours = pt[inv[our_p]].max(axis=0).astype(np.int32)                        # (M,)
    oppv = pt[inv[opp_p]].max(axis=1).astype(np.int32)                        # (N, M)

    res = (ours[None, :] > oppv).astype(np.float32)
    res += 0.5 * (ours[None, :] == oppv)
    res *= valid
    nv = valid.sum(axis=1).astype(np.float32)
    nv_safe = np.maximum(nv, 1.0)
    eq_n = res.sum(axis=1) / nv_safe

    # opp strength: rank of its value among the valid sampled hands, per runout
    ov = np.where(valid, oppv, -1)
    order = np.argsort(ov, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(N)[:, None], axis=0)
    ninv_col = (~valid).sum(axis=0)
    nval_col = np.maximum(N - ninv_col, 1)
    pct = (ranks - ninv_col[None, :] + 0.5) / nval_col[None, :]
    s_n = (np.where(valid, pct, 0.0).sum(axis=1) / nv_safe).astype(np.float32)

    # hands with no valid runout (tiny probability): neutral
    bad = nv == 0
    if bad.any():
        eq_n[bad] = 0.5
        s_n[bad] = 0.5

    m = RangeMatrix()
    m.res, m.valid, m.eq_n, m.s_n = res, valid, eq_n.astype(np.float32), s_n
    m.pre_n = preflop_pct_many(opp)
    m.N, m.M = N, M
    m.opp_hands = opp
    return m


def strength_of(hole, board, n_cmp: int, n_run: int, rng: np.random.Generator) -> float:
    """Equity of `hole` vs a random hand on `board` (0..5 cards)."""
    if len(board) == 0:
        return preflop_pct(hole)
    m = build_matrix(hole, board, n_cmp, n_run, rng)
    return float(m.eq_n.mean())


def river_percentile(hole, board, n_cmp: int, rng: np.random.Generator) -> float:
    m = build_matrix(hole, board, n_cmp, 1, rng)
    return float(m.eq_n.mean())
