"""
strategy.py - the decision maker.

Pipeline for every decision
  1. Build (or reuse, per street) a RangeMatrix: N sampled opponent hands x
     M runouts -> our equity vs each hand, and each hand's own strength.
  2. Infer the opponent's RANGE: weight every sampled hand by the likelihood
     of the actions the opponent actually took this hand. The likelihood
     curves are LEARNED online from this opponent's showdowns (what strength
     did they really have when they checked / called / bet half / bet pot on
     each street?). A maniac's bets barely move the range; a nit's bets
     narrow it to the nuts.
  3. Learn this opponent's fold / re-raise frequencies against our bets.
  4. Choose the action with the highest expected value in chips:
     fold, check/call, half-pot raise, pot raise - each scored with the
     inferred range, the learned fold equity, re-raise risk and an equity
     realisation factor for streets still to come.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from game import Action, Street
from . import engine

F, CC, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
PRE, FLOP, TURN, RIVER = Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER

# action categories for the opponent model
CHECK, CALL, BET_H, BET_P = 0, 1, 2, 3

DEFAULT_PARAMS = dict(
    # prior mean strength of hands taking each action, per street
    # (strength = equity vs a random hand at that street)
    m_pre=(0.45, 0.56, 0.66, 0.70),
    m_post=(0.45, 0.62, 0.70, 0.74),
    prior_n=6.0,
    # fold-equity priors (folds to our bet): [pre/post][level 0/1+][half/pot]
    fe_pre=((0.30, 0.36), (0.22, 0.28)),
    fe_post=((0.36, 0.44), (0.28, 0.34)),
    fe_n=5.0,
    rr_prior=0.22,             # P(opponent re-raises | continues)
    rr_n=5.0,
    rho=0.85,                  # how strength-sorted opponent folds are
    # equity realisation factors
    r_pre_ip=0.98, r_pre_oop=0.90,
    r_flop_ip=0.97, r_flop_oop=0.90,
    r_turn_ip=0.98, r_turn_oop=0.93,
    raise_bias=0.0,            # aggression knob (A/B: >0 beats fixed bots but is exploitable by adaptive ones)
    min_ess=25.0,
    # range-aware fold model: opponent folds hands below a learned strength
    # threshold (per street and bet size) -> fold prob = share of their
    # CURRENT inferred range below it
    fe_model="thr",
    thr_prior=(0.30, 0.37),    # prior threshold for half / pot bets
    thr_n=6.0,
    thr_beta=1.0,
    # sample sizes
    n_flop=220, m_flop=24, n_turn=200, m_turn=27, n_river=400,
    # Was 0.55. Measured on a 2-core container, 0.55 overruns the 8 s time
    # bank on a machine ~3x slower than this one, and the tournament VM shares
    # 96 cores among many bots. 0.24 stays under the bank even at 4x slower
    # (7.2 s of think time per half) and costs nothing measurable in strength.
    budget_frac=0.24,
    # Check pricing. Value a first-to-act check by what follows it: with the
    # learned probability that this opponent bets, we must fold or call that
    # bet, instead of assuming free equity realisation. Without it, a bot that
    # bets every time we check beat us. Always on (look_on = look_full = 0):
    # a gate at 55% let a copy of us stab just below it; always-on tested best.
    look=1.0,
    look_on=0.0,
    look_full=0.0,
    xb_prior=0.45,             # prior P(opponent bets | we checked first)
    xb_n=6.0,
    # Over budget: keep tiny matrices (they mostly fit inside the judge's free
    # 1 ms per request) until this share of the bank, then tables only.
    # Tables-only ignores the board on the flop/turn and loses ~2 bb/deal.
    # Keep this <= 0.45: on a contended CPU the judge charges ~1.4 ms of
    # per-request overhead we can't see, and 0.55 reached 6.0 s of the 8 s bank.
    hard_frac=0.45,
)


# ---------------------------------------------------------------------------
_TUNED = None


def _tuned_params():
    """Tuned parameters from brain/data/params.json, if present.

    offline/tune.py writes its winner to offline/best_params.json, but nothing
    was reading it, so a tuning run could never reach the bot that actually
    plays. Copy (or symlink) that file to brain/data/params.json and it takes
    effect; a missing or broken file leaves DEFAULT_PARAMS untouched.
    """
    global _TUNED
    if _TUNED is None:
        _TUNED = {}
        try:
            import json
            import os as _os
            path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 "data", "params.json")
            if _os.path.exists(path):
                with open(path) as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    _TUNED = {k: v for k, v in loaded.items() if k in DEFAULT_PARAMS}
        except Exception:
            _TUNED = {}
    return _TUNED


def _envf(name, default):
    """Float from the environment, or `default` if unset or unparsable."""
    try:
        v = os.environ.get(name)
        return float(default) if v is None else float(v)
    except Exception:
        return float(default)


def new_half(config, memory, params=None):
    p = dict(DEFAULT_PARAMS)
    p.update(_tuned_params())
    if params:
        p.update(params)
    # Keep the tuned raise bias: deleting it lost to the published bot.
    # In a pot of 10 chips or less that bias is a large share of the pot,
    # so those bets are wider than the strength curve. Pay a chip to call them.
    # Env overrides are for local A/B tests only. A bad value must never raise
    # here: an exception in new_half drops the bot into the fallback strategy.
    p["raise_bias"] = _envf("HM_BIAS", p["raise_bias"])
    p["call_edge"] = _envf("HM_CALL", 1.0)
    p["call_pot"] = _envf("HM_CALL_POT", 10.0)
    # Aurora's blueprint bluffs the river. We were folding 91% of rivers.
    p["river_call"] = _envf("HM_RIVER_CALL", 0.0)
    p["turn_call"] = _envf("HM_TURN_CALL", 0.0)
    p["open_tax"] = _envf("HM_OPEN_TAX", 0.0)
    p["pf_fold"] = _envf("HM_PF_FOLD", 0.0)
    p["look"] = _envf("HM_LOOK", p["look"])
    p["look_on"] = _envf("HM_LOOK_ON", p["look_on"])
    p["look_full"] = _envf("HM_LOOK_FULL", p["look_full"])
    p["budget_frac"] = _envf("HM_BUDGET", p["budget_frac"])
    memory["p"] = p
    memory["rng"] = np.random.default_rng(20261108)
    memory["tb"] = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
    memory["nh"] = int(getattr(config, "num_hands", 1000) or 1000)
    memory["bb"] = int(getattr(config, "big_blind", 2) or 2)
    memory["used"] = 0.0
    # opponent action model: (street, cat) -> [sum_strength, count]
    memory["am"] = {}
    # opponent bet frequency: (street, ctx) -> [bets, opportunities], see _ctx
    memory["xb"] = {}
    # fold equity: (is_pre, level, size) -> [folds, n]; rr: [reraises, continues]
    memory["fe"] = {}
    memory["rr"] = [0.0, 0.0]
    memory["thr"] = {}
    memory["cur"] = None
    memory["stats"] = {"hands": 0, "sd": 0}


# ---------------------------------------------------------------------------
def _cat(oa) -> int:
    if oa.action == RH:
        return BET_H
    if oa.action == RP:
        return BET_P
    if oa.action == CC:
        return CALL if oa.amount > 0 else CHECK
    return -1


def _prior_m(p, street, cat):
    return (p["m_pre"] if street == PRE else p["m_post"])[cat]


def _mean_strength(mem, street, cat):
    p = mem["p"]
    st = 0 if street == PRE else int(street)
    s, n = mem["am"].get((st, cat), (0.0, 0.0))
    n0 = p["prior_n"]
    return (s + n0 * _prior_m(p, street, cat)) / (n + n0)


def _ctx(history, i):
    """Context of the opponent action history[i] on a post-flop street:
    'x' = it answers our check, 'f' = it is the first action of the street,
    None = anything else (it faces our bet)."""
    oa = history[i]
    for j in range(i - 1, -1, -1):
        prev = history[j]
        if prev.street != oa.street:
            break
        if prev.is_me and prev.action == CC and prev.amount == 0:
            return "x"
        return None
    return "f"


def _bet_freq(mem, street, ctx):
    p = mem["p"]
    b, n = mem["xb"].get((int(street), ctx), (0.0, 0.0))
    return (b + p["xb_n"] * p["xb_prior"]) / (n + p["xb_n"])


def _likelihood(mem, street, cat, s):
    """Likelihood of the opponent taking `cat` given strength array s."""
    return _lk_from_m(_mean_strength(mem, street, cat), s)


def _lk_from_m(m, s):
    """Likelihood curve over strengths s for an action whose takers have
    mean strength m (None = flat, the action says nothing)."""
    m = min(max(m, 0.12), 0.93)
    if m >= 0.5:
        k = min((2 * m - 1) / (1 - m), 10.0)
        if k < 1e-3:
            return None
        return np.power(np.maximum(s, 1e-3), k)
    j = min((1 - 2 * m) / m, 4.0)
    if j < 1e-3:
        return None
    return np.power(np.maximum(1.0 - s, 1e-3), j)


def _pre_exponent(mem, history):
    """Preflop range as an exponent k on percentile (for the eq tables)."""
    k = 0.0
    for oa in history:
        if oa.is_me or oa.street != PRE:
            continue
        c = _cat(oa)
        if c < 0:
            continue
        m = min(max(_mean_strength(mem, PRE, c), 0.12), 0.93)
        if m >= 0.5:
            k += (2 * m - 1) / (1 - m)
        else:
            k -= 0.5 * (1 - 2 * m) / m       # weak signal, can't go below 0 anyway
    return max(0.0, k)


def _fold_prob(mem, is_pre, level, size):
    p = mem["p"]
    lv = 0 if level <= 0 else 1
    pri = (p["fe_pre"] if is_pre else p["fe_post"])[lv][size]
    f, n = mem["fe"].get((is_pre, lv, size), (0.0, 0.0))
    n0 = p["fe_n"]
    return (f + n0 * pri) / (n + n0)


def _rr_prob(mem):
    p = mem["p"]
    r, n = mem["rr"]
    return (r + p["rr_n"] * p["rr_prior"]) / (n + p["rr_n"])


GRID = np.linspace(0.0, 1.0, 41)


def _cdf(s, w):
    """Weighted CDF of strengths s at GRID points: P(s < g)."""
    order = np.argsort(s)
    ss = s[order]
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    tot = cw[-1]
    if tot <= 0:
        return GRID.copy()
    return cw[np.searchsorted(ss, GRID, side="left")] / tot


def _threshold(mem, street, size):
    p = mem["p"]
    rec = mem["thr"].get((int(street), size))
    n0 = p["thr_n"]
    t0 = p["thr_prior"][size]
    if rec is None:
        A = np.zeros_like(GRID)
        k = 0.0
    else:
        A, _, k = rec
    pred = A + n0 * GRID
    target = k + n0 * t0
    return float(np.interp(target, pred, GRID))


# ---------------------------------------------------------------------------
def _temper(w, min_ess):
    """Flatten weights until the effective sample size is acceptable."""
    if w is None:
        return None
    for _ in range(6):
        sw = w.sum()
        if sw <= 0:
            return None
        ess = sw * sw / max((w * w).sum(), 1e-30)
        if ess >= min_ess:
            return w
        w = np.sqrt(w)
    return w


def _weighted_quantile(vals, w, q):
    order = np.argsort(vals)
    cw = np.cumsum(w[order])
    tot = cw[-1]
    if tot <= 0:
        return 0.5
    i = int(np.searchsorted(cw, q * tot))
    i = min(max(i, 0), len(order) - 1)
    return float(vals[order[i]])


# ---------------------------------------------------------------------------
def _sample_scale(mem, obs):
    """Time-bank governor: shrink sample sizes when we're spending too fast."""
    p = mem["p"]
    budget = mem["tb"] * p["budget_frac"]
    used = mem["used"]
    left_hands = max(mem["nh"] - obs.hand_index, 1)
    if used > budget * 1.35:
        if used < mem["tb"] * p["hard_frac"]:
            return 0.1                               # minimal matrices (60 x 8-10)
        return 0.0                                   # emergency: tables only
    per_hand = (budget - used) / left_hands
    # a typical hand with postflop play costs ~4 ms at scale 1.0
    scale = per_hand / 0.004
    return float(min(1.0, max(0.25, scale))) if used < budget else 0.25


def _matrix(mem, obs, scale):
    key = (obs.hand_index, int(obs.street))
    cur = mem["cur"]
    if cur is not None and cur.get("mkey") == key:
        return cur["m"]
    p = mem["p"]
    nb = len(obs.board)
    if nb == 3:
        n, m = p["n_flop"], p["m_flop"]
    elif nb == 4:
        n, m = p["n_turn"], p["m_turn"]
    else:
        n, m = p["n_river"], 1
    n = max(60, int(n * scale))
    if nb < 5:
        m = max(8, int(m * scale)) if nb == 3 else (m if scale >= 0.6 else max(10, int(m * scale)))
    mat = engine.build_matrix(obs.hole_cards, obs.board, n, m, mem["rng"])
    if cur is not None:
        cur["mkey"] = key
        cur["m"] = mat
    return mat


def _postflop_weights(mem, obs, mat):
    w = None
    for oa in obs.action_history:
        if oa.is_me:
            continue
        c = _cat(oa)
        if c < 0:
            continue
        s = mat.pre_n if oa.street == PRE else mat.s_n
        lk = _likelihood(mem, oa.street, c, s)
        if lk is None:
            continue
        w = lk if w is None else w * lk
    if w is None:
        return np.ones(mat.N, dtype=np.float64)
    w = _temper(w.astype(np.float64), mem["p"]["min_ess"])
    if w is None:
        return np.ones(mat.N, dtype=np.float64)
    return w


# ---------------------------------------------------------------------------
def _realisation(p, obs, ip):
    st = obs.street
    if st == RIVER:
        return 1.0
    if st == PRE:
        return p["r_pre_ip"] if ip else p["r_pre_oop"]
    if st == FLOP:
        return p["r_flop_ip"] if ip else p["r_flop_oop"]
    return p["r_turn_ip"] if ip else p["r_turn_oop"]


def _decide(obs, mem):
    p = mem["p"]
    la = obs.legal_actions
    if len(la) == 1:
        return la[0]
    P = obs.pot
    Cc = obs.to_call
    my_stack, opp_stack = obs.my_stack, obs.opp_stack
    ip = obs.is_button          # button acts last post-flop
    R = _realisation(p, obs, ip)
    is_pre = obs.street == PRE

    scale = _sample_scale(mem, obs)

    # ---- equity estimates: overall, vs continuing range, vs re-raising range
    f_by_size = {}
    for a in (RH, RP):
        if a in la:
            f_by_size[a] = _fold_prob(mem, is_pre, obs.raises_this_street, 0 if a == RH else 1)
    rho = p["rho"]

    if is_pre:
        k = _pre_exponent(mem, obs.action_history)
        eq = engine.preflop_equity(obs.hole_cards, k)
        eqc, eqr = {}, {}
        for a, f in f_by_size.items():
            kc = k + rho * 2 * f / max(1 - f, 0.05)
            eqc[a] = engine.preflop_equity(obs.hole_cards, kc)
            eqr[a] = engine.preflop_equity(obs.hole_cards, kc + 3.0)
    elif scale <= 0.0:
        # emergency mode: no matrices, cheap strength proxy
        eq = engine.preflop_pct(obs.hole_cards) if len(obs.board) < 5 else 0.5
        if len(obs.board) == 5:
            eq = engine.river_percentile(obs.hole_cards, obs.board, 80, mem["rng"])
        eqc = {a: eq * 0.85 for a in f_by_size}
        eqr = {a: eq * 0.7 for a in f_by_size}
    else:
        mat = _matrix(mem, obs, scale)
        w = _postflop_weights(mem, obs, mat)
        eq = mat.equity(w)
        eqc, eqr = {}, {}
        s = mat.s_n
        use_thr = p["fe_model"] == "thr"
        if use_thr and f_by_size:
            cdf = _cdf(s, w)
            mem["cur"]["cdf"] = (cdf, int(obs.street))
            beta = p["thr_beta"]
            for a in list(f_by_size):
                size = 0 if a == RH else 1
                t = _threshold(mem, obs.street, size)
                ft = float(np.interp(t, GRID, cdf))
                f_by_size[a] = min(0.95, max(0.02, beta * ft + (1 - beta) * f_by_size[a]))
        for a, f in f_by_size.items():
            q = _weighted_quantile(s, w, f)
            cont = (1 - rho) + rho * (s >= q)
            wc = w * cont
            eqc[a] = mat.equity(wc)
            wr = _temper(wc * np.power(np.maximum(s, 1e-3), 3.0), p["min_ess"])
            eqr[a] = mat.equity(wr) if wr is not None else eqc[a] * 0.8

    # ---- EVs (chips, relative to folding now)
    ev = {}
    if F in la:
        ev[F] = 0.0
    c = min(Cc, my_stack)
    refund = max(0, Cc - my_stack)
    all_in_call = c >= my_stack or opp_stack == 0
    Rc = 1.0 if all_in_call else R
    ev[CC] = Rc * eq * (P + c - refund) - c
    if Cc == 0:
        ev[CC] = R * eq * P
    elif p.get("call_edge", 0.0) and (p.get("call_pot", 0) <= 0 or P <= p["call_pot"]):
        # Shift close calls. Positive calls more, negative folds more.
        ev[CC] += p["call_edge"]
    if Cc > 0 and obs.street == RIVER and p.get("river_call", 0.0):
        ev[CC] += p["river_call"]
    # Aurora 3-bets half-pot hundreds of times. Calling those and folding
    # later is the chip leak. A positive pf_fold makes the preflop call worse.
    if Cc > obs.big_blind and obs.street == PRE and p.get("pf_fold", 0.0):
        ev[CC] -= p["pf_fold"]
    if Cc > 0 and obs.street == TURN and p.get("turn_call", 0.0):
        ev[CC] += p["turn_call"]

    # ---- check vs a frequent stabber: one-step lookahead (first to act only)
    if (p["look"] > 0 and Cc == 0 and not is_pre and scale > 0.0
            and opp_stack > 0 and my_stack > 0
            and not any(oa.street == obs.street for oa in obs.action_history)):
        fb = _bet_freq(mem, obs.street, "x")
        act = min(1.0, max(0.0, (fb - p["look_on"]) / max(p["look_full"] - p["look_on"], 1e-6)))
        if act > 0:
            B = min(P, opp_stack, my_stack)                 # assume a pot-sized stab
            # their stab range is no stronger than the top-fb slice
            m_b = min(_mean_strength(mem, obs.street, BET_P), 1.0 - 0.5 * fb)
            lk = _lk_from_m(m_b, mat.s_n)
            wb = w if lk is None else _temper(w * lk, p["min_ess"])
            eq_b = mat.equity(wb) if wb is not None else eq
            Rb = 1.0 if B >= my_stack else R
            ev_resp = max(0.0, Rb * eq_b * (P + 2 * B) - B)   # fold or call the stab
            ev_look = (1 - fb) * R * eq * P + fb * ev_resp
            ev[CC] = (1 - act) * ev[CC] + act * ev_look

    rr = _rr_prob(mem)
    for a in (RH, RP):
        if a not in la:
            continue
        T = obs.raise_to[a]
        X = T - obs.my_bet                      # chips we add
        Y = min(T - obs.opp_bet, opp_stack)     # chips opp adds to call
        f = f_by_size[a]
        allin = (X >= my_stack) or (Y >= opp_stack)
        Rr = 1.0 if allin else R
        pot_called = P + X + Y
        ev_called = Rr * eqc[a] * pot_called - X
        # re-raise branch (only possible if neither is all-in and cap allows)
        can_rr = (not allin) and (obs.raises_this_street + 1 < 4)
        if can_rr:
            pot_after = P + X
            to_call_opp = T - obs.opp_bet
            T2 = min(T + pot_after + to_call_opp, obs.opp_bet + opp_stack)
            extra = min(T2 - T, my_stack - X)
            pot_rr = P + X + (T - obs.opp_bet) + 2 * extra
            ev_rr = max(-X, eqr[a] * pot_rr - (X + extra))
            ev_cont = (1 - rr) * ev_called + rr * ev_rr
        else:
            ev_cont = ev_called
        ev[a] = f * P + (1 - f) * ev_cont + p["raise_bias"]
        if obs.street == PRE and Cc == 0 and p.get("open_tax", 0.0):
            ev[a] -= p["open_tax"]

    best = max(ev, key=lambda k_: ev[k_])
    cur = mem["cur"]
    if cur is not None:
        cur["last_eq"] = eq
    return best


# ---------------------------------------------------------------------------
def _start_hand(mem, obs):
    mem["cur"] = {"hi": obs.hand_index, "hole": tuple(obs.hole_cards), "hist": (),
                  "pend": None, "my_last": None, "my_last_street": None,
                  "my_last_tocall": 0}


def _resolve_pending(mem, obs):
    cur = mem["cur"]
    pend = cur.get("pend") if cur else None
    if not pend:
        return
    idx = pend["idx"]
    h = obs.action_history
    # our raise sits at idx; opponent's answer at idx + 1
    if len(h) > idx + 1 and not h[idx + 1].is_me:
        ans = h[idx + 1]
        key = pend["key"]
        rec = mem["fe"].setdefault(key, [0.0, 0.0])
        rec[1] += 1.0
        _thr_update(mem, pend, ans.action == F)
        if ans.action == F:
            rec[0] += 1.0
        else:
            mem["rr"][1] += 1.0
            if ans.action in (RH, RP):
                mem["rr"][0] += 1.0
        cur["pend"] = None


def _thr_update(mem, pend, folded):
    cdf = pend.get("cdf")
    if cdf is None:
        return
    key = (int(pend["street"]), pend["size"])
    rec = mem["thr"].get(key)
    if rec is None:
        rec = mem["thr"][key] = [np.zeros_like(GRID), 0.0, 0.0]
    rec[0] = rec[0] + cdf
    rec[1] += 1.0
    if folded:
        rec[2] += 1.0


def choose_action(obs, mem):
    t0 = time.perf_counter()
    try:
        cur = mem.get("cur")
        if cur is None or cur.get("hi") != obs.hand_index:
            _start_hand(mem, obs)
        _resolve_pending(mem, obs)
        a = _decide(obs, mem)
        cur = mem["cur"]
        cur["hist"] = tuple(obs.action_history)
        cur["my_last"] = a
        cur["my_last_street"] = obs.street
        cur["my_last_tocall"] = obs.to_call
        cur["board"] = tuple(obs.board)
        if a in (RH, RP):
            cur["pend"] = {"idx": len(obs.action_history),
                           "key": (obs.street == PRE, 0 if obs.raises_this_street <= 0 else 1,
                                   0 if a == RH else 1),
                           "street": obs.street,
                           "cdf": (cur.get("cdf") or (None, None))[0]
                           if (cur.get("cdf") or (None, -1))[1] == int(obs.street) else None,
                           "size": 0 if a == RH else 1}
        else:
            cur["pend"] = None
    finally:
        mem["used"] = mem.get("used", 0.0) + (time.perf_counter() - t0)
    return a


def on_hand_end(result, mem):
    t0 = time.perf_counter()
    try:
        _hand_end(result, mem)
    finally:
        mem["used"] = mem.get("used", 0.0) + (time.perf_counter() - t0)


def _hand_end(result, mem):
    cur = mem.get("cur")
    mem["stats"]["hands"] += 1
    if cur is None or cur.get("hi") != result.hand_index:
        mem["cur"] = None
        return
    # unresolved raise -> opponent folded or called
    pend = cur.get("pend")
    if pend:
        key = pend["key"]
        rec = mem["fe"].setdefault(key, [0.0, 0.0])
        rec[1] += 1.0
        _thr_update(mem, pend, result.winner == "me" and not result.went_to_showdown)
        if result.winner == "me" and not result.went_to_showdown:
            rec[0] += 1.0
        else:
            mem["rr"][1] += 1.0
    _count_bet_freq(mem, cur, result)
    if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
        mem["stats"]["sd"] += 1
        sd = mem["stats"]["sd"]
        # plenty of evidence already: learn from every 3rd showdown only (saves time)
        if sd <= 150 or sd % 3 == 0:
            _learn_showdown(result, mem, cur)
    mem["cur"] = None


def _count_bet_freq(mem, cur, result):
    """Count how often the opponent bets when we check to them ('x') and when
    they act first on a street ('f'), from the history we saw this hand."""
    xb = mem.setdefault("xb", {})
    hist = cur.get("hist") or ()
    for i, oa in enumerate(hist):
        if oa.is_me or oa.street == PRE:
            continue
        cx = _ctx(hist, i)
        if cx is None:
            continue
        rec = xb.setdefault((int(oa.street), cx), [0.0, 0.0])
        rec[1] += 1.0
        if oa.action in (RH, RP):
            rec[0] += 1.0
    # We checked first on the river and the hand went to showdown with no
    # reply in our history: the opponent checked behind.
    if (result.went_to_showdown and cur.get("my_last") == CC
            and cur.get("my_last_tocall", 1) == 0 and cur.get("my_last_street") == RIVER
            and not any(oa.street == RIVER for oa in hist)):
        rec = xb.setdefault((int(RIVER), "x"), [0.0, 0.0])
        rec[1] += 1.0


def _learn_showdown(result, mem, cur):
    opp = tuple(result.opp_hole)
    board = tuple(result.board)
    acts = [(oa.street, _cat(oa)) for oa in cur["hist"] if not oa.is_me]
    # infer the opponent's final action that we never saw
    my_last = cur.get("my_last")
    if my_last in (RH, RP):
        acts.append((cur["my_last_street"], CALL))
    elif my_last == CC and cur.get("my_last_tocall", 0) == 0 and cur.get("my_last_street") == RIVER:
        acts.append((RIVER, CHECK))
    acts = [(s, c) for s, c in acts if c >= 0]
    if not acts:
        return
    p = mem["p"]
    budget = mem["tb"] * p["budget_frac"]
    cheap = mem["used"] > budget * 0.8
    cache = {}
    rng = mem["rng"]
    for st, c in acts:
        st = int(st)
        if st not in cache:
            if st == 0:
                v = engine.preflop_pct(opp)
            elif st == 3 or cheap:
                if 3 not in cache:
                    cache[3] = engine.river_percentile(opp, board, 160, rng)
                v = cache[3]
            else:
                nb = 3 if st == 1 else 4
                v = engine.strength_of(opp, board[:nb], 90, 12 if nb == 3 else 27, rng)
            cache[st] = v
        rec = mem["am"].setdefault((st, c), [0.0, 0.0])
        rec[0] += cache[st]
        rec[1] += 1.0
