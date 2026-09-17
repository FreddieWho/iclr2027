#!/usr/bin/env python3
"""Shared path contract for last15h routes N01-N10.

One path = parent scene + edit operator; evaluated at identical sample
points by oracle and model. Records parent_id, source_match (if any),
action_family, t values, oracle labels, predicted classes/logits,
edit cost, validity, training-use status.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PACK = Path(__file__).resolve().parents[3] / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
from core.relations import segment_relation  # noqa: E402


def oracle_at(x):
    """(label, margin, ambiguous) for a (4,2) scene."""
    r = segment_relation(np.asarray(x, dtype=float).reshape(4, 2))
    return int(r["label"]), float(r["margin"]), bool(r["ambiguous"])


def scan_linear(x, e, predict_fn, n_scan=17):
    """Scan gamma(t) = x + t*e at n_scan uniform points.

    predict_fn takes (M,4,2) float32 array -> (logits (M,), labels (M,)).
    Returns dict with t grid, oracle labels/margins/ambiguous flags,
    pred labels/logits, edit cost.
    """
    t = np.linspace(0.0, 1.0, n_scan)
    xs = (x[None, :, :] + t[:, None, None] * e[None, :, :]).astype(np.float32)
    olab = np.zeros(n_scan, dtype=int)
    omar = np.zeros(n_scan)
    oamb = np.zeros(n_scan, dtype=bool)
    for i in range(n_scan):
        try:
            l, m, a = oracle_at(xs[i])
        except ValueError:
            l, m, a = -1, 0.0, True
        olab[i], omar[i], oamb[i] = l, m, a
    logits, plab = predict_fn(xs)
    return {"t": t, "oracle_label": olab, "oracle_margin": omar,
            "ambiguous": oamb, "pred_label": np.asarray(plab),
            "logit": np.asarray(logits, dtype=float),
            "cost": float(np.linalg.norm(e))}


def _bisect_turn(x, e, t_lo, t_hi, fn, n_iter=12):
    """Bisect a sign/label change between t_lo and t_hi. fn(t)->scalar or int."""
    flo, fhi = fn(t_lo), fn(t_hi)
    for _ in range(n_iter):
        tm = 0.5 * (t_lo + t_hi)
        fm = fn(tm)
        if (fm > 0) == (fhi > 0) if isinstance(fm, float) else (fm == fhi):
            t_hi, fhi = tm, fm
        else:
            t_lo, flo = tm, fm
    return 0.5 * (t_lo + t_hi)


def locate_oracle_turns(x, e, scan, n_bisect=12):
    """Refine each oracle label-change interval by bisection. Returns list of
    (t_star, old_label, new_label). Ambiguous-adjacent intervals flagged."""
    out = []
    olab = scan["oracle_label"]
    t = scan["t"]
    for i in range(len(t) - 1):
        if olab[i] < 0 or olab[i + 1] < 0:
            continue
        if olab[i] == olab[i + 1]:
            continue
        def fn(tt, _x=x, _e=e):
            try:
                l, _, _ = oracle_at(_x + tt * _e)
            except ValueError:
                l = -1
            return l
        # only bisect clean 0/1 flips
        ts = _bisect_turn(x, e, t[i], t[i + 1],
                          lambda tt: 1.0 if fn(tt) == olab[i + 1] else -1.0,
                          n_iter=n_bisect)
        out.append({"t_star": float(ts), "old": int(olab[i]),
                    "new": int(olab[i + 1]),
                    "near_ambiguous": bool(scan["ambiguous"][i] or scan["ambiguous"][i + 1])})
    return out


def locate_model_turns(x, e, scan, predict_fn, n_bisect=12):
    """Bisect each predicted-label change interval via logit sign. Returns
    list of (t_theta, old_pred, new_pred). Empty list = missing."""
    out = []
    plab = scan["pred_label"]
    t = scan["t"]
    xa = np.asarray(x, dtype=np.float32)

    def logit_at(tt):
        xx = (xa + np.float32(tt) * np.asarray(e, dtype=np.float32))[None]
        lg, _ = predict_fn(xx)
        return float(np.asarray(lg)[0])

    for i in range(len(t) - 1):
        if plab[i] == plab[i + 1]:
            continue
        lo, hi = (logit_at(t[i]), logit_at(t[i + 1]))
        a, b = t[i], t[i + 1]
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            if (logit_at(m) > 0) == (hi > 0):
                b = m
            else:
                a = m
        out.append({"t_theta": float(0.5 * (a + b)), "old": int(plab[i]),
                    "new": int(plab[i + 1])})
    return out


def match_turns(oracle_turns, model_turns, tol=0.02):
    """Match each oracle turn to nearest same-direction model turn.
    Returns list of dicts: status in {ok, early, late, missing} + error."""
    res = []
    for ot in oracle_turns:
        cands = [mt for mt in model_turns
                 if mt["old"] == ot["old"] and mt["new"] == ot["new"]]
        if not cands:
            # wrong-target turns exist but none in right direction
            res.append({"t_star": ot["t_star"], "status": "missing",
                        "error": None, "n_model_turns": len(model_turns)})
            continue
        mt = min(cands, key=lambda m: abs(m["t_theta"] - ot["t_star"]))
        err = mt["t_theta"] - ot["t_star"]
        status = "ok" if abs(err) <= tol else ("early" if err < 0 else "late")
        res.append({"t_star": ot["t_star"], "t_theta": mt["t_theta"],
                    "status": status, "error": float(err),
                    "n_model_turns": len(model_turns)})
    return res


# ---------------- path / edit generators ----------------

def atomic_edits(x, rng, radius=0.10):
    """Named atomic edit families for N04. Returns list of (edit, family)."""
    out = []
    angs = rng.uniform(0, 2 * np.pi, 4)
    for node in range(4):
        e = np.zeros((4, 2))
        e[node] = radius * np.array([np.cos(angs[node]), np.sin(angs[node])])
        out.append((e, f"move_node{node}"))
    for seg, nodes in (("AB", (0, 1)), ("CD", (2, 3))):
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        e = np.zeros((4, 2))
        e[list(nodes)] = radius * d
        out.append((e, f"shift_{seg}"))
    for seg, nodes in (("AB", (0, 1)), ("CD", (2, 3))):
        cen = x[list(nodes)].mean(axis=0)
        ang = rng.choice([0.15, -0.15])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = x[list(nodes)]
        e = np.zeros((4, 2))
        e[list(nodes)] = pts @ R.T - pts
        out.append((e, f"rot_{seg}"))
    return out


def _seg_closest(p1, p2, p3, p4):
    """Closest points between segments p1p2 and p3p4. Returns (q_ab, q_cd)."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a < 1e-12 or e < 1e-12:
        return p1, p3
    c = float(d1 @ r)
    b = float(d1 @ d2)
    den = a * e - b * b
    s = 0.0 if den < 1e-12 else float(np.clip((b * f - c * e) / den, 0, 1))
    t = float(np.clip((b * s + f) / e, 0, 1))
    s2 = float(np.clip((b * t - c) / a, 0, 1))
    return p1 + s2 * d1, p3 + t * d2


def _check_episode(x, e, n_dense=129, min_dwell=5, min_mid_margin=0.01):
    """Dense oracle scan; returns episode (i0,j0,t) or None, or 'multi'/'none'."""
    t = np.linspace(0, 1, n_dense)
    labs = np.zeros(n_dense, dtype=int)
    mars = np.zeros(n_dense)
    for i in range(n_dense):
        try:
            l, m, _ = oracle_at(x + t[i] * e)
        except ValueError:
            return None
        labs[i], mars[i] = l, m
    if labs[0] != 0:
        return None
    eps, i = [], 0
    while i < n_dense:
        if labs[i] == 1 and (i == 0 or labs[i - 1] == 0):
            j = i
            while j + 1 < n_dense and labs[j + 1] == 1:
                j += 1
            if j + 1 < n_dense and labs[j + 1] == 0:
                eps.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(eps) != 1:
        return None
    i0, j0 = eps[0]
    if (j0 - i0 + 1) < min_dwell or mars[i0:j0 + 1].min() < min_mid_margin:
        return None
    return i0, j0, t


def gen_event_paths(parents, rng, n_want=256, n_dense=129, seed=0,
                    min_dwell=5, min_mid_margin=0.01):
    """N03: aimed CD/AB sweeps with 0->1->0 oracle episodes.

    Direction aims the moving segment at the other segment (closest-point
    axis), lengths overshoot the gap so the sweep enters and exits.
    Geometric rule first, model never consulted.
    """
    rng = np.random.default_rng(seed)
    events, controls = [], []
    order = rng.permutation(len(parents))
    for pi in order:
        if len(events) >= n_want and len(controls) >= n_want:
            break
        x = np.asarray(parents[pi], dtype=float)
        try:
            if oracle_at(x)[0] != 0:
                continue
        except ValueError:
            continue
        q_ab, q_cd = _seg_closest(x[0], x[1], x[2], x[3])
        gap = float(np.linalg.norm(q_ab - q_cd))
        if gap < 1e-6:
            continue
        axis = (q_ab - q_cd) / gap
        made_event = False
        # try moving CD toward/through AB, then AB toward/through CD
        for nodes, tgt in (((2, 3), 1.0), ((0, 1), -1.0)):
            if made_event:
                break
            d = axis * tgt
            for over in (0.15, 0.30, 0.45):
                e = np.zeros((4, 2))
                e[list(nodes)] = (gap + over) * d
                r = _check_episode(x, e, n_dense, min_dwell, min_mid_margin)
                if r is not None:
                    i0, j0, t = r
                    events.append({
                        "parent_id": int(pi), "edit": e,
                        "action_family": f"aimed_shift_{'CD' if nodes == (2, 3) else 'AB'}",
                        "ep_start": float(t[i0]), "ep_end": float(t[j0])})
                    made_event = True
                    break
        if made_event:
            continue
        # control: sweep perpendicular (should stay negative)
        if len(controls) < n_want:
            perp = np.array([-axis[1], axis[0]])
            e = np.zeros((4, 2))
            e[2, :] = 0.3 * perp
            e[3, :] = 0.3 * perp
            t = np.linspace(0, 1, n_dense)
            try:
                labs = [oracle_at(x + tt * e)[0] for tt in t]
            except ValueError:
                continue
            if all(l == 0 for l in labs):
                controls.append({"parent_id": int(pi), "edit": e,
                                 "action_family": "perp_sweep_control"})
    return events, controls
    return events, controls


def scan_polyline(waypoints, predict_fn, n_per_leg=33):
    """Scan a piecewise-linear path. waypoints: list of (4,2) arrays.
    Returns same dict form with t in [0,1] over the whole polyline."""
    legs = []
    for k in range(len(waypoints) - 1):
        s = np.linspace(0, 1, n_per_leg)
        legs.append((1 - s)[:, None, None] * waypoints[k][None] +
                    s[:, None, None] * waypoints[k + 1][None])
    xs = np.concatenate(legs, axis=0).astype(np.float32)
    n = len(xs)
    t = np.linspace(0, 1, n)
    olab = np.zeros(n, dtype=int)
    omar = np.zeros(n)
    oamb = np.zeros(n, dtype=bool)
    for i in range(n):
        try:
            l, m, a = oracle_at(xs[i])
        except ValueError:
            l, m, a = -1, 0.0, True
        olab[i], omar[i], oamb[i] = l, m, a
    logits, plab = predict_fn(xs)
    return {"t": t, "oracle_label": olab, "oracle_margin": omar,
            "ambiguous": oamb, "pred_label": np.asarray(plab),
            "logit": np.asarray(logits, dtype=float), "cost": None}


def locate_turns_xt(x_of_t, scan, is_oracle, predict_fn=None, n_bisect=12):
    """Turn localization on a generic x(t) callable. is_oracle: bisect on
    oracle labels; else on predicted-logit sign (needs predict_fn)."""
    out = []
    t = scan["t"]
    if is_oracle:
        seq = scan["oracle_label"]

        def val(tt):
            try:
                l, _, _ = oracle_at(x_of_t(tt))
            except ValueError:
                l = -1
            return l
    else:
        seq = scan["pred_label"]

        def val(tt):
            xx = np.asarray(x_of_t(tt), dtype=np.float32)[None]
            lg, _ = predict_fn(xx)
            return float(np.asarray(lg)[0])
    for i in range(len(t) - 1):
        if seq[i] < 0 or seq[i + 1] < 0 or seq[i] == seq[i + 1]:
            continue
        a, b = t[i], t[i + 1]
        if is_oracle:
            tgt = seq[i + 1]
            lo, hi = a, b
            for _ in range(n_bisect):
                m = 0.5 * (lo + hi)
                if (1.0 if val(m) == tgt else -1.0) > 0:
                    hi = m
                else:
                    lo = m
            out.append({"t_star": float(0.5 * (lo + hi)), "old": int(seq[i]),
                        "new": int(seq[i + 1])})
        else:
            lo, hi = (val(a), val(b))
            aa, bb = a, b
            for _ in range(n_bisect):
                m = 0.5 * (aa + bb)
                if (val(m) > 0) == (hi > 0):
                    bb = m
                else:
                    aa = m
            out.append({"t_theta": float(0.5 * (aa + bb)), "old": int(seq[i]),
                        "new": int(seq[i + 1])})
    return out


def action_library():
    """N06 fixed candidate library: translate AB/CD in 8 dirs x 3 radii + noop."""
    lib = []
    angs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for seg, nodes in (("AB", (0, 1)), ("CD", (2, 3))):
        for r in (0.05, 0.10, 0.16):
            for a in angs:
                e = np.zeros((4, 2))
                e[list(nodes)] = r * np.array([np.cos(a), np.sin(a)])
                lib.append({"edit": e, "cost": float(r * np.sqrt(2)),
                            "family": f"shift_{seg}"})
    lib.append({"edit": np.zeros((4, 2)), "cost": 0.0, "family": "noop"})
    return lib
