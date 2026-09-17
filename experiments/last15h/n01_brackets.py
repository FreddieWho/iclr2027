#!/usr/bin/env python3
"""N01 round2: two-sided bracket training (v1 bilateral labels, no order yet).

Train paths: aimed single-turn 0->1 paths on train parents, truncated
mid-episode. Each path contributes x-, x+ (delta safety), far points.
Control: flipmine with the SAME number of labeled states.
Eval: single-turn 0->1 paths on eval parents (path-integral error =
|tau_theta - tau_star|), miss|flip, clean error. 1->0 reported as transfer.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
import paths
from paths import oracle_at, scan_linear, locate_oracle_turns, locate_model_turns
from common import load_model, preprocess
from coord_mlp import CoordMLP

ART = ROOT / "artifacts" / "discovery_campaign"
DELTA = 0.05


def aimed_single_turns(X, rng, n_want, want_label0=True, seed=0):
    """Aimed sweeps ending inside the flipped region -> single-turn paths.
    No exit required: the path ends at label 1 (or 0 for reverse).
    Exactly one oracle turn verified by rescan."""
    rng = np.random.default_rng(seed)
    out = []
    want = 0 if want_label0 else 1
    for pi in rng.permutation(len(X)):
        if len(out) >= n_want:
            break
        x = np.asarray(X[pi], dtype=float)
        try:
            if oracle_at(x)[0] != want:
                continue
        except ValueError:
            continue
        q_ab, q_cd = paths._seg_closest(x[0], x[1], x[2], x[3])
        gap = float(np.linalg.norm(q_ab - q_cd))
        if gap < 1e-6:
            continue
        axis = (q_ab - q_cd) / gap
        for nodes, tgt in (((2, 3), 1.0), ((0, 1), -1.0)):
            if len(out) >= n_want:
                break
            d = axis * tgt
            for over in (0.10, 0.20, 0.35, 0.55, 0.80):
                e = np.zeros((4, 2))
                e[list(nodes)] = (gap + over) * d
                try:
                    y_end, m_end, _ = oracle_at(x + e)
                except ValueError:
                    continue
                if y_end == want or m_end < 0.005:
                    continue
                sc = scan_linear(x, e, lambda xs: (np.zeros(len(xs)), np.zeros(len(xs), dtype=int)), n_scan=17)
                ot = locate_oracle_turns(x, e, sc)
                if len(ot) != 1:
                    continue
                out.append({"parent_id": int(pi), "edit": e, "t_star": float(ot[0]["t_star"]),
                            "fam": f"aimed_{'CD' if nodes == (2, 3) else 'AB'}"})
                break
    return out


def states_from_path(x, e, t_star):
    t0 = max(0.0, t_star - DELTA)
    t1 = min(1.0, t_star + DELTA)
    pts = []
    for tt, role in ((0.0, "far0"), (t0, "minus"), (t1, "plus"), (1.0, "far1")):
        try:
            y, m, _ = oracle_at(x + tt * e)
        except ValueError:
            continue
        if m < 0.005:
            continue
        pts.append((x + tt * e, y, role))
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_train_paths", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    p.add_argument("--v2", action="store_true",
                   help="add directional order margin on bracket pairs")
    p.add_argument("--dir_margin", type=float, default=1.0)
    a = p.parse_args()
    rng = np.random.default_rng(101)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)

    tr_paths = aimed_single_turns(X, rng, a.n_train_paths, seed=101)
    tr_states = []
    tr_pairs = []  # (x_minus, x_plus, y_old, y_new) for v2 directional loss
    for tp in tr_paths:
        x = X[tp["parent_id"]]
        e = tp["edit"]
        ts = tp["t_star"]
        t0 = max(0.0, ts - DELTA)
        t1 = min(1.0, ts + DELTA)
        pair = {}
        for tt, role in ((0.0, "far0"), (t0, "minus"), (t1, "plus"), (1.0, "far1")):
            try:
                yy, mm, _ = oracle_at(x + tt * e)
            except ValueError:
                continue
            if mm < 0.005:
                continue
            tr_states.append((x + tt * e, yy))
            pair[role] = (x + tt * e, yy)
        if "minus" in pair and "plus" in pair:
            (xm, ym), (xp, yp) = pair["minus"], pair["plus"]
            if ym != yp:
                tr_pairs.append((xm, xp, ym, yp))
    n_states = len(tr_states)
    # control: same number of flipmine-style flip states from pool
    mine = np.load(ART / "r04c_budget" / "mined.npz")
    meta = mine["meta"]
    pool = []
    for i in range(len(meta)):
        if f"edit_{i}" not in mine:
            break
        pi = int(meta[i, 0])
        e = np.asarray(mine[f"edit_{i}"], dtype=float)
        try:
            y0, _, _ = oracle_at(X[pi])
            y1, m1, _ = oracle_at(X[pi] + e)
        except ValueError:
            continue
        if y1 != y0 and m1 >= 0.005:
            pool.append((X[pi] + e, y1))
    ctl = [pool[i] for i in rng.choice(len(pool), size=min(n_states, len(pool)), replace=False)]

    def train(extra, seed, pairs=None):
        torch.manual_seed(seed)
        m = CoordMLP(64, 32)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd).astype(np.float32))
        yc = torch.from_numpy(y)
        bce = nn.BCEWithLogitsLoss()
        Xe_ = torch.from_numpy(np.stack([s.reshape(-1) for s, _ in extra]).astype(np.float32)) if extra else None
        ye_ = torch.from_numpy(np.array([t for _, t in extra], dtype=np.float32)) if extra else None
        if Xe_ is not None:
            Xe_ = torch.from_numpy((((Xe_.numpy().reshape(-1, 8)) - mu) / sd).astype(np.float32))
        Pm = Pp = ya = yb = None
        if pairs:
            xm = torch.from_numpy(np.stack([p[0].reshape(-1) for p in pairs]).astype(np.float32))
            xp = torch.from_numpy(np.stack([p[1].reshape(-1) for p in pairs]).astype(np.float32))
            ya = torch.from_numpy(np.array([p[2] for p in pairs], dtype=np.float32))
            yb = torch.from_numpy(np.array([p[3] for p in pairs], dtype=np.float32))
            Pm = torch.from_numpy((((xm.numpy().reshape(-1, 8)) - mu) / sd).astype(np.float32))
            Pp = torch.from_numpy((((xp.numpy().reshape(-1, 8)) - mu) / sd).astype(np.float32))
        m.train()
        for _ in range(300):
            opt.zero_grad()
            loss = bce(m(Xc), yc)
            if Xe_ is not None:
                loss = loss + bce(m(Xe_), ye_)
            if Pm is not None:
                lm, lp = m(Pm), m(Pp)
                # s = f_new - f_old with logits: for binary, f_1-f_0 = 2*logit gap direction
                sgn = (2 * yb - 1)  # +1 if new class is 1
                gap = sgn * (lp - lm)
                loss = loss + torch.relu(a.dir_margin - gap).mean()
            loss.backward()
            opt.step()
        m.eval()
        return m

    st = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    arms = {}
    for sd_ in a.seeds:
        tag = "bracketv2" if a.v2 else "bracket"
        arms[f"{tag}_s{sd_}"] = train([(s.astype(np.float32), t) for s, t in tr_states], sd_,
                                        pairs=tr_pairs if a.v2 else None)
        arms[f"flipmine_matched_s{sd_}"] = train([(s.astype(np.float32), t) for s, t in ctl], sd_)
    m_clean, _ = load_model(ART / "r04b_s11" / "clean")
    arms["clean"] = m_clean

    # eval: fresh single-turn 0->1 paths on eval parents
    ev_paths = aimed_single_turns(Xe, rng, 200, seed=202)

    def predict_of(m):
        def f(xs):
            with torch.no_grad():
                lg = m(preprocess(xs, st)).numpy()
            return lg, (lg > 0).astype(int)
        return f

    res = {"n_train_paths": len(tr_paths), "n_bracket_states": n_states,
           "n_control_states": len(ctl), "n_eval_paths": len(ev_paths)}
    for name, m in arms.items():
        pf = predict_of(m)
        errs, miss, integ = [], 0, []
        for ep in ev_paths:
            x = Xe[ep["parent_id"]]
            sc = scan_linear(x, ep["edit"], pf, n_scan=17)
            ot = locate_oracle_turns(x, ep["edit"], sc)
            if len(ot) != 1:
                continue
            # path-uniform integral classification error (missing fully counts)
            integ.append(float((sc["pred_label"] != sc["oracle_label"]).mean()))
            mt = locate_model_turns(x, ep["edit"], sc, pf)
            same = [t for t in mt if t["old"] == 0 and t["new"] == 1]
            if not same:
                miss += 1
                errs.append(None)
            else:
                errs.append(abs(same[0]["t_theta"] - ot[0]["t_star"]))
        got = [e for e in errs if e is not None]
        res[name] = {"turn_miss_rate": miss / len(errs) if errs else None,
                     "mean_abs_terr": float(np.mean(got)) if got else None,
                     "mean_integral_err": float(np.mean(integ)) if integ else None,
                     "n": len(errs)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"SAW {k}: turn_miss={v['turn_miss_rate']} mean|terr|={v['mean_abs_terr']} integ={v['mean_integral_err']} (n={v['n']})")
    print("NEXT: if bracket beats matched flipmine on turn_miss/integral -> "
          "v2 directional order + T5R3; else bracket idea dead")
    print("CLAIM: teaching where to turn, not just what the endpoint is")


if __name__ == "__main__":
    main()
