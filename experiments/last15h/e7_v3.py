#!/usr/bin/env python3
"""E7 v3 falsification net: does anything constrain the interpolant shape?

Arms (shared paths/seeds/eval; integral error primary):
  flipmine        - baseline rerun (clean + oracle flip states)
  flipmix_u       - B-1: dense oracle labels, K=8 uniform t per path
  flipmix_c       - B-1: dense oracle labels, 50% t concentrated near tau*
  vanilla_mixup   - control (expected null on |terr|): soft-label mixup pairs
  antiflat_01/10  - A-1: slope floor inside brackets + flatness outside
  sdf             - D-1: signed-distance regression through same logit
  tangent         - C-1: match teacher tangent-slope profile along path
  noise_only      - C-1 control: flipmine + input noise (S&F equivalence check)
  coupled         - A-4: v2 gap/slope coupling (blocks scale-up shortcut)
  globalgrad      - control (expected neutral-harmful): global slope penalty
  ban             - control (expected null): one self-distillation round (T=2)

Falsification bar per arm: beat flipmine integral, 5 seeds same direction;
kill lines per E7_background.md. Winners -> E6 (10 seeds).
"""
import argparse, json, sys, math
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
import paths
from paths import oracle_at, scan_linear, locate_oracle_turns, locate_model_turns
from common import preprocess
from coord_mlp import CoordMLP

ART = ROOT / "artifacts" / "discovery_campaign"
DELTA = 0.05
ARMS = ["flipmine", "flipmix_u", "flipmix_c", "vanilla_mixup",
        "antiflat_01", "antiflat_10", "sdf", "tangent", "noise_only",
        "coupled", "globalgrad", "ban"]
SEEDS = [11, 23, 47, 101, 202]


def build_data(n_train_paths=256, seed_paths=101, kmix=8):
    rng = np.random.default_rng(seed_paths)
    d = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    X = d["positions"].astype(float)
    y = d["labels"].astype(np.float32)
    mu = X.reshape(-1, 8).mean(0)
    sd = X.reshape(-1, 8).std(0) + 1e-8
    # need aimed_single_turns from n01_brackets without running its main
    sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "n01b", str(ROOT / "experiments" / "last15h" / "n01_brackets.py"))
    n01b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n01b)
    from n01_brackets import aimed_single_turns
    tr_paths = aimed_single_turns(X, rng, n_train_paths, seed=seed_paths)
    # flipmine pool (same as N01 round2b control)
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
    return {"X": X, "y": y, "mu": mu, "sd": sd, "tr_paths": tr_paths,
            "pool": pool, "kmix": kmix}


def dense_oracle_states(path_x, e, t_star, K, concentrated, rng):
    """(states, labels) densely graded along path. Skip ambiguous."""
    if concentrated:
        tu = rng.uniform(0, 1, K // 2)
        tc = np.clip(rng.uniform(t_star - DELTA, t_star + DELTA, K - K // 2), 0, 1)
        t = np.concatenate([tu, tc])
    else:
        t = rng.uniform(0, 1, K)
    out = []
    for tt in t:
        try:
            yy, mm, amb = oracle_at(path_x + tt * e)
        except ValueError:
            continue
        if amb:
            continue
        out.append((path_x + tt * e, yy))
    return out


def teacher_slope(t, t_star, tau=0.05):
    s = 1.0 / (1.0 + np.exp(-(t - t_star) / tau))
    return s * (1 - s) / tau


def sdf_target(t, t_star, new_side_fn, clip=0.2):
    s = np.where(new_side_fn(t), 1.0, -1.0)
    return np.clip(s * np.abs(t - t_star), -clip, clip)


def train_arm(arm, D, seed, epochs=300, lam_mix=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    X, y, mu, sd = D["X"], D["y"], D["mu"], D["sd"]
    mu_t = torch.from_numpy(mu.astype(np.float32))
    sd_t = torch.from_numpy(sd.astype(np.float32))
    m = CoordMLP(64, 32)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    Xc = torch.from_numpy(((X.reshape(-1, 8) - mu) / sd).astype(np.float32))
    yc = torch.from_numpy(y)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def std_raw(A):
        return (torch.from_numpy(np.asarray(A, dtype=np.float32).reshape(len(A), -1))
                - mu_t) / sd_t

    # --- arm data ---
    flip = None
    if arm in ("flipmine", "noise_only", "globalgrad", "ban", "tangent", "coupled",
                 "antiflat_01", "antiflat_10", "sdf"):
        idx = np.random.default_rng(101).choice(
            len(D["pool"]), size=min(len(D["pool"]), 4 * len(D["tr_paths"])),
            replace=False)
        # match N01 round2b control size: same # states as bracket set (~4/path)
        flip = [D["pool"][i] for i in idx]
    mix_extra = None
    if arm in ("flipmix_u", "flipmix_c"):
        mix_extra = []
        for tp in D["tr_paths"]:
            x = X[tp["parent_id"]]
            mix_extra += dense_oracle_states(
                x, tp["edit"], tp["t_star"], D["kmix"],
                concentrated=(arm == "flipmix_c"), rng=rng)
    van_extra = None
    if arm == "vanilla_mixup":
        van_extra = []
        lam = rng.beta(1.0, 1.0, size=len(mix_extra or []) or 2048)
        n_mix = len(lam)
        ii = rng.integers(0, len(X), size=(n_mix, 2))
        for k in range(n_mix):
            l = float(lam[k])
            van_extra.append((l * X[ii[k, 0]] + (1 - l) * X[ii[k, 1]],
                              l * y[ii[k, 0]] + (1 - l) * y[ii[k, 1]]))
    # path grids for slope/sdf/tangent arms (17-pt scan grid per path)
    grid = None
    if arm in ("antiflat_01", "antiflat_10", "sdf", "tangent", "coupled", "globalgrad"):
        grid = []
        for tp in D["tr_paths"]:
            x = X[tp["parent_id"]]
            t = np.linspace(0, 1, 17)
            xs = np.stack([x + tt * tp["edit"] for tt in t])
            try:
                ol = np.array([oracle_at(xx)[0] for xx in xs])
            except ValueError:
                continue
            y_end = int(ol[-1])
            grid.append({"xs": xs, "t": t, "t_star": tp["t_star"],
                         "new": y_end, "oracle": ol})

    def slope_at(model, xs_raw, e_raw):
        """ds/dt along path, NUMPY/no-grad (diagnostics only)."""
        h = 0.02
        e8 = np.asarray(e_raw, dtype=np.float32).reshape(-1)
        Xb = np.asarray(xs_raw, dtype=np.float32).reshape(len(xs_raw), -1)
        mu_n, sd_n = mu_t.numpy(), sd_t.numpy()
        Xp = torch.from_numpy(((Xb + h * e8 - mu_n) / sd_n).astype(np.float32))
        Xm = torch.from_numpy(((Xb - h * e8 - mu_n) / sd_n).astype(np.float32))
        with torch.no_grad():
            sp = model(Xp).numpy()
            sm = model(Xm).numpy()
        return (sp - sm) / (2 * h)

    def slope_at_grad(model, Z, dZdt):
        """ds/dt with autograd graph intact (training). Z: (T,8) tensor,
        dZdt: (8,) tensor = raw edit / sd. Central difference, 2 forwards."""
        h = 0.02
        sp = model(Z + h * dZdt)
        sm = model(Z - h * dZdt)
        return (sp - sm) / (2 * h)

    def grid_tensors(g):
        Z = ((torch.from_numpy(np.asarray(g["xs"], dtype=np.float32).reshape(len(g["xs"]), -1)) - mu_t) / sd_t)
        dZ = torch.from_numpy(np.asarray((g["xs"][-1] - g["xs"][0]), dtype=np.float32).reshape(-1)) / sd_t
        inside = torch.from_numpy((np.abs(g["t"] - g["t_star"]) <= DELTA))
        return Z, dZ, inside

    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(m(Xc), yc)
        if flip is not None and arm != "ban":
            Xf = std_raw([s for s, _ in flip])
            yf = torch.from_numpy(np.array([t for _, t in flip], dtype=np.float32))
            if arm == "noise_only":
                Xf = Xf + torch.randn_like(Xf) * 0.01
                loss = bce(m(Xc + torch.randn_like(Xc) * 0.01), yc) + \
                    bce(m(Xf), yf)
            else:
                loss = loss + bce(m(Xf), yf)
        if mix_extra:
            Xm = std_raw([s for s, _ in mix_extra])
            ym = torch.from_numpy(np.array([t for _, t in mix_extra], dtype=np.float32))
            loss = loss + lam_mix * bce(m(Xm), ym)
        if van_extra:
            Xv = std_raw([s for s, _ in van_extra])
            yv = torch.from_numpy(np.array([t for _, t in van_extra], dtype=np.float32))
            loss = loss + bce(m(Xv), yv)
        if grid is not None and arm in ("antiflat_01", "antiflat_10"):
            lam_in = 0.1 if arm == "antiflat_01" else 1.0
            g0, lam_out = 1.0, 0.01
            li = lo = 0.0
            for g in grid:
                Z, dZ, inside = grid_tensors(g)
                sl = slope_at_grad(m, Z, dZ).abs()
                if inside.any():
                    li = li + torch.relu(g0 - sl[inside]).mean()
                if (~inside).any():
                    lo = lo + (sl[~inside] ** 2).mean()
            loss = loss + lam_in * li / max(1, len(grid)) + lam_out * lo / max(1, len(grid))
        if grid is not None and arm == "sdf":
            ls = 0.0
            for g in grid:
                newside = g["oracle"] == g["new"]
                tgt = sdf_target(g["t"], g["t_star"],
                                 lambda tt, _n=newside, _t=g["t"]: np.interp(tt, _t, _n.astype(float)) > 0.5)
                Xs = std_raw(g["xs"])
                ls += mse(m(Xs), torch.from_numpy(tgt.astype(np.float32)))
            loss = loss + 1.0 * ls / max(1, len(grid))
        if grid is not None and arm == "tangent":
            lt = 0.0
            for g in grid:
                Z, dZ, _ = grid_tensors(g)
                sl = slope_at_grad(m, Z, dZ)
                tt_ = torch.from_numpy(teacher_slope(g["t"], g["t_star"]).astype(np.float32))
                lt = lt + ((sl - tt_) ** 2).mean()
            loss = loss + 0.1 * lt / max(1, len(grid))
        if grid is not None and arm == "coupled":
            lc = 0.0
            w, eps = DELTA, 0.1
            for g in grid:
                Z, dZ, _ = grid_tensors(g)
                sl = slope_at_grad(m, Z, dZ)
                j = int(np.argmin(np.abs(g["t"] - g["t_star"])))
                l0 = m(Z[0:1]).squeeze(0)
                l1 = m(Z[-1:]).squeeze(0)
                # gap toward new class over local slope
                sgn = 1.0 if g["new"] == 1 else -1.0
                gap = sgn * (l1 - l0)
                lc = lc + torch.relu(w - gap / torch.clamp(sl[j].abs(), min=eps))
            loss = loss + 1.0 * lc / max(1, len(grid))
        if grid is not None and arm == "globalgrad":
            lg_ = 0.0
            for g in grid:
                Z, dZ, _ = grid_tensors(g)
                sl = slope_at_grad(m, Z, dZ)
                lg_ = lg_ + (sl ** 2).mean()
            loss = loss + 0.01 * lg_ / max(1, len(grid))
        loss.backward()
        opt.step()
    m.eval()
    diag = {}
    if grid is not None and arm in ("antiflat_01", "antiflat_10", "tangent", "coupled", "globalgrad"):
        ins, outs = [], []
        for g in grid:
            e = (g["xs"][-1] - g["xs"][0])
            sl = np.abs(slope_at(m, g["xs"], e))
            inside = np.abs(g["t"] - g["t_star"]) <= DELTA
            ins.append(sl[inside].mean() if inside.any() else np.nan)
            outs.append(sl[~inside].mean() if (~inside).any() else np.nan)
        diag = {"mean_inbracket_slope": float(np.nanmean(ins)), "mean_outbracket_slope": float(np.nanmean(outs))}
    if arm == "ban":
        # stage 2: fresh net on stage-1 soft outputs (T=2)
        with torch.no_grad():
            soft_c = torch.softmax(torch.stack([m(Xc), torch.zeros_like(m(Xc))], 1) / 2, 1)[:, 0]
            soft_f = None
            if flip:
                Xf0 = std_raw([s for s, _ in flip])
                soft_f = torch.softmax(torch.stack([m(Xf0), torch.zeros_like(m(Xf0))], 1) / 2, 1)[:, 0]
        torch.manual_seed(seed + 7919)
        m2 = CoordMLP(64, 32)
        opt2 = torch.optim.Adam(m2.parameters(), lr=0.01)
        m2.train()
        for _ in range(epochs):
            opt2.zero_grad()
            l2 = bce(m2(Xc), soft_c.detach())
            if flip:
                l2 = l2 + bce(m2(Xf0), soft_f.detach())
            l2.backward()
            opt2.step()
        m2.eval()
        return m2, diag
    return m, diag


def evaluate(m, D, st, n_eval_paths=200, seed_eval=202):
    sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
    from n01_brackets import aimed_single_turns
    de = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe = de["positions"].astype(float)
    ev_paths = aimed_single_turns(Xe, np.random.default_rng(seed_eval),
                                  n_eval_paths, seed=seed_eval)

    def predict(xs):
        with torch.no_grad():
            lg = m(preprocess(xs, st)).numpy()
        return lg, (lg > 0).astype(int)

    errs, miss, integ = [], 0, []
    for ep in ev_paths:
        x = Xe[ep["parent_id"]]
        sc = scan_linear(x, ep["edit"], predict, n_scan=17)
        ot = locate_oracle_turns(x, ep["edit"], sc)
        if len(ot) != 1:
            continue
        integ.append(float((sc["pred_label"] != sc["oracle_label"]).mean()))
        mt = locate_model_turns(x, ep["edit"], sc, predict)
        same = [t for t in mt if t["old"] == 0 and t["new"] == 1]
        if not same:
            miss += 1
            errs.append(None)
        else:
            errs.append(abs(same[0]["t_theta"] - ot[0]["t_star"]))
    got = [e for e in errs if e is not None]
    return {"turn_miss_rate": miss / len(errs) if errs else None,
            "mean_abs_terr": float(np.mean(got)) if got else None,
            "mean_integral_err": float(np.mean(integ)) if integ else None,
            "n": len(errs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n_train_paths", type=int, default=256)
    p.add_argument("--n_eval_paths", type=int, default=200)
    a = p.parse_args()
    D = build_data(n_train_paths=a.n_train_paths)
    st = {"mu": D["mu"].astype(np.float32), "sd": D["sd"].astype(np.float32)}
    res = {"arm": a.arm, "epochs": a.epochs, "n_train_paths": len(D["tr_paths"]),
           "n_flip_states": len(D["pool"])}
    for sd_ in a.seeds:
        m, dg = train_arm(a.arm, D, sd_, epochs=a.epochs)
        v = evaluate(m, D, st, n_eval_paths=a.n_eval_paths)
        v.update(dg)
        res[f"s{sd_}"] = v
        print(f"SAW {a.arm} s{sd_}: turn_miss={v['turn_miss_rate']} "
              f"mean|terr|={v['mean_abs_terr']} integ={v['mean_integral_err']} (n={v['n']})",
              flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"NEXT: compare {a.arm} integral vs flipmine 0.229 across seeds; "
          f"kill lines per E7_background.md")


if __name__ == "__main__":
    main()
