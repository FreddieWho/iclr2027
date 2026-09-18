#!/usr/bin/env python3
"""E1 one-shot battery on freshly minted holdout_909 (preregistered).

Protocols isomorphic to originals; see reports/last15h/E1_PREREG.md.
RNG: n06->6, n04->4, n09->9, n03->3 (mirrors), n01/e1-specific->909.
Models: r04b_s11/{clean,flipmine} + mlpB (static only). Zero training.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(2)
from paths import (oracle_at, scan_linear, scan_polyline, locate_oracle_turns,
                   locate_model_turns, match_turns, atomic_edits, action_library)
from r02_search import candidates_for_scene
from common import load_model, preprocess
from n01_brackets import aimed_single_turns
from n03_events import build_oscillating, build_control, episodes_of

ART = ROOT / "artifacts" / "discovery_campaign"
POOL = "holdout_909"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = np.load(ART / "scenes" / POOL / "scenes.npz")
    X = d["positions"].astype(np.float32)
    y = d["labels"].astype(int)
    res = {"pool": POOL, "n": len(X)}
    loaded = {}
    for k, mp in (("clean", ART / "r04b_s11" / "clean"),
                  ("flipmine", ART / "r04b_s11" / "flipmine"),
                  ("mlpB", ART / "models" / "mlpB_h32f16")):
        m, s = load_model(mp)
        m.eval()
        loaded[k] = (m, s)

    def acc(model_key):
        m, s = loaded[model_key]
        with torch.no_grad():
            lg = m(preprocess(X, s)).numpy()
        return float((((lg > 0).astype(int)) != y).mean())

    def predict_of(model_key):
        m, s = loaded[model_key]

        def f(xs):
            with torch.no_grad():
                lg = m(preprocess(xs, s)).numpy()
            return lg, (lg > 0).astype(int)
        return f

    def pred1(model_key, x):
        m, s = loaded[model_key]
        with torch.no_grad():
            lg = m(preprocess(x[None].astype(np.float32), s)).numpy()[0]
        return int(lg > 0), float(lg)

    # ---- 1. static ----
    res["static_err"] = {k: acc(k) for k in loaded}
    print("SAW static_err:", res["static_err"], flush=True)

    # ---- 2. flips (R02 protocol, margin>=0.03) ----
    rng = np.random.default_rng(909)
    flips = []  # (x, e, y1)
    for pi in rng.permutation(len(X)):
        x = X[pi].astype(float)
        for e, _ in candidates_for_scene(x, rng, n_random=64):
            e = np.asarray(e, dtype=float)
            try:
                y0, _, _ = oracle_at(x)
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 != y0 and m1 >= 0.03:
                flips.append((x, e, y1))
                break
    res["n_flips"] = len(flips)
    for k in ("clean", "flipmine"):
        m, s = loaded[k]
        xs = np.stack([(x + e).astype(np.float32) for x, e, _ in flips])
        ys = np.array([t for _, _, t in flips])
        with torch.no_grad():
            lg = m(preprocess(xs, s)).numpy()
        res[f"miss_flip_{k}"] = float((((lg > 0).astype(int)) != ys).mean())
    print(f"SAW flips n={len(flips)} miss_clean={res['miss_flip_clean']:.3f} "
          f"miss_flipmine={res['miss_flip_flipmine']:.3f}", flush=True)

    # ---- 3. n06 actions (rng 6, lam 0/2) ----
    rng6 = np.random.default_rng(6)
    lib = action_library()
    cross = [i for i in rng6.permutation(len(X))[:512]
             if oracle_at(X[i])[0] == 1][:256]
    tasks = []
    for pi in cross:
        x = X[pi].astype(float)
        cands = []
        for act in lib:
            try:
                yy, _, _ = oracle_at(x + act["edit"])
            except ValueError:
                continue
            cands.append((act, yy))
        if not any(c[1] == 0 for c in cands):
            continue
        tasks.append((x, cands))
    res["n06"] = {"n_tasks": len(tasks)}
    for k in ("clean", "flipmine"):
        m, s = loaded[k]
        frames = []
        for x, cands in tasks:
            for act, _ in cands:
                frames.append((x + act["edit"]).astype(np.float32))
        with torch.no_grad():
            lg = m(preprocess(np.stack(frames), s)).numpy()
        kk = 0
        for lam in (0.0, 2.0):
            kk = 0
            inv = suc = 0
            for x, cands in tasks:
                nc = len(cands)
                P0 = 1 / (1 + np.exp(lg[kk:kk + nc]))
                kk += nc
                costs = np.array([c[0]["cost"] for c in cands])
                j = int(np.argmax(P0 - lam * costs))
                if cands[j][1] != 0:
                    inv += 1
                else:
                    suc += 1
            tot = inv + suc
            res["n06"][f"{k}_lam{lam:g}"] = {
                "invalid_rate": inv / tot if tot else None,
                "success_rate": suc / tot if tot else None, "n": tot}
    print("SAW n06:", {kk: vv["invalid_rate"] for kk, vv in res["n06"].items()
                       if isinstance(vv, dict)}, flush=True)

    # ---- 4. n04 combos (rng 4) ----
    rng4 = np.random.default_rng(4)
    order = rng4.permutation(len(X))
    for k in ("clean", "flipmine"):
        atom_ok = atom_n = combo_n = combo_e = 0
        emerg_n = emerg_m = emerg_cn = emerg_cm = 0
        for pi in order:
            x = X[pi].astype(float)
            try:
                y0, _, _ = oracle_at(x)
            except ValueError:
                continue
            results = []
            for e, fam in atomic_edits(x, rng4):
                try:
                    yy, _, _ = oracle_at(x + e)
                except ValueError:
                    continue
                p1, _ = pred1(k, x + e)
                atom_n += 1
                atom_ok += (p1 == yy)
                results.append((e, yy))
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    (ea, ya), (eb, yb) = results[i], results[j]
                    if ya != y0 or yb != y0:
                        continue
                    try:
                        yc, _, _ = oracle_at(x + ea + eb)
                    except ValueError:
                        continue
                    pc, _ = pred1(k, x + eb + ea)
                    combo_n += 1
                    combo_e += (pc != yc)
                    if yc != y0:
                        emerg_n += 1
                        emerg_m += (pc != yc)
                    pa, _ = pred1(k, x + ea)
                    pb, _ = pred1(k, x + eb)
                    if pa == ya and pb == yb:
                        if yc != y0:
                            emerg_cn += 1
                            emerg_cm += (pc != yc)
        # displacement-matched singles
        emerg_costs = []
        for pi in order:
            x = X[pi].astype(float)
            try:
                y0, _, _ = oracle_at(x)
            except ValueError:
                continue
            rr = []
            for e, fam in atomic_edits(x, rng4):
                try:
                    yy, _, _ = oracle_at(x + e)
                except ValueError:
                    continue
                rr.append((e, yy))
            for i in range(len(rr)):
                for j in range(i + 1, len(rr)):
                    (ea, ya), (eb, yb) = rr[i], rr[j]
                    if ya != y0 or yb != y0:
                        continue
                    try:
                        yc, _, _ = oracle_at(x + ea + eb)
                    except ValueError:
                        continue
                    if yc != y0:
                        emerg_costs.append(float(np.linalg.norm(ea + eb)))
        matched_m = matched_n = 0
        for pi in order:
            x = X[pi].astype(float)
            for e, _ in candidates_for_scene(x, rng4, n_random=32):
                e = np.asarray(e, dtype=float)
                c = float(np.linalg.norm(e))
                if not emerg_costs or min(abs(c - ec) for ec in emerg_costs) > 0.02:
                    continue
                try:
                    y0, _, _ = oracle_at(x)
                    y1, m1, _ = oracle_at(x + e)
                except ValueError:
                    continue
                if y1 == y0 or m1 < 0.005:
                    continue
                p1, _ = pred1(k, x + e)
                matched_n += 1
                matched_m += (p1 != y1)
                break
        res[f"n04_{k}"] = {
            "atomic_acc": atom_ok / atom_n if atom_n else None,
            "combo_err": combo_e / combo_n if combo_n else None,
            "emergent_n": emerg_n,
            "emergent_miss": emerg_m / emerg_n if emerg_n else None,
            "emergent_cond_n": emerg_cn,
            "emergent_cond_miss": emerg_cm / emerg_cn if emerg_cn else None,
            "matched_n": matched_n,
            "matched_miss": matched_m / matched_n if matched_n else None}
    print("SAW n04:", {k: (v["emergent_cond_miss"], v["emergent_cond_n"])
                       for k, v in res.items() if k.startswith("n04")}, flush=True)

    # ---- 5. n09 C0/C1 (rng 9) ----
    rng9 = np.random.default_rng(9)
    C1 = [(x, e, t) for x, e, t in flips
          if oracle_at(x + e)[1] >= 0.005]
    costs = sorted(float(np.linalg.norm(e)) for _, e, _ in C1)
    C0 = []
    for (x, _, _), c in zip(C1, costs):
        y0, _, _ = oracle_at(x)
        for e, _ in candidates_for_scene(x, rng9, n_random=32):
            e = np.asarray(e, dtype=float)
            if abs(float(np.linalg.norm(e)) - c) > 0.02:
                continue
            try:
                yy, mm, _ = oracle_at(x + e)
            except ValueError:
                continue
            if yy == y0 and mm >= 0.005:
                C0.append((x, e, y0))
                break
    res["n_C1"], res["n_C0"] = len(C1), len(C0)
    for k in ("clean", "flipmine"):
        m, s = loaded[k]

        def err(pairs):
            xs = np.stack([(x + e).astype(np.float32) for x, e, _ in pairs])
            ys = np.array([t for _, _, t in pairs])
            with torch.no_grad():
                lg = m(preprocess(xs, s)).numpy()
            return float((((lg > 0).astype(int)) != ys).mean())

        res[f"C1_miss_{k}"] = err(C1) if C1 else None
        res[f"C0_FA_{k}"] = err(C0) if C0 else None
    print(f"SAW n09 C1 clean={res['C1_miss_clean']:.3f} flip={res['C1_miss_flipmine']:.3f} "
          f"C0 clean={res['C0_FA_clean']:.3f} flip={res['C0_FA_flipmine']:.3f}", flush=True)

    # ---- 6. n01 turns (want 64, seed 909; underpowered, wide CI) ----
    paths = aimed_single_turns(X.astype(float), np.random.default_rng(909),
                               64, seed=909)
    res["n01_n_paths"] = len(paths)
    for k in ("clean", "flipmine"):
        pf = predict_of(k)
        miss, integ, n = 0, [], 0
        for ep in paths:
            x = X[ep["parent_id"]].astype(float)
            sc = scan_linear(x, ep["edit"], pf, n_scan=17)
            ot = locate_oracle_turns(x, ep["edit"], sc)
            if len(ot) != 1:
                continue
            n += 1
            integ.append(float((sc["pred_label"] != sc["oracle_label"]).mean()))
            mt = locate_model_turns(x, ep["edit"], sc, pf)
            same = [t for t in mt if t["old"] == 0 and t["new"] == 1]
            if not same:
                miss += 1
        res[f"n01_{k}"] = {"turn_miss": miss / n if n else None,
                           "mean_integral": float(np.mean(integ)) if integ else None,
                           "n": n}
    print("SAW n01:", {k: v["turn_miss"] for k, v in res.items()
                       if k.startswith("n01_") and isinstance(v, dict)}, flush=True)

    # ---- 7. n03 events (want 128/128, rng 3 mirror) ----
    rng3 = np.random.default_rng(3)
    ev_wps, ct_wps = [], []
    for pi in rng3.permutation(len(X)):
        if len(ev_wps) >= 128 and len(ct_wps) >= 128:
            break
        x = X[pi].astype(float)
        try:
            if oracle_at(x)[0] != 0:
                continue
        except ValueError:
            continue
        if len(ev_wps) < 128:
            r = build_oscillating(x, rng3)
            if r is not None:
                ev_wps.append(r[0])
        if len(ct_wps) < 128:
            wps = build_control(x, rng3)
            if wps is not None:
                ct_wps.append(wps)
    res["n03"] = {"n_events": len(ev_wps), "n_controls": len(ct_wps)}
    for k in ("clean", "flipmine"):
        pf = predict_of(k)
        cond_n = cond_m = full = 0
        for wps in ev_wps:
            sc = scan_polyline([w.astype(np.float32) for w in wps], pf, n_per_leg=65)
            eps = [ep for ep in episodes_of(sc["oracle_label"])
                   if (ep[1] - ep[0] + 1) >= 3
                   and sc["oracle_margin"][ep[0]:ep[1] + 1].min() >= 0.005]
            if not eps:
                continue
            if sc["pred_label"][0] != 0:
                continue
            cond_n += 1
            if (sc["pred_label"] == sc["oracle_label"]).all():
                full += 1
            if not any(sc["pred_label"][i0:j0 + 1].max() == 1 for i0, j0 in eps):
                cond_m += 1
        fa = 0
        for wps in ct_wps:
            sc = scan_polyline([w.astype(np.float32) for w in wps], pf, n_per_leg=65)
            if sc["pred_label"].max() == 1:
                fa += 1
        res["n03"][k] = {"cond_n": cond_n,
                         "cond_miss": cond_m / cond_n if cond_n else None,
                         "full_rate": full / len(ev_wps) if ev_wps else None,
                         "FA": fa / len(ct_wps) if ct_wps else None}
    print("SAW n03:", {k: v["cond_miss"] for k, v in res["n03"].items()
                       if isinstance(v, dict)}, flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("E1 DONE")


if __name__ == "__main__":
    main()
