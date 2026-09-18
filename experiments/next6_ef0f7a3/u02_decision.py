#!/usr/bin/env python3
"""U02 round1: decision interface vs representation in action selection.

后续分析 (seen pool eval_202, new rules): cache per-(scene,action) logits,
costs, oracle feasibility once; compare rules:
 (a) score = P0 - lam*cost (original);
 (b) lexicographic: cheapest among predicted-feasible (P0>=0.5), else refuse;
 (c) calibrated: temperature T fit on DEV scenes, then cheapest among
     P0_cal>=0.5 (fixed tau; T chosen on dev only).
Hard framing: oracle-feasible scenes only (coverage reported separately);
no-op scenes where oracle has nothing feasible are NOT model errors.
Mechanism: T in {0.5, 2.0} on frozen logits (assert argmax unchanged per
input) -> changed-action fraction + success delta under cost tradeoff.
Lambdas {0,1,2,5} (never single-lambda). Models: clean + flipmine (coord).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import action_library, oracle_at
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=206)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(float)
    lib = action_library()
    costs = np.array([c["cost"] for c in lib])
    cross = [i for i in rng.permutation(len(X))[:512]
             if oracle_at(X[i])[0] == 1][:256]
    # cache
    frames, feas, scene_of, act_of = [], [], [], []
    for pi in cross:
        x = X[pi]
        for ai, act in enumerate(lib):
            try:
                yy, _, _ = oracle_at(x + act["edit"])
            except ValueError:
                continue
            frames.append((x + act["edit"]).astype(np.float32))
            feas.append(yy == 0)
            scene_of.append(pi)
            act_of.append(ai)
    frames = np.stack(frames)
    feas = np.array(feas)
    scene_of = np.array(scene_of)
    acost = costs[np.array(act_of)]
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "cache.npz",
             frames=frames, feas=feas, scene_of=scene_of, cross=np.array(cross))
    loaded = {}
    for k, mp in (("clean", ART / "r04b_s11" / "clean"),
                  ("flipmine", ART / "r04b_s11" / "flipmine")):
        m, s = load_model(mp)
        m.eval()
        with torch.no_grad():
            loaded[k] = m(preprocess(frames, s)).numpy()
    # scene split: dev (fit T) / test
    uniq = np.array(sorted(set(scene_of)))
    rng.shuffle(uniq)
    dev_scenes = set(uniq[:len(uniq) // 3])
    te_scenes = [s for s in uniq if s not in dev_scenes]
    dev_m = np.isin(scene_of, list(dev_scenes))

    res = {"n_scenes": len(uniq), "n_test_scenes": len(te_scenes),
           "lambdas": [0, 1, 2, 5]}
    for k, lg in loaded.items():
        P0 = 1 / (1 + np.exp(lg))
        # fit T on dev (NLL on feasibility labels)
        T = nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.LBFGS([T], max_iter=50)

        def closure():
            opt.zero_grad()
            p = torch.sigmoid(torch.from_numpy(lg[dev_m]).float() / T.clamp(0.05, 20))
            loss = nn.BCELoss()(p, torch.from_numpy(feas[dev_m]).float())
            loss.backward()
            return loss
        opt.step(closure)
        Tfit = float(T.clamp(0.05, 20))
        Pcal = 1 / (1 + np.exp(lg / Tfit))
        # oracle-best cost per test scene (feasible scenes only)
        out = {"T_dev": Tfit}
        for lam in (0, 1, 2, 5):
            s_suc, s_tot, excess = {}, {}, {}
            for rule in ("score", "lexico", "calib"):
                suc = tot = exc = 0.0
                ref = 0
                for s in te_scenes:
                    idx = np.nonzero(scene_of == s)[0]
                    of = feas[idx]
                    if not of.any():
                        continue  # no oracle-feasible action: not a model error
                    tot += 1
                    best = acost[idx][of].min()
                    if rule == "score":
                        j = idx[int(np.argmax(P0[idx] - lam * acost[idx]))]
                    elif rule == "lexico":
                        cand = idx[P0[idx] >= 0.5]
                        if len(cand) == 0:
                            ref += 1
                            continue
                        j = cand[int(np.argmin(acost[cand]))]
                    else:
                        cand = idx[Pcal[idx] >= 0.5]
                        if len(cand) == 0:
                            ref += 1
                            continue
                        j = cand[int(np.argmin(acost[cand]))]
                    if feas[j]:
                        suc += 1
                        exc += acost[j] - best
                s_suc[rule] = suc / tot if tot else None
                excess[rule] = (exc / suc) if suc else None
                s_tot[rule] = tot
                if rule != "score":
                    out[f"lam{lam}_{rule}_refuse"] = ref / tot if tot else None
            out[f"lam{lam}"] = {"success": s_suc, "excess_cost": excess, "n": s_tot}
        # temperature mechanism: argmax-invariant rescaling, selection change?
        mech = {}
        for Tm in (0.5, 2.0):
            lgT = lg / Tm
            same_argmax = True  # binary: class prediction unchanged?
            pred0 = (lg > 0)
            predT = (lgT > 0)
            assert (pred0 == predT).all(), "temperature changed argmax!"
            chg, d_suc = [], {}
            for lam in (0, 1, 2, 5):
                c0 = cT = s0 = sT = n = 0
                for s in te_scenes:
                    idx = np.nonzero(scene_of == s)[0]
                    if not feas[idx].any():
                        continue
                    n += 1
                    PT = 1 / (1 + np.exp(lgT[idx]))
                    j0 = int(np.argmax(P0[idx] - lam * acost[idx]))
                    jT = int(np.argmax(PT - lam * acost[idx]))
                    if j0 != jT:
                        c0 += 1
                    s0 += feas[idx][j0]
                    sT += feas[idx][jT]
                chg.append(c0 / n if n else None)
                d_suc[f"lam{lam}"] = (float(sT - s0) / n) if n else None
            mech[f"T{Tm}"] = {"changed_frac_per_lam": chg, "success_delta": d_suc}
        out["temperature_mechanism"] = mech
        res[k] = out
        print(f"SAW {k}: T={Tfit:.3f} " +
              "; ".join(f"lam{lam} score/lex/calib="
                        f"{out[f'lam{lam}']['success']['score']:.3f}/"
                        f"{out[f'lam{lam}']['success']['lexico']:.3f}/"
                        f"{out[f'lam{lam}']['success']['calib']:.3f}"
                        for lam in (0, 1, 2, 5)), flush=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE U02")


if __name__ == "__main__":
    main()
