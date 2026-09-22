#!/usr/bin/env python3
"""U3: does P1 confidence inversion survive effective relational repair?

Q1: fixed high-risk set = top quartile of |AB logit| under raw_clean
    (defined once per seed, never redefined per model). On this FIXED set,
    report each arm's start/end correctness, full repair and migration,
    with row/parent denominators. Secondary: own-start-correct and
    common-start-correct.
Q2: per arm, rank quartets by OWN |AB logit| into quartiles; report AB
    error per quartile (within-arm ranking only; raw logit magnitudes
    are never compared across models as one probability scale).
Preserve side: static_err / single_err / preserve_FA per arm/seed on all
dev512 singles (same definitions as relfeat_eval).

NEW outputs only:
  artifacts/next_novelty/u3_confidence/U3_SUMMARY.json
No training, no sealed-holdout reads.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402

BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "u3_confidence"
rng = np.random.default_rng(20260924)

ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_checkpoint(arm: str, seed: int) -> Path:
    from pathlib import Path as P
    art = ROOT / "artifacts" / "discovery_campaign"
    if arm == "relflip":
        return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
    if arm == "raw_clean":
        return art / ("r04b_s%d" % seed) / "clean"
    if arm == "raw_flipmine":
        return art / ("r04b_s%d" % seed) / "flipmine"
    if arm == "relfeat":
        if seed == 11:
            return art / "r04b_s11" / "relfeat"
        return art / ("r04b_s%d_relfeat" % seed) / "relfeat"
    raise ValueError(arm)


def pcb(groups, stat_fn, n_boot=2000):
    ups = np.arange(len(groups))
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        boots.append(stat_fn([groups[i] for i in draw]))
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perquartet", type=Path, default=ROOT / "artifacts" /
                   "next_novelty" / "u1_factorial" / "U1_PERQUARTET.csv")
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    Q = {}
    with open(a.perquartet) as f:
        for r in csv.DictReader(f):
            Q[(int(r["seed"]), r["arm"], int(r["qid"]))] = (
                [float(r["logit_A"]), float(r["logit_B"]),
                 float(r["logit_AB"])],
                [int(r["correct_A"]), int(r["correct_B"]),
                 int(r["correct_AB"])],
                int(r["parent"]))
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Sx, Se = np.asarray(b["Sx"], np.float32), np.asarray(b["Se"], np.float32)
    Smeta = json.loads(str(b["Smeta"]))
    y0 = np.array([m["y0"] for m in Smeta])
    y1 = np.array([m["y1"] for m in Smeta])
    fl = np.array([m["flip"] == 1 for m in Smeta])
    spar = np.array([m["parent"] for m in Smeta])

    def start_conf(seed, arm, q):
        lg = Q[(seed, arm, q)][0]
        return max(abs(lg[0]), abs(lg[1]))

    out = {"bank": a.bank, "seeds": {}}
    for seed in (11, 23, 47):
        qids = sorted({q for (s, _, q) in Q if s == seed})
        par = {q: Q[(seed, "raw_clean", q)][2] for q in qids}
        conf = np.array([start_conf(seed, "raw_clean", q) for q in qids])
        thr = float(np.quantile(conf, 0.75))
        fixed = [q for q in qids
                 if start_conf(seed, "raw_clean", q) >= thr]
        fpar = np.array([par[q] for q in fixed])
        Fgroups = [np.nonzero(fpar == u)[0]
                   for u in np.unique(fpar)]
        q1 = {}
        for arm in ARMS:
            C = {q: Q[(seed, arm, q)][1] for q in fixed}
            st = np.array([1.0 if C[q][:2] == [1, 1] else 0.0 for q in fixed])
            en = np.array([float(C[q][2]) for q in fixed])
            fu = np.array([1.0 if C[q] == [1, 1, 1] else 0.0 for q in fixed])
            mg = np.array([1.0 if (C[q][2] == 1 and C[q][:2] != [1, 1])
                           else 0.0 for q in fixed])
            q1[arm] = {
                "n": len(fixed), "n_parents": len(np.unique(fpar)),
                "start_pass": [round(float(st.mean()), 4),
                               pcb(Fgroups, lambda idx: st[np.concatenate(idx)].mean())],
                "end_correct": [round(float(en.mean()), 4),
                                pcb(Fgroups, lambda idx: en[np.concatenate(idx)].mean())],
                "full_repair": [round(float(fu.mean()), 4),
                                pcb(Fgroups, lambda idx: fu[np.concatenate(idx)].mean())],
                "migration": [round(float(mg.mean()), 4),
                              pcb(Fgroups, lambda idx: mg[np.concatenate(idx)].mean())],
            }
        own, common = {}, {}
        for arm in ARMS:
            oq = [q for q in qids if Q[(seed, arm, q)][1][:2] == [1, 1]]
            own[arm] = {"n": len(oq), "AB_err": round(float(np.mean(
                [1 - Q[(seed, arm, q)][1][2] for q in oq])), 4) if oq else None}
        com = [q for q in qids
               if all(Q[(seed, arm, q)][1][:2] == [1, 1] for arm in ARMS)]
        for arm in ARMS:
            common[arm] = {"AB_err": round(float(np.mean(
                [1 - Q[(seed, arm, q)][1][2] for q in com])), 4) if com else None}
        common_n = len(com)
        q2 = {}
        for arm in ARMS:
            c = np.array([start_conf(seed, arm, q) for q in qids])
            e = np.array([1 - Q[(seed, arm, q)][1][2] for q in qids])
            qs = np.quantile(c, [0.25, 0.5, 0.75])
            qr = {}
            bounds = [-np.inf] + list(qs) + [np.inf]
            for i in range(4):
                m = (c >= bounds[i]) & (c <= bounds[i + 1]) if i < 3 else \
                    (c >= bounds[i]) & (c <= bounds[i + 1])
                if i == 3:
                    m = (c >= bounds[3])
                qr["Q%d" % i] = {"n": int(m.sum()),
                                 "AB_err": round(float(e[m].mean()), 4) if m.sum() else None,
                                 "n_parents": len(np.unique(
                                     [par[q] for q, mm in zip(qids, m) if mm]))}
            q2[arm] = qr
        models = {}
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model, stats = load_model(mp)
            model.eval()
            models[arm] = (model, stats)
        pres = {}
        with torch.no_grad():
            for arm in ARMS:
                model, stats = models[arm]
                bl = []
                for s in range(0, len(Sx), 4096):
                    xx = (Sx[s:s + 4096].reshape(len(Sx[s:s + 4096]), -1) +
                          Se[s:s + 4096].reshape(len(Se[s:s + 4096]), -1))
                    bl.append(model(preprocess(xx.astype(np.float32),
                                               stats)).numpy().reshape(-1))
                lg1 = np.concatenate(bl)
                p1 = (lg1 > 0).astype(int)
                pres[arm] = {
                    "static_err": round(float((p1 != y1).mean()), 4),
                    "single_flip_err": round(float((p1[fl] != y1[fl]).mean()), 4) if fl.sum() else None,
                }
        # preserve_FA needs start correctness: second forward on starts
        with torch.no_grad():
            for arm in ARMS:
                model, stats = models[arm]
                bl = []
                for s in range(0, len(Sx), 4096):
                    xx = Sx[s:s + 4096].reshape(len(Sx[s:s + 4096]), -1)
                    bl.append(model(preprocess(xx.astype(np.float32),
                                               stats)).numpy().reshape(-1))
                p0 = (np.concatenate(bl) > 0).astype(int)
                st = (~fl) & (p0 == y0)
                # recompute p1
                cl = []
                for s in range(0, len(Sx), 4096):
                    xx = (Sx[s:s + 4096].reshape(len(Sx[s:s + 4096]), -1) +
                          Se[s:s + 4096].reshape(len(Se[s:s + 4096]), -1))
                    cl.append(model(preprocess(xx.astype(np.float32),
                                               stats)).numpy().reshape(-1))
                p1 = (np.concatenate(cl) > 0).astype(int)
                pres[arm]["preserve_FA"] = round(
                    float((p1[st] != p0[st]).mean()), 4) if st.sum() else None
                pres[arm]["n_preserve_eligible"] = int(st.sum())
        out["seeds"]["s%d" % seed] = {
            "fixed_set_def": "top quartile START confidence max(|A|,|B|) under raw_clean",
            "fixed_thr": round(thr, 4),
            "Q1_fixed_set": q1,
            "secondary_own_start": own,
            "secondary_common_start": {"n": common_n, "arms": common},
            "Q2_own_confidence_quartiles": q2,
            "preserve": pres,
        }
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "U3_SUMMARY.json", "w"), indent=1)
    for seed in (11, 23, 47):
        d = out["seeds"]["s%d" % seed]
        print("s%d fixed n=%d thr=%.3f" % (
            seed, d["Q1_fixed_set"]["raw_clean"]["n"], d["fixed_thr"]),
            {k: (v["end_correct"][0], v["full_repair"][0]) for k, v in
             d["Q1_fixed_set"].items()}, flush=True)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
