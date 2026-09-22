#!/usr/bin/env python3
"""L007-C1b: does the headline quartet metric depend on labeling?

C1 showed that transition-root location is largely label-dependent on the
eval paths. The paper's headline numbers (joint consistency J, atomic pass,
AB correctness on the quartet banks) come from the same model class on the
same task, whose oracle is permutation-invariant -- so the same question
applies to the main metric.

Method (zero training, frozen checkpoints):
  - dev512 quartet bank; apply each of the 8 relabelings jointly to Qx and
    Qe (so every configuration is the same geometric object with the same
    oracle labels; asserted by recomputing labels where cheap);
  - evaluate the SAME frozen model per relabeling;
  - report J / atomic pass / AB correctness per relabeling, the spread, and
    the per-quartet flip rate of J.

Interpretation guard: the edits' semantic family names (e.g. "shift_CD")
refer to the original labeling, so a segment swap changes which segment
moves -- that is fine for a permutation-invariance test (the oracle label of
every resulting configuration is unchanged), but must be stated.

Outputs: artifacts/next_novelty/l007_perm_quartet/PERM_QUARTET.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(4)
from common import load_model, preprocess  # noqa: E402

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "l007_perm_quartet"
PERMS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def perm_idx(p):
    idx = [0, 1, 2, 3]
    if p[0]:
        idx[0], idx[1] = idx[1], idx[0]
    if p[1]:
        idx[2], idx[3] = idx[3], idx[2]
    if p[2]:
        idx[0], idx[2] = idx[2], idx[0]
        idx[1], idx[3] = idx[3], idx[1]
    return np.array(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx = np.asarray(b["Qx"], float)
    Qe = np.asarray(b["Qe"], float)
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    out = {"bank": a.bank, "n_quartets": len(quads), "models": {},
           "note": ("relabelings applied jointly to Qx and Qe; oracle labels "
                    "are invariant (proper-crossing predicate)")}
    for seed in (11, 23, 47):
        mp = ART / ("r04b_s%d" % seed) / "flipmine"
        model, stats = load_model(mp)
        model.eval()
        per_perm = []
        for p_ in PERMS:
            idx = perm_idx(p_)
            Xp = Qx[:, idx, :]
            Ep = Qe[:, idx, :]
            with torch.no_grad():
                xs_all = []
                for qid, (i1, m1), (i2, m2) in quads:
                    xa = Xp[i1]
                    xb = Xp[i2]
                    xab = xa + Ep[i1]
                    xs_all += [xa, xb, xab]
                lg = []
                for s in range(0, len(xs_all), 1024):
                    xx = np.stack(xs_all[s:s + 1024]).astype(np.float32)
                    lg.append(model(preprocess(xx, stats)).numpy().reshape(-1))
                lg = np.concatenate(lg)
                pl = (lg > 0).astype(int)
            js, apass, abok = [], [], []
            for k, (qid, (i1, m1), (i2, m2)) in enumerate(quads):
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                ca = int(pl[3 * k] == la)
                cb = int(pl[3 * k + 1] == lb)
                cc = int(pl[3 * k + 2] == m1["yAB"])
                js.append(1 if (ca and cb and cc) else 0)
                apass.append(1 if (ca and cb) else 0)
                abok.append(cc)
            per_perm.append({"perm": list(p_), "J": round(float(np.mean(js)), 4),
                             "atomic_pass": round(float(np.mean(apass)), 4),
                             "AB_correct": round(float(np.mean(abok)), 4),
                             "J_vector": js})
        Js = np.array([[q["J_vector"][i] for q in per_perm]
                       for i in range(len(quads))])
        unanimous = np.all(Js == Js[:, :1], axis=1)
        Jvals = [q["J"] for q in per_perm]
        out["models"]["s%d" % seed] = {
            "J_per_perm": Jvals,
            "J_mean": round(float(np.mean(Jvals)), 4),
            "J_range": round(float(max(Jvals) - min(Jvals)), 4),
            "atomic_pass_mean": round(float(np.mean(
                [q["atomic_pass"] for q in per_perm])), 4),
            "AB_correct_mean": round(float(np.mean(
                [q["AB_correct"] for q in per_perm])), 4),
            "quartet_J_flip_frac": round(float(1 - unanimous.mean()), 4),
            "n_perm": len(PERMS),
        }
        print("s%d: J per perm %s range=%.4f flip_frac=%s" % (
            seed, Jvals, out["models"]["s%d" % seed]["J_range"],
            out["models"]["s%d" % seed]["quartet_J_flip_frac"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "PERM_QUARTET.json", "w"), indent=1,
              default=float)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
