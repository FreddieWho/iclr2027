#!/usr/bin/env python3
"""U1 factorial re-derivation: representation x flip-supervision on dev512.

Four frozen arms evaluated on the SAME dev bank with the SAME quartet order:
  raw_clean    = coordinate raw, clean-only            (r04b_s{seed}/clean)
  raw_flipmine = coordinate raw, clean + flip BCE      (r04b_s{seed}/flipmine)
  relfeat      = 10-dim relational, clean-only         (r04b relfeat)
  relflip      = 10-dim relational, clean + flip BCE   (next_novelty/relflip/s{seed})

Per quartet / seed / arm records A/B/AB logits and correctness.
Summaries: J, paired arm differences, interaction
  I = (J_relflip - J_relfeat) - (J_rawflip - J_raw)
with parent-cluster bootstrap CIs on the SAME parent resample;
A/B rates, AB accuracy, R_full / R_endpoint / M and regression rates
of each repair arm against the raw_clean H set (base110).
Per-seed results plus a descriptive mean (never a seed x quartet pool).

NEW outputs only (never overwrites frozen result JSONs):
  artifacts/next_novelty/u1_factorial/U1_PERQUARTET.csv
  artifacts/next_novelty/u1_factorial/U1_SUMMARY.json
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

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "next_novelty" / "u1_factorial"
rng = np.random.default_rng(20260922)

ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_checkpoint(arm: str, seed: int) -> Path:
    if arm == "relflip":
        return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
    if arm == "raw_clean":
        return ART / ("r04b_s%d" % seed) / "clean"
    if arm == "raw_flipmine":
        return ART / ("r04b_s%d" % seed) / "flipmine"
    if arm == "relfeat":
        if seed == 11:
            return ART / "r04b_s11" / "relfeat"
        return ART / ("r04b_s%d_relfeat" % seed) / "relfeat"
    raise ValueError(arm)


def pcb(groups, stat_fn, n_boot=2000):
    ups = np.arange(len(groups))
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(ups, size=len(ups), replace=True)
        boots.append(stat_fn([groups[i] for i in draw]))
    q = np.quantile(boots, [0.025, 0.975])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def evaluate(model_dir: Path, quads, Qx, Qe):
    model, stats = load_model(model_dir)
    model.eval()
    out = {}
    with torch.no_grad():
        xs_all, keys = [], []
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(Qx[i1], float)
            xb = np.asarray(Qx[i2], float)
            xab = xa + np.asarray(Qe[i1], float)
            xs_all += [xa, xb, xab]
            keys.append((qid, m1, m2))
        lg_all = []
        for s in range(0, len(xs_all), 512):
            xx = np.stack(xs_all[s:s + 512]).astype(np.float32)
            lg_all.append(model(preprocess(xx, stats)).numpy().reshape(-1))
        lg = np.concatenate(lg_all)
        pl = (lg > 0).astype(int)
    for k, (qid, m1, m2) in enumerate(keys):
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        out[qid] = {
            "logits": [round(float(lg[3 * k]), 6), round(float(lg[3 * k + 1]), 6),
                       round(float(lg[3 * k + 2]), 6)],
            "correct": [int(pl[3 * k] == la), int(pl[3 * k + 1] == lb),
                        int(pl[3 * k + 2] == m1["yAB"])],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    bank_path = BANK / ("bank_" + a.bank + ".npz")
    bank_sha = sha_of(bank_path)
    b = np.load(bank_path, allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items() if len(ps) == 2}
    Q = sorted(qpar)
    par = np.array([qpar[q] for q in Q])
    Pall = [np.nonzero(par == u)[0] for u in np.unique(par)]

    R = {}
    model_sha = {}
    for seed in (11, 23, 47):
        R[seed] = {}
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model_sha["%s_s%d" % (arm, seed)] = sha_of(mp / "model.pt")
            R[seed][arm] = evaluate(mp, quads, Qx, Qe)

    def Jvec(seed, arm):
        return np.array([1.0 if R[seed][arm][q]["correct"] == [1, 1, 1] else 0.0
                         for q in Q])

    summary = {"bank": a.bank, "bank_sha256": bank_sha, "n_quartets": len(Q),
               "model_sha256": model_sha, "seeds": {}}
    for seed in (11, 23, 47):
        J = {arm: Jvec(seed, arm) for arm in ARMS}
        arms = {}
        for arm in ARMS:
            C = R[seed][arm]
            arms[arm] = {
                "J": round(float(J[arm].mean()), 4),
                "J_ci_parent": pcb(Pall, lambda idx: J[arm][np.concatenate(idx)].mean()),
                "atomic_A": round(float(np.mean([C[q]["correct"][0] for q in Q])), 4),
                "atomic_B": round(float(np.mean([C[q]["correct"][1] for q in Q])), 4),
                "atomic_pass": round(float(np.mean([C[q]["correct"][:2] == [1, 1] for q in Q])), 4),
                "AB_correct": round(float(np.mean([C[q]["correct"][2] for q in Q])), 4),
            }
        pairs = {}
        names = list(ARMS)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = J[names[j]] - J[names[i]]
                pairs["%s_minus_%s" % (names[j], names[i])] = [
                    round(float(d.mean()), 4),
                    pcb(Pall, lambda idx: d[np.concatenate(idx)].mean())]
        I = (J["relflip"] - J["relfeat"]) - (J["raw_flipmine"] - J["raw_clean"])
        inter = [round(float(I.mean()), 4),
                 pcb(Pall, lambda idx: I[np.concatenate(idx)].mean())]
        C = R[seed]["raw_clean"]
        base110 = [q for q in Q if C[q]["correct"] == [1, 1, 0]]
        bp = np.array([qpar[q] for q in base110])
        G = [np.nonzero(bp == u)[0] for u in np.unique(bp)] if base110 else []
        repair = {}
        for arm in ("raw_flipmine", "relfeat", "relflip"):
            F = R[seed][arm]
            Rf = np.array([1.0 if F[q]["correct"] == [1, 1, 1] else 0.0 for q in base110])
            Re = np.array([float(F[q]["correct"][2]) for q in base110])
            Mg = np.array([1.0 if (F[q]["correct"][2] == 1 and F[q]["correct"][:2] != [1, 1])
                           else 0.0 for q in base110])
            n = len(base110)
            ar = [q for q in Q if C[q]["correct"][:2] == [1, 1]]
            ab_ok = [q for q in Q if C[q]["correct"][2] == 1]
            repair[arm] = {
                "n_H": n,
                "R_full": [round(float(Rf.mean()), 4), pcb(G, lambda idx: Rf[np.concatenate(idx)].mean())] if n else None,
                "R_endpoint": [round(float(Re.mean()), 4), pcb(G, lambda idx: Re[np.concatenate(idx)].mean())] if n else None,
                "M_migrate": [round(float(Mg.mean()), 4), pcb(G, lambda idx: Mg[np.concatenate(idx)].mean())] if n else None,
                "identity_ok": bool(abs(float(Rf.mean() + Mg.mean() - Re.mean())) < 1e-9) if n else None,
                "atomic_regression": round(float(np.mean([F[q]["correct"][:2] != [1, 1] for q in ar])), 4) if ar else None,
                "AB_regression": round(float(np.mean([F[q]["correct"][2] == 0 for q in ab_ok])), 4) if ab_ok else None,
            }
        summary["seeds"]["s%d" % seed] = {
            "arms": arms, "paired_J_diff": pairs,
            "interaction_I": inter, "repair_vs_rawclean_H": repair,
            "n_H_rawclean": len(base110),
        }
    desc = {}
    for arm in ARMS:
        desc[arm] = round(float(np.mean(
            [summary["seeds"]["s%d" % s]["arms"][arm]["J"] for s in (11, 23, 47)])), 4)
    summary["descriptive_mean_J"] = desc

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "U1_PERQUARTET.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "qid", "parent", "logit_A", "logit_B",
                    "logit_AB", "correct_A", "correct_B", "correct_AB", "J"])
        for seed in (11, 23, 47):
            for arm in ARMS:
                for q in Q:
                    r = R[seed][arm][q]
                    w.writerow([seed, arm, q, qpar[q]] + r["logits"] + r["correct"] +
                               [int(r["correct"] == [1, 1, 1])])
    json.dump(summary, open(a.out / "U1_SUMMARY.json", "w"), indent=1)
    for seed in (11, 23, 47):
        s = summary["seeds"]["s%d" % seed]
        print("s%d J:" % seed,
              {a_: s["arms"][a_]["J"] for a_ in ARMS},
              "I:", s["interaction_I"], flush=True)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
