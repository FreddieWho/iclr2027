#!/usr/bin/env python3
"""P2 §19: atomic pass + J for clean/affine/flipmine from unified tables.

Affine applied analytically to stored signed logits (no model reruns).
Quartets grouped by qid; A/B correctness from path starts, AB from endpoint.
Seeds 11/23/47. Affines from frozen P2_OPERATING fits.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UIN = ROOT / "artifacts" / "next_novelty" / "p1_unified"
OUT = ROOT / "artifacts" / "next_novelty" / "p2"
rng = np.random.default_rng(26092246)
AFF = {"s11": (1.0, -8.0), "s23": (0.5, -3.0), "s47": (0.5, -3.0)}


def quartets(seed, mname):
    d = json.load(open(UIN / ("unified_s%d_%s.json" % (seed, mname))))
    by_q = {}
    for r in d["paths"]:
        by_q.setdefault(r["qid"], {})[r["ptype"]] = r
    return [(q, v["A"], v["B"]) for q, v in by_q.items() if "A" in v and "B" in v]


def main():
    aff = {k: {"alpha": a, "b": b} for k, (a, b) in AFF.items()}
    json.dump({"fits": aff, "note": "from frozen P2_OPERATING wide-grid run"},
              open(OUT / "P2_AFFINE_FIT.json", "w"), indent=1)
    res = {}
    for seed in (11, 23, 47):
        alpha, b0 = AFF["s%d" % seed]
        out = {}
        for tag in ("clean", "flipmine"):
            quads = quartets(seed, tag)
            n = len(quads)
            a_ok = sum(1 for _, ra, rb in quads if ra["start_correct"] and rb["start_correct"])
            ab = [ra for _, ra, rb in quads for ra in (ra,)]
            j = sum(1 for _, ra, rb in quads
                    if ra["start_correct"] and rb["start_correct"] and ra["end_correct"])
            out[tag] = {"n_q": n,
                        "atomic_pass": round(a_ok / n, 4) if n else None,
                        "J": round(j / n, 4) if n else None}
        # affine from CLEAN logits
        quads = quartets(seed, "clean")
        n = len(quads)
        a_ok = ab_ok = jj = 0
        for _, ra, rb in quads:
            pa = int((ra["start_logit"] * alpha + b0) > 0) == ra["start_lab"]
            pb = int((rb["start_logit"] * alpha + b0) > 0) == rb["start_lab"]
            pc = int((ra["end_logit"] * alpha + b0) > 0) == ra["end_lab"]
            a_ok += (pa and pb)
            ab_ok += pc
            jj += (pa and pb and pc)
        out["affine"] = {"n_q": n,
                         "atomic_pass": round(a_ok / n, 4) if n else None,
                         "AB_end_acc": round(ab_ok / n, 4) if n else None,
                         "J": round(jj / n, 4) if n else None}
        # paired CI for J_affine - J_clean and atomic diffs (quartet bootstrap)
        dJ, dA = [], []
        qa = quartets(seed, "clean")
        for _ in range(2000):
            idx = rng.choice(len(qa), size=len(qa), replace=True)
            ja = sum(1 for i in idx
                     if ((qa[i][1]["start_logit"] * alpha + b0) > 0) == qa[i][1]["start_lab"]
                     and ((qa[i][2]["start_logit"] * alpha + b0) > 0) == qa[i][2]["start_lab"]
                     and ((qa[i][1]["end_logit"] * alpha + b0) > 0) == qa[i][1]["end_lab"]) / len(idx)
            jc = sum(1 for i in idx
                     if qa[i][1]["start_correct"] and qa[i][2]["start_correct"]
                     and qa[i][1]["end_correct"]) / len(idx)
            dJ.append(ja - jc)
        q = np.quantile(dJ, [0.025, 0.975])
        out["affine"]["J_minus_clean_ci"] = [round(float(q[0]), 4), round(float(q[1]), 4)]
        res["s%d" % seed] = out
        print("s%d:" % seed, {k: (v.get("atomic_pass"), v.get("J")) for k, v in out.items()}, flush=True)
    json.dump(res, open(OUT / "P2_AFFINE_ATOMS.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
