#!/usr/bin/env python3
"""P1 unified evaluation table (§2): same parents, same confidence score.

Per model (s11/s23/s47 x clean/flipmine), two row sets from frozen bank:
 PATHS: join p1_coordinate rows (turn status etc.) + bank edit vectors.
 SINGLES: fresh model forwards on capped singles (40/parent), parent-linked.
Output: unified JSON per model + manifest. No resampling across models.
"""
import argparse
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
P1OLD = ROOT / "artifacts" / "p123_upgrade" / "p1"
OUT = ROOT / "artifacts" / "next_novelty" / "p1_unified"
PER_PARENT_CAP = 40


def fwd_logits(model, stats, xs):
    xs = np.asarray(xs, np.float32)
    with torch.no_grad():
        out = []
        for s in range(0, len(xs), 512):
            out.append(model(preprocess(xs[s:s + 512], stats)).numpy().reshape(-1))
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--p1old", default=str(P1OLD))
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    P1OLDL = Path(a.p1old)
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    Sx, Se = b["Sx"], b["Se"]
    Smeta = json.loads(str(b["Smeta"]))
    # cap singles per parent (deterministic: first N per parent in bank order)
    seen = {}
    keep = []
    for i, m in enumerate(Smeta):
        c = seen.get(m["parent"], 0)
        if c < PER_PARENT_CAP:
            seen[m["parent"]] = c + 1
            keep.append(i)
    keep = np.array(keep)
    a.out.mkdir(parents=True, exist_ok=True)
    for seed in (11, 23, 47):
        for mname in ("clean", "flipmine"):
            model, stats = load_model(ART / ("r04b_s%d" % seed) / mname)
            model.eval()
            old = json.load(open(P1OLDL / ("p1coord_s%d_%s.json" % (seed, mname))))["rows"]
            old_by_q = {(r["qid"], r["ptype"]): r for r in old}
            paths = []
            for qi, m in enumerate(Qmeta):
                key = (m["qid"], m["ptype"])
                r = old_by_q.get(key)
                if r is None:
                    continue  # non-single-cross: excluded from primary (counted below)
                x = np.asarray(Qx[qi], float)
                e = np.asarray(Qe[qi], float)
                lg_end = fwd_logits(model, stats, (x + e)[None])[0]
                pred_start = m["start_lab"] if r["start_correct"] else 1 - m["start_lab"]
                s_logit = r["start_conf"] if pred_start == 1 else -r["start_conf"]
                paths.append({
                    "qid": m["qid"], "parent": m["parent"], "ptype": m["ptype"],
                    "edit_fam": m["fam_B"] if m["ptype"] == "A" else m["fam_A"],
                    "edit_norm": float(np.linalg.norm(e)),
                    "start_lab": m["start_lab"], "end_lab": m["end_lab"],
                    "start_pred": pred_start, "start_correct": r["start_correct"],
                    "start_logit": float(s_logit), "confidence": float(r["start_conf"]),
                    "end_logit": float(lg_end),
                    "end_pred": int(lg_end > 0), "end_correct": r["endpoint_correct"],
                    "semantic_change": int(m["end_lab"] != m["start_lab"]),
                    "m0": m["m0"], "mAB": m["mAB"],
                    "t_star": r["t_star"],
                    "cross_dist": float(r["t_star"] * np.linalg.norm(e)),
                    "turn": r["outcome"], "lag": r["lag"],
                })
            X0 = Sx[keep].astype(np.float32)
            E0 = Se[keep].astype(np.float32)
            M0 = [Smeta[i] for i in keep]
            lg0 = fwd_logits(model, stats, X0)
            lg1 = fwd_logits(model, stats, (X0.reshape(len(X0), -1) + E0.reshape(len(E0), -1)).reshape(-1, 4, 2))
            singles = []
            for i, m in enumerate(M0):
                p0 = int(lg0[i] > 0)
                p1 = int(lg1[i] > 0)
                singles.append({
                    "eid": m["eid"], "parent": m["parent"], "fam": m["fam"],
                    "edit_norm": float(np.linalg.norm(E0[i])),
                    "y0": m["y0"], "y1": m["y1"],
                    "pred0": p0, "pred1": p1,
                    "correct0": int(p0 == m["y0"]), "correct1": int(p1 == m["y1"]),
                    "conf0": float(abs(lg0[i])), "logit0": float(lg0[i]),
                    "logit1": float(lg1[i]),
                    "flip": m["flip"], "m0": m["m0"], "m1": m["m1"],
                })
            fn = a.out / ("unified_s%d_%s.json" % (seed, mname))
            json.dump({"paths": paths, "singles": singles}, open(fn, "w"))
            nsc = sum(1 for r in paths if r["start_correct"])
            print("SAW s%d %s: paths=%d/%d start_correct=%d singles=%d" % (
                seed, mname, len(paths), len(Qmeta), nsc, len(singles)), flush=True)
    h = hashlib.sha256()
    for f in sorted(a.out.glob("unified_*.json")):
        h.update(open(f, "rb").read())
    json.dump({"bank": a.bank, "per_parent_cap": PER_PARENT_CAP,
               "sha256_inputs": h.hexdigest()},
              open(a.out / "manifest.json", "w"), indent=1)
    print("DONE unified", a.bank)


if __name__ == "__main__":
    main()
