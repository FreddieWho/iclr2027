#!/usr/bin/env python3
"""D09 overlap audit: re-adjudicate Q1/Q2 excluding train_101 parents (<512).

D09 bank contains 7 E-parents from train_101 range (found via D03 exclusion
audit). Models were trained on train_101 static scenes, so those quartets
are not strictly unseen. Recompute gaps on parent>=512 subset only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
from u01_eval import load_arm, featurize_raw  # noqa: E402
from common import rel_features  # noqa: E402

torch.set_num_threads(4)
BANKDIR = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUTD = ROOT / "artifacts" / "f095_campaign" / "D09"


def eval_arm(mp, Qx, Qe, Qm, keep):
    model, ck = load_arm(Path(mp))
    model.eval()
    fe = ck.get("featurize")
    idx = [i for i, m in enumerate(Qm) if m["qid"] in keep]
    lgA, lgB, lgAB, lab = [], [], [], []
    with torch.no_grad():
        for qi in idx:
            m = Qm[qi]
            # find partner row (same qid, other ptype)
            xa = np.asarray(Qx[qi], float)
            if m["ptype"] == "A":
                pj = next(j for j, x in enumerate(Qm) if x["qid"] == m["qid"] and x["ptype"] == "B")
                xb = np.asarray(Qx[pj], float)
                xab = xa + np.asarray(Qe[qi], float)
                la, lb = m["yA"], Qm[pj]["yB"]
            else:
                continue
            if fe == "sixdist":
                blocks = []
                for arr in (xa, xb, xab):
                    mu = np.asarray(ck["mu"], dtype=np.float32)
                    sd = np.asarray(ck["sd"], dtype=np.float32)
                    xx = ((rel_features(arr[None])[0][:6] - mu) / sd).astype(np.float32)
                    blocks.append(model(torch.from_numpy(xx)).numpy().reshape(-1))
                la_, lb_, lab_ = blocks
                lgA.append(la_); lgB.append(lb_); lgAB.append(lab_)
            else:
                for arr, lst in ((xa, lgA), (xb, lgB), (xab, lgAB)):
                    if fe == "rel12":
                        mu = np.asarray(ck["mu"], dtype=np.float32)
                        sd = np.asarray(ck["sd"], dtype=np.float32)
                        xx = ((rel_features(arr[None])[0] - mu) / sd).astype(np.float32)
                        lst.append(model(torch.from_numpy(xx)).numpy().reshape(-1))
                    else:
                        lst.append(model(featurize_raw(arr[None], ck)).numpy().reshape(-1))
            lab.append((la, lb, m["yAB"]))
    sc = np.stack([np.concatenate(lgA), np.concatenate(lgB), np.concatenate(lgAB)], axis=1)
    pred = (sc > 0).astype(int)
    lab = np.array(lab)
    ok = (pred == lab)
    return round(float(ok.all(axis=1).mean()), 4), len(lab)


def main():
    b = np.load(BANKDIR / "bank_d09fresh665.npz", allow_pickle=True)
    Qx, Qe = np.asarray(b["Qx"], float), np.asarray(b["Qe"], float)
    Qm = json.loads(str(b["Qmeta"]))
    qids = sorted(set(m["qid"] for m in Qm if m["parent"] >= 512))
    dropped = sorted(set(m["qid"] for m in Qm if m["parent"] < 512))
    print(f"keep {len(qids)} qids, drop {len(dropped)} (train_101 parents)", flush=True)
    ART = ROOT / "artifacts"
    paths = {
        ("six", "clean"): str(ART / "f095_campaign" / "U02" / "sixdist_w64_s{seed}" / "clean"),
        ("six", "flip"): str(ART / "f095_campaign" / "U02" / "sixdist_w64_s{seed}" / "flipmine"),
        ("raw", "clean"): str(ART / "discovery_campaign" / "r04b_s{seed}" / "clean"),
        ("raw", "flip"): str(ART / "discovery_campaign" / "r04b_s{seed}" / "flipmine"),
    }
    out = {"dropped_qids": dropped, "seeds": {}}
    for seed in (11, 23, 47):
        row = {}
        for (enc, reg), pat in paths.items():
            J, n = eval_arm(pat.format(seed=seed), Qx, Qe, Qm, set(qids))
            row[f"{enc}_{reg}"] = {"J": J, "n": n}
            print(f"s{seed} {enc}-{reg} J={J} n={n}", flush=True)
        d1 = row["six_flip"]["J"] - row["raw_flip"]["J"]
        d2 = row["six_clean"]["J"] - row["raw_clean"]["J"]
        row["Q1_gap"] = round(d1, 4)
        row["Q2_gap"] = round(d2, 4)
        row["Q1_hit"] = bool(d1 >= 0.15)
        row["Q2_hit"] = bool(d2 >= 0.05)
        out["seeds"]["s%d" % seed] = row
    with open(OUTD / "D09_AUDIT.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUTD / "D09_AUDIT.json")


if __name__ == "__main__":
    main()
