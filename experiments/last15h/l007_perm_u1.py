#!/usr/bin/env python3
"""L007-C1c: is the U1 four-arm interaction stable under task relabeling?

The paper's newest claim is the super-additive interaction
  I = (J_relflip - J_relfeat) - (J_rawflip - J_raw)
measured on dev512 with ONE labeling. The oracle is permutation-invariant
(proper crossing), while the MLP is not, and C1 showed root location is
label-dependent. If I moves with the labeling, the claim needs an explicit
labeling-uncertainty caveat; if I is stable, the claim is strengthened.

Method (zero training, frozen checkpoints, same 237 quartets, 8 relabelings):
  - transform Qx and Qe jointly by each group element;
  - evaluate raw_clean / raw_flipmine / relfeat / relflip per seed;
  - report J per arm per relabeling, the per-relabeling I, and its spread,
    plus the paired (same-relabeling) arm differences.

Outputs: artifacts/next_novelty/l007_perm_u1/PERM_U1.json
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
OUT = ROOT / "artifacts" / "next_novelty" / "l007_perm_u1"
PERMS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")


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


def arm_dir(arm, seed):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="dev512")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    b = np.load(BANK / ("bank_" + a.bank + ".npz"), allow_pickle=True)
    Qx0 = np.asarray(b["Qx"], float)
    Qe0 = np.asarray(b["Qe"], float)
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    labels = []
    for qid, (i1, m1), (i2, m2) in quads:
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        labels.append((la, lb, m1["yAB"]))
    out = {"bank": a.bank, "n_quartets": len(quads), "arms": list(ARMS),
           "per_seed": {}}
    # --- oracle invariance re-verification on relabeled configurations ---
    # (pre-registered in PREREG_C1C_CONFIRM.md: recompute, do not assume)
    sys.path.insert(0, str(ROOT / "docs" / "iclr2027_discovery_campaign_20260917"))
    from core.relations import segment_relation  # noqa: E402
    mism = 0
    checked = 0
    for p_ in PERMS:
        idx = perm_idx(p_)
        Xp = Qx0[:, idx, :]
        Ep = Qe0[:, idx, :]
        for qid, (i1, m1), (i2, m2) in quads:
            la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
            lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
            for xx, want in ((Xp[i1], la), (Xp[i2], lb),
                             (Xp[i1] + Ep[i1], m1["yAB"])):
                try:
                    got = int(segment_relation(np.asarray(xx, float))["label"])
                except Exception:
                    continue
                checked += 1
                if got != int(want):
                    mism += 1
    out["oracle_invariance_check"] = {"checked": checked, "mismatches": mism}
    assert mism == 0, "oracle labels are NOT invariant under the relabelings"
    for seed in (11, 23, 47):
        models = {}
        for arm in ARMS:
            m, st = load_model(arm_dir(arm, seed))
            m.eval()
            models[arm] = (m, st)
        J = {arm: [] for arm in ARMS}
        vec = {arm: [] for arm in ARMS}
        for p_ in PERMS:
            idx = perm_idx(p_)
            Xp = Qx0[:, idx, :]
            Ep = Qe0[:, idx, :]
            for arm in ARMS:
                m, st = models[arm]
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
                        lg.append(m(preprocess(xx, st)).numpy().reshape(-1))
                    pl = (np.concatenate(lg) > 0).astype(int)
                js = []
                for k, (la, lb, lab) in enumerate(labels):
                    ca = int(pl[3 * k] == la)
                    cb = int(pl[3 * k + 1] == lb)
                    cc = int(pl[3 * k + 2] == lab)
                    js.append(1 if (ca and cb and cc) else 0)
                J[arm].append(round(float(np.mean(js)), 4))
                vec[arm].append(js)
        I = [round((J["relflip"][i] - J["relfeat"][i])
                   - (J["raw_flipmine"][i] - J["raw_clean"][i]), 4)
             for i in range(len(PERMS))]
        # paired arm differences per relabeling
        diffs = {
            "relflip_relfeat": [round(J["relflip"][i] - J["relfeat"][i], 4)
                                for i in range(len(PERMS))],
            "rawflip_raw": [round(J["raw_flipmine"][i] - J["raw_clean"][i], 4)
                            for i in range(len(PERMS))],
        }
        out["per_seed"]["s%d" % seed] = {
            "J": J, "I_per_perm": I,
            "I_identity": I[PERMS.index((0, 0, 0))],
            "I_mean": round(float(np.mean(I)), 4),
            "I_min": min(I), "I_max": max(I),
            "I_all_positive": bool(all(v > 0 for v in I)),
            "paired_diffs": diffs,
            "rawflip_raw_all_positive": bool(all(v > 0 for v in diffs["rawflip_raw"])),
            "relflip_relfeat_all_positive": bool(all(v > 0 for v in diffs["relflip_relfeat"])),
        }
        print("s%d: I per perm=%s (identity=%s, all>0=%s)" % (
            seed, I, out["per_seed"]["s%d" % seed]["I_identity"],
            out["per_seed"]["s%d" % seed]["I_all_positive"]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out / "PERM_U1.json", "w"), indent=1, default=float)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
