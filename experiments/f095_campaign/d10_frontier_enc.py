#!/usr/bin/env python3
"""D10 encoder extension: same-coverage/same-cost frontier for rel12 and
sixdist encoders. Identical logic to d10_frontier.py (helpers imported),
only the logit source changes: logits recomputed on p2_table frames with
relfeat/relflip (four-arm frozen) and sixdist clean/flipmine (U02 frozen).
Question: does the matched-coverage decision gain replicate beyond raw?
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "repair_decomposition"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "ccm_audit"))
from common import rel_features  # noqa: E402
from d10_frontier import (N_BOOT, BOOT_SEED, fit_isotonic, match_ladder,  # noqa: E402
                          paired_own_quality_ci, policy_outcomes,
                          reachable_ladder, scene_maxp0, split_scenes)
from p2_decompose import load  # noqa: E402
from u01_eval import load_arm  # noqa: E402
from u1_factorial_eval import arm_checkpoint  # noqa: E402

torch.set_num_threads(4)
ART = ROOT / "artifacts" / "f095_campaign" / "D10"
SEEDS = (11, 23, 47)


@torch.no_grad()
def logits_on(frames, mp):
    model, ck = load_arm(Path(mp))
    model.eval()
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)
    fe = ck.get("featurize")
    out = []
    for s in range(0, len(frames), 2048):
        X = np.asarray(frames[s:s + 2048], np.float32)
        if fe == "rel12":
            F = rel_features(X)
        elif fe == "sixdist":
            F = rel_features(X)[:, :6]
        else:
            F = X.reshape(len(X), -1)
        F = torch.from_numpy(((F - mu) / sd).astype(np.float32))
        out.append(model(F).numpy().reshape(-1))
    return np.concatenate(out)


def main():
    d = load()
    feas = d["feas"].astype(bool)
    scene_of = d["scene_of"]
    act_cost_row = d["costs"][d["act_idx"]]
    frames = d["frames"]
    dev, ev = split_scenes(d, feas)
    rng = np.random.default_rng(BOOT_SEED)
    res = {"logic": "identical to d10_frontier.py (helpers imported); "
                    "logits recomputed per encoder",
           "table_sha256": hashlib.sha256(
               (ROOT / "artifacts" / "p123_upgrade" / "p2" / "p2_table.npz")
               .read_bytes()).hexdigest(),
           "pairs": {}}
    pairs = {}
    for seed in SEEDS:
        pairs[f"rel12_s{seed}"] = (
            ("relflip", arm_checkpoint("relflip", seed)),
            ("relfeat", arm_checkpoint("relfeat", seed)))
        u2 = ROOT / "artifacts" / "f095_campaign" / "U02" / f"sixdist_w64_s{seed}"
        pairs[f"sixdist_s{seed}"] = (
            ("sixflip", u2 / "flipmine"), ("sixclean", u2 / "clean"))
    for pname, ((fn, fmp), (cn, cmp_)) in pairs.items():
        Lg = {"flip": logits_on(frames, fmp), "clean": logits_on(frames, cmp_)}
        P0 = {m: -np.asarray(Lg[m], float) for m in Lg}
        ladders = {m: reachable_ladder(P0[m][np.isin(scene_of, dev)]) for m in Lg}
        dev_cov, dev_cost = {}, {}
        for m in Lg:
            dc, dco = [], []
            for t in ladders[m]:
                ac, _, co = policy_outcomes(P0[m], scene_of, act_cost_row,
                                            feas, dev, float(t))
                dc.append(float(ac.mean()))
                dco.append(float(co[ac].sum()))
            dev_cov[m], dev_cost[m] = np.array(dc), np.array(dco)
        R = {"ladder_sizes": {m: int(len(ladders[m])) for m in Lg},
             "flip_model_sha": hashlib.sha256(
                 (Path(fmp) / "model.pt").read_bytes()).hexdigest(),
             "clean_model_sha": hashlib.sha256(
                 (Path(cmp_) / "model.pt").read_bytes()).hexdigest(),
             "matched_same_coverage": [], "matched_same_cost": []}
        for tf, tc, gap in match_ladder(dev_cov["flip"], ladders["flip"],
                                        dev_cov["clean"], ladders["clean"]):
            ac, suf, cof = policy_outcomes(P0["flip"], scene_of, act_cost_row,
                                           feas, ev, tf)
            acc, suc, coc = policy_outcomes(P0["clean"], scene_of, act_cost_row,
                                            feas, ev, tc)
            qf = float(suf[ac].mean()) if ac.any() else None
            qc = float(suc[acc].mean()) if acc.any() else None
            row = {"tau_f": tf, "tau_c": tc, "dev_cov_gap": gap,
                   "eval_cov_f": round(float(ac.mean()), 4),
                   "eval_cov_c": round(float(acc.mean()), 4),
                   "eval_q_f": round(qf, 4) if qf is not None else None,
                   "eval_q_c": round(qc, 4) if qc is not None else None}
            if ac.any() and acc.any():
                row["own_quality_diff_CI"] = paired_own_quality_ci(
                    suf, ac, suc, acc, ev, rng)
            R["matched_same_coverage"].append(row)
        for tf, tc, gap in match_ladder(dev_cost["flip"], ladders["flip"],
                                        dev_cost["clean"], ladders["clean"]):
            ac, suf, cof = policy_outcomes(P0["flip"], scene_of, act_cost_row,
                                           feas, ev, tf)
            acc, suc, coc = policy_outcomes(P0["clean"], scene_of, act_cost_row,
                                            feas, ev, tc)
            qf = float(suf[ac].mean()) if ac.any() else None
            qc = float(suc[acc].mean()) if acc.any() else None
            R["matched_same_cost"].append(
                {"tau_f": tf, "tau_c": tc, "dev_cost_gap": gap,
                 "eval_q_f": round(qf, 4) if qf is not None else None,
                 "eval_q_c": round(qc, 4) if qc is not None else None})
        deltas = [r["eval_q_f"] - r["eval_q_c"] for r in R["matched_same_coverage"]
                  if r["eval_q_f"] is not None and r["eval_q_c"] is not None]
        R["median_gain_same_coverage"] = round(float(np.median(deltas)), 4) \
            if deltas else None
        R["frac_positive"] = round(float(np.mean([v > 0 for v in deltas])), 4) \
            if deltas else None
        res["pairs"][pname] = R
        print(f"{pname}: n_pairs={len(deltas)} median_gain={R['median_gain_same_coverage']} "
              f"frac>0={R['frac_positive']}", flush=True)
    with open(ART / "D10_ENC_SUMMARY.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", ART / "D10_ENC_SUMMARY.json")


if __name__ == "__main__":
    main()
