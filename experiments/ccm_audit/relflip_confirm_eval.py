#!/usr/bin/env python3
"""R3: parameterized relflip-confirm reproduce entry.

Evaluates FROZEN relflip checkpoints on a chosen sample bank and records
provenance: model SHA (state_dict hash), input bank SHA, raw-prediction
source (the checkpoint files themselves), aggregation source (this script +
bank manifest). Also emits the four-arm paired table on the SAME bank when
the raw/relfeat arms are available there; confirm-bank rows that lack a
paired arm are reported as missing (never stitched from other tables).

Usage:
  python3 experiments/ccm_audit/relflip_confirm_eval.py \
      --bank confirm1007 --seeds 11 23 47 \
      --model-map '{"s11":"artifacts/next_novelty/relflip/s11",...}' \
      --out artifacts/next_novelty/relflip_v2
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

BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
DISC = ROOT / "artifacts" / "discovery_campaign"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_checkpoint(arm: str, seed: int) -> Path:
    if arm == "relflip":
        return ROOT / "artifacts" / "next_novelty" / "relflip" / ("s%d" % seed)
    if arm == "raw_clean":
        return DISC / ("r04b_s%d" % seed) / "clean"
    if arm == "raw_flipmine":
        return DISC / ("r04b_s%d" % seed) / "flipmine"
    if arm == "relfeat":
        if seed == 11:
            return DISC / "r04b_s11" / "relfeat"
        return DISC / ("r04b_s%d_relfeat" % seed) / "relfeat"
    raise ValueError(arm)


def evaluate(model_dir: Path, bank: str, qx, qe, qmeta):
    model, stats = load_model(model_dir)
    model.eval()
    by_q = {}
    for qi, m in enumerate(qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    R = {}
    with torch.no_grad():
        xs_all, keys = [], []
        for qid, (i1, m1), (i2, m2) in quads:
            xa = np.asarray(qx[i1], float)
            xb = np.asarray(qx[i2], float)
            xab = xa + np.asarray(qe[i1], float)
            xs_all += [xa, xb, xab]
            keys.append((qid, m1, m2))
        out = []
        for s in range(0, len(xs_all), 512):
            xx = np.stack(xs_all[s:s + 512]).astype(np.float32)
            out.append(model(preprocess(xx, stats)).numpy().reshape(-1))
        lg = np.concatenate(out)
        pl = (lg > 0).astype(int)
    for k, (qid, m1, m2) in enumerate(keys):
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        R[qid] = (int(pl[3 * k] == la), int(pl[3 * k + 1] == lb),
                  int(pl[3 * k + 2] == m1["yAB"]))
    return R


def quartet_metrics(R):
    n = len(R)
    aA = sum(v[0] for v in R.values()) / n
    aB = sum(v[1] for v in R.values()) / n
    apass = sum(1 for v in R.values() if v[0] and v[1]) / n
    ab = sum(v[2] for v in R.values()) / n
    J = sum(1 for v in R.values() if v == (1, 1, 1)) / n
    own = [q for q in R if R[q][0] and R[q][1]]
    cond = 1 - sum(R[q][2] for q in own) / len(own) if own else None
    return {"n_q": n, "atomic_pass": round(apass, 4), "AB_correct": round(ab, 4),
            "cond_comp_miss": round(cond, 4) if cond is not None else None,
            "cond_n": len(own), "J": round(J, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="confirm1007",
                   help="bank stem under artifacts/p123_upgrade/bank (dev512/confirm1007)")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    p.add_argument("--model-map", default=None,
                   help='JSON dict {seed: model_dir} overriding arm_checkpoint for "relflip"')
    p.add_argument("--out", type=Path,
                   default=ROOT / "artifacts" / "next_novelty" / "relflip_v2")
    p.add_argument("--arms", nargs="+",
                   default=["relflip", "raw_clean", "raw_flipmine", "relfeat"],
                   help="arms to evaluate on this bank (missing arms reported as null)")
    a = p.parse_args()
    bank_path = BANK / ("bank_" + a.bank + ".npz")
    manifest_path = BANK / ("bank_" + a.bank + ".manifest.json")
    b = np.load(bank_path, allow_pickle=True)
    qx, qe = b["Qx"], b["Qe"]
    qmeta = json.loads(str(b["Qmeta"]))
    bank_sha = sha_of(bank_path)
    manifest = json.loads(manifest_path.read_text())
    model_map = json.loads(a.model_map) if a.model_map else {}

    res = {"bank": a.bank, "bank_sha256": bank_sha,
           "bank_manifest": manifest,
           "provenance": {
               "raw_prediction_source": "frozen checkpoints (model.pt) via common.load_model",
               "aggregation_source": "experiments/ccm_audit/relflip_confirm_eval.py",
               "feature_boundary": ("relfeat/relflip arms use hand-designed label-free "
                                    "relational features: six ordered pair distances + four "
                                    "centroid radii; node-pair identity is part of the input; "
                                    "task-aware deterministic representation, not a new "
                                    "foundation model and not oracle-free.")},
           "arms": {}}
    for arm in a.arms:
        res["arms"][arm] = {}
        for seed in a.seeds:
            try:
                mp = (ROOT / model_map[str(seed)] if str(seed) in model_map
                      else arm_checkpoint(arm, seed))
                if not (mp / "model.pt").exists():
                    res["arms"][arm]["s%d" % seed] = None
                    continue
                R = evaluate(mp, a.bank, qx, qe, qmeta)
                met = quartet_metrics(R)
                met["model_dir"] = str(mp.relative_to(ROOT))
                met["model_sha256"] = sha_of(mp / "model.pt")
                res["arms"][arm]["s%d" % seed] = met
                print(arm, seed, met, flush=True)
            except Exception as e:
                res["arms"][arm]["s%d" % seed] = {"error": str(e)}
    # migration on relflip confirm: R_full / M on base110 (A/B correct & AB wrong
    # under raw_clean arm) -- requires raw_clean on the SAME bank; else missing.
    res["migration"] = {}
    if "raw_clean" in a.arms and "relflip" in a.arms:
        rng = np.random.default_rng(26092247)
        for seed in a.seeds:
            rc = res["arms"]["raw_clean"].get("s%d" % seed)
            rf_arm = res["arms"]["relflip"].get("s%d" % seed)
            if not rc or not rf_arm or "error" in (rc or {}):
                res["migration"]["s%d" % seed] = None
                continue
            C = evaluate(arm_checkpoint("raw_clean", seed), a.bank, qx, qe, qmeta)
            F = evaluate(arm_checkpoint("relflip", seed), a.bank, qx, qe, qmeta)
            by_q = {}
            for qi, m in enumerate(qmeta):
                by_q.setdefault(m["qid"], []).append((qi, m))
            qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items() if len(ps) == 2}
            base110 = [q for q in C if C[q] == (1, 1, 0)]
            Rf = np.array([1.0 if F[q] == (1, 1, 1) else 0.0 for q in base110])
            Re = np.array([float(F[q][2]) for q in base110])
            Mg = np.array([1.0 if (F[q][2] == 1 and (F[q][0] == 0 or F[q][1] == 0))
                           else 0.0 for q in base110])
            def pcb(groups, vv):
                ups = np.arange(len(groups)); boots = []
                for _ in range(2000):
                    draw = rng.choice(ups, size=len(ups), replace=True)
                    boots.append(vv[np.concatenate([groups[i] for i in draw])].mean())
                q = np.quantile(boots, [0.025, 0.975])
                return [round(float(q[0]), 4), round(float(q[1]), 4)]
            G = [np.nonzero(np.array([qpar[q] for q in base110]) == u)[0]
                 for u in np.unique([qpar[q] for q in base110])]
            res["migration"]["s%d" % seed] = {
                "n110": len(base110),
                "R_full": [round(float(Rf.mean()), 4), pcb(G, Rf)],
                "R_endpoint": [round(float(Re.mean()), 4), pcb(G, Re)],
                "M": [round(float(Mg.mean()), 4), pcb(G, Mg)],
                "M_over_R_endpoint": round(float(Mg.mean() / Re.mean()), 4)}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "RELFLIP_CONFIRM.json", "w"), indent=1)
    print("DONE ->", a.out)


if __name__ == "__main__":
    main()
