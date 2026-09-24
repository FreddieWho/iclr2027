#!/usr/bin/env python3
"""D02 orbit evaluation: group-averaged predictions on the dev bank.

For each frozen four-arm checkpoint x seed, evaluates the 8 legal
relabelings of every quartet (SAME group element for all states/edits:
permuted coordinates, features recomputed from transformed coordinates),
then logit-averages f_G(x) = mean_g f(gx). Reports group-averaged J plus
orbit statistics (per-g J max-min, all-8-correct, any-wrong, atomic pass,
orbit disagreement). Extra cost: 8x forwards, recorded in the manifest.
No training, no sealed-pool reads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "f095_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
from symmetry import GROUP, apply  # noqa: E402
from common import load_model, preprocess  # noqa: E402

import torch  # noqa: E402
torch.set_num_threads(4)

ART = ROOT / "artifacts" / "discovery_campaign"
BANK = ROOT / "artifacts" / "p123_upgrade" / "bank"
OUT = ROOT / "artifacts" / "f095_campaign" / "D02"
ARMS = ("raw_clean", "raw_flipmine", "relfeat", "relflip")


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


def forward_all(model, stats, xs: np.ndarray) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(xs), 512):
            xx = np.stack(xs[s:s + 512]).astype(np.float32)
            out.append(model(preprocess(xx, stats)).numpy().reshape(-1))
    return np.concatenate(out)


def groupavg_orbit_metrics(per_g, labels):
    """Evaluate f_G(hx) via the exact group-composition permutation of saved scores."""
    orbit = []
    for h in GROUP:
        composition = [GROUP.index(tuple(h[i] for i in g)) for g in GROUP]
        orbit.append(per_g[:, :, composition].mean(axis=2))
    orbit = np.stack(orbit, axis=2)
    delta = float(np.max(np.abs(orbit - orbit[:, :, :1])))
    assert np.allclose(orbit, orbit[:, :, :1], rtol=1e-12, atol=1e-12)
    all8 = float(((orbit > 0) == labels[:, :, None]).all(axis=(1, 2)).mean())
    identity = float(((orbit[:, :, 0] > 0) == labels).all(axis=1).mean())
    assert all8 == identity
    return all8, delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/e1a933_review/data_d02")
    p.add_argument("--bank", default="dev512")
    a = p.parse_args()
    global OUT
    OUT = a.out_dir
    if (OUT / "D02_SUMMARY.json").exists():
        raise FileExistsError("Use a fresh --out-dir")
    bank_path = BANK / ("bank_" + a.bank + ".npz")
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    b = np.load(bank_path, allow_pickle=True)
    Qx, Qe = b["Qx"], b["Qe"]
    Qmeta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for qi, m in enumerate(Qmeta):
        by_q.setdefault(m["qid"], []).append((qi, m))
    quads = [(qid, ps[0], ps[1]) for qid, ps in by_q.items() if len(ps) == 2]
    qpar = {qid: ps[0][1]["parent"] for qid, ps in by_q.items() if len(ps) == 2}
    Q = sorted(qpar)

    summary = {"bank": a.bank, "bank_sha256": bank_sha, "group": [list(g) for g in GROUP],
               "forward_multiplier": len(GROUP), "seeds": {}}
    for seed in (11, 23, 47):
        arms = {}
        for arm in ARMS:
            mp = arm_checkpoint(arm, seed)
            model, stats = load_model(mp)
            model.eval()
            msha = hashlib.sha256((mp / "model.pt").read_bytes()).hexdigest()
            # base states once; per-g transform then forward
            base = {}
            for qid, (i1, m1), (i2, m2) in quads:
                xa = np.asarray(Qx[i1], float).reshape(4, 2)
                xb = np.asarray(Qx[i2], float).reshape(4, 2)
                xab = xa + np.asarray(Qe[i1], float).reshape(4, 2)
                la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
                lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
                base[qid] = (xa, xb, xab, (la, lb, m1["yAB"]))
            # per-g logits: [n_q, 3, 8]
            per_g = np.zeros((len(Q), 3, len(GROUP)))
            order = [(qid, base[qid]) for qid in Q]
            flats = [[], [], []]  # per state index, stacked over (q, g)
            for qi, (qid, (xa, xb, xab, _)) in enumerate(order):
                for gi, g in enumerate(GROUP):
                    flats[0].append(apply(xa, g).reshape(8))
                    flats[1].append(apply(xb, g).reshape(8))
                    flats[2].append(apply(xab, g).reshape(8))
            for si in range(3):
                lg = forward_all(model, stats, flats[si])
                per_g[:, si, :] = lg.reshape(len(Q), len(GROUP))
            avg = per_g.mean(axis=2)  # [n_q, 3] group-averaged logits
            labels = np.array([base[q][3] for q in Q])
            pred_avg = (avg > 0).astype(int)
            ok_avg = (pred_avg == labels)
            J_avg = float((ok_avg.all(axis=1)).mean())
            ap_avg = float((ok_avg[:, :2].all(axis=1)).mean())
            # per-g correctness cube
            pred_g = (per_g > 0).astype(int)  # [n,3,8]
            ok_g = (pred_g == labels[:, :, None])
            J_g = ok_g.all(axis=1).mean(axis=0)  # [8]
            all8 = float(ok_g.all(axis=(1, 2)).mean())
            anywrong = float((~ok_g.all(axis=1)).any(axis=1).mean())
            atom_g = ok_g[:, :2, :].all(axis=1)  # [n,8]
            dis = float((atom_g.max(axis=1) != atom_g.min(axis=1)).mean())
            all8_avg, max_avg_diff = groupavg_orbit_metrics(per_g, labels)
            arms[arm] = {
                "model_sha256": msha,
                "J_groupavg": round(J_avg, 4),
                "atomic_pass_groupavg": round(ap_avg, 4),
                "per_g_J": [round(float(v), 4) for v in J_g],
                "per_g_J_maxmin": [round(float(J_g.max() - J_g.min()), 4)],
                "all8_raw": all8,
                "all8_groupavg": all8_avg,
                "groupavg_max_orbit_logit_difference": max_avg_diff,
                "groupavg_orbit_equality": "exact group closure; same logits multiset for every relabeling",
                "any_g_wrong_frac": round(anywrong, 4),
                "atomic_orbit_disagreement": round(dis, 4),
            }
            print(f"s{seed} {arm:12s} J_avg={J_avg:.4f} "
                  f"per-g_J=[{J_g.min():.4f},{J_g.max():.4f}] all8={all8:.4f}", flush=True)
        summary["seeds"]["s%d" % seed] = arms
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "D02_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote", OUT / "D02_SUMMARY.json")


if __name__ == "__main__":
    main()
