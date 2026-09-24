"""Re-evaluate L-015/L-006 checkpoints on the paper's dev512 quartet bank.

The training script's own miner found only 2 E pairs. That denominator is discarded.
This script uses the same 3-state joint definition as
experiments/ccm_audit/u1_factorial_eval.py on artifacts/p123_upgrade/bank/bank_dev512.npz
(237 quartets). It does not read confirm_1007 or any other sealed pool.

J = 1 iff path-A, path-B, and the composition are all correct.
Labeling sweep uses the same 8-element group as C1c. Oracle labels are invariant;
model inputs are relabeled jointly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "e1a933_review"))
spec = importlib.util.spec_from_file_location(
    "leads_train", ROOT / "experiments" / "e1a933_review" / "leads_l014_l015_l006.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

BANK = ROOT / "artifacts" / "p123_upgrade" / "bank" / "bank_dev512.npz"
TRAIN = ROOT / "artifacts" / "f095_campaign" / "D01" / "scenes_N" / "scenes.npz"


def load_quads():
    b = np.load(BANK, allow_pickle=True)
    Qx = np.asarray(b["Qx"], np.float32)
    Qe = np.asarray(b["Qe"], np.float32)
    meta = json.loads(str(b["Qmeta"]))
    by_q = {}
    for i, m in enumerate(meta):
        by_q.setdefault(m["qid"], []).append((i, m))
    quads = []
    for qid, rows in by_q.items():
        if len(rows) != 2:
            continue
        quads.append((qid, rows[0], rows[1]))
    return Qx, Qe, quads


def j_vec(model, arm, stats, Qx, Qe, quads, perm):
    idx = list(perm)
    Xp = Qx[:, idx, :]
    Ep = Qe[:, idx, :]
    xs = []
    labels = []
    parents = []
    for qid, (i1, m1), (i2, m2) in quads:
        xs += [Xp[i1], Xp[i2], Xp[i1] + Ep[i1]]
        la = m1["yA"] if m1["ptype"] == "A" else m1["yB"]
        lb = m2["yA"] if m2["ptype"] == "A" else m2["yB"]
        labels.append((int(la), int(lb), int(m1["yAB"])))
        parents.append(int(m1["parent"]))
    pred = T.predict(model, arm, np.stack(xs), stats) > 0
    ok = []
    for k, (la, lb, lab) in enumerate(labels):
        ok.append(int(pred[3 * k] == la and pred[3 * k + 1] == lb and pred[3 * k + 2] == lab))
    return np.array(ok, int), np.array(parents, int)


def parent_boot(values, parents, rng, n_boot=2000):
    groups = {}
    for v, p in zip(values, parents):
        groups.setdefault(int(p), []).append(float(v))
    keys = list(groups)
    mats = [np.array(groups[k]) for k in keys]
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(mats), len(mats))
        boots.append(np.concatenate([mats[i] for i in ix]).mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(np.mean(values)), "ci95": [float(lo), float(hi)],
            "n": int(len(values)), "n_parents": int(len(keys))}


def main():
    out_dir = Path(sys.argv[1])
    Qx, Qe, quads = load_quads()
    train = np.load(TRAIN)
    xtr = train["positions"].astype(np.float32)
    ident = T.G8[0]
    rows = []
    rng = np.random.default_rng(924)
    for arm in T.ARMS:
        stats = T.fit_stats(arm, xtr)
        for seed in T.SEEDS:
            ckpt = out_dir / f"{arm}_s{seed}" / "model.pt"
            if not ckpt.exists():
                raise SystemExit(f"missing {ckpt}")
            model = T.build(arm)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state"])
            model.eval()
            per = {}
            parents = None
            logit_gap = 0.0
            base_pred = None
            for perm in T.G8:
                ok, parents = j_vec(model, arm, stats, Qx, Qe, quads, perm)
                per[perm] = ok
            # invariance probe: identity vs each perm, on the three evaluated states
            j_id = per[ident]
            js = [float(v.mean()) for v in per.values()]
            changed = [float(np.mean(v != j_id)) for p, v in per.items() if p != ident]
            row = {
                "arm": arm, "seed": int(seed),
                "J": float(j_id.mean()),
                "J_min": float(min(js)), "J_max": float(max(js)),
                "J_swing": float(max(js) - min(js)),
                "status_flip_mean": float(np.mean(changed)),
                "n": int(len(j_id)), "n_parents": int(len(np.unique(parents))),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    # paired seed-level and pooled identity-labeling contrasts vs raw
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    contrasts = {}
    for name in ("permaug", "setmlp", "relfeat"):
        deltas = []
        for seed, arms in by_seed.items():
            deltas.append(arms[name]["J"] - arms["raw"]["J"])
        deltas = np.array(deltas)
        contrasts[name] = {
            "delta_mean": float(deltas.mean()),
            "delta_sd": float(deltas.std(ddof=1)),
            "n_positive": int((deltas > 0).sum()),
            "n_ge_12pp": int((deltas >= 0.12).sum()),
            "min": float(deltas.min()), "max": float(deltas.max()),
            "n_seeds": int(len(deltas)),
        }
    # parent-clustered CI on seed 11 identity labeling is not the claim;
    # pool seeds only as a descriptive mean, and also CI within each seed.
    per_seed_ci = []
    for arm in T.ARMS:
        stats = T.fit_stats(arm, xtr)
        for seed in T.SEEDS:
            model = T.build(arm)
            model.load_state_dict(torch.load(out_dir / f"{arm}_s{seed}" / "model.pt", map_location="cpu", weights_only=False)["state"])
            model.eval()
            ok, parents = j_vec(model, arm, stats, Qx, Qe, quads, ident)
            per_seed_ci.append({"arm": arm, "seed": int(seed), **parent_boot(ok, parents, rng)})
    out = {
        "bank": str(BANK.relative_to(ROOT)),
        "definition": "3-state joint J, same as u1_factorial_eval.py; identity labeling is G8[0]",
        "discarded_in_script_eval": "training script miner returned 2 quartets; those J values are not used",
        "sealed_pools": "not read",
        "rows": rows,
        "vs_raw_across_seeds": contrasts,
        "per_seed_parent_ci": per_seed_ci,
    }
    (out_dir / "REEVAL_DEV512.json").write_text(json.dumps(out, indent=2))
    print("wrote", out_dir / "REEVAL_DEV512.json", flush=True)


if __name__ == "__main__":
    main()
