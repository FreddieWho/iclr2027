#!/usr/bin/env python3
"""E3: N06 full-path actions on coord (zero training).

Action = library edit applied as a whole path [x, x+e], scanned at 17 pts.
Success requires endpoint==0 AND a single 1->0 turn with no 0->1 re-entry.
Model picks by the same endpoint rule as N06 (blind to mid-path), so
endpoint-right-but-mid-violated failures isolate exactly what N03 adds.
Models: clean vs flipmine (strongest coord repair). lam in {0, 2}.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(2)
from paths import action_library, oracle_at
from common import load_model, preprocess

ART = ROOT / "artifacts" / "discovery_campaign"
MODELS = {"clean": ART / "r04b_s11" / "clean",
          "flipmine": ART / "r04b_s11" / "flipmine"}
N_SCAN = 17


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_parents", type=int, default=256)
    p.add_argument("--lam_costs", type=float, nargs="+", default=[0.0, 2.0])
    p.add_argument("--bigrot", action="store_true",
                   help="extend library with large rotations that can re-enter")
    a = p.parse_args()
    rng = np.random.default_rng(63)
    d = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    X = d["positions"].astype(np.float32)
    lib = action_library()
    bigrot_specs = []  # (nodes, angle): materialized per scene
    if a.bigrot:
        for ang in (0.3, -0.3, 0.5, -0.5, 0.8, -0.8):
            for nodes in ((0, 1), (2, 3)):
                bigrot_specs.append((list(nodes), ang))
    loaded = {}
    for k, mp in MODELS.items():
        m, s = load_model(mp)
        m.eval()
        loaded[k] = (m, s)

    cross = [i for i in rng.permutation(len(X))[:a.n_parents * 2]
             if oracle_at(X[i])[0] == 1][:a.n_parents]
    # oracle full-path outcomes per (scene, action)
    tasks = []
    for pi in cross:
        x = X[pi].astype(float)
        alle = list(lib)
        for nodes, ang in bigrot_specs:
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            cen = x[nodes].mean(axis=0)
            e = np.zeros((4, 2))
            e[nodes] = (x[nodes] - cen) @ R.T - (x[nodes] - cen)
            alle.append({"edit": e, "cost": float(np.linalg.norm(e)),
                         "family": f"rot{''.join(map(str, nodes))}_{ang}"})
        cands = []
        for act in alle:
            t = np.linspace(0, 1, N_SCAN)
            try:
                labs = [oracle_at(x + tt * act["edit"])[0] for tt in t]
            except ValueError:
                continue
            # valid full path: ends 0 with a single 1->0 turn, no 0->1 re-entry
            reentry = sum(1 for i in range(1, len(labs))
                          if labs[i - 1] == 0 and labs[i] == 1)
            cands.append({"edit": act["edit"], "cost": act["cost"],
                          "fam": act["family"], "end": labs[-1],
                          "mid_ok": reentry == 0})
        if not any(c["end"] == 0 and c["mid_ok"] for c in cands):
            continue  # no fully-valid action in library
        tasks.append({"parent": int(pi), "cands": cands})
    print(f"tasks={len(tasks)}/{len(cross)}", flush=True)

    res = {"n_tasks": len(tasks)}
    for name, (m, s) in loaded.items():
        frames = []
        for t in tasks:
            x = X[t["parent"]].astype(np.float32)
            for c in t["cands"]:
                frames.append(x + c["edit"])
        with torch.no_grad():
            lg = m(preprocess(np.stack(frames), s)).numpy()
        res[name] = {}
        k = 0
        for lam in a.lam_costs:
            kk = 0
            end_wrong = mid_viol = suc = 0
            for t in tasks:
                nc = len(t["cands"])
                P0 = 1 / (1 + np.exp(lg[kk:kk + nc]))
                kk += nc
                costs = np.array([c["cost"] for c in t["cands"]])
                j = int(np.argmax(P0 - lam * costs))
                c = t["cands"][j]
                if c["end"] != 0:
                    end_wrong += 1
                elif not c["mid_ok"]:
                    mid_viol += 1
                else:
                    suc += 1
            tot = end_wrong + mid_viol + suc
            res[name][f"lam{lam:g}"] = {
                "endpoint_wrong_rate": end_wrong / tot if tot else None,
                "mid_violated_rate": mid_viol / tot if tot else None,
                "success_rate": suc / tot if tot else None, "n": tot}
        k = k  # noqa
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    for k in loaded:
        for lam in a.lam_costs:
            v = res[k][f"lam{lam:g}"]
            print(f"SAW {k} lam{lam:g}: end_wrong={v['endpoint_wrong_rate']} "
                  f"mid_viol={v['mid_violated_rate']} success={v['success_rate']} (n={v['n']})")
    print("NEXT: mid_violated>0 with end rule -> N03-type failures infect "
          "endpoint-based action choice; flipmine comparison decides repair")
    print("CLAIM: endpoint-correct actions can fail mid-path")


if __name__ == "__main__":
    main()
