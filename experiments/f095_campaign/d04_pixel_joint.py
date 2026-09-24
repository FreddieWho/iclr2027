#!/usr/bin/env python3
"""D04 stage A redo (CPU): retrain SmallCNN pixel arms WITH saved weights.

Same recipe as next6 u03/u03b (20ep, Adam 3e-3, batch128, MINPX=20 gate,
margin floor 0.02, U01 fixed quartets), relabeled as re-run (NOT original
weights — originals were never saved). Adds what D04 stage A requires:
same-parent same-noise-seed x/A/B/AB renders -> joint J, atomic-pass,
E-subset CCM + R_full/M, and 5-style stratification.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
sys.path.insert(0, str(ROOT / "experiments" / "next6_ef0f7a3"))
torch.set_num_threads(4)
from paths import oracle_at
from n08_visual import render, SmallCNN
from u03b_style import render_s, STYLES

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
MINPX = 20
IMG = 64


def render_q(state, qseed):
    return render(np.asarray(state, float), np.random.default_rng(qseed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=803)
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    qt = np.load(U01 / "quartets_train.npz")
    QTR = [(qt[f"q{i}"], qt["meta"][i]) for i in range(len(qt["meta"]))]
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr, ytr = dtr["positions"].astype(float), dtr["labels"].astype(int)
    R_clean, y_clean = [], []
    for i in range(min(2048, len(Xtr) * 4)):
        x = Xtr[i % len(Xtr)]
        R_clean.append(render(x, rng))
        y_clean.append(ytr[i % len(Xtr)])
    mine = np.load(ART / "r04b_s11" / "mined.npz")
    meta = mine["meta"]
    R_flip, y_flip, attempts = [], [], 0
    for i in rng.permutation(len(meta)):
        if len(R_flip) >= 920:
            break
        if f"edit_{i}" not in mine:
            continue
        pi = int(meta[i, 0])
        x, e = Xtr[pi], np.asarray(mine[f"edit_{i}"], dtype=float)
        attempts += 1
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 == y0 or m1 < 0.02:
            continue
        r0, r1 = render(x, rng), render(x + e, rng)
        if int((np.abs(r1 - r0).max(0) > 0.2).sum()) < MINPX:
            continue
        R_flip.append(r1)
        y_flip.append(y1)
    R_q, y_q = [], []
    for qi, (q, m) in enumerate(QTR):
        x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
        y0, ya, yb, yc = (int(v) for v in m)
        for st, yy in ((x, y0), (x + ea, ya), (x + eb, yb), (x + ea + eb, yc)):
            R_q.append(render_q(st, 9000 + qi))
            y_q.append(yy)

    def T(A):
        return torch.from_numpy(np.stack(A).astype(np.float32))

    arms = {"clean": (R_clean, y_clean),
            "pixflip": (R_clean + R_flip, y_clean + y_flip),
            "pixquartet": (R_clean + R_q, y_clean + y_q)}
    res = {"recipe": "u03-20ep-Adam3e3-b128-MINPX20", "seed": a.seed,
           "rerun": True, "not_original_weights": True,
           "counts": {k: len(v[0]) for k, v in arms.items()},
           "flip_attempts": attempts, "n_eval_quartets": len(QEV)}
    bce = nn.BCEWithLogitsLoss()
    for name, (R, yy) in arms.items():
        torch.manual_seed(a.seed)
        net = SmallCNN()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        Rb = T(R)
        yb = torch.from_numpy(np.array(yy, dtype=np.float32))
        net.train()
        for ep in range(a.epochs):
            perm = torch.randperm(len(Rb))
            for i in range(0, len(Rb), 128):
                b = perm[i:i + 128]
                opt.zero_grad()
                loss = bce(net(Rb[b]), yb[b])
                loss.backward()
                opt.step()
        net.eval()
        (a.out / name).mkdir(exist_ok=True)
        torch.save({"state": net.state_dict(), "seed": a.seed,
                    "recipe": res["recipe"]}, a.out / name / "model.pt")
        # full-quartet eval, same parent + same noise seed per state
        preds, labels = {}, {}
        states = {}
        for qi, (q, m) in enumerate(QEV):
            x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
            y0, ya, yb, yc = (int(v) for v in m)
            for sname, st, yl in (("x0", x, y0), ("A", x + ea, ya),
                                  ("B", x + eb, yb), ("AB", x + ea + eb, yc)):
                states.setdefault(sname, []).append((render_q(st, 50000 + qi), yl, qi))
        with torch.no_grad():
            for sname, lst in states.items():
                pl = (net(T([r for r, _, _ in lst])).numpy() > 0).astype(int)
                preds[sname] = pl
                labels[sname] = np.array([yl for _, yl, _ in lst])
        n = len(QEV)
        ok0 = preds["x0"] == labels["x0"]
        okA = preds["A"] == labels["A"]
        okB = preds["B"] == labels["B"]
        okAB = preds["AB"] == labels["AB"]
        E = (labels["A"] == labels["B"]) & (labels["B"] == labels["x0"]) & \
            (labels["AB"] != labels["x0"])
        res[name] = {
            "static_acc": round(float(ok0.mean()), 4),
            "atomic_acc": round(float((okA & okB).mean()), 4),
            "AB_acc": round(float(okAB.mean()), 4),
            "J": round(float((okA & okB & okAB).mean()), 4),
            "E_n": int(E.sum()),
            "E_J": round(float((okA & okB & okAB)[E].mean()), 4) if E.sum() else None,
            "E_CCM_denom": int(((okA & okB)[E]).sum()),
        }
        print(f"SAW {name}: J={res[name]['J']} E_J={res[name]['E_J']} E_n={res[name]['E_n']}",
              flush=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE d04_pixel_joint")


if __name__ == "__main__":
    main()
