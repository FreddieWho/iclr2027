#!/usr/bin/env python3
"""U03 round1: pixel closed loop (same U01 geometry, pixel-only models).

Geometry batch = U01 fixed quartets (train 230 / eval 547). Each quartet's 4
states rendered with the SAME nuisance (per-quartet bg seed, lw, colors).
Arms (SmallCNN, same init seed, same 20ep recipe, counts recorded):
 clean: 2048 rendered train_101 scenes;
 pixflip: clean + 920 rendered flipmine flips (equal-count control);
 pixquartet: clean + 920 rendered quartet states (230x4).
Eval (same render style): rendered U01 eval quartets -> emergent miss,
atomic acc; rendered eval_202 subset -> static err. Visibility gate MINPX=20
with attempt totals recorded; geometry margin floor 0.02 (discriminability).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(2)
from paths import oracle_at
from n08_visual import render, SmallCNN

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
MINPX = 20


def render_q(state, qseed):
    return render(np.asarray(state, float), np.random.default_rng(qseed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=803)
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    qt = np.load(U01 / "quartets_train.npz")
    QTR = [(qt[f"q{i}"], qt["meta"][i]) for i in range(len(qt["meta"]))]
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]

    # train renders: clean scenes
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr, ytr = dtr["positions"].astype(float), dtr["labels"].astype(int)
    R_clean, y_clean = [], []
    for i in range(min(2048, len(Xtr) * 4)):
        x = Xtr[i % len(Xtr)]
        R_clean.append(render(x, rng))
        y_clean.append(ytr[i % len(Xtr)])
    # flipmine flips (equal-count 920 control)
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
    # quartet states (230x4 = 920)
    R_q, y_q = [], []
    for qi, (q, m) in enumerate(QTR):
        x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
        y0, ya, yb, yc = (int(v) for v in m)
        for st, yy in ((x, y0), (x + ea, ya), (x + eb, yb), (x + ea + eb, yc)):
            R_q.append(render_q(st, 9000 + qi))
            y_q.append(yy)
    # eval renders (same style): quartet states + static subset
    dev = np.load(ART / "scenes" / "eval_202" / "scenes.npz")
    Xe, ye = dev["positions"].astype(float), dev["labels"].astype(int)
    R_stat = [render(x, rng) for x in Xe[:256]]
    y_stat = list(ye[:256])
    E_end, E_endy, E_a, E_ay, E_b, E_by = [], [], [], [], [], []
    for qi, (q, m) in enumerate(QEV):
        x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
        ya, yb, yc = (int(v) for v in m[1:4])
        E_end.append(render_q(x + ea + eb, 50000 + qi))
        E_endy.append(yc)
        E_a.append(render_q(x + ea, 50000 + qi))
        E_ay.append(ya)
        E_b.append(render_q(x + eb, 50000 + qi))
        E_by.append(yb)

    def T(A):
        return torch.from_numpy(np.stack(A).astype(np.float32))

    arms = {"clean": (R_clean, y_clean),
            "pixflip": (R_clean + R_flip, y_clean + y_flip),
            "pixquartet": (R_clean + R_q, y_clean + y_q)}
    res = {"counts": {k: len(v[0]) for k, v in arms.items()},
           "flip_attempts": attempts, "n_eval_quartets": len(QEV)}
    for name, (R, yy) in arms.items():
        torch.manual_seed(a.seed)
        net = SmallCNN()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        bce = nn.BCEWithLogitsLoss()
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
        with torch.no_grad():
            pe = (net(T(E_end)).numpy() > 0).astype(int)
            pa = (net(T(E_a)).numpy() > 0).astype(int)
            pb = (net(T(E_b)).numpy() > 0).astype(int)
            ps = (net(T(R_stat)).numpy() > 0).astype(int)
        E_endy = np.array(E_endy)
        both = (pa == np.array(E_ay)) & (pb == np.array(E_by))
        res[name] = {"emergent_miss": float((pe != E_endy).mean()),
                     "atomic_acc": float(both.mean()),
                     "cond_miss": float((pe[both] != E_endy[both]).mean()) if both.sum() else None,
                     "static_err": float((ps != np.array(y_stat)).mean())}
        print(f"SAW {name}: emerg={res[name]['emergent_miss']:.3f} "
              f"atom={res[name]['atomic_acc']:.3f} static={res[name]['static_err']:.3f}",
              flush=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE U03")


if __name__ == "__main__":
    main()
