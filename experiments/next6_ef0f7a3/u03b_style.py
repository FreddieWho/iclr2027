#!/usr/bin/env python3
"""U03b: unseen visual style transfer (card round2).

Self-contained: retrains 3 pixel arms (same recipe as u03, own seed,
后续分析) then evaluates on FIXED U01 eval-quartet geometry rendered in
5 styles: S0 same-style replication; S1 lw=3; S2 colors swapped;
S3 bright bg; S4 scale 0.9. Geometry (hence discriminability) unchanged.
Only inference on new styles; no tuning.
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
from n08_visual import SmallCNN

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
IMG = 64
MINPX = 20


def render_s(x, rng, lw=2, swap=False, bg=0.5, scale=1.0):
    img = np.full((IMG, IMG, 3), bg + rng.uniform(-0.05, 0.05), np.float32)
    pts = np.asarray(x, float) * scale
    px = ((pts + 1) / 2 * (IMG - 1)).astype(int).clip(0, IMG - 1)
    cols = [(0.9, 0.1, 0.1)] * 2 + [(0.1, 0.1, 0.9)] * 2
    if swap:
        cols = cols[2:] + cols[:2]
    for s, (p, q) in enumerate([(px[0], px[1]), (px[2], px[3])]):
        n = int(np.hypot(*(q - p)) * 2) + 1
        for t in np.linspace(0, 1, max(n, 2)):
            r, c = int(round(p[0] + t * (q[0] - p[0]))), int(round(p[1] + t * (q[1] - p[1])))
            r0, r1 = max(0, r - lw), min(IMG, r + lw + 1)
            c0, c1 = max(0, c - lw), min(IMG, c + lw + 1)
            img[c0:c1, r0:r1] = cols[2 * s]
    return img.transpose(2, 0, 1)


STYLES = {"S0_same": {}, "S1_lw3": {"lw": 3}, "S2_swap": {"swap": True},
          "S3_bright": {"bg": 0.8}, "S4_scale09": {"scale": 0.9}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=804)
    p.add_argument("--epochs", type=int, default=20)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    qt = np.load(U01 / "quartets_train.npz")
    QTR = [(qt[f"q{i}"].reshape(3, 4, 2), qt["meta"][i]) for i in range(len(qt["meta"]))]
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"].reshape(3, 4, 2), qe["meta"][i]) for i in range(len(qe["meta"]))]
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr, ytr = dtr["positions"].astype(float), dtr["labels"].astype(int)

    R_clean = [render_s(Xtr[i % len(Xtr)], rng) for i in range(2048)]
    y_clean = [ytr[i % len(Xtr)] for i in range(2048)]
    mine = np.load(ART / "r04b_s11" / "mined.npz")
    meta = mine["meta"]
    R_flip, y_flip = [], []
    for i in rng.permutation(len(meta)):
        if len(R_flip) >= 920:
            break
        if f"edit_{i}" not in mine:
            continue
        pi = int(meta[i, 0])
        x, e = Xtr[pi], np.asarray(mine[f"edit_{i}"], dtype=float)
        try:
            y0, _, _ = oracle_at(x)
            y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 == y0 or m1 < 0.02:
            continue
        r0, r1 = render_s(x, rng), render_s(x + e, rng)
        if int((np.abs(r1 - r0).max(0) > 0.2).sum()) < MINPX:
            continue
        R_flip.append(r1)
        y_flip.append(y1)
    R_q, y_q = [], []
    for qi, (q, m) in enumerate(QTR):
        x, ea, eb = q
        y0, ya, yb, yc = (int(v) for v in m)
        for st, yy in ((x, y0), (x + ea, ya), (x + eb, yb), (x + ea + eb, yc)):
            R_q.append(render_s(st, np.random.default_rng(9000 + qi)))
            y_q.append(yy)

    def T(A):
        return torch.from_numpy(np.stack(A).astype(np.float32))

    arms = {"clean": (R_clean, y_clean),
            "pixflip": (R_clean + R_flip, y_clean + y_flip),
            "pixquartet": (R_clean + R_q, y_clean + y_q)}
    nets = {}
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
        nets[name] = net
    res = {"counts": {k: len(v[0]) for k, v in arms.items()},
           "n_eval_quartets": len(QEV)}
    for sname, kw in STYLES.items():
        E_end, E_endy, E_a, E_ay, E_b, E_by = [], [], [], [], [], []
        for qi, (q, m) in enumerate(QEV):
            x, ea, eb = q
            ya, yb, yc = (int(v) for v in m[1:4])
            rs = np.random.default_rng(50000 + qi)
            E_end.append(render_s(x + ea + eb, rs, **kw))
            E_endy.append(yc)
            E_a.append(render_s(x + ea, rs, **kw))
            E_ay.append(ya)
            E_b.append(render_s(x + eb, rs, **kw))
            E_by.append(yb)
        E_endy = np.array(E_endy)
        row = {}
        for name, net in nets.items():
            with torch.no_grad():
                pe = (net(T(E_end)).numpy() > 0).astype(int)
                pa = (net(T(E_a)).numpy() > 0).astype(int)
                pb = (net(T(E_b)).numpy() > 0).astype(int)
            both = (pa == np.array(E_ay)) & (pb == np.array(E_by))
            row[name] = {"emerg": round(float((pe != E_endy).mean()), 3),
                         "atom": round(float(both.mean()), 3)}
        res[sname] = row
        print(f"SAW {sname}: " + "; ".join(
            f"{k}=emerg{v['emerg']}/atom{v['atom']}" for k, v in row.items()),
            flush=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE U03b")


if __name__ == "__main__":
    main()
