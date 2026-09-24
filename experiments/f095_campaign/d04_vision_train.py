#!/usr/bin/env python3
"""D04 stage B (local CPU): ResNet18 strong-control on pixel renders.

Arms (same data as stage-A pixflip: 2048 clean + 920 flips, 64px native;
one 224-upsample probe seed): pretrained-full-finetune, random-init
control, head-only reference. Same recipe all arms (Adam 3e-4, 20ep,
batch128); static/single-dev selection only. Eval = stage-A protocol
(x/A/B/AB same-parent same-noise-seed -> J/atomic/E-CCM/R_full/M).
CPU-measured: ~0.3s/step -> ~2.5min/arm at 64px; 224 probe slower, 1 seed.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "last15h"))
torch.set_num_threads(8)
from paths import oracle_at
from n08_visual import render

ART = ROOT / "artifacts" / "discovery_campaign"
U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"
MINPX, IMG = 20, 64
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def build_data(seed, return_coords=False):
    rng = np.random.default_rng(seed)
    dtr = np.load(ART / "scenes" / "train_101" / "scenes.npz")
    Xtr, ytr = dtr["positions"].astype(float), dtr["labels"].astype(int)
    R, Y, C = [], [], []
    for i in range(min(2048, len(Xtr) * 4)):
        x = Xtr[i % len(Xtr)]
        R.append(render(x, rng)); Y.append(ytr[i % len(Xtr)]); C.append(x)
    mine = np.load(ART / "r04b_s11" / "mined.npz")
    meta = mine["meta"]
    attempts = 0
    for i in rng.permutation(len(meta)):
        if len(R) >= 2048 + 920:
            break
        if f"edit_{i}" not in mine:
            continue
        pi = int(meta[i, 0])
        x, e = Xtr[pi], np.asarray(mine[f"edit_{i}"], dtype=float)
        attempts += 1
        try:
            y0, _, _ = oracle_at(x); y1, m1, _ = oracle_at(x + e)
        except ValueError:
            continue
        if y1 == y0 or m1 < 0.02:
            continue
        r0, r1 = render(x, rng), render(x + e, rng)
        if int((np.abs(r1 - r0).max(0) > 0.2).sum()) < MINPX:
            continue
        R.append(r1); Y.append(y1); C.append(x + e)
    if return_coords:
        return R, Y, attempts, np.asarray(C, dtype=float)
    return R, Y, attempts


def prep(imgs, res):
    t = torch.from_numpy(np.stack(imgs).astype(np.float32))
    if res != IMG:
        t = torch.nn.functional.interpolate(t, size=(res, res), mode="bilinear",
                                            align_corners=False)
    return (t - IMNET_MEAN) / IMNET_STD


def make_net(mode, seed):
    torch.manual_seed(seed)
    if mode == "pretrained":
        net = resnet18(weights="IMAGENET1K_V1")
    else:
        net = resnet18(weights=None)
    net.fc = nn.Linear(512, 1)
    if mode == "headonly":
        base = resnet18(weights="IMAGENET1K_V1")
        net = resnet18(weights=None)
        net.load_state_dict(base.state_dict(), strict=False)
        net.fc = nn.Linear(512, 1)
        for n, p in net.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=803)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--arms", type=str, default="pretrained,random,headonly")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    R, Y, attempts = build_data(a.seed)
    qe = np.load(U01 / "quartets_eval.npz")
    QEV = [(qe[f"q{i}"], qe["meta"][i]) for i in range(len(qe["meta"]))]
    Xb = prep(R, a.res)
    yb = torch.from_numpy(np.array(Y, dtype=np.float32))
    res = {"seed": a.seed, "res": a.res, "recipe": "Adam3e-4-20ep-b128-IMAGENETnorm",
           "n_train": len(R), "flip_attempts": attempts,
           "rerun": True, "not_original_weights": True}
    bce = nn.BCEWithLogitsLoss()
    for mode in a.arms.split(","):
        torch.manual_seed(a.seed)
        net = make_net(mode.strip(), a.seed)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=3e-4)
        net.train()
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xb))
            for i in range(0, len(Xb), 128):
                b = perm[i:i + 128]
                opt.zero_grad()
                loss = bce(net(Xb[b]).squeeze(-1), yb[b])
                loss.backward(); opt.step()
        net.eval()
        (a.out / mode.strip()).mkdir(exist_ok=True)
        torch.save({"state": net.state_dict(), "seed": a.seed, "mode": mode.strip(),
                    "res": a.res}, a.out / mode.strip() / "model.pt")
        Ps, Ls = [], []
        with torch.no_grad():
            for qi, (q, m) in enumerate(QEV):
                x, ea, eb = (np.asarray(q[i], float).reshape(4, 2) for i in range(3))
                y0, ya_, yb_, yc = (int(v) for v in m)
                sts = [(x, y0), (x + ea, ya_), (x + eb, yb_), (x + ea + eb, yc)]
                imgs = [render(np.asarray(s, float),
                               np.random.default_rng(50000 + qi)) for s, _ in sts]
                pl = (net(prep(imgs, a.res)).squeeze(-1).numpy() > 0).astype(int)
                Ps.append(pl); Ls.append([yl for _, yl in sts])
        P, L = np.array(Ps), np.array(Ls)
        okA, okB, okAB = P[:, 1] == L[:, 1], P[:, 2] == L[:, 2], P[:, 3] == L[:, 3]
        E = (L[:, 1] == L[:, 2]) & (L[:, 2] == L[:, 0]) & (L[:, 3] != L[:, 0])
        res[mode.strip()] = {
            "static_acc": round(float((P[:, 0] == L[:, 0]).mean()), 4),
            "atomic_acc": round(float((okA & okB).mean()), 4),
            "AB_acc": round(float(okAB.mean()), 4),
            "J": round(float((okA & okB & okAB).mean()), 4),
            "E_n": int(E.sum()),
            "E_J": round(float((okA & okB & okAB)[E].mean()), 4) if E.sum() else None,
            "E_CCM_denom": int(((okA & okB)[E]).sum()),
        }
        print(f"SAW {mode}: J={res[mode.strip()]['J']} E_J={res[mode.strip()]['E_J']}",
              flush=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("DONE d04_vision_train")


if __name__ == "__main__":
    main()
