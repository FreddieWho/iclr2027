#!/usr/bin/env python3
"""N08 round1: 64x64 rendered segments, small CNN, pixel-space boundary test.

Model sees pixels only; oracle sees geometry (train/eval side only).
Falls back to minimal two-color rendering; no downloads, no big VLM.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import oracle_at

PACK = ROOT / "docs" / "iclr2027_discovery_campaign_20260917"
sys.path.insert(0, str(PACK))
from core.relations import make_relational_scenes  # noqa: E402

IMG = 64


def render(x, rng, lw=2):
    """Two-color segments on gray bg. x in ~[-1,1]^2 -> pixels."""
    img = np.full((IMG, IMG, 3), 0.5 + rng.uniform(-0.05, 0.05), np.float32)
    px = ((np.asarray(x, float) + 1) / 2 * (IMG - 1)).astype(int).clip(0, IMG - 1)
    cols = [(0.9, 0.1, 0.1)] * 2 + [(0.1, 0.1, 0.9)] * 2  # AB red, CD blue
    segs = [(px[0], px[1]), (px[2], px[3])]
    for s, (p, q) in enumerate(segs):
        n = int(np.hypot(*(q - p)) * 2) + 1
        for t in np.linspace(0, 1, max(n, 2)):
            r, c = int(round(p[0] + t * (q[0] - p[0]))), int(round(p[1] + t * (q[1] - p[1])))
            r0, r1 = max(0, r - lw), min(IMG, r + lw + 1)
            c0, c1 = max(0, c - lw), min(IMG, c + lw + 1)
            img[c0:c1, r0:r1] = cols[2 * s]
    return img.transpose(2, 0, 1)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 16, 5, 2), nn.ReLU(),
                               nn.Conv2d(16, 32, 5, 2), nn.ReLU(),
                               nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
                               nn.AdaptiveAvgPool2d((4, 4)),
                               nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU())
        self.c = nn.Linear(64, 1)

    def forward(self, x):
        return self.c(self.f(x)).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--n_scan", type=int, default=400)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    Xtr, ytr, _ = make_relational_scenes(a.n_train, seed=a.seed, min_margin=0.02)
    Xev, yev, _ = make_relational_scenes(a.n_eval, seed=a.seed + 800, min_margin=0.02)
    Rtr = np.stack([render(x, rng) for x in Xtr])
    Rev = np.stack([render(x, rng) for x in Xev])
    T = lambda A: torch.from_numpy(A.astype(np.float32))
    yt = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    torch.manual_seed(a.seed)
    net = SmallCNN()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Rtr))
        for i in range(0, len(Rtr), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = bce(net(T(Rtr[b])), yt[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        lge = net(T(Rev)).numpy()
    clean_acc = float((((lge > 0).astype(int)) == np.asarray(yev)).mean())
    # pixel-space boundary scan: move one endpoint stepwise, re-render
    # visibility gate: renders must differ by >=MINPX pixels (else the flip
    # is sub-pixel and missing it is correct behavior, not blindness)
    MINPX = 20
    miss = tot = invis = 0
    for i in rng.permutation(len(Xev))[:a.n_scan]:
        x = np.asarray(Xev[i], float)
        node = int(rng.integers(4))
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)])
        try:
            y0, _, _ = oracle_at(x)
        except ValueError:
            continue
        found = None
        r0 = render(x, rng)
        for r in np.linspace(0.02, 0.30, 15):
            e = np.zeros((4, 2))
            e[node] = r * d
            try:
                y1, m1, _ = oracle_at(x + e)
            except ValueError:
                continue
            if y1 != y0 and m1 >= 0.02:
                r1 = render(x + e, rng)
                ndiff = int((np.abs(r1 - r0).max(0) > 0.2).sum())
                if ndiff < MINPX:
                    invis += 1
                    continue
                found = (e, y1)
                break
        if found is None:
            continue
        e, y1 = found
        with torch.no_grad():
            lp = net(T(render(x + e, rng)[None])).numpy()[0]
        tot += 1
        miss += (int(lp > 0) != int(y1))
    res = {"clean_acc": clean_acc, "n_train": a.n_train, "n_eval": a.n_eval,
           "pixel_flip_n": tot, "invisible_skipped": invis,
           "pixel_miss_flip": (miss / tot) if tot else None}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print(f"SAW: clean_acc={clean_acc:.3f} pixel_miss|flip={res['pixel_miss_flip']} (n={tot}, invis_skipped={invis})")
    print("NEXT: if pixel miss|flip high with good clean acc -> visual turn "
          "course trial; if render pipeline jams -> minimal fallback already in use")
    print("CLAIM: observation-to-turn chain lives in pixels, not fitted rules")


if __name__ == "__main__":
    main()
