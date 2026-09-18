#!/usr/bin/env python3
"""B01 bridge pilot: frozen foundation backbone + linear probe on rendered quartets.

Renders the 4 quartet states (x, x+ea, x+eb, x+ea+eb) at backbone resolution
with nuisance-locked RNG (same fresh seed per state, cf. u03b_fixed), extracts
FROZEN features (dinov2-base or clip-vit-base-patch16 via transformers, CPU),
trains a linear probe on train-quartets renders, and evaluates on eval quartets.

Primary pilot readouts ONLY:
  - atomic accuracy on held-out states (can the probe read the parts?)
  - conditional compositional miss P(AB wrong | A,B correct) + denominators
Go/no-go: atomic>=0.90 on held-out -> expand to full matrix; else stop/redesign.
No backbone training. Probe training is allowed (linear only).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "last15h" / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "discovery_campaign"))
torch.set_num_threads(4)
from paths import oracle_at  # noqa: F401  (kept for parity; labels come from meta)

U01 = ROOT / "artifacts" / "next6_ef0f7a3" / "u01"

BACKBONES = {
    "dinov2": ("facebook/dinov2-base", "AutoModel", 224),
    "clip": ("openai/clip-vit-base-patch16", "CLIPVisionModel", 224),
}


def render_big(x, rng, img=224, lw=6, bg=0.5):
    out = np.full((img, img, 3), bg + rng.uniform(-0.05, 0.05), np.float32)
    px = ((np.asarray(x, float) + 1) / 2 * (img - 1)).astype(int).clip(0, img - 1)
    cols = [(0.9, 0.1, 0.1)] * 2 + [(0.1, 0.1, 0.9)] * 2
    for s, (p, q) in enumerate([(px[0], px[1]), (px[2], px[3])]):
        n = int(np.hypot(*(q - p)) * 2) + 1
        for t in np.linspace(0, 1, max(n, 2)):
            r = int(round(p[0] + t * (q[0] - p[0])))
            c = int(round(p[1] + t * (q[1] - p[1])))
            r0, r1 = max(0, r - lw), min(img, r + lw + 1)
            c0, c1 = max(0, c - lw), min(img, c + lw + 1)
            out[c0:c1, r0:r1] = cols[2 * s]
    return out  # HWC float32


def load_quartets(name):
    d = np.load(U01 / name)
    Q = [(d[f"q{i}"].reshape(3, 4, 2), d["meta"][i]) for i in range(len(d["meta"]))]
    return Q


def main():
    from transformers import AutoModel, CLIPVisionModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", choices=list(BACKBONES), required=True)
    p.add_argument("--n_eval", type=int, default=200,
                   help="eval quartets subset (pilot scale)")
    p.add_argument("--seed", type=int, default=909)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--scale", action="store_true",
                   help="StandardScaler before probe (same frozen feats)")
    a = p.parse_args()
    model_id, cls_name, img = BACKBONES[a.backbone]
    cls = {"AutoModel": AutoModel, "CLIPVisionModel": CLIPVisionModel}[cls_name]
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    # Manual preprocessing (avoid torchvision dependency): resize + normalize.
    NORMS = {
        "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "clip": ((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
    }
    mean, std = NORMS[a.backbone]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    def preprocess(imgs):
        arr = []
        for im in imgs:
            r = Image.fromarray((im * 255).astype(np.uint8)).resize(
                (img, img), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(r, dtype=np.float32) / 255.0)
            arr.append(t.permute(2, 0, 1))
        return (torch.stack(arr) - mean_t) / std_t

    print(f"LOAD {model_id}", flush=True)
    net = cls.from_pretrained(model_id).eval()

    QTR = load_quartets("quartets_train.npz")
    QEV = load_quartets("quartets_eval.npz")
    if a.n_eval < len(QEV):
        QEV = [QEV[i] for i in rng.permutation(len(QEV))[:a.n_eval]]

    def feats_of(qlist):
        """Render 4 states per quartet (nuisance-locked), return feats+labels."""
        imgs, labels, kinds, qis = [], [], [], []
        for qi, (q, m) in enumerate(qlist):
            x, ea, eb = q
            y0, ya, yb, yc = (int(v) for v in m)
            seed = 60000 + qi
            for st, yy, kind in ((x, y0, "base"), (x + ea, ya, "A"),
                                 (x + eb, yb, "B"), (x + ea + eb, yc, "AB")):
                imgs.append(render_big(st, np.random.default_rng(seed), img=img))
                labels.append(yy)
                kinds.append(kind)
                qis.append(qi)
        with torch.no_grad():
            batch = preprocess(imgs)
            h = net(pixel_values=batch)
            Z = (h.pooler_output if h.pooler_output is not None
                 else h.last_hidden_state[:, 0]).numpy()
        return Z, np.array(labels), np.array(kinds), np.array(qis)

    print("FEAT train", flush=True)
    Ztr, ytr, _, _ = feats_of(QTR)
    print("FEAT eval", flush=True)
    Zte, yte, kte, qte = feats_of(QEV)
    np.savez(a.out / "feats.npz", Ztr=Ztr, ytr=ytr, Zte=Zte, yte=yte,
             kte=kte, qte=qte)  # cache: probe sweeps must not re-extract
    if a.scale:
        sc = StandardScaler().fit(Ztr)
        Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)

    clf = LogisticRegression(max_iter=2000, C=a.C).fit(Ztr, ytr)
    pred = clf.predict(Zte)
    P = {k: pred[kte == k] for k in ("base", "A", "B", "AB")}
    Y = {k: yte[kte == k] for k in ("base", "A", "B", "AB")}
    atom = float(np.mean([P["A"] == Y["A"], P["B"] == Y["B"]]))
    both = (P["A"] == Y["A"]) & (P["B"] == Y["B"])
    cond_n = int(both.sum())
    cond_miss = float((P["AB"][both] != Y["AB"][both]).mean()) if cond_n else None
    # compositional-relevant subset: atomics right, labels yA==yB, AB flipped
    comp = both & (Y["A"] == Y["B"]) & (Y["AB"] != Y["A"])
    res = {
        "backbone": a.backbone, "model_id": model_id,
        "C": a.C, "scale": bool(a.scale),
        "n_train_quartets": len(QTR), "n_eval_quartets": len(QEV),
        "train_acc": float(clf.score(Ztr, ytr)),
        "atomic_acc": atom,
        "single_acc": {k: float((P[k] == Y[k]).mean()) for k in P},
        "cond_n": cond_n, "cond_miss": cond_miss,
        "comp_n": int(comp.sum()),
        "comp_miss": float((P["AB"][comp] != Y["AB"][comp]).mean()) if comp.sum() else None,
        "go": bool(atom >= 0.90),
    }
    json.dump(res, open(a.out / "result.json", "w"), indent=1)
    print("SAW", json.dumps(res), flush=True)
    print("DONE B01", "GO" if res["go"] else "NOGO")


if __name__ == "__main__":
    main()
